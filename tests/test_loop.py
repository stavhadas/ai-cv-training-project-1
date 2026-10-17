"""Tests for `src/pcbi/train/loop.py`.

`RunningMean` and the result dataclasses are fully written, so their tests pass immediately. The
tests of `train_one_epoch` and `validate` fail with `NotImplementedError` until those two bodies are
filled in — that failure is the intended state, not a bug in the tests.

Everything runs on CPU, on a tiny synthetic dataset and a two-layer model: no manifest, no PNGs, no
timm, no network.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pcbi.data.datasets import Sample
from pcbi.train.loop import RunningMean, TrainEpochResult, ValResult, train_one_epoch, validate

NUM_CLASSES = 4
IMAGE_SHAPE = (3, 4, 4)
DEVICE = "cpu"


class SyntheticDataset(Dataset):
    """`n` random tiny images with labels and crop ids, as the real dataset's `Sample`."""

    def __init__(self, n: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, *IMAGE_SHAPE, generator=generator)
        self.labels = [i % NUM_CLASSES for i in range(n)]
        self.crop_ids = [f"crop_{i:03d}" for i in range(n)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return Sample(self.images[idx], self.labels[idx], self.crop_ids[idx])


def make_model(seed: int = 0) -> nn.Module:
    """A two-layer model: flatten the tiny image, one hidden layer, 4 logits."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(int(np.prod(IMAGE_SHAPE)), 16),
        nn.ReLU(),
        nn.Linear(16, NUM_CLASSES),
    )


def make_loader(n: int, batch_size: int) -> DataLoader:
    return DataLoader(SyntheticDataset(n), batch_size=batch_size, shuffle=False)


# --- RunningMean and the result dataclasses (no gap: these pass immediately) ----------------------


def test_running_mean_weights_each_batch_by_its_sample_count():
    mean = RunningMean()
    mean.update(1.0, n=3)  # three samples averaging 1.0
    mean.update(5.0, n=1)  # one sample of 5.0
    assert mean.mean == pytest.approx((3 * 1.0 + 1 * 5.0) / 4)


def test_running_mean_is_not_the_mean_of_the_batch_means():
    mean = RunningMean()
    mean.update(0.0, n=9)
    mean.update(10.0, n=1)
    assert mean.mean == pytest.approx(1.0)  # not (0 + 10) / 2


def test_running_mean_with_nothing_added_raises():
    with pytest.raises(ValueError, match="no samples"):
        _ = RunningMean().mean


def test_result_dataclasses_hold_what_they_say():
    train = TrainEpochResult(mean_loss=1.5, samples_seen=10, seconds=0.2)
    assert (train.mean_loss, train.samples_seen, train.seconds) == (1.5, 10, 0.2)

    val = ValResult(
        mean_loss=1.0, logits=np.zeros((2, 4)), labels=np.array([0, 1]), crop_ids=["a", "b"]
    )
    assert val.logits.shape == (2, 4)
    assert val.crop_ids == ["a", "b"]


# --- train_one_epoch (the first gap) -------------------------------------------------------------


def test_one_training_epoch_runs_and_reports_what_it_saw():
    model = make_model()
    loader = make_loader(n=10, batch_size=4)  # batches of 4, 4, 2
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), DEVICE)

    assert isinstance(result, TrainEpochResult)
    assert result.samples_seen == 10
    assert np.isfinite(result.mean_loss)
    assert result.seconds >= 0


def test_training_changes_the_weights():
    model = make_model()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_one_epoch(model, make_loader(8, 4), optimizer, nn.CrossEntropyLoss(), DEVICE)

    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))


def test_loss_decreases_when_the_same_small_batch_is_trained_repeatedly():
    model = make_model()
    loader = make_loader(n=8, batch_size=8)  # one batch, the same one every epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    losses = [
        train_one_epoch(model, loader, optimizer, criterion, DEVICE).mean_loss for _ in range(30)
    ]

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]  # memorising 8 samples should be far from a marginal drop


def test_train_one_epoch_leaves_the_model_in_training_mode():
    """Even when validate (eval mode) ran just before — the second epoch must train again."""
    model = make_model()
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_one_epoch(model, make_loader(8, 4), optimizer, nn.CrossEntropyLoss(), DEVICE)

    assert model.training


def test_on_step_is_called_once_per_batch_with_a_plain_float_loss():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls: list[tuple[int, float]] = []

    train_one_epoch(
        model,
        make_loader(n=10, batch_size=4),  # 3 batches
        optimizer,
        nn.CrossEntropyLoss(),
        DEVICE,
        on_step=lambda step, loss: calls.append((step, loss)),
    )

    assert [step for step, _ in calls] == [0, 1, 2]
    assert all(type(loss) is float for _, loss in calls)


def test_the_training_mean_loss_weights_the_smaller_final_batch_correctly():
    """With lr=0 the weights never change, so the epoch's mean loss must equal the loss of the
    whole dataset in one go — and only a sample-weighted mean gets that right when batches are
    4, 4 and 2 (a mean of the three batch means would not)."""
    model = make_model()
    dataset = SyntheticDataset(10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    with torch.inference_mode():
        expected = criterion(model(dataset.images), torch.tensor(dataset.labels)).item()

    result = train_one_epoch(model, make_loader(10, 4), optimizer, criterion, DEVICE)

    assert result.mean_loss == pytest.approx(expected, rel=1e-5)


# --- validate (the second gap) -------------------------------------------------------------------


def test_validate_returns_one_logits_row_per_sample_in_loader_order():
    model = make_model()
    dataset = SyntheticDataset(10)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)  # batches of 4, 4, 2

    result = validate(model, loader, nn.CrossEntropyLoss(), DEVICE)

    assert isinstance(result, ValResult)
    assert result.logits.shape == (10, NUM_CLASSES)
    assert result.labels.tolist() == dataset.labels
    assert result.crop_ids == dataset.crop_ids

    model.eval()
    with torch.inference_mode():
        expected = model(dataset.images).numpy()
    np.testing.assert_allclose(result.logits, expected, rtol=1e-5, atol=1e-6)


def test_validate_mean_loss_does_not_depend_on_how_the_samples_are_batched():
    model = make_model()
    criterion = nn.CrossEntropyLoss()

    ragged = validate(model, make_loader(10, 4), criterion, DEVICE)  # 4, 4, 2
    whole = validate(model, make_loader(10, 10), criterion, DEVICE)  # 10

    assert ragged.mean_loss == pytest.approx(whole.mean_loss, rel=1e-5)


def test_validate_does_not_change_the_weights_or_record_gradients():
    model = make_model()
    before = [p.detach().clone() for p in model.parameters()]

    validate(model, make_loader(8, 4), nn.CrossEntropyLoss(), DEVICE)

    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
    assert all(p.grad is None for p in model.parameters())


def test_validate_puts_the_model_in_eval_mode():
    model = make_model()
    model.train()

    validate(model, make_loader(8, 4), nn.CrossEntropyLoss(), DEVICE)

    assert not model.training
