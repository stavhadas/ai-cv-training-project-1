"""Tests for `src/pcbi/train/imbalance.py`.

Two functions are deliberately unwritten — `inverse_freq_weights` and `effective_number_weights`.
Everything that can be tested without them is isolated from them by monkeypatching `get_weights`
(or, for `class_counts_from_manifest`, needs neither): those tests pass immediately and check the
*mechanics* — mode dispatch, the "never touch the other mechanism" rule, per-class weights turning
into correct per-sample weights and a correct sampler. The direct tests on the two gapped functions,
and two short end-to-end tests with no monkeypatching, fail with `NotImplementedError` until they
are filled in — that failure is the intended state, not a bug in the tests.
"""

import numpy as np
import pytest
import torch
from torch import nn

from pcbi.data.datasets import CLASS_ORDER
from pcbi.train import imbalance as imbalance_mod


class FakeDataset:
    """The minimum `build_sampler` needs — no manifest, no crop directory, no PNGs on disk."""

    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)


def row(crop_id: str, label: str, split: str) -> dict[str, str]:
    return {"crop_id": crop_id, "label": label, "split": split}


# --- class_counts_from_manifest (no gap) --------------------------------------------------------


def test_class_order_is_the_one_this_module_assumes():
    """Everything below is written against this exact order; pin it so a drift is loud."""
    assert tuple(CLASS_ORDER) == ("normal", "excess", "insufficient", "spike")


def test_counts_only_the_requested_split():
    rows = [
        row("a", "normal", "train"),
        row("b", "excess", "train"),
        row("c", "excess", "train"),
        row("d", "spike", "val"),
    ]
    counts = imbalance_mod.class_counts_from_manifest(rows, split="train")
    assert counts.tolist() == [1, 2, 0, 0]  # normal, excess, insufficient, spike


def test_a_class_absent_from_the_split_is_a_genuine_zero_not_a_missing_entry():
    rows = [row("a", "spike", "train")]
    counts = imbalance_mod.class_counts_from_manifest(rows, split="train")
    assert counts.shape == (4,)
    assert counts.tolist() == [0, 0, 0, 1]


def test_an_unknown_label_raises_and_names_it():
    rows = [row("a", "bridging", "train")]
    with pytest.raises(ValueError, match="bridging"):
        imbalance_mod.class_counts_from_manifest(rows, split="train")


def test_a_split_with_no_matching_rows_raises():
    rows = [row("a", "normal", "train")]
    with pytest.raises(ValueError, match="split='val'"):
        imbalance_mod.class_counts_from_manifest(rows, split="val")


# --- get_weights dispatch (no gap: the two real functions are stubbed) ---------------------------


def test_get_weights_dispatches_inverse_freq_with_just_the_counts(monkeypatch):
    seen = {}

    def stub(counts):
        seen["counts"] = counts
        return np.array([9.0])

    monkeypatch.setattr(imbalance_mod, "inverse_freq_weights", stub)
    result = imbalance_mod.get_weights("inverse_freq", np.array([1, 2, 3]), beta=0.9)
    assert np.array_equal(seen["counts"], np.array([1, 2, 3]))
    assert np.array_equal(result, np.array([9.0]))


def test_get_weights_dispatches_effective_number_with_counts_and_beta(monkeypatch):
    seen = {}

    def stub(counts, beta):
        seen["counts"], seen["beta"] = counts, beta
        return np.array([7.0])

    monkeypatch.setattr(imbalance_mod, "effective_number_weights", stub)
    result = imbalance_mod.get_weights("effective_number", np.array([4, 5, 6]), beta=0.99)
    assert seen["beta"] == 0.99
    assert np.array_equal(result, np.array([7.0]))


def test_get_weights_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="weight_scheme"):
        imbalance_mod.get_weights("made_up", np.array([1, 2]), beta=0.9)


# --- build_criterion, structural (get_weights stubbed where it applies) --------------------------


def test_mode_none_leaves_the_criterion_unweighted():
    criterion = imbalance_mod.build_criterion({"mode": "none"}, np.array([1, 2, 3, 4]))
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is None


def test_an_omitted_mode_defaults_to_none():
    criterion = imbalance_mod.build_criterion({}, np.array([1, 2, 3, 4]))
    assert criterion.weight is None


def test_weighted_sampler_mode_leaves_the_criterion_unweighted():
    """That mode corrects the batches, not the loss — combining both would double-correct."""
    criterion = imbalance_mod.build_criterion({"mode": "weighted_sampler"}, np.array([1, 2, 3, 4]))
    assert criterion.weight is None


def test_mode_none_never_calls_get_weights_for_the_criterion(monkeypatch):
    """ "unweighted" must mean the weighting machinery was never invoked, not just discarded."""

    def explode(*args, **kwargs):
        raise AssertionError("get_weights must not be called when mode='none'")

    monkeypatch.setattr(imbalance_mod, "get_weights", explode)
    imbalance_mod.build_criterion({"mode": "none"}, np.array([1, 2, 3, 4]))


def test_class_weighted_loss_passes_the_right_arguments_to_get_weights(monkeypatch):
    stub_weights = np.array([1.0, 2.0, 0.5, 4.0])
    seen = {}

    def stub(scheme, counts, beta):
        seen["scheme"], seen["counts"], seen["beta"] = scheme, counts, beta
        return stub_weights

    monkeypatch.setattr(imbalance_mod, "get_weights", stub)
    counts = np.array([10, 20, 30, 40])
    criterion = imbalance_mod.build_criterion(
        {"mode": "class_weighted_loss", "weight_scheme": "effective_number", "beta": 0.95},
        counts,
    )

    assert seen["scheme"] == "effective_number"
    assert seen["beta"] == 0.95
    assert np.array_equal(seen["counts"], counts)
    assert torch.allclose(criterion.weight, torch.as_tensor(stub_weights, dtype=torch.float32))


def test_class_weighted_loss_uses_the_default_scheme_and_beta_when_omitted(monkeypatch):
    seen = {}

    def stub(scheme, counts, beta):
        seen["scheme"], seen["beta"] = scheme, beta
        return np.array([1.0, 1.0, 1.0, 1.0])

    monkeypatch.setattr(imbalance_mod, "get_weights", stub)
    imbalance_mod.build_criterion({"mode": "class_weighted_loss"}, np.array([1, 2, 3, 4]))

    assert seen["scheme"] == imbalance_mod.DEFAULT_WEIGHT_SCHEME
    assert seen["beta"] == imbalance_mod.DEFAULT_BETA


def test_the_weight_tensor_lands_on_the_requested_device(monkeypatch):
    monkeypatch.setattr(
        imbalance_mod, "get_weights", lambda s, c, b: np.array([1.0, 1.0, 1.0, 1.0])
    )
    criterion = imbalance_mod.build_criterion(
        {"mode": "class_weighted_loss"}, np.array([1, 2, 3, 4]), device="cpu"
    )
    assert criterion.weight.device == torch.device("cpu")
    assert criterion.weight.dtype == torch.float32


def test_build_criterion_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        imbalance_mod.build_criterion({"mode": "bogus"}, np.array([1, 2, 3, 4]))


# --- build_sampler, structural (get_weights stubbed) ----------------------------------------------


def test_mode_none_returns_no_sampler():
    sampler = imbalance_mod.build_sampler(
        {"mode": "none"}, FakeDataset([0, 1, 2, 3]), np.array([1, 1, 1, 1])
    )
    assert sampler is None


def test_class_weighted_loss_mode_returns_no_sampler():
    """That mode corrects the loss, not the batches — the sampler stays untouched."""
    sampler = imbalance_mod.build_sampler(
        {"mode": "class_weighted_loss"}, FakeDataset([0, 1, 2, 3]), np.array([1, 1, 1, 1])
    )
    assert sampler is None


def test_mode_none_never_calls_get_weights_for_the_sampler(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("get_weights must not be called when mode='none'")

    monkeypatch.setattr(imbalance_mod, "get_weights", explode)
    imbalance_mod.build_sampler({"mode": "none"}, FakeDataset([0, 1]), np.array([1, 1, 1, 1]))


def test_build_sampler_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        imbalance_mod.build_sampler({"mode": "bogus"}, FakeDataset([0]), np.array([1, 1, 1, 1]))


def test_the_sampler_is_built_with_replacement_and_one_epoch_per_pass(monkeypatch):
    monkeypatch.setattr(
        imbalance_mod, "get_weights", lambda s, c, b: np.array([1.0, 1.0, 1.0, 1.0])
    )
    dataset = FakeDataset([0, 1, 2, 3, 0, 1])
    sampler = imbalance_mod.build_sampler(
        {"mode": "weighted_sampler"}, dataset, np.array([2, 2, 1, 1])
    )
    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
    assert sampler.replacement is True
    assert sampler.num_samples == len(dataset)


def test_sampler_reproduces_intended_class_proportions_over_many_epochs(monkeypatch):
    """The weighted-sampler mechanics, isolated from whichever weight scheme is configured.

    `get_weights` is stubbed with a fixed, made-up weight vector, so this checks exactly what
    `build_sampler` is responsible for — turning a per-class weight into a per-sample weight and
    handing it to `WeightedRandomSampler` correctly — without depending on `inverse_freq_weights`
    or `effective_number_weights` being implemented.

    With per-sample weight `w_c` constant within class `c`, the probability of drawing class `c` on
    any one draw is `(count_c * w_c) / sum_j(count_j * w_j)`. 20 epochs of a 100-example dataset is
    2000 draws, which keeps the sampling noise on each class's empirical share under ~0.01 — well
    inside the tolerance used here.
    """
    counts = np.array([50, 30, 15, 5])
    class_weights = np.array([1.0, 2.0, 4.0, 8.0])
    monkeypatch.setattr(imbalance_mod, "get_weights", lambda s, c, b: class_weights)

    labels = [class_index for class_index, n in enumerate(counts) for _ in range(n)]
    dataset = FakeDataset(labels)
    sampler = imbalance_mod.build_sampler({"mode": "weighted_sampler"}, dataset, counts)

    weighted = counts * class_weights
    target = weighted / weighted.sum()

    torch.manual_seed(0)
    tally = np.zeros(len(counts))
    for _ in range(20):
        for idx in sampler:
            tally[labels[idx]] += 1

    empirical = tally / tally.sum()
    assert np.allclose(empirical, target, atol=0.05)


# --- inverse_freq_weights, the first gap ----------------------------------------------------------


def test_inverse_freq_weights_sum_to_the_number_of_classes():
    counts = np.array([10, 20, 30, 40])
    weights = imbalance_mod.inverse_freq_weights(counts)
    assert weights.shape == counts.shape
    assert np.isclose(weights.sum(), len(counts))


def test_inverse_freq_weights_are_all_equal_to_one_on_balanced_counts():
    weights = imbalance_mod.inverse_freq_weights(np.array([25, 25, 25, 25]))
    assert np.allclose(weights, 1.0)


def test_inverse_freq_weights_are_larger_for_rarer_classes():
    # Strictly decreasing counts must produce strictly increasing weights.
    weights = imbalance_mod.inverse_freq_weights(np.array([100, 50, 20, 5]))
    assert np.all(np.diff(weights) > 0)


# --- effective_number_weights, the second gap -----------------------------------------------------


def test_effective_number_weights_sum_to_the_number_of_classes():
    counts = np.array([10, 20, 30, 40])
    weights = imbalance_mod.effective_number_weights(counts, beta=0.99)
    assert weights.shape == counts.shape
    assert np.isclose(weights.sum(), len(counts))


def test_effective_number_weights_are_all_equal_to_one_on_balanced_counts():
    weights = imbalance_mod.effective_number_weights(np.array([25, 25, 25, 25]), beta=0.99)
    assert np.allclose(weights, 1.0)


def test_effective_number_weights_are_larger_for_rarer_classes():
    weights = imbalance_mod.effective_number_weights(np.array([100, 50, 20, 5]), beta=0.99)
    assert np.all(np.diff(weights) > 0)


def test_effective_number_at_beta_zero_is_uniform_regardless_of_imbalance():
    """The documented degenerate case: at beta=0 every class's effective number is defined as 1,
    so even a heavily imbalanced count vector must produce all-equal weights — unlike
    `inverse_freq_weights`, where only a *balanced* input gives that result.
    """
    weights = imbalance_mod.effective_number_weights(np.array([1000, 500, 10, 1]), beta=0.0)
    assert np.allclose(weights, 1.0)


# --- true end-to-end: no monkeypatching, both gaps have to be filled in ---------------------------


def test_class_weighted_loss_end_to_end_with_the_real_weight_function():
    counts = np.array([100, 50, 25, 25])
    criterion = imbalance_mod.build_criterion(
        {"mode": "class_weighted_loss", "weight_scheme": "inverse_freq"}, counts
    )
    assert criterion.weight is not None
    assert torch.isclose(criterion.weight.sum(), torch.tensor(4.0), atol=1e-4)


def test_weighted_sampler_end_to_end_with_the_real_weight_function():
    counts = np.array([50, 30, 15, 5])
    labels = [class_index for class_index, n in enumerate(counts) for _ in range(n)]
    sampler = imbalance_mod.build_sampler(
        {"mode": "weighted_sampler", "weight_scheme": "inverse_freq"}, FakeDataset(labels), counts
    )
    assert sampler is not None
    assert sampler.num_samples == len(labels)
