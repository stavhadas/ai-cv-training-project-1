"""Tests for `src/pcbi/train/run.py` and the sampler hook it added to `make_dataloader`.

A real `run_training` call end to end, on CPU: a dozen tiny PNGs, a small timm model built without
pretrained weights (no network), W&B disabled. Slow-ish (a few seconds), but it is the only test
that proves the pieces actually fit together.
"""

import csv

import pytest
import torch
from PIL import Image
from torch.utils.data import SequentialSampler

from pcbi.data import datasets as datasets_mod
from pcbi.data.datasets import CLASS_ORDER
from pcbi.train import run as run_mod
from pcbi.train.config import ImbalanceConfig, TrainConfig

TINY_MODEL = "resnet18"  # built with pretrained=False: random weights, no download


@pytest.fixture
def crops(tmp_path):
    """Three train and two val crops per class, each a flat colour so classes are separable."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    rows = []
    for split, per_class in (("train", 3), ("val", 2)):
        for label_index, label in enumerate(CLASS_ORDER):
            for k in range(per_class):
                crop_id = f"{split}_{label}_{k}"
                value = 30 + label_index * 60
                Image.new("RGB", (32, 32), (value, 255 - value, value)).save(
                    crops_dir / f"{crop_id}.png"
                )
                rows.append({"crop_id": crop_id, "label": label, "split": split})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest, crops_dir


def tiny_config(**overrides) -> TrainConfig:
    values = {"model": TINY_MODEL, "epochs": 2, "batch_size": 4, "num_workers": 0}
    values.update(overrides)
    return TrainConfig(**values)


def train(crops, tmp_path, cfg=None, **kwargs):
    manifest, crops_dir = crops
    return run_mod.run_training(
        cfg or tiny_config(),
        manifest,
        crops_dir,
        tmp_path / "outputs",
        use_wandb=False,
        pretrained=False,
        **kwargs,
    )


def test_a_short_run_writes_the_report_and_the_confusion_matrix(crops, tmp_path, capsys):
    run_dir = train(crops, tmp_path, max_steps=2)

    assert (run_dir / "confusion_matrix.png").is_file()
    assert (run_dir / "classification_report.txt").is_file()
    assert (run_dir / "config.yaml").is_file()
    out = capsys.readouterr().out
    assert "--max-steps 2 reached; stopping after epoch 1." in out
    for name in CLASS_ORDER:
        assert name in out


def test_max_steps_caps_the_optimizer_steps_across_epochs(crops, tmp_path, monkeypatch):
    steps = []
    real_step = torch.optim.SGD.step

    def counting_step(self, *args, **kwargs):
        steps.append(1)
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", counting_step)
    # 12 train crops at batch 4 is 3 steps an epoch; a budget of 5 must cross an epoch boundary.
    train(crops, tmp_path, cfg=tiny_config(epochs=5), max_steps=5)
    assert len(steps) == 5


def test_overfit_batch_evaluates_on_the_same_crops_it_trains_on(crops, tmp_path, capsys):
    train(crops, tmp_path, overfit_batch=4)
    out = capsys.readouterr().out
    assert "train 4 | overfit-4 4" in out
    assert "classification report (overfit-4, 4 crops)" in out


@pytest.mark.parametrize("mode", ["class_weighted_loss", "weighted_sampler"])
def test_each_imbalance_mode_trains(crops, tmp_path, mode):
    cfg = tiny_config(epochs=1, imbalance=ImbalanceConfig(mode=mode))
    assert (train(crops, tmp_path, cfg=cfg) / "confusion_matrix.png").is_file()


def test_format_confusion_labels_rows_and_columns():
    matrix = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    lines = run_mod.format_confusion(torch.tensor(matrix).numpy()).splitlines()
    assert lines[0].split()[-4:] == list(CLASS_ORDER)
    assert lines[1].split() == ["normal", "1", "0", "0", "0"]


def test_make_dataloader_refuses_shuffle_together_with_a_sampler(crops):
    dataset = datasets_mod.CropDataset(*crops, split="train")
    with pytest.raises(ValueError, match="not both"):
        datasets_mod.make_dataloader(dataset, shuffle=True, sampler=SequentialSampler(dataset))


def test_make_dataloader_uses_the_given_sampler(crops):
    dataset = datasets_mod.CropDataset(*crops, split="train")
    sampler = SequentialSampler(dataset)
    assert datasets_mod.make_dataloader(dataset, sampler=sampler).sampler is sampler
