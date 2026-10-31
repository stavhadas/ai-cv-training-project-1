"""Tests for how `pcbi train` uses `src/pcbi/train/seeding.py`.

The train loader is built the way `run_training` builds it: `make_dataloader` with the generator
from `make_generator`, shuffled, and with real worker processes.
"""

import csv

from PIL import Image

from pcbi.data import datasets as datasets_mod
from pcbi.data import transforms as transforms_mod
from pcbi.data.datasets import CLASS_ORDER
from pcbi.train.seeding import make_generator

EVAL_CONFIG = {
    "input_size": (3, 32, 32),
    "interpolation": "bilinear",
    "mean": (0.0, 0.0, 0.0),
    "std": (1.0, 1.0, 1.0),
}


def write_crops(tmp_path, per_class=4):
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    rows = []
    for label_index, label in enumerate(CLASS_ORDER):
        for k in range(per_class):
            crop_id = f"train_{label}_{k}"
            Image.new("RGB", (32, 32), (label_index * 60, k * 40, 0)).save(
                crops_dir / f"{crop_id}.png"
            )
            rows.append({"crop_id": crop_id, "label": label, "split": "train"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest, crops_dir


def batch_order(dataset, seed):
    loader = datasets_mod.make_dataloader(
        dataset,
        {"batch_size": 4, "num_workers": 2},
        shuffle=True,
        generator=make_generator(seed),
    )
    return [list(batch.crop_id) for batch in loader]


def test_the_same_seed_gives_the_same_batch_order_with_two_workers(tmp_path):
    manifest, crops_dir = write_crops(tmp_path)
    transform = transforms_mod.build_eval_transform(EVAL_CONFIG)
    dataset = datasets_mod.CropDataset(manifest, crops_dir, "train", transform)

    first = batch_order(dataset, seed=0)
    assert first == batch_order(dataset, seed=0)
    # Guards against a vacuous pass: if the loader were not shuffling, every seed would agree.
    assert first != batch_order(dataset, seed=1)
