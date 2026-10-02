"""Tests for the Stage 2 Dataset and transform layer.

Two bodies are deliberately unwritten — `CropDataset.__getitem__` and
`transforms.build_eval_transform`. Until they are filled in, the tests that exercise them fail with
`NotImplementedError` from the gap itself, not with some downstream `AttributeError` that hides
which piece is missing. The fixtures build no tensors of their own for exactly that reason: nothing
here can fail *before* reaching the gap.

Everything that does not depend on those two bodies — the manifest reading, the label validation,
the loader construction — passes immediately. So the suite tells you which half is done.

Nothing here downloads a model. `fake_config` is a hand-written stand-in for
`timm.data.resolve_data_config(model=...)`, which keeps CI off the network and lets the
normalization test pick values (mean 0, std 1) that make a hard-coded ImageNet triple visible.
"""

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from pcbi.data import datasets as datasets_mod
from pcbi.data import transforms as transforms_mod

CROP_SIDE = 300  # square, as `pcbi make-crops` writes them
INPUT_SIDE = 224

# What timm hands back for a resnet18. Pinned here rather than resolved from a model so the tests
# neither download weights nor change behaviour when timm's defaults move.
IMAGENET_CONFIG = {
    "input_size": (3, INPUT_SIDE, INPUT_SIDE),
    "interpolation": "bicubic",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

# mean 0 / std 1 is the identity normalization, so anything built from this config leaves pixel
# values in [0, 1]. That is what makes a hard-coded ImageNet mean detectable.
FAKE_CONFIG = {
    "input_size": (3, INPUT_SIDE, INPUT_SIDE),
    "interpolation": "bilinear",
    "mean": (0.0, 0.0, 0.0),
    "std": (1.0, 1.0, 1.0),
}

MANIFEST_FIELDS = ["crop_id", "label", "split", "group_id", "board", "package", "view_folder"]

# Deliberately not in CLASS_ORDER order, so a test that passes only because the fixture happened to
# be sorted the same way would fail.
ROWS = [
    ("CS1_R0805_V2_a_1", "spike", "train"),
    ("CS1_R0805_V2_a_2", "normal", "train"),
    ("CS1_R0805_V2_b_1", "insufficient", "val"),
    ("CS2_R0603_V2_c_1", "excess", "val"),
    ("CS2_R0603_V2_c_2", "normal", "val"),
    ("CS6_C0805_V2_d_1", "excess", "test"),
]


def write_manifest(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for index, (crop_id, label, split) in enumerate(rows):
            writer.writerow(
                {
                    "crop_id": crop_id,
                    "label": label,
                    "split": split,
                    "group_id": f"manual_{index:03d}",
                    "board": crop_id.split("_")[0],
                    "package": crop_id.split("_")[1],
                    "view_folder": "V2",
                }
            )


@pytest.fixture
def crops(tmp_path):
    """A manifest plus one square PNG per row, the shape `pcbi make-crops` produces."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    for index, (crop_id, _, _) in enumerate(ROWS):
        # A flat mid-grey per crop, distinct per row. Flat because these tests assert ranges and
        # ordering, and a constant image makes an expected value something you can work out by hand.
        value = 40 + index * 30
        Image.new("RGB", (CROP_SIDE, CROP_SIDE), (value, value, value)).save(
            crops_dir / f"{crop_id}.png"
        )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ROWS)
    return manifest, crops_dir


def build_dataset(crops, split="train", config=FAKE_CONFIG):
    manifest, crops_dir = crops
    return datasets_mod.CropDataset(
        manifest,
        crops_dir,
        split=split,
        transform=transforms_mod.build_eval_transform(config),
    )


# --- the manifest, the split and the labels (no gaps involved) ---------------------------------


def test_the_dataset_holds_only_its_own_split(crops):
    manifest, crops_dir = crops
    for split, expected in [("train", 2), ("val", 3), ("test", 1)]:
        dataset = datasets_mod.CropDataset(manifest, crops_dir, split=split)
        assert len(dataset) == expected


def test_rows_are_kept_in_manifest_order(crops):
    """An unshuffled loader is only reproducible if the underlying order is."""
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="val")
    assert dataset.crop_ids == [c for c, _, s in ROWS if s == "val"]


def test_labels_are_class_order_indices(crops):
    """The index is permanent, so it is pinned against CLASS_ORDER rather than against a literal."""
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="val")
    expected = [datasets_mod.CLASS_ORDER.index(label) for _, label, s in ROWS if s == "val"]
    assert dataset.labels == expected
    assert all(isinstance(label, int) for label in dataset.labels)
    assert all(0 <= label < len(datasets_mod.CLASS_ORDER) for label in dataset.labels)


def test_class_order_is_the_documented_one():
    """Reordering this silently relabels every existing checkpoint, so it is pinned."""
    assert datasets_mod.CLASS_ORDER == ("normal", "excess", "insufficient", "spike")
    assert datasets_mod.CLASS_TO_INDEX["normal"] == 0


def test_a_row_with_an_unknown_label_raises(tmp_path):
    """Caught while building the dataset, not at epoch 3 batch 700."""
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [("CS1_R0805_V2_a_1", "bridging", "train")])
    with pytest.raises(ValueError, match="not in CLASS_ORDER"):
        datasets_mod.CropDataset(manifest, tmp_path / "crops", split="train")


def test_the_offending_label_and_crop_are_named(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [("CS1_R0805_V2_a_1", "bridging", "train")])
    with pytest.raises(ValueError) as caught:
        datasets_mod.CropDataset(manifest, tmp_path / "crops", split="train")
    assert "bridging" in str(caught.value)
    assert "CS1_R0805_V2_a_1" in str(caught.value)


def test_an_empty_split_raises(crops):
    """Silently empty is worse than loud: it surfaces as an unexplainable training curve."""
    manifest, crops_dir = crops
    write_manifest(manifest, [(c, lab, "train") for c, lab, _ in ROWS])
    with pytest.raises(ValueError, match="no rows with split='val'"):
        datasets_mod.CropDataset(manifest, crops_dir, split="val")


def test_paths_are_derived_from_the_crop_id(crops):
    """There is no crop_path column; the id is the stem and the directory is a runtime argument."""
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="val")
    assert dataset.paths[0] == crops_dir / f"{dataset.crop_ids[0]}.png"
    assert all(path.is_file() for path in dataset.paths)


# --- the dataloader (no gaps involved) ---------------------------------------------------------


def test_the_loader_never_drops_a_partial_batch(crops):
    """Dropping the tail would silently shrink the denominator of every val and test metric."""
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="val")
    loader = datasets_mod.make_dataloader(dataset, {"batch_size": 2}, seed=0, shuffle=False)
    assert loader.drop_last is False
    assert sum(len(batch) for batch in loader.batch_sampler) == len(dataset)


def test_the_loader_is_seeded_from_the_given_seed(crops):
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="train")
    loader = datasets_mod.make_dataloader(dataset, seed=7, shuffle=True)
    assert loader.generator.initial_seed() == 7


def test_shuffling_is_off_by_default(crops):
    """val and test must yield one fixed order; only the train loader opts into shuffling."""
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="val")
    loader = datasets_mod.make_dataloader(dataset, seed=0)
    assert not isinstance(loader.sampler, torch.utils.data.RandomSampler)


# --- the transform config (no gaps involved) ---------------------------------------------------


def test_input_size_comes_from_the_config():
    assert transforms_mod.input_size(FAKE_CONFIG) == (INPUT_SIDE, INPUT_SIDE)
    assert transforms_mod.input_size({"input_size": (3, 384, 384)}) == (384, 384)


def test_interpolation_is_mapped_from_timms_string():
    assert transforms_mod.interpolation_mode({"interpolation": "bicubic"}) is (
        transforms_mod.v2.InterpolationMode.BICUBIC
    )
    assert transforms_mod.interpolation_mode({}) is transforms_mod.v2.InterpolationMode.BICUBIC


def test_an_unknown_interpolation_is_refused():
    with pytest.raises(ValueError, match="unknown interpolation"):
        transforms_mod.interpolation_mode({"interpolation": "magic"})


def test_augmentation_is_refused_until_stage_3():
    """The train pipeline is eval plus shuffling for now, and says so rather than ignoring it."""
    with pytest.raises(NotImplementedError, match="Stage 3"):
        transforms_mod.build_train_transform(FAKE_CONFIG, {"hflip": 0.5})


# --- the two gapped bodies ---------------------------------------------------------------------
#
# Everything below fails with NotImplementedError until the gaps are filled. The first test needs
# only `__getitem__`; the rest go through `build_eval_transform` as well. So the two gaps surface
# independently, and the suite says which one is still open rather than only the earlier one.


def test_getitem_returns_a_sample_even_with_no_transform(crops):
    """The `__getitem__` gap on its own, with no transform in the way.

    A dataset built without a transform still has to load the file and return a `Sample` carrying
    the label and the crop_id. Only the image's dtype and size are the transform's business.
    """
    manifest, crops_dir = crops
    dataset = datasets_mod.CropDataset(manifest, crops_dir, split="val", transform=None)
    sample = dataset[0]

    assert isinstance(sample, datasets_mod.Sample)
    assert sample.crop_id == "CS1_R0805_V2_b_1"
    assert sample.label == datasets_mod.CLASS_ORDER.index("insufficient")


def test_an_item_is_a_float32_chw_tensor_at_the_configured_size(crops):
    """Shape and dtype are the contract every downstream layer assumes without checking."""
    dataset = build_dataset(crops, split="val")
    sample = dataset[0]

    assert sample.image.shape == (3, INPUT_SIDE, INPUT_SIDE)
    assert sample.image.dtype == torch.float32


def test_an_item_carries_its_label_and_crop_id(crops):
    """crop_id riding along is what makes a wrong prediction traceable to a file on disk."""
    dataset = build_dataset(crops, split="val")
    sample = dataset[0]

    assert isinstance(sample.label, int)
    assert sample.label == datasets_mod.CLASS_ORDER.index("insufficient")
    assert sample.crop_id == "CS1_R0805_V2_b_1"


def test_every_label_in_a_split_is_an_int_in_range(crops):
    dataset = build_dataset(crops, split="val")
    labels = [dataset[i].label for i in range(len(dataset))]
    assert all(isinstance(label, int) for label in labels)
    assert all(0 <= label <= 3 for label in labels)
    assert labels == [datasets_mod.CLASS_ORDER.index(lab) for _, lab, s in ROWS if s == "val"]


def test_the_val_loader_yields_crop_ids_in_manifest_order_every_run(crops):
    """The property that makes two val scores comparable at all.

    Run twice, from scratch, and assert both the order and its agreement with the manifest. A
    loader that merely happens to be stable within one process is not enough — the comparison being
    protected here is against a number recorded on another day.
    """
    expected = [crop_id for crop_id, _, split in ROWS if split == "val"]

    seen = []
    for _ in range(2):
        dataset = build_dataset(crops, split="val")
        loader = datasets_mod.make_dataloader(dataset, {"batch_size": 2}, seed=0, shuffle=False)
        seen.append([crop_id for batch in loader for crop_id in batch.crop_id])

    assert seen[0] == expected
    assert seen[0] == seen[1]


def test_normalization_comes_from_the_config_not_from_a_literal(crops):
    """The test that catches a hard-coded ImageNet triple.

    The same crop through a mean=0/std=1 config must land in [0, 1]; through the ImageNet config it
    must not. If the pipeline ignores `data_config` and hard-codes the usual constants, the two are
    identical and this fails — which is the point, because nothing else about a mis-normalised model
    looks wrong. It trains, the loss falls, and the benchmark silently compares a well-fed network
    against a badly-fed one.
    """
    identity = build_dataset(crops, split="val", config=FAKE_CONFIG)[0].image
    imagenet = build_dataset(crops, split="val", config=IMAGENET_CONFIG)[0].image

    assert 0.0 <= float(identity.min()) and float(identity.max()) <= 1.0
    assert float(imagenet.min()) < 0.0 or float(imagenet.max()) > 1.0
    assert not torch.allclose(identity, imagenet)


def test_the_transform_is_deterministic(crops):
    """Val and test are measured through this; a stochastic eval score cannot be compared."""
    dataset = build_dataset(crops, split="val")
    assert torch.equal(dataset[0].image, dataset[0].image)

    transform = transforms_mod.build_eval_transform(FAKE_CONFIG)
    image = Image.new("RGB", (CROP_SIDE, CROP_SIDE), (123, 45, 67))
    assert torch.equal(transform(image), transform(image))


def test_train_with_no_augmentation_is_exactly_the_eval_pipeline(crops):
    """Stage 2's first runs must differ from eval only in shuffling."""
    manifest, crops_dir = crops
    image = Image.new("RGB", (CROP_SIDE, CROP_SIDE), (90, 90, 90))

    train = transforms_mod.build_train_transform(FAKE_CONFIG, {})
    evaluation = transforms_mod.build_eval_transform(FAKE_CONFIG)
    assert torch.equal(train(image), evaluation(image))


# --- the real crops ----------------------------------------------------------------------------

REAL_MANIFEST = Path("data/crops/manifest.csv")
REAL_CROPS = Path("data/crops")


@pytest.mark.skipif(
    not (REAL_MANIFEST.is_file() and REAL_CROPS.is_dir()),
    reason="requires the generated crops (gitignored)",
)
def test_real_crops_load_with_the_expected_split_sizes():
    """Pinned against what `pcbi split` reported, so a regenerated manifest cannot drift."""
    for split, expected in [("train", 238), ("val", 82), ("test", 80)]:
        dataset = datasets_mod.CropDataset(REAL_MANIFEST, REAL_CROPS, split=split)
        assert len(dataset) == expected
        assert all(path.is_file() for path in dataset.paths)
