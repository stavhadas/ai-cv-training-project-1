"""Stage 2: the `torch.utils.data.Dataset` over the crops `pcbi make-crops` wrote.

One row of `data/crops/manifest.csv` becomes one training example. The manifest already carries the
label and the split, so this layer does no joining, no grouping and no sampling logic of its own —
`pcbi split` froze all of that, and re-deciding any of it here would quietly unfreeze it.

Two things are worth knowing before reading on.

**`CLASS_ORDER` is permanent.** The index a class maps to is baked into every checkpoint's final
layer and every logged per-class metric. Reordering it later would not raise anything — it would
silently relabel old runs, turning a saved model's `normal` head into its `excess` head. It matches
the order the README and the product spec state, with the pass class at index 0.

**There is no path column in the manifest.** A crop's file is `crops_dir / f"{crop_id}.png"`, and
that is deliberate: the crops are mounted at a different root on Kaggle than the one they were
written at, so a path recorded at extraction time would be wrong everywhere it mattered. The id is
the stem; the directory is a runtime argument.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple
from .crops import load_source

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_MANIFEST = Path("data/crops/manifest.csv")
DEFAULT_CROPS_DIR = Path("data/crops")

# The index of each class is permanent — see the module docstring. Order is the one the README and
# the product spec use, pass class first.
CLASS_ORDER = ("normal", "excess", "insufficient", "spike")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}

SPLITS = ("train", "val", "test")


class Sample(NamedTuple):
    """One example. A `NamedTuple` so torch's `default_collate` batches it field by field.

    `crop_id` rides along all the way to the model output on purpose: without it a wrong prediction
    is a number in a confusion matrix, and with it you can open the actual PNG and look at what the
    model got wrong. That is most of error analysis.
    """

    image: torch.Tensor  # float32, (3, H, W)
    label: int  # index into CLASS_ORDER
    crop_id: str


class CropDataset(Dataset):
    """The crops for one split, in manifest order."""

    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST,
        crops_dir: Path = DEFAULT_CROPS_DIR,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        """Read the manifest, keep one split, and check every label before any training starts.

        Rows are kept in **manifest order** rather than sorted here. The manifest is already sorted
        by `crop_id`, and preserving whatever order it has means an unshuffled loader yields a
        stable, inspectable sequence — which is what makes a val run comparable with the last one.

        Validation is eager and total: every label is checked now, not lazily in `__getitem__`. A
        bad label discovered at epoch 3, batch 700 is a confusing crash; discovered here it names
        the crop.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {split!r}.")

        manifest_path, crops_dir = Path(manifest_path), Path(crops_dir)
        if not manifest_path.is_file():
            raise ValueError(f"No such manifest: {manifest_path}. Run `pcbi make-crops` first.")

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {"crop_id", "label", "split"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{manifest_path} is missing column(s): {', '.join(sorted(missing))}. "
                    f"Expected a manifest written by `pcbi make-crops`."
                )
            rows = [row for row in reader if row["split"] == split]

        unknown = sorted({row["label"] for row in rows} - set(CLASS_ORDER))
        if unknown:
            offender = next(row for row in rows if row["label"] in unknown)
            raise ValueError(
                f"{manifest_path} has {len(unknown)} label(s) not in CLASS_ORDER: "
                f"{', '.join(unknown)} (e.g. crop {offender['crop_id']}). "
                f"CLASS_ORDER is {', '.join(CLASS_ORDER)} and its indices are permanent — if a new "
                f"class is real, appending it is safe; inserting it is not."
            )

        if not rows:
            raise ValueError(
                f"{manifest_path} has no rows with split={split!r}. An empty dataset shows up much "
                f"later as an unexplainable training curve, so it is refused here."
            )

        self.manifest_path = manifest_path
        self.crops_dir = crops_dir
        self.split = split
        self.transform = transform
        self.rows = rows
        self.crop_ids = [row["crop_id"] for row in rows]
        self.labels = [CLASS_TO_INDEX[row["label"]] for row in rows]
        self.paths = [crops_dir / f"{row['crop_id']}.png" for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        """Load one crop and return it as a `Sample`.

        Args:
            idx: position in this split, `0 <= idx < len(self)`. The row it refers to is
                `self.rows[idx]`, with the convenience lists `self.paths[idx]`,
                `self.labels[idx]` and `self.crop_ids[idx]` already prepared by `__init__`.

        Returns:
            A `Sample` with exactly three fields:

            - ``image`` — a ``torch.Tensor`` of dtype ``torch.float32``, shape ``(3, H, W)``,
              channels-first. When ``self.transform`` is set, ``(H, W)`` is whatever the transform
              produces, which for `transforms.build_eval_transform` is the backbone's
              ``input_size``, normally ``(224, 224)``. The transform is what converts the image to
              a tensor and normalizes it, so it must be applied when one is set.
            - ``label`` — a plain Python ``int``, the index of this crop's class in ``CLASS_ORDER``.
              An ``int``, not a tensor: ``default_collate`` turns a batch of ints into an ``int64``
              tensor of shape ``(B,)``, which is what a cross-entropy loss expects.
            - ``crop_id`` — the ``str`` id of this crop, returned **alongside** the image and label
              rather than dropped. It is what lets a wrong prediction be traced back to a file on
              disk, and the field the tests use to assert the loader's ordering.

        The file to open is `self.paths[idx]`. Open it the way the rest of this project does — see
        `crops.load_source` — so orientation and colour mode cannot disagree between the crops that
        were written and the crops that get read back.

        When ``self.transform`` is ``None``, return the loaded image without a transform applied.
        That path exists so the loading itself can be inspected in isolation; it does not have to
        satisfy the dtype and shape contract above, which is the transform's job.
        """
        # === YOUR CODE: load self.paths[idx], apply self.transform if set, return a Sample ===
        # Split paths[idx] to root and dataset_path
        root = self.crops_dir
        dataset_path = self.paths[idx].relative_to(root)  # Get the relative path to the root directory

        image = load_source(root=root, dataset_path=dataset_path)  # Load the image using the provided load_source function
        if self.transform is not None:
            image = self.transform(image)  # Apply the transform if it's set
        label = self.labels[idx]
        crop_id = self.crop_ids[idx]
        sample = Sample(image=image, label=label, crop_id=crop_id)
        return sample

        # === END YOUR CODE ===


def worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct, reproducible seed.

    Workers are forked after the generator below is seeded, so without this every worker would
    inherit one RNG state and augmentations would repeat across them in lockstep. `initial_seed()`
    is already derived from the loader's generator, so deriving from it keeps the whole thing a
    function of the one seed the caller passed.
    """
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_dataloader(
    dataset: CropDataset,
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap a `CropDataset` in a `DataLoader` whose behaviour is a function of `seed` alone.

    `shuffle` defaults to False, which is the right default for the two splits whose numbers get
    reported: val and test must yield the same order every run or their scores are not comparable.
    Only the training loader passes `shuffle=True`.

    `drop_last=False` always. Dropping a partial final batch is a common training convenience, but
    on val and test it silently discards up to a batch of examples and quietly changes the
    denominator of every metric. With 82 val crops and a batch of 32, that would be 18 crops
    disappearing from a score without anything saying so.
    """
    cfg = cfg or {}
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False),
        persistent_workers=bool(cfg.get("num_workers", 0)) and cfg.get("persistent_workers", False),
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
