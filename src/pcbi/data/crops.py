"""Stage 1 Step 7: cut the crops a model trains on, and the manifest that describes them.

Everything before this step describes joints; nothing has produced pixels. `pcbi ingest` recorded
one row per solder joint with its polygon, `pcbi group` and the manual tagger recovered which
images show the same physical component, and `pcbi split` froze a leak-free partition over those
joints. This turns each of those rows into an actual image file plus one line of a manifest
carrying its geometry, folder metadata, label and split.

**Crops must be byte-identical when regenerated at the same margin.** A frozen split is only worth
something if the pixels behind it hold still: the split hash would keep matching while the dataset
quietly moved underneath it. So the crop is PNG (lossless — no second generation of JPEG loss on
top of already-JPEG sources), rows are processed in sorted `crop_id` order, the encoder settings
are pinned rather than left to a Pillow default, and `save_crop` copies onto a fresh canvas so no
source EXIF rides along into the training data.

The margin is a fraction of **each side independently**: `margin * bbox_w` on the left and right,
`margin * bbox_h` on the top and bottom, so the crop keeps the joint box's aspect ratio. This is
deliberately not what `group.crop_box` does — that one grows by `tolerance * max(w, h)` on all four
sides, because it feeds a fingerprint that compares whole components and wanted a comparable amount
of surrounding board regardless of box shape. Two crop rules, two jobs; see `crop_box` below.

One caveat the manifest cannot express in a column. For the joints that `pcbi ingest` merged (43 of
400 on the real download, the double-defect images in notes/s1_three_polygons.md), `points` is no
longer a traced outline — `reduce_joint_cluster` replaced it with the union *bounding box* of the
polygons it absorbed. `bbox_w` and `bbox_h` are unaffected, but `polygon_area` on those rows is a
rectangle's area and is not comparable with the other 357. `merged_count` reports how many, so the
number gets stated rather than discovered.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

from PIL import Image, ImageOps

from pcbi.data import qa_polygons as qa_mod
from pcbi.data import show as show_mod
from pcbi.data import split as split_mod

DEFAULT_POLYGONS = Path("data/interim/polygons.csv")
DEFAULT_SPLITS = Path("data/splits/split_v1.csv")
DEFAULT_OUT_DIR = Path("data/crops")
DEFAULT_SHEET_DIR = Path("reports/qa")
DEFAULT_MARGIN = 0.1
MANIFEST_NAME = "manifest.csv"

# Contact sheets are built from train crops only. Eyeballing what a class looks like is a modelling
# decision, and making it while looking at val or test is how a split leaks through a person.
SHEET_SPLIT = "train"
SHEET_MAX = 64
SHEET_COLUMNS = 8
SHEET_SEED = 2  # its own constant: qa_polygons samples with 0, group_report with 1

# Pinned rather than left to Pillow's defaults, because byte identity is a requirement here and a
# default that shifts in a future release would break it silently.
PNG_SAVE = {"format": "PNG", "compress_level": 6}

MANIFEST_FIELDS = [
    "crop_id",
    "image_file",
    "source_path",
    "board",
    "package",
    "view_folder",
    "light_direction",
    "group_id",
    "split",
    "label_raw",
    "label",
    "n_polygons_in_image",
    "joint_position",
    "bbox_w",
    "bbox_h",
    "polygon_area",
    "crop_w",
    "crop_h",
]


@dataclass(frozen=True)
class Summary:
    """What the run produced, in the terms the person checking it cares about."""

    crops: int
    margin: float
    classes: dict[str, int]
    splits: dict[str, int]
    merged_count: int  # rows whose polygon_area is a bounding box, not a traced outline
    median_size: tuple[int, int]
    sheets: dict[str, Path]


def crop_id(board: str, package: str, view_folder: str, dataset_path: str, rank: str | int) -> str:
    """The stable ID for one crop, which is also its filename stem.

    The single place this format is spelled out. `split.crop_id` owns a different one —
    `<dataset_path>#<rank>` — which is readable in an error message but contains `/` and `#` and so
    cannot name a file. Both identify the same joint; `load_splits` is where they meet.

    `V2.1` becomes `V2_1` so the filename carries one extension and not two.
    """
    stem = Path(dataset_path).stem
    return f"{board}_{package}_{view_folder.replace('.', '_')}_{stem}_{rank}"


def load_polygon_rows(path: Path) -> list[dict]:
    """Every joint in `polygons.csv`, with `points` decoded back into a list of coordinates.

    The first and only place that column is read back. `pcbi ingest` JSON-encodes it on write and
    nothing has needed it since — `split.load_crops` reads the same file and ignores it — so the
    decode lives here rather than being repeated by the next caller.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"dataset_path", "points", "polygon_rank_x"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is not a polygons CSV: no {', '.join(sorted(missing))} column. "
                f"Expected a file written by `pcbi ingest`."
            )
        rows = list(reader)

    for row in rows:
        try:
            row["points"] = json.loads(row["points"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: {row['dataset_path']} has unreadable points: {exc}") from exc
    return rows


def load_splits(path: Path) -> dict[str, tuple[str, str]]:
    """`split.crop_id -> (group_id, split)` from a frozen split CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"crop_id", "group_id", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is not a split CSV: no {', '.join(sorted(missing))} column. "
                f"Run `pcbi split` first."
            )
        return {row["crop_id"]: (row["group_id"], row["split"]) for row in reader}


def crop_box(
    bbox: tuple[float, float, float, float],
    margin: float,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """The joint box grown by `margin` of its own width and height, clamped to the frame.

    Per-axis rather than isotropic: the horizontal margin scales with the box's width and the
    vertical with its height, so the crop has the same aspect ratio as the joint box it came from.
    `group.crop_box` deliberately does the opposite for fingerprinting — read its docstring for why
    that job wanted the other rule.

    Floor on the mins and ceil on the maxes, so the integer box only ever grows to enclose the
    float one; a joint near an edge clamps and comes out asymmetric rather than padded, because
    there is no board beyond the frame to include.
    """
    x0, y0, x1, y1 = bbox
    margin_x = margin * (x1 - x0)
    margin_y = margin * (y1 - y0)
    width, height = size
    return (
        max(0, int(floor(x0 - margin_x))),
        max(0, int(floor(y0 - margin_y))),
        min(width, int(ceil(x1 + margin_x))),
        min(height, int(ceil(y1 + margin_y))),
    )


def load_source(root: Path, dataset_path: str) -> Image.Image:
    """One source photo, oriented and in RGB — the repo-wide open sequence.

    `exif_transpose` is not optional. The polygon coordinates were recorded against the displayed
    orientation, so skipping it would rotate the pixels out from under every box.
    """
    with Image.open(root / dataset_path) as raw:
        raw.load()
        return ImageOps.exif_transpose(raw).convert("RGB")


def save_crop(image: Image.Image, path: Path) -> None:
    """Write one crop as PNG, carrying nothing but pixels.

    Pasting onto a fresh canvas drops the `info` dict the crop inherited from its JPEG source
    (EXIF, JFIF, DPI). None of it describes the crop any more, and training data has no use for the
    camera settings of the frame it was cut from.
    """
    clean = Image.new("RGB", image.size)
    clean.paste(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(path, **PNG_SAVE)


def build_manifest_row(
    row: dict,
    identifier: str,
    bbox: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
    group_id: str,
    split: str,
) -> dict:
    """One manifest line. Geometry columns describe the joint *before* the margin was applied."""
    x0, y0, x1, y1 = bbox
    return {
        "crop_id": identifier,
        "image_file": row["image_name"],
        "source_path": row["dataset_path"],
        "board": row["board"],
        "package": row["package"],
        "view_folder": row["viewpoint"],
        "light_direction": row["light_direction"],
        "group_id": group_id,
        "split": split,
        "label_raw": row["raw_label"],
        "label": row["class"],
        "n_polygons_in_image": row["polygon_count"],
        "joint_position": row["joint_position"],
        "bbox_w": f"{x1 - x0:.2f}",
        "bbox_h": f"{y1 - y0:.2f}",
        "polygon_area": f"{qa_mod.polygon_area(row['points']):.2f}",
        "crop_w": box[2] - box[0],
        "crop_h": box[3] - box[1],
    }


def write_crops(
    root: Path,
    polygon_rows: Sequence[dict],
    splits: dict[str, tuple[str, str]],
    out_dir: Path,
    margin: float,
) -> list[dict]:
    """Cut every joint out of its source image and return the manifest rows.

    Source images are opened once per image rather than once per joint — each frame is 2560x1440
    and holds two joints, so the obvious loop would decode every photo twice.
    """
    prepared = [
        (
            crop_id(
                row["board"],
                row["package"],
                row["viewpoint"],
                row["dataset_path"],
                row["polygon_rank_x"],
            ),
            split_mod.crop_id(row["dataset_path"], row["polygon_rank_x"]),
            row,
        )
        for row in polygon_rows
    ]

    counts: defaultdict[str, int] = defaultdict(int)
    for identifier, _, _ in prepared:
        counts[identifier] += 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} crop id(s) are not unique, e.g. {duplicates[0]}. "
            f"(board, package, viewpoint, filename, rank) must identify a joint uniquely."
        )

    orphans = sorted(key for _, key, _ in prepared if key not in splits)
    if orphans:
        shown = ", ".join(orphans[:3])
        more = f" (and {len(orphans) - 3} more)" if len(orphans) > 3 else ""
        raise ValueError(
            f"{len(orphans)} joint(s) have no row in the split: {shown}{more}. "
            f"Re-run `pcbi split` against the same polygons CSV."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Grouped by source image so each 2560x1440 frame is decoded once, not once per joint.
    manifest: list[dict] = []
    source: Image.Image | None = None
    loaded: str | None = None
    for identifier, key, row in sorted(
        prepared, key=lambda item: (item[2]["dataset_path"], item[0])
    ):
        if row["dataset_path"] != loaded:
            source = load_source(root, row["dataset_path"])
            loaded = row["dataset_path"]
        bbox = qa_mod.bbox_of(row["points"])
        box = crop_box(bbox, margin, source.size)
        save_crop(source.crop(box), out_dir / f"{identifier}.png")
        manifest.append(build_manifest_row(row, identifier, bbox, box, *splits[key]))

    manifest.sort(key=lambda entry: entry["crop_id"])
    return manifest


def write_manifest(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheets(
    crop_dir: Path,
    manifest: Sequence[dict],
    sheet_dir: Path,
    seed: int = SHEET_SEED,
    max_per_class: int = SHEET_MAX,
) -> dict[str, Path]:
    """One `crops_<class>.png` per class, built from train crops only.

    A class with more than `max_per_class` train crops gets a seeded sample rather than the first N:
    crop ids sort by board, so the first N would show one board under one lighting and hide exactly
    the variation the sheet exists to reveal.
    """
    by_class: defaultdict[str, list[str]] = defaultdict(list)
    for row in manifest:
        if row["split"] == SHEET_SPLIT:
            by_class[row["label"]].append(row["crop_id"])

    written: dict[str, Path] = {}
    for label in sorted(by_class):
        ids = sorted(by_class[label])
        if len(ids) > max_per_class:
            ids = sorted(random.Random(seed).sample(ids, k=max_per_class))
        tiles = [
            show_mod.tile_from_image(Image.open(crop_dir / f"{name}.png"), name[:42])
            for name in ids
        ]
        sheet_dir.mkdir(parents=True, exist_ok=True)
        destination = sheet_dir / f"crops_{label}.png"
        show_mod.compose_grid(tiles, columns=SHEET_COLUMNS).save(destination, **PNG_SAVE)
        written[label] = destination
    return written


def make_crops(
    root: Path,
    polygons_path: Path = DEFAULT_POLYGONS,
    splits_path: Path = DEFAULT_SPLITS,
    out_dir: Path = DEFAULT_OUT_DIR,
    margin: float = DEFAULT_MARGIN,
    sheet_dir: Path | None = DEFAULT_SHEET_DIR,
    seed: int = SHEET_SEED,
) -> Summary:
    """Cut every crop, write the manifest, render the sheets. `sheet_dir=None` skips the sheets."""
    if margin < 0:
        raise ValueError(f"margin must be 0 or greater, got {margin:g}.")

    polygon_rows = load_polygon_rows(polygons_path)
    if not polygon_rows:
        raise ValueError(f"{polygons_path} holds no joints to crop.")
    splits = load_splits(splits_path)

    manifest = write_crops(root, polygon_rows, splits, out_dir, margin)
    write_manifest(manifest, out_dir / MANIFEST_NAME)

    sheets = contact_sheets(out_dir, manifest, sheet_dir, seed) if sheet_dir is not None else {}

    classes: defaultdict[str, int] = defaultdict(int)
    split_counts: defaultdict[str, int] = defaultdict(int)
    for row in manifest:
        classes[row["label"]] += 1
        split_counts[row["split"]] += 1

    return Summary(
        crops=len(manifest),
        margin=margin,
        classes=dict(classes),
        splits=dict(split_counts),
        merged_count=sum(1 for row in polygon_rows if row.get("merged_from")),
        median_size=(
            int(statistics.median(int(row["crop_w"]) for row in manifest)),
            int(statistics.median(int(row["crop_h"]) for row in manifest)),
        ),
        sheets=sheets,
    )
