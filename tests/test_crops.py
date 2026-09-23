"""Tests for `pcbi make-crops`, Stage 1 Step 7's training crops and their manifest.

The fixture builds a small real-JPEG tree under `tmp_path` with **hand-chosen polygon coordinates**,
so every geometry assertion here is arithmetic against a number written in the test rather than a
golden value recorded from a previous run. Crop boxes are the one thing no amount of downstream
testing can recover: a box that is off by ten pixels still produces a plausible-looking dataset.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import crops as crops_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import split as split_mod

runner = CliRunner()

FRAME = (400, 300)  # every fixture photo, so clamping cases can be built by hand


def noise_bytes(seed: int, size: tuple[int, int]) -> bytes:
    """Deterministic pixel noise, so a crop's content depends on where the box was."""
    import random

    rng = random.Random(seed)
    return bytes(rng.randrange(0, 256) for _ in range(size[0] * size[1] * 3))


def rectangle(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def polygon_row(
    board: str,
    package: str,
    viewpoint: str,
    name: str,
    rank: int,
    points: list[list[float]],
    class_name: str,
    merged_from: str = "",
) -> dict:
    return {
        "image_name": name,
        "dataset_path": f"Dataset/{board}/{package}/{viewpoint}/{name}",
        "board": board,
        "package": package,
        "viewpoint": viewpoint,
        "light_direction": "bottom_to_top" if viewpoint == "V2" else "top_to_bottom",
        "capture_timestamp": "",
        "raw_label": {"normal": "good", "excess": "exc_solder"}.get(class_name, class_name),
        "class": class_name,
        "points": points,
        "polygon_count": 2,
        "polygon_rank_x": rank,
        "joint_position": "left" if rank == 1 else "right",
        "merged_from": merged_from,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def crops_dataset(tmp_path):
    """A dataset root, a polygons CSV and a split CSV, with geometry chosen to be checkable.

    Image `a` carries the two joints every assertion about box arithmetic uses: a 100x200 box well
    inside the frame, and a box pushed against the top-left corner so clamping has to happen.
    """
    root = tmp_path / "raw"
    rows: list[dict] = []
    plan = [
        # (board, package, viewpoint, name, class, split, [(rank, bbox)])
        (
            "CS1",
            "R0805",
            "V2",
            "a.jpg",
            "normal",
            "train",
            [(1, (50, 40, 150, 240)), (2, (0, 0, 60, 50))],
        ),
        (
            "CS1",
            "R0805",
            "V2.1",
            "b.jpg",
            "normal",
            "train",
            [(1, (10, 10, 90, 90)), (2, (200, 100, 300, 200))],
        ),
        (
            "CS2",
            "R0603",
            "V2",
            "c.jpg",
            "excess",
            "train",
            [(1, (20, 20, 80, 80)), (2, (120, 30, 200, 110))],
        ),
        (
            "CS2",
            "R0603",
            "V2",
            "d.jpg",
            "excess",
            "val",
            [(1, (30, 30, 90, 90)), (2, (150, 40, 220, 120))],
        ),
        (
            "CS6",
            "C0805",
            "V2",
            "e.jpg",
            "spike",
            "test",
            [(1, (40, 50, 110, 130)), (2, (250, 60, 330, 150))],
        ),
    ]

    split_rows = []
    for index, (board, package, viewpoint, name, class_name, split, joints) in enumerate(plan):
        folder = root / "Dataset" / board / package / viewpoint
        folder.mkdir(parents=True, exist_ok=True)
        Image.frombytes("RGB", FRAME, noise_bytes(index, FRAME)).save(
            folder / name, format="JPEG", quality=95
        )
        for rank, (x0, y0, x1, y1) in joints:
            # The second joint of image `c` is merged, so polygon_area on it is a bbox area.
            merged = "spike+excess" if (name == "c.jpg" and rank == 2) else ""
            row = polygon_row(
                board, package, viewpoint, name, rank, rectangle(x0, y0, x1, y1), class_name, merged
            )
            rows.append(row)
            split_rows.append(
                {
                    "crop_id": split_mod.crop_id(row["dataset_path"], rank),
                    "dataset_path": row["dataset_path"],
                    "polygon_rank_x": rank,
                    "joint_position": row["joint_position"],
                    "board": board,
                    "package": package,
                    "cell": f"{board}/{package}",
                    "viewpoint": viewpoint,
                    "class": class_name,
                    "group_id": f"manual_{index:03d}",
                    "split": split,
                    "split_hash": "0" * 64,
                }
            )

    polygons_path = tmp_path / "polygons.csv"
    splits_path = tmp_path / "split_v1.csv"
    with polygons_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ingest_mod.CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "points": json.dumps(row["points"])})
    write_csv(splits_path, split_mod.CSV_FIELDS, split_rows)
    return root, polygons_path, splits_path


def read_manifest(out_dir: Path) -> list[dict]:
    with (out_dir / crops_mod.MANIFEST_NAME).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digests(directory: Path) -> dict[str, str]:
    return {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}


# --- byte identity, the whole point ----------------------------------------------------------


def test_regenerating_at_the_same_margin_produces_identical_bytes(crops_dataset, tmp_path):
    """The requirement this command exists to satisfy.

    A frozen split is worth nothing if the pixels behind it move: the split hash would keep matching
    while the dataset quietly changed. File bytes, not decoded pixels — decoded pixels would still
    match if the PNG encoder settings drifted, and a dataset that re-encodes on every run is not
    reproducible even when it looks identical.
    """
    root, polygons, splits = crops_dataset
    first, second = tmp_path / "first", tmp_path / "second"
    crops_mod.make_crops(root, polygons, splits, first, margin=0.1, sheet_dir=None)
    crops_mod.make_crops(root, polygons, splits, second, margin=0.1, sheet_dir=None)

    assert tree_digests(first) == tree_digests(second)
    assert len(tree_digests(first)) == 11  # 10 crops + the manifest


def test_a_different_margin_produces_different_bytes(crops_dataset, tmp_path):
    """Guards the test above: it would pass trivially on a function that ignored its margin."""
    root, polygons, splits = crops_dataset
    tight, loose = tmp_path / "tight", tmp_path / "loose"
    crops_mod.make_crops(root, polygons, splits, tight, margin=0.1, sheet_dir=None)
    crops_mod.make_crops(root, polygons, splits, loose, margin=0.3, sheet_dir=None)

    assert tree_digests(tight) != tree_digests(loose)


def test_the_crops_carry_no_metadata_from_the_source_photo(crops_dataset, tmp_path):
    """A crop's training value is its pixels; the frame's camera settings only add noise."""
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)

    with Image.open(next(out.glob("*.png"))) as img:
        assert "exif" not in img.info
        assert "jfif" not in img.info


# --- the crop box ----------------------------------------------------------------------------


def test_margin_zero_is_the_joint_box_itself():
    assert crops_mod.crop_box((50.0, 40.0, 150.0, 240.0), 0.0, FRAME) == (50, 40, 150, 240)


def test_the_margin_scales_each_axis_by_that_axis(crops_dataset):
    """The rule that separates this from group.crop_box, which uses max(w, h) on all four sides.

    A 100x200 box at margin 0.1 grows by 10px horizontally and 20px vertically — not 20 on both,
    which is what an isotropic margin would give.
    """
    assert crops_mod.crop_box((50.0, 40.0, 150.0, 240.0), 0.1, FRAME) == (40, 20, 160, 260)


def test_the_box_grows_outward_to_whole_pixels():
    """Floor the mins and ceil the maxes, so no annotated pixel is ever cut off by rounding."""
    assert crops_mod.crop_box((50.5, 40.5, 150.5, 240.5), 0.0, FRAME) == (50, 40, 151, 241)


def test_a_joint_against_the_frame_edge_clamps_rather_than_pads():
    """There is no board beyond the frame, so an edge crop is honestly asymmetric."""
    assert crops_mod.crop_box((0.0, 0.0, 60.0, 50.0), 0.5, FRAME) == (0, 0, 90, 75)
    assert crops_mod.crop_box((370.0, 280.0, 400.0, 300.0), 0.5, FRAME) == (355, 270, 400, 300)


# --- the manifest ----------------------------------------------------------------------------


def test_the_manifest_header_matches_the_declared_fields(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)
    with (out / crops_mod.MANIFEST_NAME).open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == crops_mod.MANIFEST_FIELDS


def test_every_joint_becomes_exactly_one_crop_and_one_row(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    summary = crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)

    rows = read_manifest(out)
    assert summary.crops == 10
    assert len(rows) == 10
    assert len(list(out.glob("*.png"))) == 10
    assert len({row["crop_id"] for row in rows}) == 10


def test_crop_id_is_filename_safe_and_is_the_file_stem(crops_dataset, tmp_path):
    """The manifest carries no crop_file column, so this rule is how a consumer finds the pixels."""
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)

    for row in read_manifest(out):
        assert re.fullmatch(r"[A-Za-z0-9_]+", row["crop_id"]), row["crop_id"]
        assert (out / f"{row['crop_id']}.png").is_file()
    # V2.1 must not leave a dot in the name, or the file appears to have two extensions.
    assert (out / "CS1_R0805_V2_1_b_1.png").is_file()


def test_the_recorded_crop_size_is_the_saved_image_size(crops_dataset, tmp_path):
    """crop_w/crop_h are what a dataloader will budget for; a lie here is found much later."""
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)

    for row in read_manifest(out):
        with Image.open(out / f"{row['crop_id']}.png") as img:
            assert img.size == (int(row["crop_w"]), int(row["crop_h"])), row["crop_id"]


def test_the_geometry_columns_describe_the_joint_before_the_margin(crops_dataset, tmp_path):
    """bbox_w/bbox_h must not move when the margin does — only crop_w/crop_h may."""
    root, polygons, splits = crops_dataset
    tight, loose = tmp_path / "tight", tmp_path / "loose"
    crops_mod.make_crops(root, polygons, splits, tight, margin=0.0, sheet_dir=None)
    crops_mod.make_crops(root, polygons, splits, loose, margin=0.25, sheet_dir=None)

    a = {row["crop_id"]: row for row in read_manifest(tight)}
    b = {row["crop_id"]: row for row in read_manifest(loose)}
    target = "CS1_R0805_V2_a_1"  # the 100x200 joint, well inside the frame
    assert a[target]["bbox_w"] == b[target]["bbox_w"] == "100.00"
    assert a[target]["bbox_h"] == b[target]["bbox_h"] == "200.00"
    assert int(a[target]["crop_w"]) == 100
    assert int(b[target]["crop_w"]) == 150  # 100 + 2 * 25


def test_the_manifest_carries_the_split_and_group(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    summary = crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)

    assert summary.splits == {"train": 6, "val": 2, "test": 2}
    rows = {row["crop_id"]: row for row in read_manifest(out)}
    assert rows["CS1_R0805_V2_a_1"]["split"] == "train"
    assert rows["CS1_R0805_V2_a_1"]["group_id"] == "manual_000"
    assert rows["CS6_C0805_V2_e_1"]["split"] == "test"


def test_labels_keep_both_the_raw_and_the_project_name(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)
    rows = {row["crop_id"]: row for row in read_manifest(out)}
    assert (rows["CS1_R0805_V2_a_1"]["label"], rows["CS1_R0805_V2_a_1"]["label_raw"]) == (
        "normal",
        "good",
    )


def test_merged_joints_are_counted_so_polygon_area_can_be_read_correctly(crops_dataset, tmp_path):
    """ingest replaced those points with a union bbox, so their area is a rectangle's.

    Nothing in the manifest can express that, so the count is surfaced instead of left implicit.
    """
    root, polygons, splits = crops_dataset
    summary = crops_mod.make_crops(
        root, polygons, splits, tmp_path / "crops", margin=0.1, sheet_dir=None
    )
    assert summary.merged_count == 1


def test_the_manifest_is_sorted_by_crop_id(crops_dataset, tmp_path):
    """Stable order regardless of the input CSV's, so a diff of two manifests is readable."""
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=None)
    ids = [row["crop_id"] for row in read_manifest(out)]
    assert ids == sorted(ids)


# --- reading the inputs ----------------------------------------------------------------------


def test_a_joint_with_no_row_in_the_split_is_an_error(crops_dataset, tmp_path):
    """Cropping it anyway would put an unassigned image in the training folder."""
    root, polygons, splits = crops_dataset
    rows = list(csv.DictReader(splits.open(newline="", encoding="utf-8")))
    dropped = rows.pop(0)
    trimmed = tmp_path / "partial_split.csv"
    write_csv(trimmed, split_mod.CSV_FIELDS, rows)

    with pytest.raises(ValueError, match="have no row in the split"):
        crops_mod.make_crops(root, polygons, trimmed, tmp_path / "crops", sheet_dir=None)
    assert dropped["crop_id"]


def test_a_file_that_is_not_a_polygons_csv_is_refused(tmp_path):
    wrong = tmp_path / "wrong.csv"
    write_csv(wrong, ["a", "b"], [{"a": "1", "b": "2"}])
    with pytest.raises(ValueError, match="not a polygons CSV"):
        crops_mod.load_polygon_rows(wrong)


def test_a_file_that_is_not_a_split_csv_is_refused(tmp_path):
    wrong = tmp_path / "wrong.csv"
    write_csv(wrong, ["a", "b"], [{"a": "1", "b": "2"}])
    with pytest.raises(ValueError, match="not a split CSV"):
        crops_mod.load_splits(wrong)


def test_a_negative_margin_is_refused(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    with pytest.raises(ValueError, match="0 or greater"):
        crops_mod.make_crops(root, polygons, splits, tmp_path / "crops", margin=-0.1)


# --- the contact sheets ----------------------------------------------------------------------


def test_sheets_are_built_from_train_crops_only(crops_dataset, tmp_path):
    """Choosing what a class looks like while looking at val or test is how a split leaks.

    The fixture puts `spike` in test alone, so a sheet for it would mean val/test crops reached the
    sampler.
    """
    root, polygons, splits = crops_dataset
    out, sheets = tmp_path / "crops", tmp_path / "sheets"
    summary = crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=sheets)

    assert set(summary.sheets) == {
        "normal",
        "excess",
    }  # spike is test-only, excess has a train half
    assert (sheets / "crops_normal.png").is_file()
    assert not (sheets / "crops_spike.png").exists()


def test_a_sheet_never_holds_more_than_the_cap(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out, sheets = tmp_path / "crops", tmp_path / "sheets"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=sheets)

    manifest = read_manifest(out)
    train_normal = [r for r in manifest if r["split"] == "train" and r["label"] == "normal"]
    picked = crops_mod.contact_sheets(out, manifest, tmp_path / "capped", max_per_class=2)
    assert len(train_normal) > 2  # otherwise the cap is not being exercised
    assert (tmp_path / "capped" / "crops_normal.png").is_file()
    assert set(picked) == {"normal", "excess"}


def test_the_sheets_are_reproducible(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out = tmp_path / "crops"
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=tmp_path / "a")
    crops_mod.make_crops(root, polygons, splits, out, margin=0.1, sheet_dir=tmp_path / "b")
    assert tree_digests(tmp_path / "a") == tree_digests(tmp_path / "b")


# --- the CLI ---------------------------------------------------------------------------------


def run_make_crops(**options):
    args = ["make-crops"]
    for name, value in options.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return runner.invoke(app, args)


def test_cli_writes_the_crops_and_prints_the_counts(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out, sheets = tmp_path / "crops", tmp_path / "sheets"
    result = run_make_crops(
        root=root, polygons=polygons, splits=splits, out_dir=out, sheet_dir=sheets, margin=0.1
    )
    assert result.exit_code == 0, result.output
    assert "10 crop(s)" in result.output
    assert "median crop:" in result.output
    assert "train 6" in result.output
    assert "contact sheet(s)" in result.output
    assert (out / crops_mod.MANIFEST_NAME).is_file()


def test_cli_can_skip_the_sheets(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    out, sheets = tmp_path / "crops", tmp_path / "sheets"
    result = run_make_crops(
        root=root, polygons=polygons, splits=splits, out_dir=out, sheet_dir=sheets, no_sheets=True
    )
    assert result.exit_code == 0, result.output
    assert "contact sheet(s)" not in result.output
    assert not sheets.exists()


def test_cli_rejects_a_negative_margin(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    result = run_make_crops(
        root=root, polygons=polygons, splits=splits, out_dir=tmp_path / "crops", margin=-1
    )
    assert result.exit_code == 2, result.output
    assert "--margin must be 0 or greater" in result.output


def test_cli_reports_missing_inputs(crops_dataset, tmp_path):
    root, polygons, splits = crops_dataset
    missing_root = run_make_crops(root=tmp_path / "nowhere", polygons=polygons, splits=splits)
    assert missing_root.exit_code == 1, missing_root.output
    assert "No such folder" in missing_root.output

    missing_polygons = run_make_crops(root=root, polygons=tmp_path / "no.csv", splits=splits)
    assert missing_polygons.exit_code == 1, missing_polygons.output
    assert "No such polygons file" in missing_polygons.output

    missing_splits = run_make_crops(root=root, polygons=polygons, splits=tmp_path / "no.csv")
    assert missing_splits.exit_code == 1, missing_splits.output
    assert "No such split file" in missing_splits.output


# --- the real download -----------------------------------------------------------------------

REAL_ROOT = Path("data/raw")
REAL_POLYGONS = Path("data/interim/polygons.csv")
REAL_SPLITS = Path("data/splits/split_v1.csv")


@pytest.mark.skipif(
    not (REAL_ROOT.is_dir() and REAL_POLYGONS.is_file() and REAL_SPLITS.is_file()),
    reason="requires the real SolDef_AI download (gitignored)",
)
def test_real_dataset_crops_match_the_frozen_split(tmp_path):
    """400 joints, the same class and split counts `pcbi split` reported."""
    summary = crops_mod.make_crops(
        REAL_ROOT, REAL_POLYGONS, REAL_SPLITS, tmp_path / "crops", margin=0.1, sheet_dir=None
    )
    assert summary.crops == 400
    assert summary.classes == {"excess": 157, "spike": 121, "normal": 65, "insufficient": 57}
    assert summary.splits == {"train": 238, "val": 82, "test": 80}
    assert summary.merged_count == 43
