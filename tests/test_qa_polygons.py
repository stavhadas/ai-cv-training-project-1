"""Tests for `pcbi qa-polygons`, the Stage 1 Step 2 visual QA sheets.

`joint_qa_dataset` places real (tiny) JPEGs so Pillow can actually open and draw on them — unlike
`test_ingest.py`'s fixtures, which never open image bytes. One image carries an EXIF orientation
tag, since that's the specific failure mode this module exists to guard against: a polygon that
looks right in the JSON but lands in the wrong place once the photo is displayed correctly.
"""

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import audit as audit_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import qa_polygons as qa_mod

runner = CliRunner()


def labelme_record(image_name: str, shapes: list[tuple[str, list]], width=200, height=150) -> dict:
    return {
        "version": "5.0.1",
        "imagePath": image_name,
        "imageWidth": width,
        "imageHeight": height,
        "imageData": None,
        "shapes": [
            {"label": label, "shape_type": "polygon", "points": points} for label, points in shapes
        ],
    }


def make_jpeg(path: Path, size=(200, 150), color=(40, 60, 90), exif_orientation=None) -> None:
    img = Image.new("RGB", size, color=color)
    if exif_orientation is None:
        img.save(path, format="JPEG")
        return
    exif = Image.Exif()
    exif[0x0112] = exif_orientation  # the Orientation tag
    img.save(path, format="JPEG", exif=exif)


@pytest.fixture
def joint_qa_dataset(tmp_path):
    """CS1/R0805: five V2 images — three with 2 polygons, two with 3 — plus a V1 image (excluded).

    One 3-polygon image (`rotated.jpg`) carries an EXIF orientation tag, so its saved pixel
    dimensions (200x150) differ from its displayed dimensions (150x200) — the case
    `exif_transpose` exists to handle.
    """
    root = tmp_path

    def place(name, exif_orientation=None):
        folder = root / "Dataset" / "CS1" / "R0805" / "V2"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        make_jpeg(path, exif_orientation=exif_orientation)
        return path

    v1_folder = root / "Dataset" / "CS1" / "R0805" / "V1" / "Setup1"
    v1_folder.mkdir(parents=True, exist_ok=True)
    v1_image = v1_folder / "placement.jpg"
    make_jpeg(v1_image)

    two_poly_images = [place(f"two_{i}.jpg") for i in range(3)]
    three_poly_images = [place("three_a.jpg"), place("rotated.jpg", exif_orientation=6)]

    labeled = root / "Labeled"
    labeled.mkdir()

    def annotate(image_path: Path, shapes: list[tuple[str, list]]) -> None:
        name = image_path.name
        (labeled / name).write_bytes(image_path.read_bytes())
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(labelme_record(name, shapes))
        )

    two_poly_shapes = [
        ("exc_solder", [[10, 10], [30, 10], [30, 30], [10, 30]]),
        ("spike", [[60, 60], [80, 60], [80, 80], [60, 80]]),
    ]
    three_poly_shapes = [
        ("exc_solder", [[10, 10], [30, 10], [30, 30], [10, 30]]),
        ("spike", [[60, 60], [80, 60], [80, 80], [60, 80]]),
        ("good", [[100, 100], [120, 100], [120, 120], [100, 120]]),
    ]
    for path in two_poly_images:
        annotate(path, two_poly_shapes)
    for path in three_poly_images:
        annotate(path, three_poly_shapes)
    annotate(v1_image, [("good", [[1, 1], [2, 1], [2, 2], [1, 2]])])

    return root


def test_group_by_image_buckets_rows_by_dataset_path(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    assert len(groups) == 5  # 3 two-poly + 2 three-poly images; the V1 image never reaches here
    assert all(len(v) in (2, 3) for v in groups.values())


def test_bbox_of_returns_the_axis_aligned_box():
    points = [[10, 30], [30, 10], [20, 50], [5, 25]]
    assert qa_mod.bbox_of(points) == (5, 10, 30, 50)


def test_polygon_area_of_a_known_square():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert qa_mod.polygon_area(square) == 100.0


def test_polygon_area_of_a_degenerate_shape_is_zero():
    assert qa_mod.polygon_area([[0, 0], [1, 1]]) == 0.0


def test_render_annotated_image_corrects_exif_orientation(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    rotated_path = next(p for p in groups if p.endswith("rotated.jpg"))
    img = qa_mod.render_annotated_image(result.root / rotated_path, groups[rotated_path])
    # Saved as 200x150 with a 90-degree EXIF tag; corrected display size is 150x200.
    assert img.size == (150, 200)


def test_render_annotated_image_handles_images_with_no_exif_tag(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    plain_path = next(p for p in groups if p.endswith("two_0.jpg"))
    img = qa_mod.render_annotated_image(result.root / plain_path, groups[plain_path])
    assert img.size == (200, 150)


def test_build_sheets_splits_across_multiple_sheets_when_needed(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    paths = sorted(groups)
    sheets = qa_mod.build_sheets(result.root, groups, paths, max_per_sheet=2)
    assert len(sheets) == 3  # 5 images, 2 per sheet -> 3 sheets (2, 2, 1)


def test_build_sheets_fits_everything_on_one_sheet_when_it_fits(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    paths = sorted(groups)
    sheets = qa_mod.build_sheets(result.root, groups, paths, max_per_sheet=24)
    assert len(sheets) == 1


def test_write_sheets_names_the_first_sheet_without_a_suffix(tmp_path):
    sheets = [Image.new("RGB", (10, 10)) for _ in range(3)]
    paths = qa_mod.write_sheets(sheets, tmp_path, "overlay_3poly")
    assert [p.name for p in paths] == [
        "overlay_3poly.png",
        "overlay_3poly_2.png",
        "overlay_3poly_3.png",
    ]
    assert all(p.is_file() for p in paths)


def test_bbox_stats_grouped_by_class_and_package(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    stats = qa_mod.bbox_stats(rows)
    by_class_package = {(row["class"], row["package"]) for row in stats}
    # exc_solder -> excess, spike -> spike, good (joint) -> normal; all under package R0805.
    assert by_class_package == {("excess", "R0805"), ("spike", "R0805"), ("normal", "R0805")}
    excess_row = next(row for row in stats if row["class"] == "excess")
    assert excess_row["count"] == 5  # one per joint-task image
    assert excess_row["width_mean"] == pytest.approx(20.0)
    assert excess_row["height_mean"] == pytest.approx(20.0)
    assert excess_row["area_mean"] == pytest.approx(400.0)


def test_write_qa_report_end_to_end(joint_qa_dataset, tmp_path):
    out_dir = tmp_path / "qa"
    summary = qa_mod.write_qa_report(
        joint_qa_dataset, out_dir=out_dir, sample_size=4, max_per_sheet=2
    )
    assert summary["sample_images"] == 4
    assert summary["three_poly_images"] == 2
    assert summary["three_poly_sheets"] == 1  # 2 images, max_per_sheet=2 -> 1 sheet
    assert summary["bbox_stats_rows"] == 3

    assert (out_dir / "overlay_sample.png").is_file()
    assert (out_dir / "overlay_3poly.png").is_file()
    assert not (out_dir / "overlay_3poly_2.png").exists()
    assert (out_dir / "bbox_stats.csv").is_file()


def test_write_qa_report_sampling_is_reproducible_for_a_fixed_seed(joint_qa_dataset, tmp_path):
    first = qa_mod.write_qa_report(joint_qa_dataset, out_dir=tmp_path / "a", seed=42, sample_size=3)
    second = qa_mod.write_qa_report(
        joint_qa_dataset, out_dir=tmp_path / "b", seed=42, sample_size=3
    )
    img_a = Image.open(tmp_path / "a" / "overlay_sample.png")
    img_b = Image.open(tmp_path / "b" / "overlay_sample.png")
    assert first["sample_images"] == second["sample_images"]
    assert img_a.tobytes() == img_b.tobytes()


def test_placement_images_never_appear_in_qa_output(joint_qa_dataset, tmp_path):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    assert not any("placement.jpg" in path for path in groups)


def test_cli_writes_expected_files(joint_qa_dataset, tmp_path):
    out_dir = tmp_path / "qa"
    result = runner.invoke(
        app,
        [
            "qa-polygons",
            "--root",
            str(joint_qa_dataset),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "overlay_sample.png").is_file()
    assert (out_dir / "overlay_3poly.png").is_file()
    assert (out_dir / "bbox_stats.csv").is_file()


def test_cli_fails_clearly_when_root_is_missing(tmp_path):
    result = runner.invoke(
        app,
        ["qa-polygons", "--root", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "qa")],
    )
    assert result.exit_code == 1


def test_find_dataset_path_matches_the_full_path(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    full_path = next(iter(groups))
    assert qa_mod.find_dataset_path(groups, full_path) == full_path
    # Windows-style separators, and just the bare filename, both resolve to the same key.
    assert qa_mod.find_dataset_path(groups, full_path.replace("/", "\\")) == full_path
    assert qa_mod.find_dataset_path(groups, Path(full_path).name) == full_path


def test_find_dataset_path_raises_clearly_when_not_found(joint_qa_dataset):
    result = audit_mod.audit(joint_qa_dataset)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    groups = qa_mod.group_by_image(rows)
    with pytest.raises(ValueError, match="no joint-task annotation"):
        qa_mod.find_dataset_path(groups, "nope.jpg")


def test_render_single_draws_at_full_resolution_not_a_thumbnail(joint_qa_dataset):
    dataset_path, img = qa_mod.render_single(joint_qa_dataset, "two_0.jpg")
    assert dataset_path.endswith("two_0.jpg")
    assert img.size == (200, 150)  # the source resolution, not TILE_SIZE


def test_cli_image_renders_one_file_at_full_resolution(joint_qa_dataset, tmp_path):
    out_dir = tmp_path / "qa"
    result = runner.invoke(
        app,
        [
            "qa-polygons",
            "--root",
            str(joint_qa_dataset),
            "--out-dir",
            str(out_dir),
            "--image",
            "two_0.jpg",
        ],
    )
    assert result.exit_code == 0, result.output
    out_path = out_dir / "two_0_annotated.png"
    assert out_path.is_file()
    assert Image.open(out_path).size == (200, 150)
    # Only the single-image render happened — no sample/3poly sheets.
    assert not (out_dir / "overlay_sample.png").exists()


def test_cli_image_fails_clearly_when_not_found(joint_qa_dataset, tmp_path):
    result = runner.invoke(
        app,
        [
            "qa-polygons",
            "--root",
            str(joint_qa_dataset),
            "--out-dir",
            str(tmp_path / "qa"),
            "--image",
            "nope.jpg",
        ],
    )
    assert result.exit_code == 1


REAL_ROOT = Path("data/raw")


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="requires the real SolDef_AI download (gitignored)"
)
def test_real_dataset_has_43_three_polygon_images(tmp_path):
    summary = qa_mod.write_qa_report(REAL_ROOT, out_dir=tmp_path / "qa")
    assert summary["three_poly_images"] == 43


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="requires the real SolDef_AI download (gitignored)"
)
def test_real_render_single_by_full_path():
    dataset_path, img = qa_mod.render_single(
        REAL_ROOT, "SolDef_AI/Dataset/CS1/R0805/V2/WIN_20220330_13_11_58_Pro.jpg"
    )
    assert dataset_path == "SolDef_AI/Dataset/CS1/R0805/V2/WIN_20220330_13_11_58_Pro.jpg"
    assert img.size[0] > 0 and img.size[1] > 0
