"""Tests for `pcbi ingest`, Stage 1's polygon-level CSV.

`joint_dataset` reproduces the real download's shape at a size that runs in milliseconds: a V2 and
a V2.1 image under one (board, package) cell (with distinct, orderable polygon x-positions), plus a
placement (V1) image whose labels must never reach the output.

`test_real_dataset_matches_the_audited_counts` runs against the actual SolDef_AI download and is
skipped when it isn't present (it's gitignored and not available in CI) — it is what actually
verifies the class counts against reports/data_audit.md (182 / 130 / 74 / 57).
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import audit as audit_mod
from pcbi.data import ingest as ingest_mod

runner = CliRunner()


def labelme_record(image_name: str, shapes: list[tuple[str, list]], width=1280, height=960) -> dict:
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


@pytest.fixture
def joint_dataset(tmp_path):
    """CS1/R0805: a V2 image (2 polygons), a V2.1 image (2 polygons), a V1 image (excluded)."""
    root = tmp_path

    def place(viewpoint, setup, name):
        parts = [root, "Dataset", "CS1", "R0805", viewpoint]
        if setup:
            parts.append(setup)
        folder = Path(*parts)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(b"fake jpeg bytes")
        return path

    v2_image = place("V2", None, "WIN_20220329_15_46_12_Pro.jpg")
    v21_image = place("V2.1", None, "c_45alt.jpg")
    v1_image = place("V1", "Setup1", "d_top.jpg")

    labeled = root / "Labeled"
    labeled.mkdir()

    def annotate(image_path: Path, shapes: list[tuple[str, list]]) -> None:
        name = image_path.name
        (labeled / name).write_bytes(image_path.read_bytes())
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(labelme_record(name, shapes))
        )

    # x-positions chosen so sorted-by-x order differs from shape (file) order.
    annotate(v2_image, [("exc_solder", [[50, 0]]), ("spike", [[10, 0]])])
    annotate(v21_image, [("poor_solder", [[1, 0]]), ("spike", [[99, 0]])])
    annotate(v1_image, [("good", [[1, 0]]), ("no_good", [[2, 0]])])

    return root


@pytest.fixture
def taxonomy():
    return ingest_mod.load_taxonomy()


def test_ingest_rows_cover_only_the_joint_task(joint_dataset, taxonomy):
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    assert {row["image_name"] for row in rows} == {
        "WIN_20220329_15_46_12_Pro.jpg",
        "c_45alt.jpg",
    }
    assert len(rows) == 4


def test_placement_labels_never_appear_in_the_output(joint_dataset, taxonomy):
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    raw_labels = {row["raw_label"] for row in rows}
    assert "good" not in raw_labels
    assert "no_good" not in raw_labels
    assert all(row["viewpoint"] in ("V2", "V2.1") for row in rows)


def test_class_counts_match_the_mapping(joint_dataset, taxonomy):
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    assert Counter(row["class"] for row in rows) == {
        "excess": 1,
        "spike": 2,
        "insufficient": 1,
    }


def test_polygon_rank_x_orders_left_to_right(joint_dataset, taxonomy):
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    v2_rows = {row["raw_label"]: row for row in rows if row["viewpoint"] == "V2"}
    assert v2_rows["spike"]["polygon_rank_x"] == 1  # x=10
    assert v2_rows["exc_solder"]["polygon_rank_x"] == 2  # x=50
    assert v2_rows["spike"]["polygon_count"] == 2
    assert v2_rows["exc_solder"]["polygon_count"] == 2


def test_capture_timestamp_parsed_from_camera_filenames(joint_dataset, taxonomy):
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    by_image = {row["image_name"]: row for row in rows}
    assert by_image["WIN_20220329_15_46_12_Pro.jpg"]["capture_timestamp"] == "2022-03-29T15:46:12"
    assert by_image["c_45alt.jpg"]["capture_timestamp"] == ""


def test_light_direction_follows_the_viewpoint_assumption(joint_dataset, taxonomy):
    """V2 = bottom_to_top, V2.1 = top_to_bottom — an assumed convention, not a measured fact."""
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    for row in rows:
        expected = ingest_mod.LIGHT_DIRECTION_BY_VIEWPOINT[row["viewpoint"]]
        assert row["light_direction"] == expected


def test_ingest_rows_keep_points_as_a_native_list(joint_dataset, taxonomy):
    """Callers that stay in Python (e.g. qa-polygons) get real coordinates, not a string."""
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    row = next(row for row in rows if row["raw_label"] == "poor_solder")
    assert row["points"] == [[1, 0]]


def test_csv_points_column_is_valid_json(joint_dataset, tmp_path):
    out = tmp_path / "polygons.csv"
    ingest_mod.write_polygons_csv(joint_dataset, out)
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(row for row in reader if row["raw_label"] == "poor_solder")
    assert json.loads(row["points"]) == [[1, 0]]


def test_an_unmapped_label_fails(joint_dataset, taxonomy):
    labeled = joint_dataset / "Labeled"
    (joint_dataset / "Dataset" / "CS1" / "R0805" / "V2" / "e_extra.jpg").write_bytes(b"fake")
    (labeled / "e_extra.jpg").write_bytes(b"fake")
    (labeled / "e_extra.json").write_text(
        json.dumps(labelme_record("e_extra.jpg", [("mystery_label", [[0, 0]])]))
    )
    result = audit_mod.audit(joint_dataset)
    with pytest.raises(ValueError, match="unmapped label"):
        ingest_mod.ingest_rows(result, taxonomy)


def test_cli_writes_the_csv(joint_dataset, tmp_path):
    out = tmp_path / "polygons.csv"
    result = runner.invoke(
        app,
        ["ingest", "--root", str(joint_dataset), "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(ingest_mod.CSV_FIELDS)
    assert len(lines) == 5  # header + 4 polygon rows


def test_cli_fails_clearly_when_root_is_missing(tmp_path):
    result = runner.invoke(
        app, ["ingest", "--root", str(tmp_path / "nope"), "--out", str(tmp_path / "out.csv")]
    )
    assert result.exit_code == 1


def test_cli_reports_an_unmapped_label_without_a_traceback(joint_dataset, tmp_path):
    (joint_dataset / "Labeled" / "e_extra.json").write_text(
        json.dumps(labelme_record("e_extra.jpg", [("mystery_label", [[0, 0]])]))
    )
    (joint_dataset / "Dataset" / "CS1" / "R0805" / "V2" / "e_extra.jpg").write_bytes(b"fake")
    (joint_dataset / "Labeled" / "e_extra.jpg").write_bytes(b"fake")
    result = runner.invoke(
        app,
        ["ingest", "--root", str(joint_dataset), "--out", str(tmp_path / "out.csv")],
    )
    assert result.exit_code == 1
    assert "unmapped label" in result.output


def square(cx: float, half: float) -> list[list[float]]:
    """A `2*half`-wide square polygon centered on `cx` (y is irrelevant to these tests)."""
    top = 10
    bottom = 10 + 2 * half
    return [[cx - half, top], [cx + half, top], [cx + half, bottom], [cx - half, bottom]]


@pytest.fixture
def three_poly_dataset(tmp_path):
    """notes/s1_three_polygons.md's documented shape, plus the two real exceptions it discovered.

    - merge_a: the documented pattern — spike overlapping excess on the left joint (gap 15
      between their centers), poor_solder alone on the right joint (gap 275 to the left pair).
      Expect: spike+excess merges (spike wins), poor_solder untouched.
    - merge_b: same-class duplicates (two `good`/normal squares of different sizes) on the left
      joint, spike alone on the right. Expect: normal+normal merges, keeping the larger square.
    - merge_c: an ordinary already-2-joint image; the two polygons are far enough apart that
      cluster_joints must NOT merge them.
    """
    root = tmp_path

    def place(name: str) -> Path:
        folder = root / "Dataset" / "CS1" / "R0805" / "V2"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(b"fake jpeg bytes")
        return path

    labeled = root / "Labeled"
    labeled.mkdir()

    def annotate(name: str, shapes: list[tuple[str, list]]) -> None:
        image_path = place(name)
        (labeled / name).write_bytes(image_path.read_bytes())
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(labelme_record(name, shapes))
        )

    annotate(
        "merge_a.jpg",
        [
            ("exc_solder", square(20, 10)),  # center_x=20, area=400
            ("spike", square(35, 10)),  # center_x=35, area=400 -- close to exc_solder
            ("poor_solder", square(310, 10)),  # center_x=310 -- far from the pair above
        ],
    )
    annotate(
        "merge_b.jpg",
        [
            ("good", square(20, 10)),  # normal, area=400
            ("good", square(20, 15)),  # normal, area=900 -- same center, larger
            ("spike", square(310, 10)),
        ],
    )
    annotate(
        "merge_c.jpg",
        [
            ("poor_solder", square(20, 10)),
            ("spike", square(310, 10)),
        ],
    )
    return root


def test_cluster_joints_splits_at_the_largest_gap():
    rows = [
        {"points": square(20, 10)},
        {"points": square(35, 10)},
        {"points": square(310, 10)},
    ]
    clusters = ingest_mod.cluster_joints(rows)
    assert sorted(clusters) == [[0, 1], [2]]


def test_cluster_joints_single_polygon_is_one_cluster():
    rows = [{"points": square(20, 10)}]
    assert ingest_mod.cluster_joints(rows) == [[0]]


def test_cluster_joints_two_polygons_far_apart_are_two_clusters():
    rows = [{"points": square(20, 10)}, {"points": square(310, 10)}]
    assert ingest_mod.cluster_joints(rows) == [[0], [1]]


def test_reduce_joint_cluster_leaves_a_single_row_unchanged():
    row = {"class": "excess", "points": square(20, 10)}
    result = ingest_mod.reduce_joint_cluster([row])
    assert result["class"] == "excess"
    assert result["merged_from"] == ""


def test_reduce_joint_cluster_applies_precedence_between_different_classes():
    excess = {"class": "excess", "points": square(20, 10)}
    spike = {"class": "spike", "points": square(35, 10)}
    result = ingest_mod.reduce_joint_cluster([excess, spike])
    assert result["class"] == "spike"  # spike beats excess
    assert result["merged_from"] == "spike+excess"


def test_reduce_joint_cluster_keeps_exactly_one_same_class_duplicate():
    """Which one is kept is random by design, so assert the invariant, not the winner."""
    small = {"class": "normal", "points": square(20, 10)}
    large = {"class": "normal", "points": square(20, 15)}
    result = ingest_mod.reduce_joint_cluster([small, large])
    assert result["class"] == "normal"
    assert result["merged_from"] == "normal+normal"
    # Whichever survived, the crop box still spans both polygons.
    xs = [p[0] for p in result["points"]]
    assert (min(xs), max(xs)) == (5, 35)


def test_reduce_joint_cluster_points_are_the_union_bounding_box():
    excess = {"class": "excess", "points": square(20, 10)}  # x in [10, 30]
    spike = {"class": "spike", "points": square(35, 10)}  # x in [25, 45]
    result = ingest_mod.reduce_joint_cluster([excess, spike])
    xs = [p[0] for p in result["points"]]
    assert min(xs) == 10
    assert max(xs) == 45


def test_merge_double_defect_joints_matches_the_documented_pattern(three_poly_dataset, taxonomy):
    result = audit_mod.audit(three_poly_dataset)
    raw_rows = ingest_mod.ingest_rows(result, taxonomy)
    merged_rows = ingest_mod.merge_double_defect_joints(raw_rows)

    by_image: dict[str, list[dict]] = {}
    for row in merged_rows:
        by_image.setdefault(row["image_name"], []).append(row)

    merge_a = by_image["merge_a.jpg"]
    assert len(merge_a) == 2
    merged = next(r for r in merge_a if r["merged_from"])
    assert merged["class"] == "spike"
    assert merged["merged_from"] == "spike+excess"
    untouched = next(r for r in merge_a if not r["merged_from"])
    assert untouched["class"] == "insufficient"


def test_merge_double_defect_joints_handles_same_class_pairs(three_poly_dataset, taxonomy):
    """merge_b's two `normal` polygons on one joint collapse to one, with no review needed."""
    result = audit_mod.audit(three_poly_dataset)
    raw_rows = ingest_mod.ingest_rows(result, taxonomy)
    merged_rows = ingest_mod.merge_double_defect_joints(raw_rows)
    merge_b = [r for r in merged_rows if r["image_name"] == "merge_b.jpg"]
    assert len(merge_b) == 2
    assert any(r["merged_from"] == "normal+normal" for r in merge_b)


def test_merge_double_defect_joints_leaves_already_separate_joints_alone(
    three_poly_dataset, taxonomy
):
    result = audit_mod.audit(three_poly_dataset)
    raw_rows = ingest_mod.ingest_rows(result, taxonomy)
    merged_rows = ingest_mod.merge_double_defect_joints(raw_rows)
    merge_c = [r for r in merged_rows if r["image_name"] == "merge_c.jpg"]
    assert len(merge_c) == 2
    assert all(r["merged_from"] == "" for r in merge_c)


def test_finalize_positions_assigns_left_and_right(three_poly_dataset, taxonomy):
    result = audit_mod.audit(three_poly_dataset)
    raw_rows = ingest_mod.ingest_rows(result, taxonomy)
    merged_rows = ingest_mod.merge_double_defect_joints(raw_rows)
    merge_a = {r["class"]: r for r in merged_rows if r["image_name"] == "merge_a.jpg"}
    assert merge_a["spike"]["joint_position"] == "left"
    assert merge_a["insufficient"]["joint_position"] == "right"
    assert merge_a["spike"]["polygon_count"] == 2


def test_write_polygons_csv_returns_raw_and_merged_rows(three_poly_dataset, tmp_path):
    out = tmp_path / "polygons.csv"
    raw_rows, merged_rows = ingest_mod.write_polygons_csv(three_poly_dataset, out)
    assert len(raw_rows) == 8  # merge_a: 3, merge_b: 3, merge_c: 2 polygons
    assert len(merged_rows) == 6  # merge_a: 2, merge_b: 2, merge_c: 2 joints

    with out.open(newline="", encoding="utf-8") as f:
        rows_written = list(csv.DictReader(f))
    assert len(rows_written) == len(merged_rows)
    assert {row["merged_from"] for row in rows_written} >= {"", "spike+excess", "normal+normal"}


def test_cli_prints_before_and_after_class_counts(three_poly_dataset, tmp_path):
    out = tmp_path / "polygons.csv"
    result = runner.invoke(app, ["ingest", "--root", str(three_poly_dataset), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "before merge:" in result.output
    assert "after merge (notes/s1_three_polygons.md):" in result.output


REAL_ROOT = Path("data/raw")


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="requires the real SolDef_AI download (gitignored)"
)
def test_real_dataset_matches_the_audited_counts():
    result = audit_mod.audit(REAL_ROOT)
    rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    images = {row["image_name"] for row in rows}
    assert len(images) == 200
    assert len(rows) == 443
    assert Counter(row["class"] for row in rows) == {
        "excess": 182,
        "spike": 130,
        "normal": 74,
        "insufficient": 57,
    }


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="requires the real SolDef_AI download (gitignored)"
)
def test_real_dataset_merge_matches_stage1_note():
    result = audit_mod.audit(REAL_ROOT)
    raw_rows = ingest_mod.ingest_rows(result, ingest_mod.load_taxonomy())
    merged_rows = ingest_mod.merge_double_defect_joints(raw_rows)
    assert len(merged_rows) == 400
    assert sum(1 for row in merged_rows if row["merged_from"]) == 43
    assert Counter(row["class"] for row in merged_rows) == {
        "excess": 157,
        "spike": 121,
        "normal": 65,
        "insufficient": 57,
    }
    # The two cases outside "always spike + a different class" still merge, by the same rule.
    merged_from = Counter(row["merged_from"] for row in merged_rows if row["merged_from"])
    assert merged_from["insufficient+normal"] == 1
    assert merged_from["spike+spike"] == 1
