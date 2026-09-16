"""Tests for `pcbi ingest`, Stage 1's polygon-level CSV.

`joint_dataset` reproduces the real download's shape at a size that runs in milliseconds: a V2 and
a V2.1 image under one (board, package) cell (with distinct, orderable polygon x-positions), plus a
placement (V1) image whose labels must never reach the output.

`test_real_dataset_matches_the_audited_counts` runs against the actual SolDef_AI download and is
skipped when it isn't present (it's gitignored and not available in CI) — it is what actually
verifies the class counts against reports/data_audit.md (182 / 130 / 74 / 57).
"""

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


def test_points_round_trip_as_json(joint_dataset, taxonomy):
    result = audit_mod.audit(joint_dataset)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    row = next(row for row in rows if row["raw_label"] == "poor_solder")
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
