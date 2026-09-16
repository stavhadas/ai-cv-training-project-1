"""Tests for `pcbi audit`, built on synthetic datasets.

The real SolDef_AI download is gigabytes and git-ignored, so CI cannot use it. These fixtures
reproduce its documented shape at a size that runs in milliseconds:

- `dataset` / `mirrored`: generic LabelMe layouts, used for parsing, layout-detection, and
  path/duplicate mechanics that don't depend on the board/package/viewpoint hierarchy.
- `soldef_like`: the real download's actual shape — `Dataset/<board>/<package>/<viewpoint>[/setup]`
  plus a flat `Labeled/` copy of the annotated subset — used for the board-aware sections that
  Step 4's updated spec added (task split, cell table, joint-task coverage, brightness).
"""

import base64
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import audit as audit_mod

runner = CliRunner()

# Stands in for a real embedded image: big enough that a test would notice if it were kept.
FAKE_IMAGE_DATA = base64.b64encode(b"\xff\xd8\xff" * 40_000).decode()


def labelme_record(image_name, labels, width=1280, height=960):
    return {
        "version": "5.0.1",
        "imagePath": f"..\\images\\{image_name}",  # LabelMe writes the annotator's own separators
        "imageWidth": width,
        "imageHeight": height,
        "imageData": FAKE_IMAGE_DATA,
        "shapes": [
            {"label": label, "shape_type": "polygon", "points": [[1, 2], [3, 4], [5, 6]]}
            for label in labels
        ],
    }


def make_jpeg(path: Path, brightness: int = 128, size: tuple[int, int] = (32, 32)) -> None:
    """A solid-color JPEG at approximately the given grayscale brightness (0-255)."""
    Image.new("L", size, color=brightness).save(path, format="JPEG", quality=95)


@pytest.fixture
def dataset(tmp_path):
    """A small dataset in the one-JSON-per-image layout."""
    images = tmp_path / "Solder" / "45"
    images.mkdir(parents=True)
    annotations = tmp_path / "Solder" / "annotations"
    annotations.mkdir()

    plan = {
        "R0805_01_L1_45.jpg": ["good", "good", "exc_solder"],
        "R0805_01_L2_45.jpg": ["good", "spike"],
        "C0805_02_L1_45.jpg": ["poor_solder"],
        "C0805_02_L2_45.jpg": ["good"],
    }
    for name, labels in plan.items():
        (images / name).write_bytes(b"not a real jpeg")
        (annotations / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(labelme_record(name, labels))
        )

    # An unannotated image, and a non-image file, to check they are counted separately.
    (tmp_path / "Placement" / "top").mkdir(parents=True)
    (tmp_path / "Placement" / "top" / "R1206_03_L1_top.png").write_bytes(b"not a real png")
    (tmp_path / "README.txt").write_text("dataset notes")
    return tmp_path


def test_counts_images_and_annotations(dataset):
    result = audit_mod.audit(dataset)
    assert len(result.images) == 5
    assert len(result.json_files) == 4
    assert len(result.annotations) == 4
    assert result.extension_counts == {".jpg": 4, ".png": 1}


def test_label_counts_match_the_fixture(dataset):
    result = audit_mod.audit(dataset)
    assert result.label_counts == {"good": 4, "exc_solder": 1, "spike": 1, "poor_solder": 1}


def test_detects_one_json_per_image_layout(dataset):
    result = audit_mod.audit(dataset)
    assert result.layouts == {"one JSON per image": 4}


def test_image_data_is_stripped_not_parsed(dataset):
    """The base64 blob must never reach the parsed record, let alone the report."""
    path = next((dataset / "Solder" / "annotations").iterdir())
    parsed = audit_mod.load_json_without_image_data(path)
    assert parsed["imageData"] is None
    assert FAKE_IMAGE_DATA not in audit_mod.render_markdown(audit_mod.audit(dataset))


def test_image_path_is_reduced_to_a_leaf_name(dataset):
    result = audit_mod.audit(dataset)
    names = {record.image_name for record in result.annotations}
    assert names == {
        "R0805_01_L1_45.jpg",
        "R0805_01_L2_45.jpg",
        "C0805_02_L1_45.jpg",
        "C0805_02_L2_45.jpg",
    }


def test_token_table_positions(dataset):
    result = audit_mod.audit(dataset)
    rows, token_counts = audit_mod.token_table(result.images)
    assert token_counts == {4: 5}
    assert [row["distinct"] for row in rows] == [3, 3, 2, 2]
    # Position 1 looks like a component id: 3 distinct values, shared by 2, 2 and 1 images.
    assert rows[1]["max_group"] == 2


def test_combined_list_layout_is_detected(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "R0603_07_L1_45.jpg").write_bytes(b"x")
    (tmp_path / "all.json").write_text(
        json.dumps(
            [
                labelme_record("R0603_07_L1_45.jpg", ["good"]),
                labelme_record("R0603_07_L2_45.jpg", ["no_good"]),
            ]
        )
    )
    result = audit_mod.audit(tmp_path)
    assert result.layouts == {"combined JSON (list of records)": 1}
    assert len(result.annotations) == 2
    assert result.label_counts == {"good": 1, "no_good": 1}


def test_combined_mapping_layout_is_detected(tmp_path):
    (tmp_path / "all.json").write_text(
        json.dumps({"a.jpg": labelme_record("a.jpg", ["good", "spike"])})
    )
    result = audit_mod.audit(tmp_path)
    assert result.layouts == {"combined JSON (mapping of image name to record)": 1}
    assert result.label_counts == {"good": 1, "spike": 1}


def test_malformed_json_is_reported_not_raised(dataset):
    (dataset / "broken.json").write_text("{ not json")
    result = audit_mod.audit(dataset)
    assert [source for source, _ in result.unreadable] == ["broken.json"]
    assert len(result.annotations) == 4  # the good files still parsed


def test_cli_writes_the_report(dataset, tmp_path):
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["audit", "--root", str(dataset), "--out", str(out)])
    assert result.exit_code == 0
    assert out.is_file()
    assert "5 image files, 4 JSON files" in result.stdout


def test_cli_fails_clearly_when_root_is_missing(tmp_path):
    result = runner.invoke(app, ["audit", "--root", str(tmp_path / "nope")])
    assert result.exit_code == 1


# --------------------------------------------------------------------------------------------
# The shape the real SolDef_AI download turned out to have: a deep archive organised by board /
# package / viewpoint, plus a flat `Labeled/` folder holding byte-copies of part of it next to the
# annotations. Filenames are camera timestamps, so they describe nothing about the subject.
# --------------------------------------------------------------------------------------------

TIMESTAMP = "WIN_20220329_14_30_{:02d}_Pro.jpg"

VIEWPOINTS = ("V1/Setup1", "V1/Setup2", "V2")


@pytest.fixture
def mirrored(tmp_path):
    """Returns (root, {image name: archive folder}, [annotated names])."""
    serial = iter(range(1, 99))
    archive: dict[str, str] = {}

    for board, package, components in (("CS1", "R0805", 2), ("CS2", "R1206", 1)):
        for viewpoint in VIEWPOINTS:
            folder = tmp_path.joinpath("Dataset", board, package, *viewpoint.split("/"))
            folder.mkdir(parents=True, exist_ok=True)
            for _ in range(components):
                name = TIMESTAMP.format(next(serial))
                (folder / name).write_bytes(b"jpeg")
                archive[name] = f"Dataset/{board}/{package}/{viewpoint}"

    labeled = tmp_path / "Labeled"
    labeled.mkdir()
    annotated = [name for name, folder in sorted(archive.items()) if folder.endswith("V1/Setup1")]
    for name in annotated:
        (labeled / name).write_bytes(b"jpeg")  # the same photo, filed a second time
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(labelme_record(name, ["good", "spike"]))
        )
    return tmp_path, archive, annotated


def test_a_flat_annotated_copy_is_not_counted_as_new_data(mirrored):
    root, archive, annotated = mirrored
    result = audit_mod.audit(root)
    assert len(result.images) == len(archive) + len(annotated) == 12
    assert result.distinct_names == len(archive) == 9
    assert set(result.duplicates) == set(annotated)


def test_tables_use_the_deep_copy_of_each_duplicate(mirrored):
    root, archive, _ = mirrored
    result = audit_mod.audit(root)
    chosen = audit_mod.deduplicated_images(result)
    assert len(chosen) == len(archive)
    assert not any(path.parent.name == "Labeled" for path in chosen)


def test_mirrored_folders_pairs_each_copy_with_its_source(mirrored):
    root, _, _ = mirrored
    rows = audit_mod.mirrored_folders(audit_mod.audit(root))
    assert [(row["folders"], row["shared"]) for row in rows] == [
        (("Dataset/CS1/R0805/V1/Setup1", "Labeled"), 2),
        (("Dataset/CS2/R1206/V1/Setup1", "Labeled"), 1),
    ]


def test_annotation_locations_recovers_the_folder_the_flat_copy_lost(mirrored):
    root, _, _ = mirrored
    folders, ambiguous, missing = audit_mod.annotation_locations(audit_mod.audit(root))
    assert folders == {"Dataset/CS1/R0805/V1/Setup1": 2, "Dataset/CS2/R1206/V1/Setup1": 1}
    assert ambiguous == [] and missing == []


def test_an_ambiguous_filename_join_is_flagged_not_guessed(mirrored):
    """A filename that resolves to two archive folders cannot be trusted as metadata."""
    root, _, annotated = mirrored
    clash = root.joinpath("Dataset", "CS3", "R0603", "V2")
    clash.mkdir(parents=True)
    (clash / annotated[0]).write_bytes(b"jpeg")

    result = audit_mod.audit(root)
    _, ambiguous, _ = audit_mod.annotation_locations(result)
    assert ambiguous == [annotated[0]]
    assert annotated[0] in audit_mod.render_markdown(result)


def test_an_annotation_with_no_image_file_is_reported(mirrored):
    root, _, _ = mirrored
    (root / "Labeled" / "orphan.json").write_text(
        json.dumps(labelme_record("nowhere_to_be_found.jpg", ["good"]))
    )
    _, _, missing = audit_mod.annotation_locations(audit_mod.audit(root))
    assert missing == ["nowhere_to_be_found.jpg"]


def test_path_segments_carry_what_the_filenames_do_not(mirrored):
    root, _, _ = mirrored
    result = audit_mod.audit(root)
    rows, depth_counts = audit_mod.segment_table(audit_mod.deduplicated_images(result), root)

    # Dataset/CS1/R0805/V1/Setup1 is five folders deep; V2 images stop one level short.
    assert depth_counts == {5: 6, 4: 3}
    assert [row["distinct"] for row in rows] == [1, 2, 2, 2, 2]
    assert rows[1]["examples"] == ["CS1", "CS2"]  # board
    assert set(rows[3]["examples"]) == {"V1", "V2"}  # viewpoint


def test_position_role_names_what_cannot_group(mirrored):
    root, archive, _ = mirrored
    rows, _ = audit_mod.token_table(audit_mod.deduplicated_images(audit_mod.audit(root)))
    roles = [audit_mod.position_role(row, len(archive)) for row in rows]
    # WIN / 20220329 / 14 / 30 / <seconds> / Pro
    assert roles == ["constant", "constant", "constant", "constant", "unique per file", "constant"]


def test_report_says_plainly_that_filenames_cannot_group_images(mirrored):
    root, _, _ = mirrored
    text = audit_mod.render_markdown(audit_mod.audit(root))
    assert "No filename position groups images" in text
    assert "6 of 6 positions therefore cannot group images at all" in text


def test_report_warns_that_the_file_count_double_counts(mirrored):
    root, _, _ = mirrored
    text = audit_mod.render_markdown(audit_mod.audit(root))
    assert "12 image files" in text and "9 distinct filenames" in text
    assert "not** the number of distinct photographs" in text


def test_cli_reports_the_duplicate_count(mirrored, tmp_path):
    root, _, _ = mirrored
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["audit", "--root", str(root), "--out", str(out)])
    assert result.exit_code == 0
    assert "9 distinct filenames" in result.stdout
    assert "3 appear in more" in result.stdout


def test_token_count_outliers_finds_the_odd_ones(tmp_path):
    normal_a = tmp_path / "a_1.jpg"
    normal_a.write_bytes(b"x")
    normal_b = tmp_path / "b_2.jpg"
    normal_b.write_bytes(b"x")
    odd = tmp_path / "c_3_extra.jpg"
    odd.write_bytes(b"x")
    assert audit_mod.token_count_outliers([normal_a, normal_b, odd]) == [odd]


def test_sample_duplicate_hashes_detects_a_genuine_collision(tmp_path):
    folder_a = tmp_path / "one"
    folder_a.mkdir()
    folder_b = tmp_path / "two"
    folder_b.mkdir()
    (folder_a / "same_name.jpg").write_bytes(b"AAAA")
    (folder_b / "same_name.jpg").write_bytes(b"BBBB")  # different photo, same filename

    result = audit_mod.audit(tmp_path)
    rows = audit_mod.sample_duplicate_hashes(result)
    assert rows == [{"name": "same_name.jpg", "folders": ["one", "two"], "identical": False}]


def test_sample_duplicate_hashes_confirms_a_real_copy(mirrored):
    root, _, annotated = mirrored
    result = audit_mod.audit(root)
    rows = audit_mod.sample_duplicate_hashes(result)
    assert rows and all(row["identical"] for row in rows)


def test_report_flags_a_byte_level_mismatch(tmp_path):
    folder_a = tmp_path / "one"
    folder_a.mkdir()
    folder_b = tmp_path / "two"
    folder_b.mkdir()
    (folder_a / "same_name.jpg").write_bytes(b"AAAA")
    (folder_b / "same_name.jpg").write_bytes(b"BBBB")

    text = audit_mod.render_markdown(audit_mod.audit(tmp_path))
    assert "different photos" in text
    assert "not copies" in text


# --------------------------------------------------------------------------------------------
# `classify_path`: the function everything board/package/viewpoint-aware is built on.
# --------------------------------------------------------------------------------------------


def test_classify_path_reads_board_package_viewpoint_and_setup():
    info = audit_mod.classify_path(("Dataset", "CS1", "R0805", "V1", "Setup1"))
    assert (info.board, info.package, info.viewpoint, info.setup) == (
        "CS1",
        "R0805",
        "V1",
        "Setup1",
    )
    assert info.viewpoint_folder == "V1/Setup1"


def test_classify_path_handles_a_viewpoint_with_no_setup_folder():
    info = audit_mod.classify_path(("Dataset", "CS1", "R0805", "V2.1"))
    assert info.setup is None
    assert info.viewpoint_folder == "V2.1"


def test_classify_path_does_not_split_v2_1_on_the_dot():
    """V2.1 must survive as one token — path segments are split on `/`, never on `.`."""
    info = audit_mod.classify_path(("Dataset", "CS1", "R0805", "V2.1"))
    assert info.viewpoint == "V2.1"


def test_classify_path_returns_all_none_without_a_viewpoint_folder():
    info = audit_mod.classify_path(("Labeled",))
    assert (info.board, info.package, info.viewpoint, info.setup) == (None, None, None, None)
    assert info.task is None
    assert info.in_scope is None


def test_task_and_scope_are_read_off_path_info():
    assert audit_mod.PathInfo("CS1", "R0805", "V1", "Setup1").task == "placement"
    assert audit_mod.PathInfo("CS1", "R0805", "V2", None).task == "joint"
    assert audit_mod.PathInfo("CS1", "R0805", "V2.1", None).task == "joint"
    assert audit_mod.PathInfo("CS1", "R0805", "V3", None).task is None  # axonometric, unannotated
    assert audit_mod.PathInfo("CS1", "R0805", "V1", "Setup1").in_scope is True
    assert audit_mod.PathInfo("CS3", "LED_TH", "V1", "Setup1").in_scope is False


# --------------------------------------------------------------------------------------------
# The real download's board/package/viewpoint hierarchy, with one ragged cell (a shot missing),
# one out-of-scope through-hole package, and two `V1` lighting setups at different brightness.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def soldef_like(tmp_path):
    root = tmp_path

    def place(board, package, viewpoint, setup, name, brightness=128):
        parts = [root, "Dataset", board, package, viewpoint]
        if setup:
            parts.append(setup)
        folder = Path(*parts)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        make_jpeg(path, brightness)
        return path

    placement = [
        place("CS1", "R0805", "V1", "Setup1", "img_001.jpg", brightness=60),  # darker
        place("CS1", "R0805", "V1", "Setup1", "img_002.jpg", brightness=60),
        place("CS1", "R0805", "V1", "Setup2", "img_003.jpg", brightness=200),  # brighter
        place("CS1", "R0805", "V1", "Setup2", "img_004.jpg", brightness=200),
    ]
    place("CS1", "R0805", "V3", None, "img_005.jpg")  # axonometric, never annotated

    joint_v2 = [
        place("CS1", "R0805", "V2", None, "img_006.jpg"),
        place("CS1", "R0805", "V2", None, "img_007.jpg"),
    ]
    joint_v21 = [
        place("CS1", "R0805", "V2.1", None, "img_008.jpg"),  # one short of V2 -> ragged cell
    ]

    # Through-hole: out of the paper's four SMD packages, so out of project scope.
    place("CS3", "LED_TH", "V1", "Setup1", "img_100.jpg")

    labeled = root / "Labeled"
    labeled.mkdir()

    def annotate(image_path: Path, labels: list[str]) -> None:
        name = image_path.name
        (labeled / name).write_bytes(image_path.read_bytes())
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(labelme_record(name, labels))
        )

    annotate(placement[0], ["good"])
    annotate(placement[1], ["no_good"])
    for path in joint_v2:
        annotate(path, ["exc_solder", "spike"])
    for path in joint_v21:
        annotate(path, ["good"])

    return root


def test_annotations_by_task_splits_joint_from_placement(soldef_like):
    tasks = audit_mod.annotations_by_task(audit_mod.audit(soldef_like))
    assert len(tasks["joint"]) == 3
    assert len(tasks["placement"]) == 2
    assert not tasks.get("unclassified")


def test_task_buckets_keep_the_shared_label_good_apart(soldef_like):
    """`good` means different things in each task; pooling them would be the exact bug Step 4
    calls out."""
    tasks = audit_mod.annotations_by_task(audit_mod.audit(soldef_like))

    placement_labels: Counter[str] = Counter()
    for record in tasks["placement"]:
        placement_labels.update(record.labels)
    assert placement_labels == {"good": 1, "no_good": 1}

    joint_labels: Counter[str] = Counter()
    for record in tasks["joint"]:
        joint_labels.update(record.labels)
    assert joint_labels == {"exc_solder": 2, "spike": 2, "good": 1}


def test_an_unresolvable_annotation_lands_in_unclassified_not_dropped(soldef_like):
    (soldef_like / "Labeled" / "ghost.json").write_text(
        json.dumps(labelme_record("ghost.jpg", ["mystery"]))
    )
    tasks = audit_mod.annotations_by_task(audit_mod.audit(soldef_like))
    assert len(tasks["unclassified"]) == 1
    assert tasks["unclassified"][0].image_name == "ghost.jpg"


def test_cell_table_flags_the_ragged_cell_and_the_out_of_scope_package(soldef_like):
    result = audit_mod.audit(soldef_like)
    rows, unclassified = audit_mod.cell_table(audit_mod.deduplicated_images(result), soldef_like)
    by_key = {(row["board"], row["package"]): row for row in rows}

    assert by_key[("CS1", "R0805")]["ragged"] is True
    assert by_key[("CS1", "R0805")]["in_scope"] is True
    assert by_key[("CS3", "LED_TH")]["ragged"] is False
    assert by_key[("CS3", "LED_TH")]["in_scope"] is False
    assert unclassified == 0  # every image here sits under a recognizable viewpoint folder


def test_joint_task_coverage_reports_v2_and_v21_together(soldef_like):
    result = audit_mod.audit(soldef_like)
    cells, directions = audit_mod.joint_task_coverage(result)
    assert cells == [{"board": "CS1", "package": "R0805", "images": 3}]
    assert directions == [{"board": "CS1", "package": "R0805", "counts": {"V2": 2, "V2.1": 1}}]


def test_brightness_table_distinguishes_the_two_lighting_setups(soldef_like):
    result = audit_mod.audit(soldef_like)
    rows, available = audit_mod.brightness_table(audit_mod.deduplicated_images(result), soldef_like)
    assert available is True
    by_setup = {(row["board"], row["setup"]): row["mean_brightness"] for row in rows}
    assert by_setup[("CS1", "Setup1")] < 100
    assert by_setup[("CS1", "Setup2")] > 150


def test_brightness_table_only_measures_the_v1_viewpoint(soldef_like):
    result = audit_mod.audit(soldef_like)
    rows, _ = audit_mod.brightness_table(audit_mod.deduplicated_images(result), soldef_like)
    boards_measured = {row["board"] for row in rows}
    assert boards_measured == {"CS1", "CS3"}  # both have a V1 shot; V2/V2.1/V3 never contribute


def test_brightness_table_skips_a_file_that_is_not_a_real_image(soldef_like):
    (soldef_like / "Dataset" / "CS1" / "R0805" / "V1" / "Setup1" / "fake.jpg").write_bytes(
        b"not a real jpeg"
    )
    result = audit_mod.audit(soldef_like)
    rows, available = audit_mod.brightness_table(audit_mod.deduplicated_images(result), soldef_like)
    assert available is True  # one bad file must not take down the whole section
    by_setup = {(row["board"], row["setup"]): row["images"] for row in rows}
    assert by_setup[("CS1", "Setup1")] == 2  # the fake file was skipped, not counted


def test_brightness_table_reports_when_pillow_is_unavailable(monkeypatch, soldef_like):
    import builtins

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)
    result = audit_mod.audit(soldef_like)
    rows, available = audit_mod.brightness_table(audit_mod.deduplicated_images(result), soldef_like)
    assert rows == []
    assert available is False


def test_report_has_the_12_step4_sections(soldef_like):
    text = audit_mod.render_markdown(audit_mod.audit(soldef_like))
    for heading in (
        "## 1. Folder tree",
        "## 2. Image counts",
        "## 3. Annotation layout",
        "## 4. Path segments",
        "## 5. Filename tokens",
        "## 6. Annotated-image join",
        "## 7. Label counts per task",
        "## 8. Polygons per image per task",
        "## 9. Cell table",
        "## 10. Joint-task coverage",
        "## 11. Image brightness",
        "## 12. Metadata table",
    ):
        assert heading in text
    assert "Joint-quality task" in text
    assert "Placement task" in text
    assert "Where encoded" in text and "Status" in text


def _cell_table_rows(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "## 9. Cell table")
    end = next(i for i, line in enumerate(lines) if line.startswith("## 10.") and i > start)
    return [line for line in lines[start:end] if line.startswith("| CS")]


def test_report_marks_the_through_hole_package_out_of_scope(soldef_like):
    text = audit_mod.render_markdown(audit_mod.audit(soldef_like))
    led_row = next(line for line in _cell_table_rows(text) if "LED_TH" in line)
    assert "out of scope" in led_row


def test_report_flags_the_ragged_cell(soldef_like):
    text = audit_mod.render_markdown(audit_mod.audit(soldef_like))
    r0805_row = next(line for line in _cell_table_rows(text) if "R0805" in line)
    assert "yes" in r0805_row


def test_report_notes_which_setup_is_brighter(soldef_like):
    text = audit_mod.render_markdown(audit_mod.audit(soldef_like))
    assert "is brightest" in text
    assert "Setup2" in text


def test_cli_prints_task_breakdown_instead_of_pooled_labels(soldef_like, tmp_path):
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["audit", "--root", str(soldef_like), "--out", str(out)])
    assert result.exit_code == 0
    assert "joint: 3 images" in result.stdout
    assert "placement: 2 images" in result.stdout
    assert "exc_solder: 2" in result.stdout
    assert "no_good: 1" in result.stdout
