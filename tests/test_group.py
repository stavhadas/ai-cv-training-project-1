"""Tests for `pcbi group`, Stage 1 Step 4's candidate groupings.

The pure logic (union-find, distance thresholds, id assignment) is tested directly on hand-built
inputs, with no images and no torch. `grouped_dataset` builds a small real-JPEG tree for the paths
that must actually open files, and the `embed` path takes its embedder as an argument so it can be
exercised without downloading pretrained weights in CI.
"""

import csv
import json
import random
from pathlib import Path

import pytest
from PIL import Image, ImageStat
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import audit as audit_mod
from pcbi.data import group as group_mod
from pcbi.data import ingest as ingest_mod

runner = CliRunner()


def labelme_record(image_name: str, shapes: list[tuple[str, list]]) -> dict:
    return {
        "version": "5.0.1",
        "imagePath": image_name,
        "imageWidth": 64,
        "imageHeight": 64,
        "imageData": None,
        "shapes": [
            {"label": label, "shape_type": "polygon", "points": points} for label, points in shapes
        ],
    }


def noise_bytes(seed: int, low: int, high: int, size: tuple[int, int]) -> bytes:
    """Deterministic pixel noise in a brightness band.

    Noise rather than a flat colour because a perceptual hash of a uniform image is degenerate —
    two different solid colours can hash identically, which would make the phash tests meaningless.
    """
    rng = random.Random(seed)
    return bytes(rng.randrange(low, high) for _ in range(size[0] * size[1] * 3))


def make_jpeg(path: Path, data: bytes, size: tuple[int, int] = (64, 64)) -> None:
    Image.frombytes("RGB", size, data).save(path, format="JPEG", quality=95)


@pytest.fixture
def grouped_dataset(tmp_path):
    """Four joint-task images across two cells, with known visual relationships.

    - `a` and `b` are byte-identical, so any working similarity method must merge them. `b` also
      carries a `_C7` suffix, the only component signal the real dataset ever has.
    - `c` and `d` are distinct, and sit in *different* cells while sharing a brightness band — so a
      method that merges them proves grouping is compared globally, not clamped within a cell.
    """
    root = tmp_path
    dark = noise_bytes(seed=1, low=0, high=90, size=(64, 64))
    bright_c = noise_bytes(seed=2, low=170, high=255, size=(64, 64))
    bright_d = noise_bytes(seed=3, low=170, high=255, size=(64, 64))

    labeled = root / "Labeled"
    labeled.mkdir()

    def place(board: str, package: str, viewpoint: str, name: str, data: bytes) -> None:
        folder = root / "Dataset" / board / package / viewpoint
        folder.mkdir(parents=True, exist_ok=True)
        make_jpeg(folder / name, data)
        make_jpeg(labeled / name, data)
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(
                labelme_record(
                    name,
                    [
                        ("spike", [[5, 5], [15, 5], [15, 15], [5, 15]]),
                        ("exc_solder", [[40, 5], [55, 5], [55, 15], [40, 15]]),
                    ],
                )
            )
        )

    place("CS1", "R0805", "V2", "a_WIN_20220330_13_11_58_Pro.jpg", dark)
    place("CS1", "R0805", "V2.1", "b_WIN_20220330_13_12_00_Pro_C7.jpg", dark)  # identical to `a`
    place("CS1", "R0805", "V2", "c_WIN_20220330_13_13_00_Pro.jpg", bright_c)
    place("CS2", "R0603", "V2", "d_WIN_20220330_16_07_40_Pro.jpg", bright_d)
    return root


@pytest.fixture
def images(grouped_dataset):
    result = audit_mod.audit(grouped_dataset)
    return group_mod.collect_images(result, ingest_mod.load_taxonomy())


def brightness_embedder(loaded: list[Image.Image]) -> list[list[float]]:
    """A stand-in for the CNN: dark images embed one way, bright images the other.

    Lets the whole embed path (load -> embed -> cosine -> union-find) run with torch never imported.
    """
    return [
        [1.0, 0.0] if ImageStat.Stat(image.convert("L")).mean[0] < 128 else [0.0, 1.0]
        for image in loaded
    ]


def letters(components, images) -> list[list[str]]:
    """Components as their images' name initials, so assertions don't depend on index order."""
    return sorted(
        sorted(images[index].image_name[0] for index in members) for members in components
    )


# --- pure logic, no images -------------------------------------------------------------------


def test_parse_component_hint_reads_the_suffix():
    assert group_mod.parse_component_hint("WIN_20220408_13_50_25_Pro_C12") == "C12"


def test_parse_component_hint_is_empty_when_there_is_no_suffix():
    assert group_mod.parse_component_hint("WIN_20220330_13_11_58_Pro") == ""


def test_union_find_merges_transitively():
    """A~B and B~C must put all three together, without A ever being compared to C."""
    union_find = group_mod.UnionFind(4)
    union_find.union(0, 1)
    union_find.union(1, 2)
    assert union_find.find(0) == union_find.find(2)
    assert union_find.find(3) != union_find.find(0)


def test_union_find_components_covers_every_index_exactly_once():
    components = group_mod.union_find_components(5, [(0, 1), (1, 2)])
    assert sorted(sorted(members) for members in components) == [[0, 1, 2], [3], [4]]


def test_hamming_pairs_respects_the_threshold():
    # 0x0 and 0x1 differ by one bit; 0xf differs from 0x0 by four.
    fingerprints = {0: "0", 1: "1", 2: "f"}
    assert group_mod.hamming_pairs(fingerprints, 1) == [(0, 1)]
    assert sorted(group_mod.hamming_pairs(fingerprints, 4)) == [(0, 1), (0, 2), (1, 2)]
    assert group_mod.hamming_pairs(fingerprints, 0) == []


def test_cosine_pairs_respects_the_threshold():
    vectors = {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]}
    assert group_mod.cosine_pairs(vectors, 0.99) == [(0, 1)]
    assert sorted(group_mod.cosine_pairs(vectors, -1.0)) == [(0, 1), (0, 2), (1, 2)]


def test_cosine_pairs_ignores_vector_magnitude():
    """Cosine similarity is about direction, so an unnormalized embedder still works."""
    vectors = {0: [1.0, 0.0], 1: [7.0, 0.0]}
    assert group_mod.cosine_pairs(vectors, 0.99) == [(0, 1)]


def test_assign_group_ids_is_prefixed_and_ordered_by_smallest_path():
    records = [
        group_mod.ImageRecord("z.jpg", "z.jpg", "CS1", "R0805", "V2", ""),
        group_mod.ImageRecord("a.jpg", "a.jpg", "CS1", "R0805", "V2", ""),
    ]
    # Components deliberately passed in the "wrong" order; ids must not depend on that.
    ids = group_mod.assign_group_ids([[0], [1]], records, "phash")
    assert ids[1] == "phash_001"  # a.jpg sorts first
    assert ids[0] == "phash_002"


# --- cropping --------------------------------------------------------------------------------


def test_crop_box_at_zero_tolerance_is_the_joint_box_itself():
    assert group_mod.crop_box((5.0, 5.0, 55.0, 15.0), 0.0, (64, 64)) == (5, 5, 55, 15)


def test_crop_box_rounds_outward_so_no_annotated_pixel_is_cut_off():
    """floor/ceil, not round: a crop that clips the polygon defeats the point of cropping to it."""
    assert group_mod.crop_box((5.4, 5.6, 54.4, 14.6), 0.0, (64, 64)) == (5, 5, 55, 15)


def test_crop_box_margin_scales_with_the_longer_side():
    """A wide, flat box and a tall, narrow one keep a comparable amount of board around them.

    Scaling each axis by its own length would stretch this 50x10 box into a squarish crop, which
    changes what the fingerprint sees without anyone asking for it.
    """
    x0, y0, x1, y1 = group_mod.crop_box((25.0, 25.0, 75.0, 35.0), 0.2, (200, 200))
    assert (x0, y0, x1, y1) == (15, 15, 85, 45)  # 0.2 * 50 = 10px on every side


def test_crop_box_clamps_to_the_frame_instead_of_going_negative():
    """There is no board beyond the frame, so an edge component gets an asymmetric crop."""
    assert group_mod.crop_box((5.0, 5.0, 55.0, 15.0), 1.0, (64, 64)) == (0, 0, 64, 64)


def test_collect_images_spans_every_polygon_not_just_the_first(images):
    """The crop box has to cover both leads; taking the first row's polygon would miss one."""
    record = next(r for r in images if r.image_name.startswith("a"))
    assert record.joint_box == (5.0, 5.0, 55.0, 15.0)


def test_load_image_crops_when_given_a_tolerance_and_not_otherwise(grouped_dataset, images):
    record = next(r for r in images if r.image_name.startswith("a"))
    assert group_mod.load_image(grouped_dataset, record).size == (64, 64)
    assert group_mod.load_image(grouped_dataset, record, tolerance=0.0).size == (50, 10)


def test_cropping_changes_what_gets_fingerprinted(grouped_dataset, images):
    """The whole premise: a hash of the component is not a hash of the frame around it."""
    full = group_mod.compute_signatures(grouped_dataset, images, "phash", tolerance=None)
    cropped = group_mod.compute_signatures(grouped_dataset, images, "phash", tolerance=0.0)
    assert full.keys() == cropped.keys()
    assert any(full[index] != cropped[index] for index in full)


def test_cropping_flows_through_the_embed_path_too(grouped_dataset, images):
    """`embed` must see the crop as well, or the two methods disagree about what they measured."""
    seen: list[tuple[int, int]] = []

    def recording_embedder(loaded):
        seen.extend(image.size for image in loaded)
        return brightness_embedder(loaded)

    group_mod.embed_all(images, grouped_dataset, recording_embedder, tolerance=0.0)
    assert seen == [(50, 10)] * 4


# --- with images -----------------------------------------------------------------------------


def test_collect_images_returns_each_joint_image_once_in_a_stable_order(images):
    paths = [record.dataset_path for record in images]
    assert len(images) == 4
    assert paths == sorted(paths)  # the stable order that makes group ids reproducible
    by_letter = {record.image_name[0]: record for record in images}
    assert by_letter["b"].component_hint == "C7"
    assert by_letter["a"].component_hint == ""
    assert by_letter["d"].cell == "CS2/R0603"


def test_coarse_components_group_by_cell(images):
    components = group_mod.coarse_components(images)
    assert letters(components, images) == [["a", "b", "c"], ["d"]]


def test_phash_merges_byte_identical_images_and_keeps_others_apart(grouped_dataset, images):
    components, fingerprints = group_mod.group_images(grouped_dataset, images, "phash", 0)
    by_letter = {images[index].image_name[0]: value for index, value in fingerprints.items()}
    assert by_letter["a"] == by_letter["b"]  # `a` and `b` are the same bytes
    assert letters(components, images) == [["a", "b"], ["c"], ["d"]]


def test_embed_path_runs_with_an_injected_embedder(grouped_dataset, images):
    """Dark `a`/`b` merge. `c` and `d` embed identically but sit in different cells, so they
    must NOT merge — comparisons never cross a cell boundary."""
    components, fingerprints = group_mod.group_images(
        grouped_dataset, images, "embed", 0.99, embedder=brightness_embedder
    )
    assert letters(components, images) == [["a", "b"], ["c"], ["d"]]
    assert fingerprints == {}  # a 512-float vector has no place in a CSV cell


def test_identical_images_in_different_cells_are_never_merged(grouped_dataset, images):
    """The strongest possible similarity signal must still lose to the cell boundary.

    `a` and `b` are byte-identical and share a cell, so they merge. A threshold loose enough to
    merge everything cannot pull `d` (a different cell) in with them.
    """
    components, _ = group_mod.group_images(grouped_dataset, images, "phash", 256)
    for members in components:
        assert len({images[index].cell for index in members}) == 1
    assert letters(components, images) == [["a", "b", "c"], ["d"]]


def test_embed_all_returns_one_vector_per_image_index(grouped_dataset, images):
    vectors = group_mod.embed_all(images, grouped_dataset, brightness_embedder, batch_size=2)
    assert sorted(vectors) == [0, 1, 2, 3]  # survives being split across batches


def test_group_images_rejects_an_unknown_method(grouped_dataset, images):
    with pytest.raises(ValueError, match="unknown method"):
        group_mod.group_images(grouped_dataset, images, "telepathy", None)


# --- CSV and CLI -----------------------------------------------------------------------------


def test_write_groups_csv_has_the_declared_header_and_one_row_per_image(grouped_dataset, tmp_path):
    out = tmp_path / "groups.csv"
    rows = group_mod.write_groups_csv(grouped_dataset, out, "coarse")
    assert len(rows) == 4
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(group_mod.CSV_FIELDS)
    assert len(lines) == 5


def test_written_csv_carries_everything_step_five_needs(grouped_dataset, tmp_path):
    """cell, component_hint and viewpoint must be present, or the checks need the tree again."""
    out = tmp_path / "groups.csv"
    group_mod.write_groups_csv(grouped_dataset, out, "coarse")
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_name = {row["image_name"][0]: row for row in rows}
    assert by_name["b"]["component_hint"] == "C7"
    assert by_name["b"]["cell"] == "CS1/R0805"
    assert by_name["b"]["viewpoint"] == "V2.1"
    assert by_name["a"]["group_id"] == by_name["c"]["group_id"]  # same cell
    assert by_name["a"]["group_size"] == "3"


def test_coarse_leaves_the_threshold_column_empty(grouped_dataset, tmp_path):
    out = tmp_path / "groups.csv"
    rows = group_mod.write_groups_csv(grouped_dataset, out, "coarse", threshold=None)
    assert {row["threshold"] for row in rows} == {""}


def test_the_csv_records_which_crop_produced_it(grouped_dataset, tmp_path):
    """Each CSV carries its own provenance, so two of them can never be confused later."""
    out = tmp_path / "groups.csv"
    rows = group_mod.write_groups_csv(grouped_dataset, out, "phash", threshold=0, tolerance=0.5)
    assert {row["crop_tolerance"] for row in rows} == {0.5}


def test_a_full_frame_run_leaves_the_crop_column_empty(grouped_dataset, tmp_path):
    out = tmp_path / "groups.csv"
    rows = group_mod.write_groups_csv(grouped_dataset, out, "phash", threshold=0, tolerance=None)
    assert {row["crop_tolerance"] for row in rows} == {""}


def test_coarse_reports_no_crop_even_when_one_is_passed(grouped_dataset, tmp_path):
    """`coarse` never opens an image, so claiming a crop tolerance in its CSV would be a lie."""
    out = tmp_path / "groups.csv"
    rows = group_mod.write_groups_csv(grouped_dataset, out, "coarse", tolerance=0.5)
    assert {row["crop_tolerance"] for row in rows} == {""}


def test_write_groups_csv_is_deterministic(grouped_dataset, tmp_path):
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    group_mod.write_groups_csv(grouped_dataset, first, "phash", threshold=0)
    group_mod.write_groups_csv(grouped_dataset, second, "phash", threshold=0)
    assert first.read_bytes() == second.read_bytes()


def test_cli_writes_the_csv(grouped_dataset, tmp_path):
    out = tmp_path / "groups_coarse.csv"
    result = runner.invoke(
        app, ["group", "--root", str(grouped_dataset), "--method", "coarse", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "4 image(s) in 2 group(s) by coarse" in result.output


def test_cli_rejects_an_unknown_method(grouped_dataset, tmp_path):
    result = runner.invoke(app, ["group", "--root", str(grouped_dataset), "--method", "telepathy"])
    assert result.exit_code == 2


def test_cli_fails_clearly_when_root_is_missing(tmp_path):
    result = runner.invoke(
        app, ["group", "--root", str(tmp_path / "nope"), "--out", str(tmp_path / "g.csv")]
    )
    assert result.exit_code == 1


def test_cli_says_when_threshold_is_ignored(grouped_dataset, tmp_path):
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(grouped_dataset),
            "--method",
            "coarse",
            "--threshold",
            "5",
            "--out",
            str(tmp_path / "g.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ignored for --method coarse" in result.output


def test_cli_rejects_a_negative_crop_tolerance(grouped_dataset, tmp_path):
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(grouped_dataset),
            "--method",
            "phash",
            "--crop-tolerance",
            "-1",
            "--out",
            str(tmp_path / "g.csv"),
        ],
    )
    assert result.exit_code == 2
    assert "must be 0 or greater" in result.output


def test_cli_crop_tolerance_reaches_the_csv(grouped_dataset, tmp_path):
    out = tmp_path / "g.csv"
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(grouped_dataset),
            "--method",
            "phash",
            "--threshold",
            "0",
            "--crop-tolerance",
            "0.25",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "crop tolerance: 0.25" in result.output
    with out.open(newline="", encoding="utf-8") as f:
        assert {row["crop_tolerance"] for row in csv.DictReader(f)} == {"0.25"}


def test_cli_full_frame_skips_the_crop_stage(grouped_dataset, tmp_path):
    out = tmp_path / "g.csv"
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(grouped_dataset),
            "--method",
            "phash",
            "--threshold",
            "0",
            "--full-frame",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "crop: full frame" in result.output
    with out.open(newline="", encoding="utf-8") as f:
        assert {row["crop_tolerance"] for row in csv.DictReader(f)} == {""}


REAL_ROOT = Path("data/raw")


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="requires the real SolDef_AI download (gitignored)"
)
def test_real_dataset_coarse_gives_seven_groups(tmp_path):
    rows = group_mod.write_groups_csv(REAL_ROOT, tmp_path / "groups.csv", "coarse")
    assert len(rows) == 200
    assert len({row["group_id"] for row in rows}) == 7
