"""Tests for `pcbi group --report`, the grouping evidence report.

Most of what the report does is arithmetic over (components, images), so those tests build
`ImageRecord` lists by hand and touch no files at all. Only the grids and the CLI need a real tree.
"""

import json
import random
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import group as group_mod
from pcbi.data import group_report as report_mod

runner = CliRunner()


def record(name: str, cell: str, viewpoint: str, hint: str = "") -> group_mod.ImageRecord:
    board, package = cell.split("/")
    return group_mod.ImageRecord(
        image_name=name,
        dataset_path=f"Dataset/{board}/{package}/{viewpoint}/{name}",
        board=board,
        package=package,
        viewpoint=viewpoint,
        component_hint=hint,
    )


@pytest.fixture
def records() -> list[group_mod.ImageRecord]:
    """Two light directions in CS1/R0805 (so partners are possible), one-directional CS2/R0603.

    `a` and `c` share component number C1 across the two folders — a reference pair.
    """
    return [
        record("a.jpg", "CS1/R0805", "V2", "C1"),  # 0
        record("b.jpg", "CS1/R0805", "V2", ""),  # 1
        record("c.jpg", "CS1/R0805", "V2.1", "C1"),  # 2  -- pairs with 0
        record("d.jpg", "CS1/R0805", "V2.1", ""),  # 3
        record("e.jpg", "CS2/R0603", "V2", ""),  # 4  -- no V2.1 in this cell
    ]


# --- pure metrics ------------------------------------------------------------------------------


def test_size_histogram_counts_groups_by_size():
    assert report_mod.size_histogram([[0], [1], [2, 3, 4]]) == {1: 2, 3: 1}


def test_cell_violations_flags_a_group_spanning_two_cells(records):
    violations = report_mod.cell_violations([[0, 4], [1], [2], [3]], records)
    assert violations == [[0, 4]]  # CS1/R0805 merged with CS2/R0603


def test_cell_violations_are_empty_when_every_group_stays_in_one_cell(records):
    assert report_mod.cell_violations([[0, 1, 2, 3], [4]], records) == []


def test_reference_pairs_finds_the_shared_component_number(records):
    assert report_mod.reference_pairs(records) == [(0, 2)]


def test_reference_pairs_ignores_a_component_number_with_no_partner():
    lonely = [record("a.jpg", "CS1/R0805", "V2", "C9")]
    assert report_mod.reference_pairs(lonely) == []


def test_cross_lit_cells_excludes_single_direction_cells(records):
    assert report_mod.cross_lit_cells(records) == {"CS1/R0805"}


def test_measure_counts_a_merged_reference_pair(records):
    merged = report_mod.measure([[0, 2], [1], [3], [4]], records)
    assert merged.reference_merged == 1
    assert merged.reference_total == 1

    apart = report_mod.measure([[0], [1], [2], [3], [4]], records)
    assert apart.reference_merged == 0


def test_partner_rate_excludes_images_that_could_not_have_a_partner(records):
    """`e` sits in a V2-only cell, so it is not counted against the rate."""
    metrics = report_mod.measure([[0, 2], [1, 3], [4]], records)
    assert (metrics.partnered, metrics.partner_total) == (4, 4)  # not 4/5
    assert metrics.partner_rate == 1.0


def test_partner_rate_is_zero_when_nothing_merges(records):
    metrics = report_mod.measure([[0], [1], [2], [3], [4]], records)
    assert (metrics.partnered, metrics.partner_total) == (0, 4)


def test_partner_rate_needs_the_other_folder_not_just_any_partner(records):
    """Two V2 images grouped together are not partnered — the partner must come from V2.1."""
    metrics = report_mod.measure([[0, 1], [2], [3], [4]], records)
    assert metrics.partnered == 0


def test_partner_ceiling_is_limited_by_the_smaller_folder():
    """A cell with 3 V2 and 1 V2.1 can form only one pair, so only 2 images can be partnered."""
    lopsided = [
        record("a.jpg", "CS1/R0805", "V2"),
        record("b.jpg", "CS1/R0805", "V2"),
        record("c.jpg", "CS1/R0805", "V2"),
        record("d.jpg", "CS1/R0805", "V2.1"),
    ]
    ceiling, rows = report_mod.partner_ceiling(lopsided)
    assert ceiling == 2
    assert rows == [{"cell": "CS1/R0805", "v2": 3, "v2_1": 1, "pairs": 1}]


def test_partner_ceiling_ignores_single_direction_cells(records):
    ceiling, rows = report_mod.partner_ceiling(records)
    assert ceiling == 4  # CS1/R0805 has 2 and 2; CS2/R0603 contributes nothing
    assert [row["cell"] for row in rows] == ["CS1/R0805"]


def test_ceiling_can_sit_below_a_hundred_percent():
    """The point of the ceiling: a perfect fine-grained grouping still won't reach 100%."""
    lopsided = [
        record("a.jpg", "CS1/R0805", "V2"),
        record("b.jpg", "CS1/R0805", "V2"),
        record("c.jpg", "CS1/R0805", "V2.1"),
    ]
    metrics = report_mod.measure([[0, 2], [1]], lopsided)
    assert metrics.partner_ceiling == 2
    assert metrics.partner_total == 3
    assert metrics.ceiling_rate == pytest.approx(2 / 3)
    assert metrics.partnered == 2  # already at the ceiling, despite reading as 67%


def test_measure_reports_sizes_and_singletons(records):
    metrics = report_mod.measure([[0, 1, 2], [3], [4]], records)
    assert (metrics.groups, metrics.largest, metrics.singletons) == (3, 3, 2)


# --- sweep and sampling ------------------------------------------------------------------------


def test_sweep_returns_one_row_per_threshold(records):
    """Indices 0-3 are CS1/R0805, index 4 is CS2/R0603, and the two never merge."""
    fingerprints = {0: "0", 1: "0", 2: "0", 3: "f", 4: "f"}
    rows = report_mod.sweep("phash", fingerprints, records, thresholds=(0, 4))
    assert [threshold for threshold, _ in rows] == [0, 4]
    assert rows[0][1].groups == 3  # {0,1,2}, {3}, and {4} alone in its own cell
    assert rows[1][1].groups == 2  # {0,1,2,3} within 4 bits, {4} still separate


def test_sweep_never_merges_across_cells(records):
    """Index 3 and index 4 share a fingerprint but not a cell, so no threshold unites them."""
    fingerprints = {0: "0", 1: "0", 2: "0", 3: "f", 4: "f"}
    for _, metrics in report_mod.sweep("phash", fingerprints, records, thresholds=(0, 4, 256)):
        assert metrics.cell_violations == 0


def test_sweep_is_empty_for_coarse(records):
    assert report_mod.sweep("coarse", {}, records) == []


def test_choose_groups_takes_the_largest_and_never_repeats_them():
    components = [[i] for i in range(30)]
    components[5] = list(range(100, 140))  # one clearly biggest group
    top, picked = report_mod.choose_groups(components, sample=5, largest=2)
    assert top[0] == 5
    assert not set(top) & set(picked)
    assert len(picked) == 5


def test_choose_groups_is_deterministic():
    components = [[i] for i in range(40)]
    assert report_mod.choose_groups(components) == report_mod.choose_groups(components)


def test_choose_groups_copes_with_fewer_groups_than_requested():
    components = [[0], [1]]
    top, picked = report_mod.choose_groups(components, sample=20, largest=10)
    assert len(top) == 2
    assert picked == []  # nothing left over to sample


# --- rendering, grids and CLI --------------------------------------------------------------------


@pytest.fixture
def report_dataset(tmp_path):
    """Two cells of real JPEGs, annotated so `ingest_rows` sees them as joint-task images."""
    root = tmp_path
    labeled = root / "Labeled"
    labeled.mkdir()

    def place(board, package, viewpoint, name, seed):
        folder = root / "Dataset" / board / package / viewpoint
        folder.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        data = bytes(rng.randrange(256) for _ in range(32 * 32 * 3))
        for destination in (folder / name, labeled / name):
            Image.frombytes("RGB", (32, 32), data).save(destination, format="JPEG")
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(
            json.dumps(
                {
                    "imagePath": name,
                    "imageWidth": 32,
                    "imageHeight": 32,
                    "shapes": [
                        {
                            "label": "spike",
                            "shape_type": "polygon",
                            "points": [[2, 2], [8, 2], [8, 8], [2, 8]],
                        },
                        {
                            "label": "exc_solder",
                            "shape_type": "polygon",
                            "points": [[20, 2], [28, 2], [28, 8], [20, 8]],
                        },
                    ],
                }
            )
        )

    place("CS1", "R0805", "V2", "a_WIN_20220330_13_11_58_Pro_C1.jpg", 1)
    place("CS1", "R0805", "V2.1", "b_WIN_20220330_13_12_00_Pro_C1.jpg", 2)
    place("CS2", "R0603", "V2", "c_WIN_20220330_16_07_40_Pro.jpg", 3)
    return root


def test_write_group_grids_writes_one_png_per_group(report_dataset, tmp_path):
    images = group_mod.load_images(report_dataset)
    components = group_mod.coarse_components(images)
    group_ids = group_mod.assign_group_ids(components, images, "coarse")
    grids = report_mod.write_group_grids(
        report_dataset, images, components, group_ids, tmp_path / "grids"
    )
    assert len(grids) == len(components)
    assert all(path.is_file() for path in grids.values())


def test_report_has_every_requested_section(report_dataset, tmp_path):
    rows, metrics = report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "coarse",
        grid_dir=tmp_path / "grids",
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Cropping stage",
        "## 2. Group size histogram",
        "## 3. Groups crossing cell boundaries",
        "## 4. `_C##` reference pairs",
        "## 5. V2 / V2.1 partner rate",
        "## 6. Threshold sweep",
        "## 7. Group grids",
    ):
        assert heading in text
    assert len(rows) == 3
    assert metrics.groups == 2


def test_report_records_the_reference_pair_result(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "coarse",
        grid_dir=tmp_path / "grids",
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "**1 of 1 known pairs merged.**" in text  # C1 spans V2 and V2.1 in one cell


def test_report_for_phash_includes_a_sweep_table(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "phash",
        threshold=0,
        grid_dir=tmp_path / "grids",
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| Threshold | Groups | Largest |" in text
    assert "**←**" in text  # the row the groups were actually built at


def test_report_says_coarse_has_no_threshold_to_sweep(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "coarse",
        grid_dir=tmp_path / "grids",
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "there is no threshold" in text


def test_phash_without_a_threshold_writes_only_the_sweep(report_dataset, tmp_path):
    """First pass: the sweep exists so you can choose, so nothing may claim a grouping yet."""
    csv_out, report_out = tmp_path / "groups.csv", tmp_path / "report.md"
    rows, metrics = report_mod.write_report(
        report_dataset,
        csv_out,
        report_out,
        "phash",
        threshold=None,
        grid_dir=tmp_path / "grids",
        tolerance_sweep=(0.0, 1.0),
    )
    assert rows == []
    assert metrics is None
    assert not csv_out.exists()  # no threshold means no grouping to write
    # Crop examples are written (you need them to choose a tolerance); group grids are not.
    assert sorted(p.name for p in (tmp_path / "grids").iterdir()) == ["crop_t0.png", "crop_t1.png"]

    text = report_out.read_text(encoding="utf-8")
    assert "Nothing chosen" in text
    assert "## 2. Crop tolerance × threshold" in text
    assert "--threshold N --crop-tolerance T" in text
    # None of the single-setting sections may appear.
    assert "Group size histogram" not in text
    assert "known pairs merged" not in text
    assert "are grouped with a partner" not in text


def test_the_sweep_only_report_still_carries_the_threshold_free_context(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "phash",
        threshold=None,
        grid_dir=tmp_path / "grids",
        tolerance_sweep=(0.0, 1.0),
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## 4. What the partner rate can reach" in text  # the ceiling is threshold-independent
    assert "## 5. `_C##` reference pairs available" in text
    assert "| Cell | V2 | V2.1 | Max pairs |" in text


def test_a_chosen_threshold_gives_the_full_report(report_dataset, tmp_path):
    """Second pass: with a threshold, every section describes that one grouping."""
    csv_out, report_out = tmp_path / "groups.csv", tmp_path / "report.md"
    rows, metrics = report_mod.write_report(
        report_dataset, csv_out, report_out, "phash", threshold=0, grid_dir=tmp_path / "grids"
    )
    assert metrics is not None
    assert len(rows) == 3
    assert csv_out.is_file()
    text = report_out.read_text(encoding="utf-8")
    assert "## 2. Group size histogram" in text
    assert "every other section of this report describes" in text


def test_coarse_needs_no_threshold_to_produce_a_full_report(report_dataset, tmp_path):
    """`coarse` has no threshold at all, so it must not fall into the sweep-only path."""
    csv_out = tmp_path / "groups.csv"
    _, metrics = report_mod.write_report(
        report_dataset, csv_out, tmp_path / "report.md", "coarse", grid_dir=tmp_path / "grids"
    )
    assert metrics is not None
    assert csv_out.is_file()


def test_cli_without_threshold_explains_the_second_pass(report_dataset, tmp_path):
    csv_out = tmp_path / "groups.csv"
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(report_dataset),
            "--method",
            "phash",
            "--out",
            str(csv_out),
            "--report",
            "--report-out",
            str(tmp_path / "report.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "parameter sweep only" in result.output
    assert "--threshold N --crop-tolerance T" in result.output
    assert not csv_out.exists()


def metrics(reference_merged: int, largest: int) -> report_mod.Metrics:
    return report_mod.Metrics(
        groups=1,
        largest=largest,
        singletons=0,
        cell_violations=0,
        reference_merged=reference_merged,
        reference_total=4,
        partnered=0,
        partner_total=0,
        partner_ceiling=0,
    )


def test_best_row_prefers_the_threshold_that_finds_more_reference_pairs():
    rows = [(10.0, metrics(2, 3)), (20.0, metrics(4, 9))]
    assert report_mod.best_row(rows)[0] == 20.0


def test_best_row_breaks_a_tie_toward_the_tighter_grouping():
    """A setting that merged everything also scores 4/4; it must not win on that alone."""
    rows = [(10.0, metrics(4, 60)), (20.0, metrics(4, 3)), (30.0, metrics(4, 7))]
    assert report_mod.best_row(rows)[0] == 20.0


def test_best_row_of_an_empty_sweep_is_none():
    assert report_mod.best_row([]) is None


def test_the_verdict_calls_out_full_marks_won_by_merging_everything():
    """4/4 with a cell-sized group is the failure this whole report exists to make visible."""
    lines = report_mod.verdict_lines([(0.0, [(100.0, metrics(4, 56))])], ceiling_total=168)
    assert any("No swept setting separated" in line for line in lines)
    assert any("56 images" in line for line in lines)


def test_the_verdict_recognises_a_genuinely_tight_grouping():
    lines = report_mod.verdict_lines([(0.0, [(100.0, metrics(4, 2))])], ceiling_total=168)
    assert any("behaves like twin-finding" in line for line in lines)


def test_crop_examples_show_the_frame_and_the_crop_side_by_side(report_dataset, tmp_path):
    images = group_mod.load_images(report_dataset)
    grid = report_mod.crop_example_grid(report_dataset, images, tolerance=0.0, count=2)
    assert grid.width > 0 and grid.height > 0


def test_the_full_report_states_its_crop_tolerance(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "phash",
        threshold=0,
        grid_dir=tmp_path / "grids",
        tolerance=0.25,
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- **Crop tolerance:** 0.25" in text
    assert (tmp_path / "grids" / "crop_examples.png").is_file()
    assert "![crop examples, tolerance 0.25]" in text


def test_the_full_report_says_so_when_nothing_was_cropped(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "phash",
        threshold=0,
        grid_dir=tmp_path / "grids",
        tolerance=None,
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "full frame (no crop)" in text


def test_coarse_reports_that_it_never_opens_an_image(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "coarse",
        grid_dir=tmp_path / "grids",
        tolerance=0.5,
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "never opens an image" in text


def test_the_sweep_report_covers_every_tolerance(report_dataset, tmp_path):
    report_mod.write_report(
        report_dataset,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "phash",
        threshold=None,
        grid_dir=tmp_path / "grids",
        tolerance_sweep=(0.0, 0.5, 2.0),
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for value in ("0", "0.5", "2"):
        assert f"### Crop tolerance {value}" in text
        assert (tmp_path / "grids" / f"crop_t{value}.png").is_file()
        assert f"![crop examples, tolerance {value}]" in text


def test_write_report_rejects_an_unknown_method(report_dataset, tmp_path):
    with pytest.raises(ValueError, match="unknown method"):
        report_mod.write_report(report_dataset, tmp_path / "g.csv", tmp_path / "r.md", "telepathy")


def test_cli_report_flag_writes_both_files(report_dataset, tmp_path):
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(report_dataset),
            "--method",
            "coarse",
            "--out",
            str(tmp_path / "groups.csv"),
            "--report",
            "--report-out",
            str(tmp_path / "report.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "groups.csv").is_file()
    assert (tmp_path / "report.md").is_file()
    assert "wrote" in result.output


def test_cli_without_report_flag_writes_only_the_csv(report_dataset, tmp_path):
    report_out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "group",
            "--root",
            str(report_dataset),
            "--method",
            "coarse",
            "--out",
            str(tmp_path / "groups.csv"),
            "--report-out",
            str(report_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not report_out.exists()


REAL_ROOT = Path("data/raw")


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="requires the real SolDef_AI download (gitignored)"
)
def test_real_dataset_coarse_report_matches_the_known_evidence(tmp_path):
    _, metrics = report_mod.write_report(
        REAL_ROOT,
        tmp_path / "groups.csv",
        tmp_path / "report.md",
        "coarse",
        grid_dir=tmp_path / "grids",
    )
    assert metrics.groups == 7
    assert metrics.cell_violations == 0
    assert (metrics.reference_merged, metrics.reference_total) == (4, 4)
    assert (metrics.partnered, metrics.partner_total) == (180, 180)
    # 20 + 10 + 16 + 7 + 31 = 84 possible pairs, so 168 of 180 images -- coarse overshoots it
    # by merging whole cells rather than twins.
    assert metrics.partner_ceiling == 168
