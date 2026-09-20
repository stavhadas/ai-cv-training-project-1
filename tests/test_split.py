"""Tests for `pcbi split`, Stage 1 Step 5's frozen train/val/test partition.

Everything here works on hand-written CSV text under `tmp_path`: unlike grouping, splitting never
opens an image, so a fixture that built real JPEGs would only slow the suite down. `synthetic_data`
deals a small dataset whose class counts comfortably clear the gates; `starved_data` deals one that
cannot, so the "nothing passed" path is exercised rather than assumed.
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import group as group_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import split as split_mod

runner = CliRunner()

CLASSES = ["normal", "excess", "insufficient", "spike"]


def write_dataset(directory: Path, class_cycle: list[str], groups: int = 100) -> tuple[Path, Path]:
    """A groups CSV and a polygons CSV describing `groups` components, 4 crops each.

    Each group is one component photographed under V2 and V2.1 — two images — and each image holds
    two joints, which is the real dataset's shape. Classes are dealt from `class_cycle` per group,
    so every crop of a group shares a label and the stratifier is working with the same coupling it
    faces on the real data.

    The default 100 groups is not arbitrary: the gates are absolute crop floors, so a fixture too
    small to reach them would fail for its own size rather than for anything under test. 100 groups
    is 400 crops, the same scale as the real download.
    """
    directory.mkdir(parents=True, exist_ok=True)
    group_rows = []
    polygon_rows = []
    for index in range(groups):
        group_id = f"manual_{index:03d}"
        class_name = class_cycle[index % len(class_cycle)]
        for viewpoint in ("V2", "V2.1"):
            name = f"IMG_{index:03d}_{viewpoint}.jpg"
            dataset_path = f"Dataset/CS1/R0805/{viewpoint}/{name}"
            group_rows.append(
                {
                    "image_name": name,
                    "dataset_path": dataset_path,
                    "board": "CS1",
                    "package": "R0805",
                    "cell": "CS1/R0805",
                    "viewpoint": viewpoint,
                    "component_hint": "",
                    "method": "manual",
                    "threshold": "",
                    "crop_tolerance": "",
                    "fingerprint": "",
                    "group_id": group_id,
                    "group_size": 2,
                }
            )
            for rank, position in ((1, "left"), (2, "right")):
                polygon_rows.append(
                    {
                        "image_name": name,
                        "dataset_path": dataset_path,
                        "board": "CS1",
                        "package": "R0805",
                        "viewpoint": viewpoint,
                        "light_direction": "bottom_to_top",
                        "capture_timestamp": "",
                        "raw_label": class_name,
                        "class": class_name,
                        "points": "[]",
                        "polygon_count": 2,
                        "polygon_rank_x": rank,
                        "joint_position": position,
                        "merged_from": "",
                    }
                )

    groups_path = directory / "groups.csv"
    polygons_path = directory / "polygons.csv"
    write_csv(groups_path, group_mod.CSV_FIELDS, group_rows)
    write_csv(polygons_path, ingest_mod.CSV_FIELDS, polygon_rows)
    return groups_path, polygons_path


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_data(tmp_path):
    """100 groups, 400 crops, classes dealt evenly — every gate is comfortably reachable."""
    return write_dataset(tmp_path / "balanced", CLASSES)


@pytest.fixture
def starved_data(tmp_path):
    """One group in 19 carries `normal`, so no split can put 12 normal crops in val and in test."""
    cycle = ["excess", "spike", "insufficient"] * 6 + ["normal"]
    return write_dataset(tmp_path / "starved", cycle)


def crops_of(groups_path: Path, polygons_path: Path) -> list[split_mod.Crop]:
    return split_mod.load_crops(polygons_path, split_mod.load_groups(groups_path))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_split(**options):
    """`pcbi split` with `--flag value` pairs; a list value becomes several values after one flag.

    Keeps the CLI tests readable — a flat list of alternating flags and strings formats into one
    line per token, which buries what each test is actually varying.
    """
    args = ["split"]
    for name, value in options.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):  # a switch carries no value; False means leave it off
            if value:
                args.append(flag)
            continue
        args.append(flag)
        args.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return runner.invoke(app, args)


# --- the leakage test -----------------------------------------------------------------------


def test_no_group_id_lands_in_more_than_one_split(synthetic_data, tmp_path):
    """The property the whole split exists to guarantee.

    Its limit is worth stating: this can only check the group IDs it is handed. If the grouping
    itself under-merged — one physical component split across two group IDs — both halves look like
    separate groups here and this test passes while the data still leaks. That is why the grouping
    was verified visually in Step 4/5 before being frozen; this guards the split, not the grouping.
    """
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "splits.csv"
    split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20)

    seen: dict[str, set[str]] = {}
    for row in read_rows(out):
        seen.setdefault(row["group_id"], set()).add(row["split"])
    straddling = {group: names for group, names in seen.items() if len(names) > 1}
    assert straddling == {}


def test_the_real_split_is_leak_free_and_covers_every_crop(synthetic_data, tmp_path):
    """Every crop is assigned exactly once — a split that silently drops rows is not a split."""
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "splits.csv"
    split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20)

    rows = read_rows(out)
    assert len(rows) == len(read_rows(polygons_path))
    assert len({row["crop_id"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == set(split_mod.SPLITS)


# --- ratios and fold arithmetic -------------------------------------------------------------


def test_the_standard_ratios_become_five_folds_dealt_three_one_one():
    assert split_mod.fold_counts((0.6, 0.2, 0.2)) == (5, 3, 1, 1)


def test_a_half_quarter_quarter_split_becomes_four_folds():
    assert split_mod.fold_counts((0.5, 0.25, 0.25)) == (4, 2, 1, 1)


def test_ratios_that_do_not_sum_to_one_are_refused():
    """Silently renormalising would freeze a split nobody asked for."""
    with pytest.raises(ValueError, match="must sum to 1"):
        split_mod.fold_counts((0.6, 0.2, 0.1))


def test_a_zero_ratio_is_refused():
    with pytest.raises(ValueError, match="greater than 0"):
        split_mod.fold_counts((0.8, 0.2, 0.0))


def test_an_even_three_way_split_becomes_three_folds():
    """Thirds look awkward in decimal but are exactly one fold each."""
    assert split_mod.fold_counts((1 / 3, 1 / 3, 1 / 3)) == (3, 1, 1, 1)


def test_inexpressible_ratios_are_refused_with_the_nearest_ones_named():
    """0.61 needs 100 folds. Refuse, and say what can be had instead of rounding in silence."""
    with pytest.raises(ValueError, match="nearest expressible ratios are 0.6 0.2 0.2"):
        split_mod.fold_counts((0.61, 0.20, 0.19))


# --- the gates ------------------------------------------------------------------------------


def assignment_for(crops, chooser) -> dict[str, str]:
    return {crop.crop_id: chooser(crop) for crop in crops}


def test_a_split_short_on_normals_fails_g2(synthetic_data):
    """G2 exists because a val set with three normal crops cannot support a per-class number."""
    crops = crops_of(*synthetic_data)
    starved = assignment_for(
        crops, lambda crop: "train" if crop.class_name == split_mod.NORMAL_CLASS else "val"
    )
    # everything non-normal in val, so test is empty too; G2 must name val explicitly
    reasons = split_mod.gate(crops, starved)
    assert any(reason.startswith("G2:") and "val has 0 normal" in reason for reason in reasons)


def test_a_split_missing_a_defect_class_fails_g3(synthetic_data):
    crops = crops_of(*synthetic_data)
    by_index = {crop.crop_id: index for index, crop in enumerate(crops)}

    def place(crop):
        if crop.class_name == "spike":
            return "train"
        return "val" if by_index[crop.crop_id] % 2 else "test"

    reasons = split_mod.gate(crops, assignment_for(crops, place))
    assert any(reason.startswith("G3:") and "spike" in reason for reason in reasons)


def test_a_leaking_split_fails_g1(synthetic_data):
    """G1 is redundant given StratifiedGroupKFold — which is exactly why it is checked."""
    crops = crops_of(*synthetic_data)
    leaking = {crop.crop_id: ("train" if crop.viewpoint == "V2" else "test") for crop in crops}
    reasons = split_mod.gate(crops, leaking)
    assert any(reason.startswith("G1:") for reason in reasons)


def test_a_balanced_split_passes_every_gate(synthetic_data):
    groups_path, polygons_path = synthetic_data
    crops = crops_of(groups_path, polygons_path)
    n_splits, *counts = split_mod.fold_counts(split_mod.DEFAULT_RATIOS)
    choice = split_mod.choose(crops, n_splits, counts, split_mod.DEFAULT_RATIOS, 20, 0)
    assert split_mod.gate(crops, choice.assignment) == []


def test_no_candidate_passes_when_the_data_cannot_support_the_gates(starved_data, tmp_path):
    """The right answer is an error naming the gate, not a quietly weakened floor."""
    groups_path, polygons_path = starved_data
    with pytest.raises(ValueError, match="none of the .* candidate split"):
        split_mod.write_splits_csv(
            groups_path, polygons_path, tmp_path / "splits.csv", candidates=5
        )


# --- the split hash -------------------------------------------------------------------------


def test_the_hash_is_over_the_sorted_pairs_only():
    """Pinned against a hand-computed digest, so a refactor cannot redefine the fingerprint."""
    assignment = {"b#1": "val", "a#2": "train", "a#1": "train"}
    expected = hashlib.sha256(b"a#1,train\na#2,train\nb#1,val").hexdigest()
    assert split_mod.split_hash(assignment) == expected


def test_row_order_does_not_change_the_hash():
    forward = {"a#1": "train", "b#1": "val", "c#1": "test"}
    backward = {"c#1": "test", "b#1": "val", "a#1": "train"}
    assert split_mod.split_hash(forward) == split_mod.split_hash(backward)


def test_moving_one_crop_changes_the_hash():
    before = {"a#1": "train", "b#1": "val"}
    after = {"a#1": "train", "b#1": "test"}
    assert split_mod.split_hash(before) != split_mod.split_hash(after)


def test_the_hash_ignores_unrelated_columns(synthetic_data, tmp_path):
    """An edit to a column that is not (crop_id, split) must leave the fingerprint alone.

    This is the whole reason the hash is not taken over the file: a manifest gains columns over a
    project's life, and a fingerprint that churns with them stops being checked.
    """
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "splits.csv"
    *_, hash_hex, _ = split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20)
    rows = read_rows(out)
    rederived = split_mod.split_hash({row["crop_id"]: row["split"] for row in rows})
    assert rederived == hash_hex

    for row in rows:  # rewrite every unrelated field; the partition is untouched
        row["joint_position"] = "edited"
        row["viewpoint"] = "edited"
    assert split_mod.split_hash({row["crop_id"]: row["split"] for row in rows}) == hash_hex


def test_the_hash_is_written_on_every_row(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "splits.csv"
    *_, hash_hex, _ = split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20)
    assert {row["split_hash"] for row in read_rows(out)} == {hash_hex}


# --- determinism ----------------------------------------------------------------------------


def test_the_same_seed_writes_byte_identical_files(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    split_mod.write_splits_csv(groups_path, polygons_path, first, candidates=20, seed=7)
    split_mod.write_splits_csv(groups_path, polygons_path, second, candidates=20, seed=7)
    assert first.read_bytes() == second.read_bytes()


def test_a_different_seed_searches_a_different_set_of_candidates(synthetic_data, tmp_path):
    """Not that the result must differ — that the seed actually reaches the generator."""
    groups_path, polygons_path = synthetic_data
    crops = crops_of(groups_path, polygons_path)
    n_splits, *counts = split_mod.fold_counts(split_mod.DEFAULT_RATIOS)
    first = split_mod.candidate_split(crops, n_splits, counts, seed=0)
    second = split_mod.candidate_split(crops, n_splits, counts, seed=1)
    assert first != second


# --- reading the inputs ---------------------------------------------------------------------


def test_the_header_matches_the_declared_fields(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "splits.csv"
    split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20)
    with out.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == split_mod.CSV_FIELDS


def test_crop_ids_join_back_to_the_polygons_csv(synthetic_data, tmp_path):
    """The ID has to be reconstructible from polygons.csv, or the split cannot be used."""
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "splits.csv"
    split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20)
    expected = {
        split_mod.crop_id(row["dataset_path"], row["polygon_rank_x"])
        for row in read_rows(polygons_path)
    }
    assert {row["crop_id"] for row in read_rows(out)} == expected


def test_an_image_with_no_group_is_an_error_not_a_singleton(synthetic_data, tmp_path):
    """Inventing a group for an orphan is how a leak gets in unnoticed."""
    groups_path, polygons_path = synthetic_data
    rows = read_rows(groups_path)
    dropped = rows.pop(0)["dataset_path"]
    trimmed = tmp_path / "partial_groups.csv"
    write_csv(trimmed, group_mod.CSV_FIELDS, rows)

    with pytest.raises(ValueError, match="have no group"):
        split_mod.load_crops(polygons_path, split_mod.load_groups(trimmed))
    assert dropped  # the message names it


def test_a_file_that_is_not_a_groups_csv_is_refused(tmp_path):
    wrong = tmp_path / "wrong.csv"
    write_csv(wrong, ["a", "b"], [{"a": "1", "b": "2"}])
    with pytest.raises(ValueError, match="not a groups CSV"):
        split_mod.load_groups(wrong)


# --- the manifest metadata ------------------------------------------------------------------


def test_the_manifest_records_the_hash_and_what_produced_it(synthetic_data, tmp_path):
    """A run logging only the hash has to be traceable back to the inputs and the search."""
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "split_v1.csv"
    meta_out = tmp_path / "manifest_meta.json"
    *_, hash_hex, _ = split_mod.write_splits_csv(
        groups_path, polygons_path, out, candidates=20, seed=3, meta_out=meta_out
    )

    meta = json.loads(meta_out.read_text(encoding="utf-8"))
    assert meta["split_hash"] == hash_hex
    assert meta["splits_csv"] == out.as_posix()
    assert meta["inputs"]["groups_csv"] == groups_path.as_posix()
    assert meta["inputs"]["crops"] == 400
    assert meta["inputs"]["groups"] == 100
    assert meta["search"]["split_seed"] == 3
    assert meta["search"]["candidates"] == 20
    assert meta["search"]["folds"] == {"train": 3, "val": 1, "test": 1}
    assert meta["gates"]["min_normal_in_val_and_test"] == split_mod.MIN_NORMAL


def test_the_manifest_counts_match_the_csv(synthetic_data, tmp_path):
    """Two renderings of one table must not drift apart."""
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "split_v1.csv"
    meta_out = tmp_path / "manifest_meta.json"
    split_mod.write_splits_csv(groups_path, polygons_path, out, candidates=20, meta_out=meta_out)

    meta = json.loads(meta_out.read_text(encoding="utf-8"))
    rows = read_rows(out)
    for name in split_mod.SPLITS:
        in_csv = [row for row in rows if row["split"] == name]
        assert meta["counts"][name]["crops"] == len(in_csv)
        assert meta["counts"][name]["groups"] == len({row["group_id"] for row in in_csv})
        for class_name in CLASSES:
            expected = sum(1 for row in in_csv if row["class"] == class_name)
            assert meta["counts"][name][class_name] == expected
    assert meta["counts"]["total"]["crops"] == len(rows)


def test_no_manifest_is_written_when_none_is_asked_for(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    split_mod.write_splits_csv(groups_path, polygons_path, tmp_path / "split_v1.csv", candidates=20)
    assert not (tmp_path / "manifest_meta.json").exists()


def test_the_count_table_totals_every_column(synthetic_data):
    crops = crops_of(*synthetic_data)
    assignment = dict.fromkeys((crop.crop_id for crop in crops), "train")
    headers, rows = split_mod.count_table(crops, assignment)
    assert headers[:3] == ["split", "groups", "crops"]
    assert rows[-1][0] == "total"
    assert rows[-1][2] == len(crops)
    assert rows[-1][1] == len({crop.group_id for crop in crops})


def test_gate_margins_put_the_tightest_first(synthetic_data):
    crops = crops_of(*synthetic_data)
    n_splits, *counts = split_mod.fold_counts(split_mod.DEFAULT_RATIOS)
    choice = split_mod.choose(crops, n_splits, counts, split_mod.DEFAULT_RATIOS, 20, 0)
    margins = split_mod.gate_margins(crops, choice.assignment)
    slacks = [count - floor for _, count, floor in margins]
    assert slacks == sorted(slacks)
    assert all(slack >= 0 for slack in slacks)  # the winner passed every gate


# --- the data-audit section -----------------------------------------------------------------


def test_the_section_is_appended_to_the_audit_report(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    report = tmp_path / "data_audit.md"
    report.write_text("# Data audit\n\n## 12. Metadata table\n\nexisting content.\n", "utf-8")

    crops, choice, _, meta = split_mod.write_splits_csv(
        groups_path, polygons_path, tmp_path / "split_v1.csv", candidates=20
    )
    section = split_mod.render_audit_section(crops, choice, meta, "pcbi split")
    assert split_mod.update_audit_report(report, section) is False

    text = report.read_text(encoding="utf-8")
    assert "existing content." in text  # the audit's own sections survive
    assert "## 13. Frozen train/val/test split" in text
    assert meta["split_hash"] in text
    assert text.count(split_mod.AUDIT_BEGIN) == 1


def test_rerunning_replaces_the_section_instead_of_stacking_copies(synthetic_data, tmp_path):
    """Appending blindly would grow the report by a section on every run."""
    groups_path, polygons_path = synthetic_data
    report = tmp_path / "data_audit.md"
    report.write_text("# Data audit\n\nexisting content.\n", encoding="utf-8")

    for seed in (0, 1):
        crops, choice, _, meta = split_mod.write_splits_csv(
            groups_path, polygons_path, tmp_path / "split_v1.csv", candidates=20, seed=seed
        )
        section = split_mod.render_audit_section(crops, choice, meta, "pcbi split")
        replaced = split_mod.update_audit_report(report, section)

    assert replaced is True
    text = report.read_text(encoding="utf-8")
    assert text.count(split_mod.AUDIT_BEGIN) == 1
    assert text.count("## 13. Frozen train/val/test split") == 1
    assert "existing content." in text
    assert meta["split_hash"] in text  # the second run's hash, not the first's


def test_the_section_carries_the_counts_and_the_gates(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    crops, choice, _, meta = split_mod.write_splits_csv(
        groups_path, polygons_path, tmp_path / "split_v1.csv", candidates=20
    )
    section = split_mod.render_audit_section(crops, choice, meta, "pcbi split --groups g.csv")

    assert "pcbi split --groups g.csv" in section  # provenance: the command that made it
    assert f"at least {split_mod.MIN_NORMAL}" in section
    assert f"at least {split_mod.MIN_DEFECT}" in section
    assert "Tightest margin" in section
    assert "| **total** |" in section
    for class_name in CLASSES:
        assert class_name in section


def test_a_missing_audit_report_is_an_error_not_a_new_file(tmp_path):
    """`pcbi split` adds a section to the audit; it does not own a report of its own."""
    missing = tmp_path / "data_audit.md"
    with pytest.raises(ValueError, match="Run `pcbi audit` first"):
        split_mod.update_audit_report(missing, "section")
    assert not missing.exists()


# --- the CLI --------------------------------------------------------------------------------


def test_cli_writes_the_split_and_prints_the_count_table(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "split_v1.csv"
    meta_out = tmp_path / "manifest_meta.json"
    report = tmp_path / "data_audit.md"
    report.write_text("# Data audit\n\nexisting content.\n", encoding="utf-8")

    result = run_split(
        groups=groups_path,
        polygons=polygons_path,
        out=out,
        meta_out=meta_out,
        audit_out=report,
        ratios=[0.6, 0.2, 0.2],
        candidates=20,
        split_seed=0,
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert meta_out.is_file()
    assert "100 group(s), 400 crop(s)" in result.output
    assert "tightest gate margin:" in result.output
    assert "split hash:" in result.output
    for class_name in CLASSES:
        assert class_name in result.output
    for name in split_mod.SPLITS:
        assert name in result.output
    assert f"wrote {out}" in result.output
    assert f"wrote {meta_out}" in result.output
    assert f"appended the split section in {report}" in result.output
    assert "## 13. Frozen train/val/test split" in report.read_text(encoding="utf-8")


def test_cli_skips_the_report_when_asked_to(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    report = tmp_path / "data_audit.md"
    report.write_text("# Data audit\n", encoding="utf-8")
    result = run_split(
        groups=groups_path,
        polygons=polygons_path,
        out=tmp_path / "split_v1.csv",
        meta_out=tmp_path / "manifest_meta.json",
        audit_out=report,
        candidates=20,
        no_report=True,
    )
    assert result.exit_code == 0, result.output
    assert report.read_text(encoding="utf-8") == "# Data audit\n"


def test_cli_still_writes_the_split_when_the_audit_report_is_missing(synthetic_data, tmp_path):
    """The split is the real artifact; a missing write-up must not fail the command."""
    groups_path, polygons_path = synthetic_data
    out = tmp_path / "split_v1.csv"
    result = run_split(
        groups=groups_path,
        polygons=polygons_path,
        out=out,
        meta_out=tmp_path / "manifest_meta.json",
        audit_out=tmp_path / "nowhere.md",
        candidates=20,
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "Run `pcbi audit` first" in result.output


def test_cli_rejects_ratios_that_do_not_sum_to_one(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    result = run_split(
        groups=groups_path,
        polygons=polygons_path,
        out=tmp_path / "splits.csv",
        ratios=[0.6, 0.2, 0.1],
    )
    assert result.exit_code == 2, result.output
    assert "must sum to 1" in result.output


def test_cli_rejects_zero_candidates(synthetic_data, tmp_path):
    groups_path, polygons_path = synthetic_data
    result = run_split(
        groups=groups_path,
        polygons=polygons_path,
        out=tmp_path / "splits.csv",
        candidates=0,
    )
    assert result.exit_code == 2, result.output
    assert "--candidates must be at least 1" in result.output


def test_cli_reports_a_missing_groups_file(tmp_path):
    result = run_split(groups=tmp_path / "nope.csv", out=tmp_path / "s.csv")
    assert result.exit_code == 1, result.output
    assert "No such groups file" in result.output


def test_cli_reports_a_missing_polygons_file(synthetic_data, tmp_path):
    groups_path, _ = synthetic_data
    result = run_split(groups=groups_path, polygons=tmp_path / "nope.csv", out=tmp_path / "s.csv")
    assert result.exit_code == 1, result.output
    assert "No such polygons file" in result.output


def test_cli_reports_when_no_candidate_passes_the_gates(starved_data, tmp_path):
    groups_path, polygons_path = starved_data
    result = run_split(
        groups=groups_path,
        polygons=polygons_path,
        out=tmp_path / "splits.csv",
        candidates=5,
    )
    assert result.exit_code == 1, result.output
    assert "passed the gates" in result.output
    assert "G2" in result.output


# --- the real download ----------------------------------------------------------------------

REAL_GROUPS = Path("data/interim/groups_manual.csv")
REAL_POLYGONS = Path("data/interim/polygons.csv")


@pytest.mark.skipif(
    not (REAL_GROUPS.is_file() and REAL_POLYGONS.is_file()),
    reason="requires the real SolDef_AI artifacts (gitignored)",
)
def test_real_dataset_split_passes_every_gate(tmp_path):
    """The committed split, regenerated: 400 crops in 130 groups, all gates clear."""
    crops, choice, hash_hex, meta = split_mod.write_splits_csv(
        REAL_GROUPS, REAL_POLYGONS, tmp_path / "splits.csv", candidates=200, seed=0
    )
    assert len(crops) == 400
    assert len({crop.group_id for crop in crops}) == 130
    assert split_mod.gate(crops, choice.assignment) == []
    assert len(hash_hex) == 64

    counts = split_mod.class_counts(crops, choice.assignment)
    for name in split_mod.SPLITS:
        assert all(count > 0 for count in counts[name].values()), (name, counts[name])
