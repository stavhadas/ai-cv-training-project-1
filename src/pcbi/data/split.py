"""Stage 1 Step 5: freeze one train/val/test split, so every later run measures the same thing.

Stage 2 compares backbones against each other. That comparison only means something if every run
trains and evaluates on exactly the same partition, so the partition is generated once, written to
disk, and fingerprinted. A run that reports a different hash did not use this split.

Two properties have to hold at once, and they pull against each other:

- **Grouped.** SolDef_AI photographs each physical component twice, under `V2` and `V2.1` lighting,
  and `pcbi group` / the manual tagger recovered which images belong together. Every crop of a group
  must land in the same split, or the test score is measuring memorisation of a part the model
  already saw from the other side. The group is the indivisible unit here, not the image and
  certainly not the crop.
- **Stratified.** Each split should hold roughly the overall class proportions. On this dataset
  that is genuinely hard: 400 crops sit in only 130 groups, and `normal` is the scarcest class at
  65 of 400, so a group carrying two `normal` crops moves ~3% of the entire class at once.

`StratifiedGroupKFold` does both, approximately. With this few groups "approximately" is doing real
work — different seeds give visibly different splits — so this module generates many candidates and
keeps one, judged by `gate` below. The gates read **label counts only**. Choosing a split by how a
model scores on it would tune the test set to flatter that model, which is the one thing a frozen
split exists to prevent, so no model ever runs in this file.

The ratios choose the fold arithmetic rather than being approximated by it: `0.6 0.2 0.2` becomes
`n_splits=5` with the folds dealt 3/1/1, so the requested proportions are hit exactly at the fold
level and only the stratifier's own group-packing moves them.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_GROUPS = Path("data/interim/groups_manual.csv")
DEFAULT_POLYGONS = Path("data/interim/polygons.csv")

# The split is versioned in its filename rather than overwritten. A frozen split that quietly
# changes under a fixed path is the failure this whole module exists to prevent; a v2 gets a new
# name, a new hash, and the runs that used v1 stay interpretable.
DEFAULT_OUT = Path("data/splits/split_v1.csv")
DEFAULT_META = Path("data/manifest_meta.json")
DEFAULT_AUDIT = Path("reports/data_audit.md")

DEFAULT_RATIOS = (0.6, 0.2, 0.2)
DEFAULT_CANDIDATES = 200
META_VERSION = 1

# The split section is delimited so re-running `pcbi split` replaces it instead of stacking copies.
# `pcbi audit` rewrites the whole report and drops it; the section says so itself.
AUDIT_BEGIN = "<!-- pcbi split: begin -->"
AUDIT_END = "<!-- pcbi split: end -->"

SPLITS = ("train", "val", "test")
NORMAL_CLASS = "normal"

# The acceptance gates. A split that fails any of these is thrown away no matter how well balanced
# it looks elsewhere: a val or test split too thin in a class cannot produce a per-class number
# anyone should act on. Floors rather than proportions, because what breaks a metric is the absolute
# count of crops behind it.
MIN_NORMAL = 12  # G2: normal crops in val, and in test
MIN_DEFECT = 8  # G3: crops of each defect class in val, and in test

# Ratios are matched against K-ths with this slack, which is float-representation noise (0.6 * 5 is
# 3.0000000000000004), not a tolerance for ratios that nearly fit.
RATIO_TOLERANCE = 1e-6
MAX_FOLDS = 20

CSV_FIELDS = [
    "crop_id",
    "dataset_path",
    "polygon_rank_x",
    "joint_position",
    "board",
    "package",
    "cell",
    "viewpoint",
    "class",
    "group_id",
    "split",
    "split_hash",
]


@dataclass(frozen=True)
class Crop:
    """One solder joint, carrying the label it is split on and the group that keeps it honest."""

    crop_id: str
    dataset_path: str
    polygon_rank_x: str
    joint_position: str
    board: str
    package: str
    viewpoint: str
    class_name: str
    group_id: str

    @property
    def cell(self) -> str:
        return f"{self.board}/{self.package}"


@dataclass(frozen=True)
class Choice:
    """The candidate that won, plus enough about the others to explain why it did."""

    assignment: dict[str, str]  # crop_id -> split
    index: int  # which candidate, 0-based
    seed: int  # the random_state that produced it
    deviation: tuple[float, float]
    accepted: int  # how many candidates passed every gate
    considered: int
    rejections: dict[str, int]  # gate name -> how many candidates it rejected


def crop_id(dataset_path: str, polygon_rank_x: str | int) -> str:
    """The identifier for one joint. The single place this format is spelled out.

    `polygons.csv` has no ID column of its own — a joint is identified by which image it is in and
    which lead it is, left-to-right. Joining those with `#` keeps the dataset path readable inside
    the ID, so a crop that turns up in an error message can be found without a lookup.
    """
    return f"{dataset_path}#{polygon_rank_x}"


def load_groups(path: Path) -> dict[str, str]:
    """`dataset_path -> group_id` from any `groups_*.csv`.

    All four grouping methods write `group.CSV_FIELDS`, so this deliberately does not care which one
    produced the file — only that the two columns it needs are there.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"dataset_path", "group_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is not a groups CSV: no {', '.join(sorted(missing))} column. "
                f"Expected a file written by `pcbi group`."
            )
        return {row["dataset_path"]: row["group_id"] for row in reader}


def load_crops(polygons_path: Path, groups: dict[str, str]) -> list[Crop]:
    """Every joint in `polygons.csv`, tagged with the group its image belongs to.

    An image in the polygons CSV with no row in the groups CSV is a hard error rather than a
    singleton. It means the two files were generated from different data, and silently inventing a
    group for the orphan is exactly how a leak gets in.
    """
    with polygons_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    orphans = sorted({row["dataset_path"] for row in rows} - set(groups))
    if orphans:
        shown = ", ".join(orphans[:3])
        more = f" (and {len(orphans) - 3} more)" if len(orphans) > 3 else ""
        raise ValueError(
            f"{len(orphans)} image(s) in {polygons_path} have no group: {shown}{more}. "
            f"Re-run `pcbi group` and `pcbi ingest` against the same dataset root."
        )

    crops = [
        Crop(
            crop_id=crop_id(row["dataset_path"], row["polygon_rank_x"]),
            dataset_path=row["dataset_path"],
            polygon_rank_x=row["polygon_rank_x"],
            joint_position=row["joint_position"],
            board=row["board"],
            package=row["package"],
            viewpoint=row["viewpoint"],
            class_name=row["class"],
            group_id=groups[row["dataset_path"]],
        )
        for row in rows
    ]

    seen = defaultdict(int)
    for crop in crops:
        seen[crop.crop_id] += 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} duplicate crop id(s) in {polygons_path}, e.g. {duplicates[0]}. "
            f"(dataset_path, polygon_rank_x) must identify a joint uniquely."
        )
    return sorted(crops, key=lambda crop: crop.crop_id)


def fold_counts(
    ratios: Sequence[float], tolerance: float = RATIO_TOLERANCE
) -> tuple[int, int, int, int]:
    """`(n_splits, train_folds, val_folds, test_folds)` for the requested ratios.

    Picks the smallest number of folds that expresses all three ratios as whole folds, so the
    requested proportions are exact at the fold level instead of being approximated by a K somebody
    guessed. Ratios that no reasonable K expresses are refused rather than rounded quietly — a split
    that is not the one you asked for is worse than an error, because it is frozen afterwards.
    """
    if len(ratios) != 3:
        raise ValueError(f"--ratios takes exactly 3 values (train val test), got {len(ratios)}.")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"every ratio must be greater than 0, got {format_ratios(ratios)}.")
    if abs(sum(ratios) - 1.0) > tolerance:
        raise ValueError(f"ratios must sum to 1, got {format_ratios(ratios)} = {sum(ratios):g}.")

    for n_splits in range(2, MAX_FOLDS + 1):
        folds = [ratio * n_splits for ratio in ratios]
        rounded = [round(value) for value in folds]
        pairs = zip(folds, rounded, strict=True)
        if all(abs(value - whole) <= tolerance for value, whole in pairs):
            if all(whole >= 1 for whole in rounded) and sum(rounded) == n_splits:
                return (n_splits, *rounded)

    nearest = nearest_ratios(ratios)
    raise ValueError(
        f"no split into at most {MAX_FOLDS} folds gives the ratios {format_ratios(ratios)}. "
        f"The nearest expressible ratios are {format_ratios(nearest)}."
    )


def format_ratios(ratios: Sequence[float]) -> str:
    return " ".join(f"{ratio:g}" for ratio in ratios)


def nearest_ratios(ratios: Sequence[float]) -> tuple[float, ...]:
    """The closest ratios `fold_counts` would accept, for the error message."""
    best: tuple[float, tuple[float, ...]] | None = None
    for n_splits in range(2, MAX_FOLDS + 1):
        folds = [max(1, round(ratio * n_splits)) for ratio in ratios]
        if sum(folds) != n_splits:
            continue
        candidate = tuple(count / n_splits for count in folds)
        error = sum(abs(a - b) for a, b in zip(candidate, ratios, strict=True))
        if best is None or error < best[0]:
            best = (error, candidate)
    return best[1] if best else tuple(ratios)


def candidate_split(
    crops: Sequence[Crop], n_splits: int, counts: Sequence[int], seed: int
) -> dict[str, str]:
    """One candidate assignment of `crop_id -> split`.

    `StratifiedGroupKFold` produces `n_splits` folds that no group crosses; the folds are then dealt
    out in order, `counts` of them per split. Dealing in fold order rather than picking folds at
    random keeps the candidate a pure function of its seed.
    """
    labels = [crop.class_name for crop in crops]
    groups = [crop.group_id for crop in crops]
    placeholder = [0] * len(crops)  # the splitter only reads len(X); the features are never used
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [test_index for _, test_index in splitter.split(placeholder, labels, groups)]

    assignment: dict[str, str] = {}
    start = 0
    for name, take in zip(SPLITS, counts, strict=True):
        for fold in folds[start : start + take]:
            for index in fold:
                assignment[crops[index].crop_id] = name
        start += take
    return assignment


def class_counts(crops: Sequence[Crop], assignment: dict[str, str]) -> dict[str, dict[str, int]]:
    """`split -> class -> crops`, with every split and class present even at zero.

    Zeroes are filled in on purpose: an absent class is the thing the gates are looking for, and a
    missing dictionary key would read as "no data" rather than "none of them landed here".
    """
    classes = sorted({crop.class_name for crop in crops})
    counts = {name: dict.fromkeys(classes, 0) for name in SPLITS}
    for crop in crops:
        counts[assignment[crop.crop_id]][crop.class_name] += 1
    return counts


def group_counts(crops: Sequence[Crop], assignment: dict[str, str]) -> dict[str, int]:
    """`split -> distinct groups`."""
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for crop in crops:
        groups[assignment[crop.crop_id]].add(crop.group_id)
    return {name: len(groups[name]) for name in SPLITS}


def defect_classes(crops: Sequence[Crop]) -> list[str]:
    """Every class except `normal`, read from the data rather than hard-coded."""
    return sorted({crop.class_name for crop in crops} - {NORMAL_CLASS})


def gate(crops: Sequence[Crop], assignment: dict[str, str]) -> list[str]:
    """Why this candidate is unacceptable — empty means it passed.

    Reasons rather than a bool, so that when nothing passes the command can say which gate did the
    rejecting instead of just reporting failure.
    """
    reasons: list[str] = []

    by_group: defaultdict[str, set[str]] = defaultdict(set)
    for crop in crops:
        by_group[crop.group_id].add(assignment[crop.crop_id])
    leaking = sorted(group for group, names in by_group.items() if len(names) > 1)
    if leaking:
        reasons.append(f"G1: {len(leaking)} group(s) span more than one split, e.g. {leaking[0]}")

    counts = class_counts(crops, assignment)
    for name in ("val", "test"):
        found = counts[name].get(NORMAL_CLASS, 0)
        if found < MIN_NORMAL:
            reasons.append(f"G2: {name} has {found} {NORMAL_CLASS} crop(s), need {MIN_NORMAL}")
    for name in ("val", "test"):
        for class_name in defect_classes(crops):
            found = counts[name].get(class_name, 0)
            if found < MIN_DEFECT:
                reasons.append(f"G3: {name} has {found} {class_name} crop(s), need {MIN_DEFECT}")
    return reasons


def deviation(
    crops: Sequence[Crop], assignment: dict[str, str], ratios: Sequence[float]
) -> tuple[float, float]:
    """How far this candidate drifts from the overall class mix, then from the asked-for sizes.

    Both terms are label counts and sizes only — nothing here knows a model exists. The class term
    leads because an unbalanced class mix distorts every per-class metric, while a split a few crops
    off its target size does not.
    """
    counts = class_counts(crops, assignment)
    total = len(crops)
    overall: defaultdict[str, int] = defaultdict(int)
    for crop in crops:
        overall[crop.class_name] += 1

    class_drift = 0.0
    size_drift = 0.0
    for name, ratio in zip(SPLITS, ratios, strict=True):
        size = sum(counts[name].values())
        size_drift += abs(size / total - ratio)
        if size == 0:
            class_drift += len(overall)  # every class is missing; the worst score available
            continue
        for class_name, count in overall.items():
            class_drift += abs(counts[name][class_name] / size - count / total)
    return (class_drift, size_drift)


def split_hash(assignment: dict[str, str]) -> str:
    """SHA-256 over the sorted `(crop_id, split)` pairs, and nothing else.

    Deliberately not a hash of the file. Adding a column, reordering rows or fixing a typo in some
    unrelated field would all change a file hash while leaving the actual partition untouched, and a
    fingerprint that changes when the split has not is a fingerprint nobody keeps checking.
    """
    payload = "\n".join(f"{key},{assignment[key]}" for key in sorted(assignment))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose(
    crops: Sequence[Crop],
    n_splits: int,
    counts: Sequence[int],
    ratios: Sequence[float],
    candidates: int,
    seed: int,
) -> Choice:
    """Generate `candidates` splits and keep the best one that passes every gate.

    Candidate `i` uses `random_state = seed + i`, so the whole search is reproducible from the two
    numbers recorded on the command line. Among the candidates that pass, the winner is the one
    closest to the overall class mix, ties broken by size and then by candidate order — never by
    anything a model produced.
    """
    if candidates < 1:
        raise ValueError(f"--candidates must be at least 1, got {candidates}.")
    distinct_groups = len({crop.group_id for crop in crops})
    if distinct_groups < n_splits:
        raise ValueError(
            f"{distinct_groups} group(s) cannot be dealt into {n_splits} folds. "
            f"Use coarser ratios or a finer grouping."
        )

    rejections: defaultdict[str, int] = defaultdict(int)
    best: tuple[tuple[float, float], int, dict[str, str]] | None = None
    accepted = 0

    for index in range(candidates):
        assignment = candidate_split(crops, n_splits, counts, seed + index)
        reasons = gate(crops, assignment)
        if reasons:
            for name in sorted({reason.split(":")[0] for reason in reasons}):
                rejections[name] += 1
            continue
        accepted += 1
        score = deviation(crops, assignment, ratios)
        if best is None or score < best[0]:
            best = (score, index, assignment)

    if best is None:
        tally = ", ".join(f"{name} rejected {count}" for name, count in sorted(rejections.items()))
        raise ValueError(
            f"none of the {candidates} candidate split(s) passed the gates ({tally}). "
            f"This is a fact about the data, not a seed to keep hunting: with "
            f"{distinct_groups} group(s) the class floors may be unreachable. Widen "
            f"--candidates, coarsen --ratios, or revisit the gates deliberately."
        )

    score, index, assignment = best
    return Choice(
        assignment=assignment,
        index=index,
        seed=seed + index,
        deviation=score,
        accepted=accepted,
        considered=candidates,
        rejections=dict(rejections),
    )


def build_rows(crops: Sequence[Crop], assignment: dict[str, str], hash_hex: str) -> list[dict]:
    """One row per crop. `split_hash` repeats on every row so the file carries its own fingerprint.

    The hash is computed from the pairs before any row is built, so the column is derived from the
    split and never from itself.
    """
    return [
        {
            "crop_id": crop.crop_id,
            "dataset_path": crop.dataset_path,
            "polygon_rank_x": crop.polygon_rank_x,
            "joint_position": crop.joint_position,
            "board": crop.board,
            "package": crop.package,
            "cell": crop.cell,
            "viewpoint": crop.viewpoint,
            "class": crop.class_name,
            "group_id": crop.group_id,
            "split": assignment[crop.crop_id],
            "split_hash": hash_hex,
        }
        for crop in crops
    ]


def write_rows_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --- the count table, in both renderings -----------------------------------------------------


def ordered_classes(crops: Sequence[Crop]) -> list[str]:
    """Classes most-frequent first, the ordering `pcbi ingest` prints its counts in."""
    overall: defaultdict[str, int] = defaultdict(int)
    for crop in crops:
        overall[crop.class_name] += 1
    return sorted(overall, key=lambda name: (-overall[name], name))


def count_table(crops: Sequence[Crop], assignment: dict[str, str]) -> tuple[list[str], list[list]]:
    """`(headers, rows)` for the per-split class counts, with a total row.

    One function behind both the terminal table and the one in the report, so the two can never
    disagree about a number. The cells stay as ints; each renderer formats them its own way.
    """
    classes = ordered_classes(crops)
    counts = class_counts(crops, assignment)
    sizes = group_counts(crops, assignment)
    headers = ["split", "groups", "crops", *classes]
    rows: list[list] = [
        [name, sizes[name], sum(counts[name].values()), *(counts[name][c] for c in classes)]
        for name in SPLITS
    ]
    totals = [sum(row[index] for row in rows) for index in range(1, len(headers))]
    rows.append(["total", *totals])
    return headers, rows


def gate_margins(crops: Sequence[Crop], assignment: dict[str, str]) -> list[tuple[str, int, int]]:
    """`(label, count, floor)` for every G2/G3 floor, tightest margin first.

    What the gates *passed by* is the number worth reading. "All gates cleared" hides whether the
    scarcest class made it by one crop or by thirty.
    """
    counts = class_counts(crops, assignment)
    margins = []
    for name in ("val", "test"):
        margins.append((f"{name} {NORMAL_CLASS}", counts[name].get(NORMAL_CLASS, 0), MIN_NORMAL))
        for class_name in defect_classes(crops):
            margins.append((f"{name} {class_name}", counts[name].get(class_name, 0), MIN_DEFECT))
    return sorted(margins, key=lambda item: item[1] - item[2])


# --- the artifacts ---------------------------------------------------------------------------


def build_meta(
    crops: Sequence[Crop],
    choice: Choice,
    hash_hex: str,
    out: Path,
    groups_path: Path,
    polygons_path: Path,
    ratios: Sequence[float],
    n_splits: int,
    counts: Sequence[int],
    seed: int,
) -> dict:
    """The provenance record that sits next to the split.

    Deliberately not a second copy of the assignment — that is the CSV's job. This answers "which
    split is this, and what produced it", so a run logging only the hash can be traced back to the
    inputs and the search that chose it.
    """
    headers, rows = count_table(crops, choice.assignment)
    return {
        "version": META_VERSION,
        "generated": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "split_hash": hash_hex,
        "splits_csv": out.as_posix(),
        "inputs": {
            "groups_csv": groups_path.as_posix(),
            "polygons_csv": polygons_path.as_posix(),
            "crops": len(crops),
            "groups": len({crop.group_id for crop in crops}),
        },
        "search": {
            "ratios": list(ratios),
            "n_splits": n_splits,
            "folds": dict(zip(SPLITS, counts, strict=True)),
            "candidates": choice.considered,
            "split_seed": seed,
            "chosen_index": choice.index,
            "chosen_seed": choice.seed,
            "accepted": choice.accepted,
            "rejections": choice.rejections,
            "deviation": {"class_mix": choice.deviation[0], "size": choice.deviation[1]},
        },
        "gates": {
            "min_normal_in_val_and_test": MIN_NORMAL,
            "min_defect_in_val_and_test": MIN_DEFECT,
            "margins": [
                {"what": label, "count": count, "floor": floor, "slack": count - floor}
                for label, count, floor in gate_margins(crops, choice.assignment)
            ],
        },
        "counts": {row[0]: dict(zip(headers[1:], row[1:], strict=True)) for row in rows},
    }


def write_meta(meta: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def render_audit_section(crops: Sequence[Crop], choice: Choice, meta: dict, command: str) -> str:
    """The `## 13.` section `pcbi split` adds to the data audit, between its markers."""
    out: list[str] = []
    add = out.append
    inputs, search = meta["inputs"], meta["search"]

    add(AUDIT_BEGIN)
    add("## 13. Frozen train/val/test split")
    add("")
    add(f"Written by `{command}` on {meta['generated'].replace('T', ' ').rstrip('Z')} UTC.")
    add(
        "Re-running that command replaces this section in place. `pcbi audit` rewrites the whole "
        "report and drops it, so regenerate the split afterwards if you want it back."
    )
    add("")
    add(
        f"The split is **grouped**: all {inputs['crops']} crops sit in {inputs['groups']} groups, "
        f"and no group spans two splits — so no physical joint appears in both train and test. It "
        f"is **stratified only approximately**: {inputs['groups']} groups is few, and a group "
        f"carrying two `{NORMAL_CLASS}` crops moves several percent of the scarcest class at once. "
        f"That is why the counts are printed here rather than assumed."
    )
    add("")
    add(f"- **Split file:** `{meta['splits_csv']}`")
    add(f"- **Manifest metadata:** `{DEFAULT_META.as_posix()}`")
    add(f"- **Grouping:** `{inputs['groups_csv']}`")
    add(f"- **Labels:** `{inputs['polygons_csv']}`")
    add(
        f"- **Ratios:** {format_ratios(search['ratios'])} ({search['n_splits']} folds, dealt "
        f"{'/'.join(str(search['folds'][name]) for name in SPLITS)})"
    )
    add(f"- **Split hash:** `{meta['split_hash']}`")
    add("")

    headers, rows = count_table(crops, choice.assignment)
    # The first three headers are labels; the rest are class names, so they stay verbatim in
    # backticks — `normal` is a value in the CSV's `class` column, not a word to title-case.
    labelled = [head.capitalize() for head in headers[:3]] + [f"`{c}`" for c in headers[3:]]
    add("| " + " | ".join(labelled) + " |")
    add("| :--- |" + " ---: |" * (len(headers) - 1))
    for row in rows:
        cells = (
            [f"**{row[0]}**"] + [f"**{value}**" for value in row[1:]]
            if row[0] == "total"
            else [str(value) for value in row]
        )
        add("| " + " | ".join(cells) + " |")
    add("")

    add(
        f"Candidate {choice.index + 1} of {choice.considered} (seed {choice.seed}) won, chosen on "
        f"label counts alone — closest to the overall class mix among the "
        f"{choice.accepted} that passed every gate. **No model result entered this choice**, "
        f"because a test set picked to flatter a model is not a test set."
    )
    add("")
    add("The gates, all on counts:")
    add("")
    add("- **G1** — no group ID appears in more than one split.")
    add(f"- **G2** — val and test each hold at least {MIN_NORMAL} `{NORMAL_CLASS}` crops.")
    add(f"- **G3** — val and test each hold at least {MIN_DEFECT} crops of every defect class.")
    add("")
    tightest = meta["gates"]["margins"][0]
    add(
        f"Tightest margin: **{tightest['what']} at {tightest['count']} against a floor of "
        f"{tightest['floor']}** ({tightest['slack']:+d}). A future regrouping that narrows this "
        f"further is a signal to widen the gates deliberately, not to lower them quietly."
    )
    add("")
    add(
        "**Every training run records the split hash.** Two runs reporting different hashes did "
        "not train on the same data. The hash covers the sorted (crop ID, split) pairs only — not "
        "the manifest — so adding a column to the CSV leaves it unchanged, and a crop moving "
        "between splits changes it."
    )
    add(AUDIT_END)
    return "\n".join(out)


def update_audit_report(path: Path, section: str) -> bool:
    """Append `section` to the audit report, or replace the one already there.

    Returns True if an existing section was replaced. The file has to exist: this adds a section to
    `pcbi audit`'s report rather than owning a report of its own, and creating a stub named
    `data_audit.md` that holds only a split table would be a confusing thing to leave behind.
    """
    if not path.is_file():
        raise ValueError(f"No such report: {path}. Run `pcbi audit` first.")

    text = path.read_text(encoding="utf-8")
    start, end = text.find(AUDIT_BEGIN), text.find(AUDIT_END)
    if start != -1 and end != -1:
        updated = text[:start] + section + text[end + len(AUDIT_END) :]
        replaced = True
    else:
        updated = text.rstrip("\n") + "\n\n" + section + "\n"
        replaced = False
    path.write_text(updated, encoding="utf-8")
    return replaced


def write_splits_csv(
    groups_path: Path,
    polygons_path: Path,
    out: Path,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    candidates: int = DEFAULT_CANDIDATES,
    seed: int = 0,
    meta_out: Path | None = None,
) -> tuple[list[Crop], Choice, str, dict]:
    """Freeze one split to `out`, and its provenance to `meta_out` when given.

    Returns the crops, the winning candidate, its hash and the manifest metadata — the last so a
    caller can render the same numbers without recomputing them.
    """
    n_splits, *counts = fold_counts(ratios)
    crops = load_crops(polygons_path, load_groups(groups_path))
    if not crops:
        raise ValueError(f"{polygons_path} holds no crops to split.")

    choice = choose(crops, n_splits, counts, ratios, candidates, seed)
    hash_hex = split_hash(choice.assignment)
    write_rows_csv(build_rows(crops, choice.assignment, hash_hex), out)

    meta = build_meta(
        crops, choice, hash_hex, out, groups_path, polygons_path, ratios, n_splits, counts, seed
    )
    if meta_out is not None:
        write_meta(meta, meta_out)
    return crops, choice, hash_hex, meta
