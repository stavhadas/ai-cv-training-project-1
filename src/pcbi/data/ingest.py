"""Stage 1 Step 1: one row per joint-quality polygon, joined back to board/package/lighting.

Reuses the audit's filename join (`resolve_annotation_image`) and path classifier (`classify`) so
this and `pcbi audit` never disagree about where an image lives. Only images that resolve to the
joint task (V2 / V2.1) are read here — `annotations_by_task` already keeps the placement vocabulary
out of this bucket, and the extra viewpoint check below exists so a change to that split fails
loudly here instead of quietly mixing two label vocabularies that happen to share the string
`good` (see reports/data_audit.md section 7).
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from pcbi.data import audit as audit_mod

DEFAULT_TAXONOMY = Path("configs/data/taxonomy.yaml")
JOINT_VIEWPOINTS = {"V2", "V2.1"}

# reports/data_audit.md section 12 confirmed V2 and V2.1 are genuinely different light directions
# but could not recover *which* is which from the images alone (no physical reference photo). This
# assignment is a working assumption, not a measured fact — if it's ever contradicted by physical
# inspection of the bench setup, this is the one place to fix it.
LIGHT_DIRECTION_BY_VIEWPOINT = {"V2": "bottom_to_top", "V2.1": "top_to_bottom"}

# SolDef_AI's camera names files WIN_YYYYMMDD_HH_MM_SS_Pro.jpg. This is a capture-order hint, not a
# component ID — reports/data_audit.md section 12 rules out timestamp order as an ID.
TIMESTAMP_RE = re.compile(r"^WIN_(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})_Pro$")

CSV_FIELDS = [
    "image_name",
    "dataset_path",
    "board",
    "package",
    "viewpoint",
    "light_direction",
    "capture_timestamp",
    "raw_label",
    "class",
    "points",
    "polygon_count",
    "polygon_rank_x",
    "joint_position",
    "merged_from",
]

# notes/s1_three_polygons.md: 43 of 200 joint images carry a third polygon marking a second defect
# on a joint that already has one — a single-label-per-crop classifier can't represent that, so
# every joint's polygons collapse into one, by fixed class precedence (earlier wins).
MERGE_PRECEDENCE = ["insufficient", "spike", "excess", "normal"]


@dataclass
class Taxonomy:
    classes: dict[str, str]
    exclude: set[str]


def load_taxonomy(path: Path = DEFAULT_TAXONOMY) -> Taxonomy:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Taxonomy(
        classes=dict(data.get("classes") or {}),
        exclude=set(data.get("exclude") or []),
    )


def parse_capture_timestamp(stem: str) -> str | None:
    """The capture time encoded in a SolDef_AI filename stem, as an ISO string, or None."""
    match = TIMESTAMP_RE.match(stem)
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


def polygon_centroid_x(points: list) -> float:
    xs = [point[0] for point in points if len(point) >= 1]
    return statistics.mean(xs) if xs else 0.0


def rank_by_x(shapes: list[dict]) -> dict[int, int]:
    """Each shape's 1-based rank by centroid x-position, left to right, among shapes in one image.

    Ties keep filing order (Python's sort is stable), which is the only sane tiebreak: nothing in
    the data says two polygons at the same x are in any other order.
    """
    order = sorted(range(len(shapes)), key=lambda i: polygon_centroid_x(shapes[i]["points"]))
    return {shape_index: rank + 1 for rank, shape_index in enumerate(order)}


def ingest_rows(result: audit_mod.AuditResult, taxonomy: Taxonomy) -> list[dict]:
    """One row per polygon in every joint-task annotation."""
    joint = audit_mod.annotations_by_task(result).get("joint", [])
    rows: list[dict] = []

    for record in joint:
        path, _ = audit_mod.resolve_annotation_image(result, record)
        if path is None:
            continue
        info = audit_mod.classify(path, result.root)
        if info.viewpoint not in JOINT_VIEWPOINTS:
            raise ValueError(
                f"joint-task annotation resolved to viewpoint {info.viewpoint!r} ({path}); "
                f"expected one of {sorted(JOINT_VIEWPOINTS)}."
            )

        ranks = rank_by_x(record.shapes)
        for index, shape in enumerate(record.shapes):
            raw_label = shape["label"]
            if raw_label not in taxonomy.classes:
                raise ValueError(
                    f"unmapped label {raw_label!r} in {record.source} — add it to "
                    f"{DEFAULT_TAXONOMY} or fix the annotation."
                )
            rows.append(
                {
                    "image_name": path.name,
                    "dataset_path": path.relative_to(result.root).as_posix(),
                    "board": info.board,
                    "package": info.package,
                    "viewpoint": info.viewpoint,
                    "light_direction": LIGHT_DIRECTION_BY_VIEWPOINT.get(info.viewpoint, ""),
                    "capture_timestamp": parse_capture_timestamp(path.stem) or "",
                    "raw_label": raw_label,
                    "class": taxonomy.classes[raw_label],
                    "points": shape["points"],  # native list; CSV writing JSON-encodes it
                    "polygon_count": len(record.shapes),
                    "polygon_rank_x": ranks[index],
                }
            )

    return rows


def _bbox_center_x(points: list[list[float]]) -> float:
    xs = [p[0] for p in points]
    return (min(xs) + max(xs)) / 2


def union_bbox_points(points_list: list[list[list[float]]]) -> list[list[float]]:
    """The union of one or more polygons' bounding boxes, as a 4-corner rectangle — the merged
    joint's crop box."""
    xs = [p[0] for points in points_list for p in points]
    ys = [p[1] for points in points_list for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def cluster_joints(rows: list[dict]) -> list[list[int]]:
    """Split one image's polygon rows into physical joints by bounding-box center x.

    Every component in this dataset has exactly two leads (notes/s1_three_polygons.md), so two
    polygons on the same joint sit close together in x, while polygons on different joints are
    separated by the much larger gap of the component body between them. Splitting the sorted
    centers at their single largest gap finds that boundary without requiring the polygons to
    overlap at all — unlike an overlap-area test, this also catches two same-class duplicates
    drawn side by side on one joint with no overlap between them (a real case in this dataset).
    """
    n = len(rows)
    if n <= 1:
        return [list(range(n))]
    order = sorted(range(n), key=lambda i: _bbox_center_x(rows[i]["points"]))
    centers = [_bbox_center_x(rows[i]["points"]) for i in order]
    gaps = [centers[k + 1] - centers[k] for k in range(n - 1)]
    split = gaps.index(max(gaps))
    return [order[: split + 1], order[split + 1 :]]


def reduce_joint_cluster(rows: list[dict]) -> dict:
    """Collapse every polygon on one physical joint into a single row.

    Two phases, both from notes/s1_three_polygons.md, and neither assumes which classes are
    involved: same-class duplicates keep one polygon at random (the rule only says "keep one"),
    and the remaining, now-distinct classes are resolved by precedence: insufficient > spike >
    excess > normal. Covers any combination, so this never needs manual review. The kept row's
    `points` become the union of every merged polygon's bounding box.
    """
    if len(rows) == 1:
        return dict(rows[0], merged_from="")

    precedence = {name: i for i, name in enumerate(MERGE_PRECEDENCE)}

    by_class: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    dropped: list[dict] = []
    representatives: list[dict] = []
    for group in by_class.values():
        keeper = random.choice(group)
        dropped.extend(r for r in group if r is not keeper)
        representatives.append(keeper)

    representatives.sort(key=lambda r: precedence[r["class"]])
    kept, *outclassed = representatives
    dropped.extend(outclassed)

    merged = dict(kept)
    merged["points"] = union_bbox_points([kept["points"], *(d["points"] for d in dropped)])
    merged["merged_from"] = "+".join([kept["class"], *(d["class"] for d in dropped)])
    return merged


def merge_double_defect_joints(rows: list[dict]) -> list[dict]:
    """Apply the merge rule in notes/s1_three_polygons.md, image by image.

    Every image's polygons are clustered into (at most two) physical joints by `cluster_joints`,
    then each cluster with more than one polygon is collapsed by `reduce_joint_cluster`. Both
    steps handle any combination of classes, so this always succeeds — no case needs manual
    review. Returns rows after merging: one row per joint, with `merged_from` set on the ones
    that merged.
    """
    by_image: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["dataset_path"]].append(row)

    merged_rows: list[dict] = []
    for image_rows in by_image.values():
        for cluster in cluster_joints(image_rows):
            cluster_rows = [image_rows[i] for i in cluster]
            merged_rows.append(reduce_joint_cluster(cluster_rows))

    _finalize_positions(merged_rows)
    return merged_rows


def _finalize_positions(rows: list[dict]) -> None:
    """Recompute polygon_count/polygon_rank_x/joint_position per image, after merging.

    Every image now holds exactly two joints by construction (`cluster_joints` always returns at
    most two groups), so this is really just left/right labeling — but it's recomputed over
    whatever actually remains rather than assumed, so it can't silently drift out of sync.
    """
    by_image: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["dataset_path"]].append(row)

    for image_rows in by_image.values():
        image_rows.sort(key=lambda r: _bbox_center_x(r["points"]))
        n = len(image_rows)
        for rank, row in enumerate(image_rows, start=1):
            row["polygon_count"] = n
            row["polygon_rank_x"] = rank
            row["joint_position"] = "left" if rank == 1 else ("right" if rank == n else "middle")


def write_polygons_csv(
    root: Path, out: Path, taxonomy_path: Path = DEFAULT_TAXONOMY
) -> tuple[list[dict], list[dict]]:
    """Audit `root`, apply the notes/s1_three_polygons.md merge rule, and write the manifest.

    Only the merged rows (one per joint) are written to `out` — that's the training manifest.
    Returns (raw per-polygon rows before merging, final rows after merging) so a caller can report
    before/after class counts without re-auditing the dataset.
    """
    taxonomy = load_taxonomy(taxonomy_path)
    result = audit_mod.audit(root)
    raw_rows = ingest_rows(result, taxonomy)
    merged_rows = merge_double_defect_joints(raw_rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({**row, "points": json.dumps(row["points"])})

    return raw_rows, merged_rows
