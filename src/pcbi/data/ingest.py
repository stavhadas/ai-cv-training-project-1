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
import re
import statistics
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
]


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
                    "points": json.dumps(shape["points"]),
                    "polygon_count": len(record.shapes),
                    "polygon_rank_x": ranks[index],
                }
            )

    return rows


def write_polygons_csv(root: Path, out: Path, taxonomy_path: Path = DEFAULT_TAXONOMY) -> list[dict]:
    """Audit `root`, then write one CSV row per joint-task polygon to `out`."""
    taxonomy = load_taxonomy(taxonomy_path)
    result = audit_mod.audit(root)
    rows = ingest_rows(result, taxonomy)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows
