"""Stage 1 Step 2: polygon overlay sheets for visual QA, before any cropping happens.

A polygon count and a class label are just numbers; only looking at the polygon drawn on the
actual photo catches a misaligned box, a rotated image, or an annotation that doesn't line up with
what the JSON says. This draws exactly what `pcbi ingest` produced — the same points, class, and
x-rank — directly onto the source image, so a bad join or a bad crop shows up here first.

Two things can silently misplace a polygon on the overlay:

- **EXIF orientation.** A camera can tag a JPEG "display this rotated" instead of storing pixels
  already rotated. If the annotation tool applied that tag when the annotator drew polygons but
  this module doesn't, every polygon lands in the wrong place. `ImageOps.exif_transpose` corrects
  for it before anything is drawn.
- **BGR vs. RGB.** Only relevant if something here used OpenCV — it doesn't. Everything stays on
  PIL, which loads RGB throughout, so there is no channel order to get wrong.
"""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from pcbi.data import audit as audit_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data.show import compose_grid, tile_from_image

TILE_SIZE = (360, 360)
COLUMNS = 6
MAX_TILES_PER_SHEET = 24
SAMPLE_SIZE = 12
SAMPLE_SEED = 0  # fixed, so a "random" sample is the same PNG on every run

# These are the *final*, post-thumbnail pixel sizes a sheet should show. The source photos (up to
# 2560x1440) get letterboxed down to TILE_SIZE for the grid — a ~7x shrink — so a fixed-width
# outline drawn at that scale would all but vanish. render_annotated_image scales these up by the
# image's own downscale factor before drawing, so the polygon a viewer sees is a consistent
# thickness regardless of the source resolution.
OUTLINE_WIDTH = 3
BBOX_WIDTH = 2
FONT_SIZE = 15
BBOX_COLOR = (255, 255, 255)
CLASS_COLORS = {
    "normal": (90, 210, 90),
    "excess": (230, 170, 40),
    "insufficient": (230, 70, 70),
    "spike": (90, 170, 230),
}
DEFAULT_COLOR = (230, 230, 230)

BBOX_STATS_FIELDS = [
    "class",
    "package",
    "count",
    "width_mean",
    "width_median",
    "height_mean",
    "height_median",
    "area_mean",
    "area_median",
]


def group_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    """Every polygon row, bucketed by the image it belongs to."""
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["dataset_path"]].append(row)
    return dict(groups)


def bbox_of(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _int_points(points: list[list[float]]) -> list[tuple[int, int]]:
    # PIL's outline drawing (width > 1) requires integer coordinates; sub-pixel precision buys
    # nothing for a QA overlay anyway.
    return [(round(x), round(y)) for x, y in points]


def polygon_area(points: list[list[float]]) -> float:
    """The shoelace formula — the polygon's own area, not its bounding box's."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 9.2, no scalable default font
        return ImageFont.load_default()


def render_annotated_image(
    path: Path, polygons: list[dict], target_size: tuple[int, int] | None = TILE_SIZE
) -> Image.Image:
    """Open one image, correct its EXIF orientation, and draw every polygon/bbox/label on it.

    Outline and font sizes are scaled up by how much `target_size` will later shrink this image
    (see the module-level note by OUTLINE_WIDTH) so annotations stay legible on the finished sheet
    instead of thinning out to nothing. Pass `target_size=None` for a full-resolution single-image
    render (`pcbi qa-polygons --image ...`) — there's no later shrink to compensate for.
    """
    with Image.open(path) as raw:
        raw.load()
        img = ImageOps.exif_transpose(raw).convert("RGB")

    if target_size is None:
        scale = 1.0
    else:
        scale = max(img.width / target_size[0], img.height / target_size[1], 1.0)
    outline_width = max(1, round(OUTLINE_WIDTH * scale))
    bbox_width = max(1, round(BBOX_WIDTH * scale))
    font_size = max(10, round(FONT_SIZE * scale))

    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    for row in polygons:
        points = row["points"]
        color = CLASS_COLORS.get(row["class"], DEFAULT_COLOR)
        if len(points) >= 2:
            draw.polygon(_int_points(points), outline=color, width=outline_width)
        bbox = bbox_of(points)
        draw.rectangle([round(v) for v in bbox], outline=BBOX_COLOR, width=bbox_width)
        label = f"{row['class']} #{row['polygon_rank_x']}"
        text_xy = (round(bbox[0]), max(round(bbox[1]) - font_size - 2, 0))
        draw.text(text_xy, label, fill=color, font=font)
    return img


def find_dataset_path(groups: dict[str, list[dict]], image: str) -> str:
    """Resolve a user-supplied image reference to the exact key `groups` uses.

    Accepts the full dataset-relative path exactly as it appears in `polygons.csv` (either slash
    direction), or just a bare filename for convenience — which can be ambiguous if the same
    filename happens to exist under more than one (board, package), so that case is reported
    rather than picked for the caller.
    """
    normalized = image.replace("\\", "/")
    if normalized in groups:
        return normalized
    matches = sorted(key for key in groups if Path(key).name == normalized)
    if not matches:
        raise ValueError(f"{image!r} has no joint-task annotation under this root.")
    if len(matches) > 1:
        raise ValueError(f"{image!r} matches more than one image: {', '.join(matches)}")
    return matches[0]


def render_single(
    root: Path, image: str, taxonomy_path: Path = ingest_mod.DEFAULT_TAXONOMY
) -> tuple[str, Image.Image]:
    """Render one joint-task image at full resolution, by dataset-relative path or filename."""
    taxonomy = ingest_mod.load_taxonomy(taxonomy_path)
    result = audit_mod.audit(root)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    groups = group_by_image(rows)
    dataset_path = find_dataset_path(groups, image)
    annotated = render_annotated_image(
        result.root / dataset_path, groups[dataset_path], target_size=None
    )
    return dataset_path, annotated


def build_sheets(
    root: Path,
    groups: dict[str, list[dict]],
    dataset_paths: list[str],
    max_per_sheet: int = MAX_TILES_PER_SHEET,
) -> list[Image.Image]:
    """One or more grid sheets covering `dataset_paths`, in the order given."""
    sheets = []
    for start in range(0, len(dataset_paths), max_per_sheet):
        chunk = dataset_paths[start : start + max_per_sheet]
        tiles = []
        for dataset_path in chunk:
            polygons = groups[dataset_path]
            img = render_annotated_image(root / dataset_path, polygons)
            first = polygons[0]
            caption = f"{first['board']}/{first['package']} {Path(dataset_path).name}"
            tiles.append(tile_from_image(img, caption, size=TILE_SIZE))
        sheets.append(compose_grid(tiles, columns=COLUMNS))
    return sheets


def write_sheets(sheets: list[Image.Image], out_dir: Path, stem: str) -> list[Path]:
    """Save each sheet as `{stem}.png`, `{stem}_2.png`, `{stem}_3.png`, ... in order."""
    paths = []
    for i, sheet in enumerate(sheets, start=1):
        name = f"{stem}.png" if i == 1 else f"{stem}_{i}.png"
        path = out_dir / name
        sheet.save(path)
        paths.append(path)
    return paths


def bbox_stats(rows: list[dict]) -> list[dict]:
    """Bounding-box width/height and polygon area, summarized per (class, package)."""
    groups: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["class"], row["package"])].append(row)

    stats_rows = []
    for (class_name, package), group_rows in sorted(groups.items()):
        widths, heights, areas = [], [], []
        for row in group_rows:
            x0, y0, x1, y1 = bbox_of(row["points"])
            widths.append(x1 - x0)
            heights.append(y1 - y0)
            areas.append(polygon_area(row["points"]))
        stats_rows.append(
            {
                "class": class_name,
                "package": package,
                "count": len(group_rows),
                "width_mean": statistics.mean(widths),
                "width_median": statistics.median(widths),
                "height_mean": statistics.mean(heights),
                "height_median": statistics.median(heights),
                "area_mean": statistics.mean(areas),
                "area_median": statistics.median(areas),
            }
        )
    return stats_rows


def write_qa_report(
    root: Path,
    out_dir: Path = Path("reports/qa"),
    taxonomy_path: Path = ingest_mod.DEFAULT_TAXONOMY,
    seed: int = SAMPLE_SEED,
    sample_size: int = SAMPLE_SIZE,
    max_per_sheet: int = MAX_TILES_PER_SHEET,
) -> dict:
    """Audit `root`, then write the sample overlay, the 3-polygon overlay, and bbox_stats.csv."""
    taxonomy = ingest_mod.load_taxonomy(taxonomy_path)
    result = audit_mod.audit(root)
    rows = ingest_mod.ingest_rows(result, taxonomy)
    groups = group_by_image(rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(groups)
    sample_paths = random.Random(seed).sample(all_paths, k=min(sample_size, len(all_paths)))
    sample_sheets = build_sheets(result.root, groups, sample_paths, max_per_sheet=max_per_sheet)
    write_sheets(sample_sheets, out_dir, "overlay_sample")

    three_poly_paths = sorted(p for p in all_paths if groups[p][0]["polygon_count"] == 3)
    three_poly_sheets = build_sheets(
        result.root, groups, three_poly_paths, max_per_sheet=max_per_sheet
    )
    write_sheets(three_poly_sheets, out_dir, "overlay_3poly")

    stats_rows = bbox_stats(rows)
    stats_path = out_dir / "bbox_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BBOX_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(stats_rows)

    return {
        "sample_images": len(sample_paths),
        "three_poly_images": len(three_poly_paths),
        "three_poly_sheets": len(three_poly_sheets),
        "bbox_stats_rows": len(stats_rows),
    }
