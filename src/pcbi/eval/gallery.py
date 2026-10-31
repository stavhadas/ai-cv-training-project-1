"""Stage 3: `pcbi gallery` — the confusion matrix, drawn with the crops themselves.

A confusion matrix says *how many* ``insufficient`` joints were called ``normal``. This puts those
joints on screen, so you can ask *why*: a lighting setup, one board, a merged double-defect label,
a crop that is genuinely ambiguous. Two images:

- ``<name>.png``: rows are the true class and columns the predicted class, in `CLASS_ORDER`, up to
  ``per_cell`` crops in each cell. The layout matches the confusion matrix `pcbi train` prints.
- ``<name>_mistakes.png``: only the off-diagonal cells, one band each, largest first.

**Which crops, and in what order.** Within a cell, crops are sorted by the defect score
(1 − P(normal)), most confident first: descending in the columns that predicted a defect,
ascending in the ``normal`` column. For a mistake that puts the most confidently wrong crops at the
front, which are the ones most worth looking at. Ties break on ``crop_id``, so a rerun on the same
predictions draws the same crops in the same places, byte for byte.

``merged_from`` is not a manifest column; it lives in ``polygons.csv``, joined through
`crops.crop_id` — the same function that named the crop files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pcbi.data import crops as crops_mod
from pcbi.data import show as show_mod
from pcbi.data.datasets import CLASS_ORDER, CLASS_TO_INDEX

DEFAULT_OUT_DIR = Path("reports/gallery")
DEFAULT_PER_CELL = 8

THUMB = (160, 160)
CELL_COLUMNS = 4  # tiles per row inside one cell
GAP = 4  # between tiles inside a cell
CELL_GAP = 16  # between cells
CAPTION_LINES = 2
LINE_HEIGHT = 13
HEADER_HEIGHT = 22
ROW_LABEL_WIDTH = 150
WRONG_COLOR = (90, 30, 30)  # off-diagonal cell background, so mistakes stand out at a glance
RIGHT_COLOR = (30, 60, 35)


@dataclass(frozen=True)
class Entry:
    crop_id: str
    true: int  # index into CLASS_ORDER
    pred: int
    score: float  # defect score, 1 - P(normal)
    board: str
    package: str
    merged_from: str


@dataclass(frozen=True)
class Cell:
    true: int
    pred: int
    total: int  # crops in this cell before the per_cell cap
    shown: list[Entry]


def load_merged_from(polygons: Path) -> dict[str, str] | None:
    """`crop_id -> merged_from` (empty string when not merged), or None if there is no file."""
    if not polygons.is_file():
        return None
    with polygons.open(newline="", encoding="utf-8") as handle:
        return {
            crops_mod.crop_id(
                row["board"],
                row["package"],
                row["viewpoint"],
                row["dataset_path"],
                row["polygon_rank_x"],
            ): row.get("merged_from", "")
            for row in csv.DictReader(handle)
        }


def join(
    crop_ids: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    manifest: Path,
    merged: dict[str, str] | None,
) -> list[Entry]:
    """One `Entry` per prediction, with board/package from the manifest.

    Raises:
        ValueError: a predicted crop is not in the manifest, or its label there disagrees with the
            label saved alongside the predictions. Either means the predictions came from a
            different crop set than the one on disk, and every image drawn would be a guess.
    """
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row["crop_id"]: row for row in csv.DictReader(handle)}

    missing = [crop_id for crop_id in crop_ids if crop_id not in rows]
    if missing:
        raise ValueError(
            f"{len(missing)} predicted crop(s) are not in {manifest}, e.g. {missing[0]}. "
            f"Were the predictions made on a different crop set?"
        )

    entries = []
    for crop_id, label, pred, score in zip(crop_ids, labels, preds, scores, strict=True):
        row = rows[crop_id]
        if CLASS_TO_INDEX[row["label"]] != int(label):
            raise ValueError(
                f"{crop_id}: the predictions say {CLASS_ORDER[int(label)]}, {manifest} says "
                f"{row['label']}. The predictions and the manifest come from different crop sets."
            )
        entries.append(
            Entry(
                crop_id=crop_id,
                true=int(label),
                pred=int(pred),
                score=float(score),
                board=row["board"],
                package=row["package"],
                merged_from=(merged or {}).get(crop_id, ""),
            )
        )
    return entries


def build_cells(entries: list[Entry], per_cell: int) -> dict[tuple[int, int], Cell]:
    """Every (true, pred) cell, including empty ones, with its `per_cell` most confident crops."""
    normal = CLASS_TO_INDEX["normal"]
    cells = {}
    for true in range(len(CLASS_ORDER)):
        for pred in range(len(CLASS_ORDER)):
            members = [e for e in entries if e.true == true and e.pred == pred]
            # Most confident first: high defect score where a defect was predicted, low where
            # normal was. crop_id breaks ties so the order never depends on input order.
            sign = 1.0 if pred == normal else -1.0
            members.sort(key=lambda e: (sign * e.score, e.crop_id))
            cells[(true, pred)] = Cell(true, pred, len(members), members[:per_cell])
    return cells


def mistake_order(cells: dict[tuple[int, int], Cell]) -> list[Cell]:
    """Off-diagonal, non-empty cells: largest first, then in CLASS_ORDER."""
    wrong = [c for c in cells.values() if c.true != c.pred and c.total]
    return sorted(wrong, key=lambda c: (-c.total, c.true, c.pred))


def fit(draw: ImageDraw.ImageDraw, text: str, width: int) -> str:
    """`text`, cut with an ellipsis until it fits in `width` pixels."""
    font = show_mod._font()
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def tile(entry: Entry, crops_dir: Path) -> Image.Image:
    """One crop with its caption: score and board/package, then merged_from when set."""
    thumb = show_mod._thumbnail(crops_dir / f"{entry.crop_id}.png", THUMB)
    caption_h = CAPTION_LINES * LINE_HEIGHT + 4
    out = Image.new("RGB", (THUMB[0], THUMB[1] + caption_h), color=show_mod.BACKGROUND)
    out.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(out)
    lines = [f"{entry.score:.2f}  {entry.board}/{entry.package}"]
    if entry.merged_from:
        lines.append(f"merged {entry.merged_from}")
    for i, line in enumerate(lines):
        draw.text(
            (3, THUMB[1] + 2 + i * LINE_HEIGHT),
            fit(draw, line, THUMB[0] - 6),
            fill=show_mod.TEXT_COLOR,
            font=show_mod._font(),
        )
    return out


def cell_size(per_cell: int) -> tuple[int, int]:
    tile_w = THUMB[0]
    tile_h = THUMB[1] + CAPTION_LINES * LINE_HEIGHT + 4
    cols = min(CELL_COLUMNS, per_cell)
    rows = -(-per_cell // cols)
    return (
        cols * tile_w + (cols + 1) * GAP,
        HEADER_HEIGHT + rows * tile_h + (rows + 1) * GAP,
    )


def draw_cell(cell: Cell, crops_dir: Path, per_cell: int, header: str) -> Image.Image:
    """A fixed-size block: a header line, then up to `per_cell` tiles. Empty cells stay blank."""
    width, height = cell_size(per_cell)
    color = RIGHT_COLOR if cell.true == cell.pred else WRONG_COLOR
    out = Image.new("RGB", (width, height), color=color if cell.total else show_mod.BACKGROUND)
    draw = ImageDraw.Draw(out)
    draw.text((GAP, 5), fit(draw, header, width - 2 * GAP), fill=show_mod.TEXT_COLOR)

    cols = min(CELL_COLUMNS, per_cell)
    for i, entry in enumerate(cell.shown):
        piece = tile(entry, crops_dir)
        row, col = divmod(i, cols)
        out.paste(
            piece,
            (GAP + col * (piece.width + GAP), HEADER_HEIGHT + GAP + row * (piece.height + GAP)),
        )
    return out


def shown_text(cell: Cell) -> str:
    if len(cell.shown) == cell.total:
        return f"{cell.total}"
    return f"{len(cell.shown)} of {cell.total}"


def render_matrix(
    cells: dict[tuple[int, int], Cell], crops_dir: Path, per_cell: int, title: str
) -> Image.Image:
    n = len(CLASS_ORDER)
    cell_w, cell_h = cell_size(per_cell)
    width = ROW_LABEL_WIDTH + n * cell_w + (n + 1) * CELL_GAP
    height = 2 * HEADER_HEIGHT + n * cell_h + (n + 1) * CELL_GAP
    out = Image.new("RGB", (width, height), color=show_mod.BACKGROUND)
    draw = ImageDraw.Draw(out)
    draw.text((CELL_GAP, 5), title, fill=show_mod.TEXT_COLOR)

    for pred, name in enumerate(CLASS_ORDER):
        x = ROW_LABEL_WIDTH + CELL_GAP + pred * (cell_w + CELL_GAP)
        draw.text((x, HEADER_HEIGHT + 5), f"predicted {name}", fill=show_mod.TEXT_COLOR)
    for true, name in enumerate(CLASS_ORDER):
        y = 2 * HEADER_HEIGHT + CELL_GAP + true * (cell_h + CELL_GAP)
        row_total = sum(cells[(true, p)].total for p in range(n))
        draw.text((CELL_GAP, y + 5), f"true {name}", fill=show_mod.TEXT_COLOR)
        draw.text((CELL_GAP, y + 5 + LINE_HEIGHT), f"{row_total} crops", fill=show_mod.TEXT_COLOR)
        for pred in range(n):
            cell = cells[(true, pred)]
            block = draw_cell(cell, crops_dir, per_cell, shown_text(cell))
            out.paste(block, (ROW_LABEL_WIDTH + CELL_GAP + pred * (cell_w + CELL_GAP), y))
    return out


def render_mistakes(
    cells: dict[tuple[int, int], Cell], crops_dir: Path, per_cell: int, title: str
) -> Image.Image:
    bands = mistake_order(cells)
    cell_w, cell_h = cell_size(per_cell)
    width = max(cell_w + 2 * CELL_GAP, 420)
    height = HEADER_HEIGHT + max(len(bands), 1) * (cell_h + CELL_GAP) + CELL_GAP
    out = Image.new("RGB", (width, height), color=show_mod.BACKGROUND)
    draw = ImageDraw.Draw(out)
    draw.text((CELL_GAP, 5), title, fill=show_mod.TEXT_COLOR)
    if not bands:
        draw.text((CELL_GAP, HEADER_HEIGHT + CELL_GAP), "No mistakes.", fill=show_mod.TEXT_COLOR)
        return out
    for i, cell in enumerate(bands):
        header = (
            f"true {CLASS_ORDER[cell.true]} -> predicted {CLASS_ORDER[cell.pred]}: "
            f"{shown_text(cell)}"
        )
        block = draw_cell(cell, crops_dir, per_cell, header)
        out.paste(block, (CELL_GAP, HEADER_HEIGHT + CELL_GAP + i * (cell_h + CELL_GAP)))
    return out


@dataclass(frozen=True)
class Result:
    matrix: Path
    mistakes: Path
    crops: int
    wrong: int
    merged_known: bool  # False when polygons.csv was missing and merged_from could not be shown


def make_gallery(
    preds_path: Path,
    manifest: Path,
    crops_dir: Path,
    out: Path,
    per_cell: int = DEFAULT_PER_CELL,
    polygons: Path = crops_mod.DEFAULT_POLYGONS,
) -> Result:
    """Write the matrix image to `out` and the mistakes image beside it as `<stem>_mistakes.png`."""
    # Imported here: product pulls in torch and matplotlib, and `pcbi --help` must not pay for it.
    import torch

    from pcbi.eval.predictions import load_predictions
    from pcbi.eval.product import defect_score

    if per_cell < 1:
        raise ValueError(f"per_cell must be at least 1, got {per_cell}.")
    result = load_predictions(preds_path)
    scores = defect_score(torch.from_numpy(result.logits)).numpy()
    preds = result.logits.argmax(axis=1)

    merged = load_merged_from(polygons)
    entries = join(result.crop_ids, result.labels, preds, scores, manifest, merged)
    cells = build_cells(entries, per_cell)

    name = out.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix = render_matrix(
        cells, crops_dir, per_cell, f"{name}: {len(entries)} crops, rows true, columns predicted"
    )
    matrix.save(out, **crops_mod.PNG_SAVE)

    mistakes_path = out.with_name(f"{name}_mistakes.png")
    wrong = sum(1 for e in entries if e.true != e.pred)
    mistakes = render_mistakes(
        cells, crops_dir, per_cell, f"{name}: {wrong} mistakes of {len(entries)}, largest first"
    )
    mistakes.save(mistakes_path, **crops_mod.PNG_SAVE)

    return Result(out, mistakes_path, len(entries), wrong, merged is not None)
