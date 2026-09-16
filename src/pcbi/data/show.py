"""Build image grids for the Stage 0 Step 6 visual check.

A count table proves two viewpoint folders hold the same number of images; it cannot prove they
show the same physical part, or that a `Setup` folder really is the brighter lighting. This module
puts the actual pixels in front of a person (or a model that can look at images) so that can be
checked instead of assumed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pcbi.data.audit import AuditResult, classify

THUMB_SIZE = (220, 220)
PADDING = 8
LABEL_HEIGHT = 18
BACKGROUND = (24, 24, 24)
TEXT_COLOR = (230, 230, 230)
ERROR_COLOR = (120, 20, 20)
MAX_COLUMNS = 6


def select_cell(
    result: AuditResult,
    board: str,
    package: str,
    viewpoint: str,
    setup: str | None = None,
) -> list[Path]:
    """Every image in one (board, package, viewpoint[, setup]) cell, in filename order.

    Filename order is timestamp order on this dataset — a real fact about capture sequence, not a
    guess about component identity. Sorting by it is what makes `--pair` meaningful: it is the only
    thing that lets a V2 image line up with its V2.1 counterpart at all, and it should be read as
    exactly that assumption, not as a confirmed pairing.
    """
    matches = []
    for path in result.images:
        info = classify(path, result.root)
        if info.board != board or info.package != package or info.viewpoint != viewpoint:
            continue
        if setup is not None and info.setup != setup:
            continue
        matches.append(path)
    return sorted(matches, key=lambda p: p.name)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _thumbnail_image(img: Image.Image, size: tuple[int, int] = THUMB_SIZE) -> Image.Image:
    """Letterbox an already-loaded image onto a fixed-size canvas."""
    rgb = img.convert("RGB")
    rgb.thumbnail(size)
    canvas = Image.new("RGB", size, color=BACKGROUND)
    offset = ((size[0] - rgb.width) // 2, (size[1] - rgb.height) // 2)
    canvas.paste(rgb, offset)
    return canvas


def _thumbnail(path: Path, size: tuple[int, int] = THUMB_SIZE) -> Image.Image:
    """One tile's image area: the photo, letterboxed onto a fixed-size canvas.

    A file that fails to decode becomes a visibly red tile with the error message on it, rather
    than aborting the whole grid — one bad image should not hide the other 23.
    """
    try:
        with Image.open(path) as img:
            img.load()
            return _thumbnail_image(img, size)
    except Exception as exc:  # noqa: BLE001 - a bad file must not abort the whole grid
        canvas = Image.new("RGB", size, color=ERROR_COLOR)
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"unreadable:\n{exc}", fill=TEXT_COLOR, font=_font())
        return canvas


def tile_from_image(
    img: Image.Image, caption: str, size: tuple[int, int] = THUMB_SIZE
) -> Image.Image:
    """A labeled grid tile built from an already-loaded image — e.g. one qa-polygons annotated."""
    thumb = _thumbnail_image(img, size)
    tile = Image.new("RGB", (size[0], size[1] + LABEL_HEIGHT), color=BACKGROUND)
    tile.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((4, size[1] + 2), caption[:42], fill=TEXT_COLOR, font=_font())
    return tile


def _tile(image_path: Path, caption: str) -> Image.Image:
    thumb = _thumbnail(image_path)
    tile = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), color=BACKGROUND)
    tile.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((4, THUMB_SIZE[1] + 2), caption[:42], fill=TEXT_COLOR, font=_font())
    return tile


def compose_grid(tiles: list[Image.Image], columns: int = MAX_COLUMNS) -> Image.Image:
    """Lay already-built tiles out left to right, top to bottom."""
    if not tiles:
        canvas = Image.new("RGB", (420, 60), color=BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 20), "No images matched.", fill=TEXT_COLOR, font=_font())
        return canvas

    tile_w, tile_h = tiles[0].size
    cols = min(columns, len(tiles))
    rows = -(-len(tiles) // cols)  # ceil division
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * PADDING, rows * tile_h + (rows + 1) * PADDING),
        color=BACKGROUND,
    )
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        x = PADDING + col * (tile_w + PADDING)
        y = PADDING + row * (tile_h + PADDING)
        canvas.paste(tile, (x, y))
    return canvas


def build_grid(
    image_paths: list[Path],
    captions: list[str] | None = None,
    columns: int = MAX_COLUMNS,
) -> Image.Image:
    """Lay thumbnails out left to right, top to bottom, each labeled with its filename."""
    if captions is None:
        captions = [p.name for p in image_paths]
    tiles = [_tile(path, caption) for path, caption in zip(image_paths, captions, strict=True)]
    return compose_grid(tiles, columns)


def build_pair_grid(
    left_paths: list[Path],
    right_paths: list[Path],
    left_label: str,
    right_label: str,
) -> Image.Image:
    """`left` and `right` images side by side, matched by position within each sorted list.

    The row count is the shorter of the two lists; the matching itself is the timestamp-order
    assumption `select_cell` documents, not recorded evidence linking the two images.
    """
    n = min(len(left_paths), len(right_paths))
    interleaved: list[Path] = []
    captions: list[str] = []
    for i in range(n):
        interleaved.append(left_paths[i])
        captions.append(f"{left_label} #{i}: {left_paths[i].name}")
        interleaved.append(right_paths[i])
        captions.append(f"{right_label} #{i}: {right_paths[i].name}")
    return build_grid(interleaved, captions, columns=2)
