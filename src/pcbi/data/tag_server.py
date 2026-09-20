"""The browser front end for `pcbi tag`: a local page that shows a cell and takes the pairing.

A grid of 63 photographs with a click-to-pair interaction is a browser's job, so this serves one
from the standard library — no new dependency, no build step, nothing to install. The page is a
single HTML file next to this module; everything below it is plumbing.

Three things are deliberate:

- **Loopback only.** The server binds `127.0.0.1`, so nothing outside this machine can reach it.
  It is also the reason there is no authentication: there is no remote to authenticate.
- **Indices, not paths, in the image URL.** `/img/17` is looked up in the in-memory image list, so
  no request can name a file — a URL cannot walk out of the dataset root because it never contains
  a path at all.
- **Every click is saved.** A mutation writes both the pairing JSON and the grouping CSV before it
  answers. Tagging 84 pairs is an hour of a person's attention; it must not depend on them
  remembering to press save, or on this process exiting cleanly.

Thumbnails are cropped to the component using the same `group.load_image` the fingerprints used, so
what a person judges by is what a model would have seen — one view of the data, not two.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from pcbi.data import group as group_mod
from pcbi.data import tag as tag_mod

PAGE = Path(__file__).with_name("tag_page.html")

THUMB_WIDTH = 460  # the tile as it is rendered; wide enough to read a solder fillet
ZOOM_WIDTH = 1400  # the lightbox, for when two candidates look the same at tile size
JPEG_QUALITY = 82

HOST = "127.0.0.1"
DEFAULT_PORT = 8713


@dataclass
class TagState:
    """The pairing, the images it refers to, and where both get written.

    Everything that touches `groups` goes through `mutate`, which holds a lock and saves — the
    server is threaded (so slow image decodes do not block a click), and two clicks arriving at once
    must not each write a version of the state computed from the same stale copy.
    """

    root: Path
    images: list[group_mod.ImageRecord]
    cells: list[tag_mod.Cell]
    groups: list[list[str]]
    pairs_path: Path
    out_path: Path
    tolerance: float | None = tag_mod.DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, str, int], bytes] = {}

    def mutate(self, change) -> dict:
        """Apply `change(groups) -> groups`, persist the result, and return the new state."""
        with self._lock:
            self.groups = tag_mod.save_groups(self.pairs_path, change(self.groups))
            tag_mod.write_manual_csv(self.images, self.groups, self.out_path)
            return self.payload()

    def payload(self) -> dict:
        """Everything the page needs to draw badges: the groups and the progress counts."""
        summary = tag_mod.summarize(self.images, self.cells, self.groups)
        return {
            "groups": self.groups,
            "summary": {
                "images": summary.images,
                "groups": summary.groups,
                "tagged": summary.tagged,
                "untagged": summary.untagged,
                "pairable": summary.pairable,
                "unknown": summary.unknown,
                "percent": round(summary.percent, 1),
            },
        }

    def manifest(self) -> dict:
        """The fixed half of the state: which images exist and how they are laid out."""
        return {
            "images": [
                {
                    "index": index,
                    "path": image.dataset_path,
                    "name": image.image_name,
                    "label": label_of(image),
                    "cell": image.cell,
                    "viewpoint": image.viewpoint,
                    "hint": image.component_hint,
                }
                for index, image in enumerate(self.images)
            ],
            "cells": [
                {
                    "name": cell.name,
                    "board": cell.board,
                    "package": cell.package,
                    "size": cell.size,
                    "pairable": cell.pairable,
                    "balanced": cell.balanced,
                    "hinted": sum(
                        1
                        for _, indices in cell.columns
                        for index in indices
                        if self.images[index].component_hint
                    ),
                    "columns": [
                        {"viewpoint": viewpoint, "images": indices}
                        for viewpoint, indices in cell.columns
                    ],
                }
                for cell in self.cells
            ],
            "tolerances": [
                {"value": "full" if value is None else value, "label": tolerance_label(value)}
                for value in tag_mod.TOLERANCE_CHOICES
            ],
            "tolerance": "full" if self.tolerance is None else self.tolerance,
            "out": str(self.out_path),
            "pairs": str(self.pairs_path),
            **self.payload(),
        }

    def thumbnail(self, index: int, tolerance: float | None, width: int) -> bytes:
        """One rendered tile, cached — the same crop is requested every time a cell is redrawn."""
        key = (index, "full" if tolerance is None else f"{tolerance:g}", width)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Decoding happens outside the lock on purpose: a 2560x1440 JPEG takes long enough that
        # serialising it would make the grid load one tile at a time. Two threads racing on the same
        # tile just render it twice and agree on the answer.
        rendered = render_thumbnail(self.root, self.images[index], tolerance, width)
        self._cache[key] = rendered
        return rendered


def label_of(image: group_mod.ImageRecord) -> str:
    """The part of the filename that differs between images: `13_33_34`, plus any `_C##`."""
    stem = Path(image.dataset_path).stem.removesuffix("_Pro")
    parts = stem.split("_")
    label = "_".join(parts[-3:]) if len(parts) >= 3 else stem
    return f"{label}  {image.component_hint}" if image.component_hint else label


def tolerance_label(value: float | None) -> str:
    if value is None:
        return "whole frame"
    return "joints only" if value == 0 else f"+{value:g}x"


def render_thumbnail(
    root: Path, record: group_mod.ImageRecord, tolerance: float | None, width: int
) -> bytes:
    """Crop to the component the way grouping does, shrink to `width`, encode as JPEG."""
    image = group_mod.load_image(root, record, tolerance)
    if image.width > width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def parse_tolerance(raw: str | None, fallback: float | None) -> float | None:
    """`"full"` means no crop; anything unparseable falls back rather than 500-ing a tile."""
    if raw is None:
        return fallback
    if raw == "full":
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return fallback


def make_handler(state: TagState) -> type[BaseHTTPRequestHandler]:
    """Bind one `TagState` into a request handler class for `ThreadingHTTPServer`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "pcbi-tag"
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            """Silence the access log — a cell redraw is 63 image requests of pure noise."""

        # --- replies ---

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if content_type.startswith("image/"):
                self.send_header("Cache-Control", "max-age=3600")
            else:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

        def _fail(self, message: str, status: int = 400) -> None:
            self._json({"error": message}, status)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError:
                return {}

        def _cell(self, name) -> tag_mod.Cell | None:
            return next((cell for cell in state.cells if cell.name == name), None)

        # --- routes ---

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(state.manifest())
            elif url.path.startswith("/img/"):
                self._image(url.path.removeprefix("/img/"), query)
            else:
                self._fail("not found", 404)

        def _image(self, raw_index: str, query: dict) -> None:
            try:
                index = int(raw_index)
            except ValueError:
                self._fail("bad image index", 400)
                return
            if not 0 <= index < len(state.images):
                self._fail("no such image", 404)
                return
            tolerance = parse_tolerance(query.get("t", [None])[0], state.tolerance)
            width = ZOOM_WIDTH if query.get("zoom") else THUMB_WIDTH
            try:
                self._send(state.thumbnail(index, tolerance, width), "image/jpeg")
            except OSError as exc:  # a missing or corrupt file must not kill the whole grid
                self._fail(f"could not read image: {exc}", 500)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
            url = urlparse(self.path)
            body = self._body()
            if url.path == "/api/link":
                self._link(body)
            elif url.path == "/api/unlink":
                self._unlink(body)
            elif url.path == "/api/hint":
                self._per_cell(body, tag_mod.pair_by_hint)
            elif url.path == "/api/guess":
                self._per_cell(body, tag_mod.guess_by_order)
            elif url.path == "/api/clear":
                self._per_cell(body, tag_mod.clear_cell)
            else:
                self._fail("not found", 404)

        def _paths(self, body: dict, *keys: str) -> list[str] | None:
            known = {image.dataset_path for image in state.images}
            values = [body.get(key) for key in keys]
            if any(value not in known for value in values):
                self._fail("unknown image path", 400)
                return None
            return values  # type: ignore[return-value]

        def _link(self, body: dict) -> None:
            paths = self._paths(body, "a", "b")
            if paths is None:
                return
            a, b = paths
            self._json(state.mutate(lambda groups: tag_mod.link(groups, a, b)))

        def _unlink(self, body: dict) -> None:
            paths = self._paths(body, "path")
            if paths is None:
                return
            (path,) = paths
            self._json(state.mutate(lambda groups: tag_mod.unlink(groups, path)))

        def _per_cell(self, body: dict, action) -> None:
            cell = self._cell(body.get("cell"))
            if cell is None:
                self._fail("unknown cell", 400)
                return
            self._json(state.mutate(lambda groups: action(state.images, cell, groups)))

    return Handler


def build_state(
    root: Path,
    pairs_path: Path = tag_mod.DEFAULT_PAIRS,
    out_path: Path = tag_mod.DEFAULT_OUT,
    tolerance: float | None = tag_mod.DEFAULT_TOLERANCE,
    taxonomy_path: Path | None = None,
) -> TagState:
    """Audit the dataset, load any saved pairing, and hold both ready to serve."""
    images = (
        group_mod.load_images(root)
        if taxonomy_path is None
        else group_mod.load_images(root, taxonomy_path)
    )
    return TagState(
        root=root,
        images=images,
        cells=tag_mod.build_cells(images),
        groups=tag_mod.load_groups(pairs_path),
        pairs_path=pairs_path,
        out_path=out_path,
        tolerance=tolerance,
    )


def make_server(state: TagState, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """A loopback server ready to `serve_forever`. `port=0` picks a free one."""
    server = ThreadingHTTPServer((HOST, port), make_handler(state))
    server.daemon_threads = True
    return server
