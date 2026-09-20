"""Stage 1 Step 4: candidate groupings of "images that show the same physical component".

SolDef_AI records no component ID anywhere — `reports/data_audit.md` section 12 tested filenames,
path depth, cross-viewpoint ordering, cell parity and the rare `_C##` suffix, and concluded
`not recoverable`. But the train/test split needs one: two photographs of the same physical joint
(the same part under `V2` and `V2.1` lighting) must never land on opposite sides, or the test score
measures memorization.

So the grouping has to be *constructed*. This module builds candidates three ways and writes each to
its own CSV. It deliberately does not judge them — that comes later, by reading the CSVs, which is
why every column needed for that judgement (`cell`, `component_hint`, `viewpoint`) is written out
even though nothing here reads it back.

Which direction to err in, when a method is unsure:

- **Over-merging** (two different parts in one group) is harmless. Groups get bigger, splits
  coarser, and nothing leaks.
- **Under-merging** (one part split across two groups) is dangerous — the halves can straddle the
  split, which is the exact leak this exists to prevent.

Before any fingerprint is taken, `phash` and `embed` crop to the component. `pcbi ingest` records
two joint polygons per image, one per lead; the box containing both, grown by a configurable
tolerance, *is* the component. Without that crop the fingerprint is dominated by 2560x1440 pixels of
the same board under the same framing — see the note on DEFAULT_THRESHOLDS for the measured damage.

So when in doubt, merge — but only within a (board, package) cell. A physical component never moves
between boards or packages, so comparing across cells could only ever produce a false merge, and
`compare_within_cells` restricts every pairwise comparison accordingly. The trade-off is worth
stating: this makes a cell-spanning group structurally impossible, so the "does this grouping cross
a cell boundary?" check can no longer fail. It stays in the report as a guard that the restriction
is actually holding, not as evidence that a method works.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

from pcbi.data import audit as audit_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import qa_polygons as qa_mod

METHODS = ("coarse", "phash", "embed")

# Measured on the real download against the four usable `_C##` reference pairs (same physical
# component under V2 and V2.1), fingerprinting the **whole 2560x1440 frame**:
#
#   phash, 256 bits   same component 124-142 apart | different components in a cell, min 42
#   embed, cosine     same component 0.836-0.890   | different components in a cell, up to 0.940
#
# In both cases the same component was *farther apart* than many pairs of different components, so
# no threshold separated them — these defaults produce 200 singletons rather than a wrong answer.
# The crop stage below exists because of exactly this: on a full frame, roughly half of every pixel
# a fingerprint sees is board that every image in the cell shares.
DEFAULT_THRESHOLDS: dict[str, float | None] = {"coarse": None, "phash": 10.0, "embed": 0.98}

# How much board to keep around the component, as a fraction of its box's longer side, added on all
# four sides. Expressed as a fraction rather than pixels so one value means the same thing for an
# R0603 and an R1206.
#
# The sweep stops at 0.5 because of what these photographs actually are. They are macro shots: the
# component box is about 1000-1250px across in a 2560x1440 frame, so a tolerance of 0.5 already adds
# ~500px on each side and clamps against the frame edge. Anything beyond that is the full frame
# under a different name — measured, not assumed, from the crop-example sheets.
DEFAULT_CROP_TOLERANCE = 0.1
TOLERANCE_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.5)

# The one component signal that survives into any filename: `WIN_..._Pro_C12.jpg`. Only 20 files
# carry it, and among the annotated joint-task images only four numbers (C10, C12, C13, C14, all in
# CS6/C0805) appear under both V2 and V2.1 — those four pairs are the entire ground truth available
# for checking any of these methods.
COMPONENT_HINT_RE = re.compile(r"_C(\d+)$")

# 256-bit hashes, not imagehash's 64-bit default. Every image in a cell shares board, package and
# framing, so the low-frequency content these hashes keep is mostly the part they have in common;
# 64 bits leaves too few to separate one component from its neighbour.
PHASH_SIZE = 16

EMBED_MODEL = "resnet18"
EMBED_BATCH = 16

CSV_FIELDS = [
    "image_name",
    "dataset_path",
    "board",
    "package",
    "cell",
    "viewpoint",
    "component_hint",
    "method",
    "threshold",
    "crop_tolerance",
    "fingerprint",
    "group_id",
    "group_size",
]


@dataclass(frozen=True)
class ImageRecord:
    """One joint-task image, with the metadata a later analysis needs to judge its grouping."""

    image_name: str
    dataset_path: str  # relative to the dataset root, posix separators
    board: str
    package: str
    viewpoint: str
    component_hint: str  # "C12", or "" for the images with no suffix
    joint_box: tuple[float, float, float, float] | None = None  # union bbox of every polygon

    @property
    def cell(self) -> str:
        return f"{self.board}/{self.package}"


class UnionFind:
    """Merges indices into groups, transitively: A~B and B~C puts all three together.

    That transitivity is the point. A method never has to compare A against C directly to group
    them, which matters when the link between two photographs of one part runs through a third.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:  # attach the smaller tree to the larger
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def parse_component_hint(stem: str) -> str:
    """The `_C##` component number in a filename stem, or "" where there isn't one."""
    match = COMPONENT_HINT_RE.search(stem)
    return f"C{match.group(1)}" if match else ""


def collect_images(
    result: audit_mod.AuditResult, taxonomy: ingest_mod.Taxonomy
) -> list[ImageRecord]:
    """One record per distinct joint-task image, in a stable dataset_path order.

    Built from `ingest_rows` rather than by walking the tree again: those rows already carry board,
    package and viewpoint, so this and `pcbi ingest` can never disagree about where an image sits.

    `joint_box` spans *every* polygon in the image, which is what the crop stage needs. The raw
    pre-merge rows are used deliberately: `reduce_joint_cluster` sets a merged row's points to the
    union bbox of everything it absorbed, so the union over the two merged joints and the union over
    all the raw polygons are the same rectangle — and reading the raw rows keeps this clear of the
    random tie-break inside the merge.
    """
    by_image = qa_mod.group_by_image(ingest_mod.ingest_rows(result, taxonomy))
    images = []
    for dataset_path in sorted(by_image):
        rows = by_image[dataset_path]
        first = rows[0]
        images.append(
            ImageRecord(
                image_name=first["image_name"],
                dataset_path=dataset_path,
                board=first["board"],
                package=first["package"],
                viewpoint=first["viewpoint"],
                component_hint=parse_component_hint(Path(dataset_path).stem),
                joint_box=qa_mod.bbox_of([p for row in rows for p in row["points"]]),
            )
        )
    return images


def coarse_components(images: Sequence[ImageRecord]) -> list[list[int]]:
    """Group by (board, package) — read straight from the folder tree.

    Certainly leak-free, because a physical part never moves between boards or packages. Also very
    blunt: on the joint task it yields exactly 7 groups.
    """
    by_cell: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        by_cell[(image.board, image.package)].append(index)
    return [sorted(members) for members in by_cell.values()]


def hamming_pairs(fingerprints: dict[int, str], threshold: int) -> list[tuple[int, int]]:
    """Index pairs whose hex fingerprints differ by at most `threshold` bits.

    Works on the hex strings rather than imagehash objects so the comparison stays a plain integer
    popcount — no library needed to reproduce or test it.
    """
    indices = sorted(fingerprints)
    bits = {index: int(fingerprints[index], 16) for index in indices}
    return [
        (a, b)
        for position, a in enumerate(indices)
        for b in indices[position + 1 :]
        if (bits[a] ^ bits[b]).bit_count() <= threshold
    ]


def cosine_pairs(vectors: dict[int, Sequence[float]], threshold: float) -> list[tuple[int, int]]:
    """Index pairs whose feature vectors have cosine similarity at least `threshold`."""
    indices = sorted(vectors)
    if not indices:
        return []
    matrix = np.asarray([vectors[index] for index in indices], dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    similarity = matrix @ matrix.T
    return [
        (indices[i], indices[j])
        for i in range(len(indices))
        for j in range(i + 1, len(indices))
        if similarity[i, j] >= threshold
    ]


def union_find_components(n: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Every index 0..n-1 placed into a group, merging transitively across `pairs`."""
    union_find = UnionFind(n)
    for a, b in pairs:
        union_find.union(a, b)
    by_root: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(n):
        by_root[union_find.find(index)].append(index)
    return [sorted(members) for members in by_root.values()]


def assign_group_ids(
    components: Sequence[Sequence[int]], images: Sequence[ImageRecord], method: str
) -> dict[int, str]:
    """Stable `<method>_NNN` ids, so the same input always writes the same CSV.

    Components are ordered by their smallest member's dataset_path, which depends only on the data
    and not on iteration order. The method prefix means CSVs from different methods can be
    concatenated without their ids colliding.
    """
    ordered = sorted(
        components, key=lambda members: min(images[index].dataset_path for index in members)
    )
    return {
        index: f"{method}_{number:03d}"
        for number, members in enumerate(ordered, start=1)
        for index in members
    }


def crop_box(
    joint_box: tuple[float, float, float, float],
    tolerance: float,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """The component's box, grown by `tolerance * max(w, h)` on each side and clamped to the frame.

    The margin scales with the *longer* side so a tall narrow box and a short wide one both keep a
    comparable amount of surrounding board — scaling each axis independently would stretch a thin
    box into a square-ish crop and quietly change what the fingerprint sees.

    Clamping means a component near an edge gets an asymmetric crop rather than a padded one. That
    is the honest version: there is no board beyond the frame to include.
    """
    x0, y0, x1, y1 = joint_box
    margin = tolerance * max(x1 - x0, y1 - y0)
    width, height = size
    return (
        max(0, int(floor(x0 - margin))),
        max(0, int(floor(y0 - margin))),
        min(width, int(ceil(x1 + margin))),
        min(height, int(ceil(y1 + margin))),
    )


def load_image(root: Path, record: ImageRecord, tolerance: float | None = None) -> Image.Image:
    """The pixels a fingerprint is computed from — the whole frame, or the component crop.

    The single place grouping opens an image, so `phash` and `embed` can never disagree about EXIF
    orientation, colour mode or how much of the photo they are looking at. `tolerance=None` keeps
    the full frame; anything else crops to `crop_box`.
    """
    with Image.open(root / record.dataset_path) as raw:
        raw.load()
        image = ImageOps.exif_transpose(raw).convert("RGB")
    if tolerance is None or record.joint_box is None:
        return image
    return image.crop(crop_box(record.joint_box, tolerance, image.size))


def phash_fingerprint(image: Image.Image, hash_size: int = PHASH_SIZE) -> str:
    """The perceptual hash of one already-loaded image, as hex."""
    return str(imagehash.phash(image, hash_size=hash_size))


def load_embedder(
    model_name: str = EMBED_MODEL,
) -> Callable[[list[Image.Image]], list[list[float]]]:
    """Build a `batch of images -> feature vectors` callable, loading the model once.

    torch and timm are imported here rather than at module scope because they take seconds to
    import, and every other `pcbi` command would pay that cost just to reach its own code.
    """
    import timm
    import torch
    from torchvision import transforms

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    config = timm.data.resolve_data_config({}, model=model)
    height, width = config["input_size"][-2:]

    # Resize whatever comes in, squashing the aspect ratio, rather than timm's own transform, which
    # resizes and then centre-crops. Both halves of that matter here and for opposite reasons: on a
    # full frame the centre crop would discard most of the photograph, and on a component crop
    # (`load_image` with a tolerance) it would cut into the very thing being fingerprinted. A plain
    # Resize to a 2-tuple takes the whole input either way — on a crop that means *upscaling* a few
    # hundred pixels to fill the input, so the component occupies the frame instead of ~30px of it.
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )

    def embed(images: list[Image.Image]) -> list[list[float]]:
        batch = torch.stack([transform(image) for image in images])
        with torch.no_grad():
            return model(batch).tolist()

    return embed


def embed_all(
    images: Sequence[ImageRecord],
    root: Path,
    embedder: Callable[[list[Image.Image]], list[list[float]]],
    batch_size: int = EMBED_BATCH,
    tolerance: float | None = None,
) -> dict[int, list[float]]:
    """Feature vector per image index. `embedder` is passed in so this is testable without torch."""
    vectors: dict[int, list[float]] = {}
    for start in range(0, len(images), batch_size):
        chunk = range(start, min(start + batch_size, len(images)))
        loaded = [load_image(root, images[index], tolerance) for index in chunk]
        for index, vector in zip(chunk, embedder(loaded), strict=True):
            vectors[index] = vector
    return vectors


def compute_signatures(
    root: Path,
    images: Sequence[ImageRecord],
    method: str,
    embedder: Callable[[list[Image.Image]], list[list[float]]] | None = None,
    tolerance: float | None = None,
) -> dict | None:
    """The per-image data a threshold gets applied to — hashes or vectors; None for coarse.

    Split out from the grouping itself because it is the expensive half (embedding 200 images means
    loading a CNN), and sweeping a range of thresholds only needs it computed once. Sweeping a range
    of *tolerances* does not get that discount: each one is a different crop, so each one is a fresh
    pass over the pixels.
    """
    if method == "phash":
        return {
            index: phash_fingerprint(load_image(root, image, tolerance))
            for index, image in enumerate(images)
        }
    if method == "embed":
        return embed_all(images, root, embedder or load_embedder(), tolerance=tolerance)
    return None


def compare_within_cells(
    pair_function: Callable[[dict, float], list[tuple[int, int]]],
    signatures: dict,
    threshold: float,
    images: Sequence[ImageRecord],
) -> list[tuple[int, int]]:
    """Run a pairwise comparison separately inside each (board, package) cell.

    A component never moves between boards or packages, so a cross-cell comparison could only ever
    produce a false merge — there is nothing to gain by making it. Blocking by cell also cuts the
    work sharply: 200 images are 19,900 global pairs but only a few thousand within cells.
    """
    pairs: list[tuple[int, int]] = []
    for members in coarse_components(images):
        block = {index: signatures[index] for index in members}
        pairs.extend(pair_function(block, threshold))
    return pairs


def components_at_threshold(
    method: str, signatures: dict, threshold: float, images: Sequence[ImageRecord]
) -> list[list[int]]:
    """Group images from precomputed signatures, at one threshold, cell by cell."""
    if method == "phash":
        pairs = compare_within_cells(hamming_pairs, signatures, int(threshold), images)
    elif method == "embed":
        pairs = compare_within_cells(cosine_pairs, signatures, float(threshold), images)
    else:
        raise ValueError(f"{method!r} does not use a threshold.")
    return union_find_components(len(images), pairs)


def group_images(
    root: Path,
    images: Sequence[ImageRecord],
    method: str,
    threshold: float | None,
    embedder: Callable[[list[Image.Image]], list[list[float]]] | None = None,
    signatures: dict | None = None,
    tolerance: float | None = None,
) -> tuple[list[list[int]], dict[int, str]]:
    """Run one grouping method. Returns (components, fingerprints by image index)."""
    if method == "coarse":
        return coarse_components(images), {}
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}.")

    if signatures is None:
        signatures = compute_signatures(root, images, method, embedder, tolerance)
    components = components_at_threshold(method, signatures, threshold, images)
    return components, signatures if method == "phash" else {}


def build_rows(
    images: Sequence[ImageRecord],
    components: Sequence[Sequence[int]],
    fingerprints: dict[int, str],
    method: str,
    threshold: float | None,
    tolerance: float | None = None,
) -> list[dict]:
    """One row per image, carrying its group and everything needed to judge that group later."""
    group_ids = assign_group_ids(components, images, method)
    sizes = {index: len(members) for members in components for index in members}
    return [
        {
            "image_name": image.image_name,
            "dataset_path": image.dataset_path,
            "board": image.board,
            "package": image.package,
            "cell": image.cell,
            "viewpoint": image.viewpoint,
            "component_hint": image.component_hint,
            "method": method,
            "threshold": "" if threshold is None else threshold,
            "crop_tolerance": "" if tolerance is None else tolerance,
            "fingerprint": fingerprints.get(index, ""),
            "group_id": group_ids[index],
            "group_size": sizes[index],
        }
        for index, image in enumerate(images)
    ]


def resolve_threshold(method: str, threshold: float | None) -> float | None:
    """Fill in the method's default, and round phash's to whole bits."""
    threshold = DEFAULT_THRESHOLDS[method] if threshold is None else threshold
    if method == "phash" and threshold is not None:
        threshold = int(threshold)  # a Hamming distance is a whole number of bits
    return threshold


def load_images(root: Path, taxonomy_path: Path = ingest_mod.DEFAULT_TAXONOMY) -> list[ImageRecord]:
    """Audit `root` and return its joint-task images, in a stable order."""
    taxonomy = ingest_mod.load_taxonomy(taxonomy_path)
    return collect_images(audit_mod.audit(root), taxonomy)


def write_rows_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_groups_csv(
    root: Path,
    out: Path,
    method: str,
    threshold: float | None = None,
    taxonomy_path: Path = ingest_mod.DEFAULT_TAXONOMY,
    embedder: Callable[[list[Image.Image]], list[list[float]]] | None = None,
    tolerance: float | None = None,
) -> list[dict]:
    """Group the joint-task images under `root` and write one row per image to `out`."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}.")

    threshold = resolve_threshold(method, threshold)
    if method == "coarse":
        tolerance = None  # coarse never opens an image, so no crop was applied
    images = load_images(root, taxonomy_path)
    components, fingerprints = group_images(
        root, images, method, threshold, embedder, tolerance=tolerance
    )
    rows = build_rows(images, components, fingerprints, method, threshold, tolerance)
    write_rows_csv(rows, out)
    return rows
