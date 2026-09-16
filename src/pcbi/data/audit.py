"""Dataset inventory: walk a raw dataset and write down what is actually in it.

This runs before any design decision, because the expensive mistakes in this project are the ones
made from assumptions about the data — above all, letting two photographs of the same physical
component land on opposite sides of a train/test split.

Most of this comes from the filesystem or from LabelMe JSON fields, which keeps it fast. One
section — image brightness — opens image files, because it is the only way to get evidence for a
question (which light setup is brighter) that the filesystem cannot answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Filename token separators. The extension is stripped first, so `R0805_12_L2_45.jpg`
# yields R0805 / 12 / L2 / 45.
TOKEN_SPLIT = re.compile(r"[_\-.]")

# LabelMe embeds the whole source image as base64 in `imageData`. Decoding that for every file is
# slow and pointless here. base64 contains no quotes and no backslash escapes, so matching the
# value with a non-greedy character class is safe in a way that regexing JSON usually is not.
IMAGE_DATA_RE = re.compile(rb'"imageData"\s*:\s*"[^"]*"')

# The folder that names the viewpoint always looks like V1, V2, V2.1, V3 — never split this on
# "." or a name like "V2.1" breaks into "V2" and "1". Path segments are only ever split on "/".
VIEWPOINT_RE = re.compile(r"^V\d+(?:\.\d+)?$")
SETUP_RE = re.compile(r"^Setup\d+$")

# SolDef_AI's four SMD packages, per the paper. A package folder that names anything else (e.g. a
# through-hole part) is out of this project's scope, but is still measured, not discarded, so the
# report can show what was excluded and why.
SMD_PACKAGES = {"C0805", "R0603", "R0805", "R1206"}

# This dataset's two annotated viewpoints map onto two different label vocabularies. A viewpoint
# absent from this table (the axonometric V3 shot, in practice) is never annotated here.
TASK_BY_VIEWPOINT = {"V1": "placement", "V2": "joint", "V2.1": "joint"}

MAX_EXAMPLES = 4
TREE_DEPTH = 5

# Duplicate-name and cell tables list folder groups, not every individual name, because a dataset
# that ships a second annotated copy of part of itself produces hundreds of duplicated names and
# only a couple dozen folder groups.
MAX_TABLE_ROWS = 40

# Hashing every duplicate would mean re-reading a large fraction of the dataset for a fact that a
# sample already settles: filenames sharing a name are either a copy (same hash) or a genuine
# collision (different hash). One representative per *group* of folders that share names is enough
# to tell which — a rare genuine collision must not be masked by hundreds of same-photo copies.
MD5_SAMPLE_SIZE = 30

# JPEG's DCT lets a decoder produce a small image directly, far faster than decoding full
# resolution and downscaling after. Brightness only needs a coarse estimate.
BRIGHTNESS_DRAFT_SIZE = (256, 256)


@dataclass
class Annotation:
    """One annotated image, as described by a LabelMe record."""

    image_name: str
    width: int | None
    height: int | None
    labels: Counter[str]
    source: str  # the annotation file this came from, relative to root

    @property
    def polygon_count(self) -> int:
        return sum(self.labels.values())

    @property
    def size(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"


@dataclass
class PathInfo:
    """Where one image sits in the board / package / viewpoint / lighting hierarchy.

    Built by finding the viewpoint folder (the one unambiguous marker, `V1`/`V2`/`V2.1`/`V3`) and
    reading board and package from its known position relative to it. A path with no such folder —
    `Labeled/`, for instance — yields an all-`None` PathInfo rather than a guess.
    """

    board: str | None
    package: str | None
    viewpoint: str | None
    setup: str | None

    @property
    def viewpoint_folder(self) -> str | None:
        if self.viewpoint is None:
            return None
        return f"{self.viewpoint}/{self.setup}" if self.setup else self.viewpoint

    @property
    def task(self) -> str | None:
        """Which annotated label vocabulary applies here, or None if none does."""
        return TASK_BY_VIEWPOINT.get(self.viewpoint) if self.viewpoint else None

    @property
    def in_scope(self) -> bool | None:
        """Whether the package is one of the paper's four SMD types, or None if unknown."""
        return self.package in SMD_PACKAGES if self.package else None


@dataclass
class AuditResult:
    root: Path
    images: list[Path] = field(default_factory=list)
    extension_counts: Counter[str] = field(default_factory=Counter)
    images_per_folder: Counter[str] = field(default_factory=Counter)
    # Every path each filename occupies. A dataset that ships an annotated copy of part of itself
    # has the same photo in two places, and counting files then overstates how much data exists.
    images_by_name: dict[str, list[Path]] = field(default_factory=dict)
    json_files: list[Path] = field(default_factory=list)
    layouts: Counter[str] = field(default_factory=Counter)
    annotations: list[Annotation] = field(default_factory=list)
    label_counts: Counter[str] = field(default_factory=Counter)
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def distinct_names(self) -> int:
        return len(self.images_by_name)

    @property
    def duplicates(self) -> dict[str, list[Path]]:
        """Filenames that exist in more than one folder."""
        return {name: paths for name, paths in self.images_by_name.items() if len(paths) > 1}

    def rel_folder(self, path: Path) -> str:
        return path.parent.relative_to(self.root).as_posix() or "."


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_json_without_image_data(path: Path) -> object:
    """Parse a LabelMe JSON file with the embedded base64 image stripped out first."""
    raw = path.read_bytes()
    raw = IMAGE_DATA_RE.sub(b'"imageData":null', raw)
    return json.loads(raw)


def labels_of(record: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for shape in record.get("shapes") or []:
        if isinstance(shape, dict):
            counts[str(shape.get("label", "<missing label>"))] += 1
    return counts


def annotation_from_record(record: dict, source: str, fallback_name: str) -> Annotation:
    image_name = record.get("imagePath") or fallback_name
    # LabelMe writes imagePath with the annotator's own path separators; keep the leaf only.
    image_name = str(image_name).replace("\\", "/").rsplit("/", 1)[-1]
    return Annotation(
        image_name=image_name,
        width=record.get("imageWidth"),
        height=record.get("imageHeight"),
        labels=labels_of(record),
        source=source,
    )


def parse_annotation_file(path: Path, source: str) -> tuple[str, list[Annotation]]:
    """Return the detected layout and the records found.

    The dataset's layout is documented only as "a JSON file", so all four plausible shapes are
    handled and the report says which one was actually found.
    """
    data = load_json_without_image_data(path)
    stem = path.stem

    if isinstance(data, dict) and "shapes" in data:
        return "one JSON per image", [annotation_from_record(data, source, stem)]

    if isinstance(data, list):
        records = [
            annotation_from_record(item, source, f"{stem}[{i}]")
            for i, item in enumerate(data)
            if isinstance(item, dict) and "shapes" in item
        ]
        if records:
            return "combined JSON (list of records)", records

    if isinstance(data, dict):
        if "annotations" in data and "images" in data:
            return "COCO-style JSON", parse_coco(data, source)
        records = [
            annotation_from_record(item, source, key)
            for key, item in data.items()
            if isinstance(item, dict) and "shapes" in item
        ]
        if records:
            return "combined JSON (mapping of image name to record)", records

    return "unrecognized JSON", []


def parse_coco(data: dict, source: str) -> list[Annotation]:
    """Handle the COCO export shape, in case the dataset ships one instead of raw LabelMe."""
    categories = {
        c.get("id"): str(c.get("name", "<unnamed>"))
        for c in data.get("categories", [])
        if isinstance(c, dict)
    }
    by_image: defaultdict[object, Counter[str]] = defaultdict(Counter)
    for ann in data.get("annotations", []):
        if isinstance(ann, dict):
            by_image[ann.get("image_id")][categories.get(ann.get("category_id"), "<unknown>")] += 1

    out = []
    for image in data.get("images", []):
        if not isinstance(image, dict):
            continue
        out.append(
            Annotation(
                image_name=str(image.get("file_name", "<unnamed>")),
                width=image.get("width"),
                height=image.get("height"),
                labels=by_image.get(image.get("id"), Counter()),
                source=source,
            )
        )
    return out


def audit(root: Path) -> AuditResult:
    """Walk `root` and collect every fact the report needs."""
    result = AuditResult(root=root)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parent = path.parent.relative_to(root).as_posix() or "."
        if is_image(path):
            result.images.append(path)
            result.extension_counts[path.suffix.lower()] += 1
            result.images_per_folder[rel_parent] += 1
            result.images_by_name.setdefault(path.name, []).append(path)
        elif path.suffix.lower() == ".json":
            result.json_files.append(path)

    for path in result.json_files:
        source = path.relative_to(root).as_posix()
        try:
            layout, records = parse_annotation_file(path, source)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            result.unreadable.append((source, f"{type(exc).__name__}: {exc}"))
            continue
        result.layouts[layout] += 1
        result.annotations.extend(records)
        for record in records:
            result.label_counts.update(record.labels)

    return result


def classify_path(parts: tuple[str, ...]) -> PathInfo:
    """Read board / package / viewpoint / lighting-setup out of a folder path.

    Anchored on the viewpoint folder rather than a fixed depth, because `Labeled/` sits at a
    different depth than `Dataset/...` despite holding the same kind of image.
    """
    for i, part in enumerate(parts):
        if VIEWPOINT_RE.match(part):
            setup = parts[i + 1] if i + 1 < len(parts) and SETUP_RE.match(parts[i + 1]) else None
            package = parts[i - 1] if i >= 1 else None
            board = parts[i - 2] if i >= 2 else None
            return PathInfo(board=board, package=package, viewpoint=part, setup=setup)
    return PathInfo(board=None, package=None, viewpoint=None, setup=None)


def classify(path: Path, root: Path) -> PathInfo:
    return classify_path(path.parent.relative_to(root).parts)


def resolve_annotation_image(result: AuditResult, record: Annotation) -> tuple[Path | None, bool]:
    """The image path an annotation resolves to, and whether that match was ambiguous.

    Preferring a match outside the annotation's own folder is what lets a flat `Labeled/` copy
    resolve to its `Dataset/` original instead of to itself.
    """
    annotation_folder = str(PurePosixPath(record.source).parent)
    matches = result.images_by_name.get(record.image_name, [])
    elsewhere = [p for p in matches if result.rel_folder(p) != annotation_folder]
    candidates = elsewhere or matches
    if not candidates:
        return None, False
    return sorted(candidates)[0], len(candidates) > 1


def annotation_locations(result: AuditResult) -> tuple[Counter[str], list[str], list[str]]:
    """Where each annotated image also lives outside the folder its annotation sits in.

    An annotated copy filed flat has lost the directory structure that described it. Joining back
    on filename restores it — which is only sound where the filename resolves to one place, so the
    names that resolve to several are returned separately rather than silently counted.

    Returns (images per source folder, ambiguous names, names with no image file).
    """
    folders: Counter[str] = Counter()
    ambiguous: list[str] = []
    missing: list[str] = []

    for record in result.annotations:
        path, is_ambiguous = resolve_annotation_image(result, record)
        if path is None:
            missing.append(record.image_name)
            continue
        if is_ambiguous:
            ambiguous.append(record.image_name)
        folders[result.rel_folder(path)] += 1

    return folders, sorted(set(ambiguous)), sorted(set(missing))


def annotations_by_task(result: AuditResult) -> dict[str, list[Annotation]]:
    """Bucket each annotation by the task its source viewpoint belongs to.

    An annotation whose image cannot be resolved, or whose viewpoint this dataset never assigns a
    task to, lands under "unclassified" rather than being silently folded into whichever bucket the
    JSON happened to sit near — the same trap as pooling `good` across both label vocabularies.
    """
    buckets: defaultdict[str, list[Annotation]] = defaultdict(list)
    for record in result.annotations:
        path, _ = resolve_annotation_image(result, record)
        info = classify(path, result.root) if path else PathInfo(None, None, None, None)
        buckets[info.task or "unclassified"].append(record)
    return buckets


def position_rows(per_position: dict[int, Counter[str]]) -> list[dict]:
    """Distinct values, examples, and group sizes at each position of a split-up name.

    The group-size columns are the ones that matter for Step 6: if a position really identifies a
    physical component, every value should be shared by the same small number of images.
    """
    rows = []
    for position in sorted(per_position):
        values = per_position[position]
        group_sizes = list(values.values())
        rows.append(
            {
                "position": position,
                "distinct": len(values),
                "examples": [v for v, _ in values.most_common(MAX_EXAMPLES)],
                "median_group": statistics.median(group_sizes) if group_sizes else 0,
                "min_group": min(group_sizes, default=0),
                "max_group": max(group_sizes, default=0),
            }
        )
    return rows


def position_role(row: dict, total: int) -> str:
    """Whether a position can group images at all, said without interpreting what it means.

    A position with one value everywhere, or a different value in every file, cannot put two
    photographs of the same component in the same bucket no matter what it stands for.
    """
    if row["distinct"] <= 1:
        return "constant"
    if row["distinct"] >= total:
        return "unique per file"
    return "groups files"


def token_table(image_paths: list[Path]) -> tuple[list[dict], Counter[int]]:
    """Group sizes at each filename token position."""
    per_position: defaultdict[int, Counter[str]] = defaultdict(Counter)
    token_counts: Counter[int] = Counter()

    for path in image_paths:
        tokens = [t for t in TOKEN_SPLIT.split(path.stem) if t]
        token_counts[len(tokens)] += 1
        for i, token in enumerate(tokens):
            per_position[i][token] += 1

    return position_rows(per_position), token_counts


def token_count_outliers(image_paths: list[Path]) -> list[Path]:
    """Images whose filename splits into a different number of tokens than most filenames do.

    On SolDef_AI these are the `_C##` component-suffix files — the one place the authors' own
    component numbering survives into a filename.
    """

    def count(path: Path) -> int:
        return len([t for t in TOKEN_SPLIT.split(path.stem) if t])

    counts = Counter(count(p) for p in image_paths)
    if not counts:
        return []
    modal = counts.most_common(1)[0][0]
    return sorted((p for p in image_paths if count(p) != modal), key=lambda p: p.name)


def segment_table(image_paths: list[Path], root: Path) -> tuple[list[dict], Counter[int]]:
    """The same table, but over folder names on the path from `root` down to each image.

    Filenames are not the only place a dataset stores metadata, and often not the place it stores
    the most: a camera that names files by timestamp pushes every fact about the subject into the
    directory the photographer filed it under.
    """
    per_depth: defaultdict[int, Counter[str]] = defaultdict(Counter)
    depth_counts: Counter[int] = Counter()

    for path in image_paths:
        parts = path.parent.relative_to(root).parts
        depth_counts[len(parts)] += 1
        for i, part in enumerate(parts):
            per_depth[i][part] += 1

    return position_rows(per_depth), depth_counts


def deduplicated_images(result: AuditResult) -> list[Path]:
    """One path per distinct filename, keeping the most deeply nested copy.

    Where a dataset ships a second, flatter copy of part of itself, the deep copy is the one that
    still carries folder metadata, and counting both would inflate every group size in the tables
    below — the exact double-count these tables exist to avoid. This is a heuristic, not a fact;
    the duplicate-filename table shows what it set aside.
    """
    return [
        max(paths, key=lambda path: (len(path.parts), str(path)))
        for _, paths in sorted(result.images_by_name.items())
    ]


def mirrored_folders(result: AuditResult) -> list[dict]:
    """Which sets of folders hold copies of the same filenames, and how many they share."""
    groups: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for name, paths in result.duplicates.items():
        folders = tuple(sorted({result.rel_folder(path) for path in paths}))
        groups[folders].append(name)

    rows = [
        {"folders": folders, "shared": len(names), "examples": sorted(names)[:2]}
        for folders, names in groups.items()
    ]
    rows.sort(key=lambda row: (-row["shared"], row["folders"]))
    return rows


def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - identity check, not security


def sample_duplicate_hashes(result: AuditResult, sample_size: int = MD5_SAMPLE_SIZE) -> list[dict]:
    """MD5-verify one name from every duplicate-folder-group, then spend any leftover budget on
    more names from the largest groups.

    A name shared by two files means either a copy (same bytes) or a genuine collision between two
    different photographs that happen to share a filename — and only reading the bytes tells them
    apart. Sampling one name *per folder-group* rather than picking names overall means a rare but
    real collision (two different `Dataset/` folders, not a `Labeled/` copy) is never left
    unchecked just because hundreds of same-photo copies come first alphabetically.
    """
    duplicates = result.duplicates
    names_by_group: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for name, paths in duplicates.items():
        folders = tuple(sorted({result.rel_folder(p) for p in paths}))
        names_by_group[folders].append(name)

    def group_size(folders: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
        return (-len(names_by_group[folders]), folders)

    ordered_groups = sorted(names_by_group, key=group_size)
    picked = [sorted(names_by_group[folders])[0] for folders in ordered_groups][:sample_size]

    remaining = sample_size - len(picked)
    if remaining > 0:
        already_picked = set(picked)
        leftovers = sorted(name for name in duplicates if name not in already_picked)
        picked.extend(leftovers[:remaining])

    rows = []
    for name in picked:
        paths = duplicates[name]
        try:
            hashes = {hash_file(p) for p in paths}
            identical: bool | None = len(hashes) == 1
        except OSError:
            identical = None
        rows.append(
            {
                "name": name,
                "folders": sorted({result.rel_folder(p) for p in paths}),
                "identical": identical,
            }
        )
    return rows


def cell_table(images: list[Path], root: Path) -> tuple[list[dict], int]:
    """Image counts for every (board, package, viewpoint folder), flagging inconsistent rows.

    An inconsistent ("ragged") cell — one viewpoint short of the others for the same component
    set — is a warning that some shots are simply missing from the download, not that the folder
    structure means something different there.

    Returns (rows, count of images with no recognizable viewpoint folder).
    """
    cells: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    unclassified = 0

    for path in images:
        info = classify(path, root)
        if info.board is None or info.package is None or info.viewpoint_folder is None:
            unclassified += 1
            continue
        cells[(info.board, info.package)][info.viewpoint_folder] += 1

    rows = [
        {
            "board": board,
            "package": package,
            "in_scope": package in SMD_PACKAGES,
            "counts": dict(sorted(counts.items())),
            "ragged": len(set(counts.values())) > 1,
        }
        for (board, package), counts in sorted(cells.items())
    ]
    return rows, unclassified


def joint_task_coverage(result: AuditResult) -> tuple[list[dict], list[dict]]:
    """Where the joint-task (45 degree) annotations sit: by (board, package), and by light
    direction (V2 vs. V2.1).

    This is the evidence for how many boards the joint-quality label set actually covers, and where
    the same physical components appear twice under different lighting — both of them ways a naive
    random split leaks.
    """
    joint = annotations_by_task(result).get("joint", [])
    by_cell: Counter[tuple[str, str]] = Counter()
    by_direction: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for record in joint:
        path, _ = resolve_annotation_image(result, record)
        if path is None:
            continue
        info = classify(path, result.root)
        if info.board is None or info.package is None:
            continue
        by_cell[(info.board, info.package)] += 1
        by_direction[(info.board, info.package)][info.viewpoint] += 1

    cell_rows = [{"board": b, "package": p, "images": n} for (b, p), n in sorted(by_cell.items())]
    direction_rows = [
        {"board": b, "package": p, "counts": dict(sorted(c.items()))}
        for (b, p), c in sorted(by_direction.items())
    ]
    return cell_rows, direction_rows


def brightness_table(images: list[Path], root: Path) -> tuple[list[dict], bool]:
    """Mean grayscale brightness per (board, lighting-setup folder), restricted to the top (V1)
    viewpoint — the only one with more than one lighting condition per board.

    This is direct, objective evidence for the one thing Step 6 cannot get from the filesystem:
    which `Setup` folder is the brighter lighting. Returns (rows, whether Pillow was available).
    """
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return [], False

    samples: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for path in images:
        info = classify(path, root)
        if info.viewpoint != "V1" or info.board is None:
            continue
        setup = info.setup or "<no setup folder>"
        try:
            with Image.open(path) as img:
                img.draft("L", BRIGHTNESS_DRAFT_SIZE)  # fast approximate JPEG decode; no-op else
                brightness = ImageStat.Stat(img.convert("L")).mean[0]
        except Exception:  # noqa: BLE001 - a bad or fake image file must not abort the audit
            continue
        samples[(info.board, setup)].append(brightness)

    rows = [
        {
            "board": board,
            "setup": setup,
            "images": len(values),
            "mean_brightness": statistics.mean(values),
        }
        for (board, setup), values in sorted(samples.items())
    ]
    return rows, True


def tree_lines(root: Path, max_depth: int = TREE_DEPTH) -> list[str]:
    """An indented directory listing, `max_depth` levels below `root`."""
    lines = [f"{root.name}/"]

    def describe(directory: Path) -> str:
        images = json_files = other = 0
        for child in directory.iterdir():
            if not child.is_file():
                continue
            if is_image(child):
                images += 1
            elif child.suffix.lower() == ".json":
                json_files += 1
            else:
                other += 1
        parts = []
        if images:
            parts.append(f"{images} images")
        if json_files:
            parts.append(f"{json_files} json")
        if other:
            parts.append(f"{other} other")
        return f"  — {', '.join(parts)}" if parts else ""

    def walk(directory: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        children = sorted(p for p in directory.iterdir() if p.is_dir())
        for i, child in enumerate(children):
            last = i == len(children) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{child.name}/{describe(child)}")
            walk(child, depth + 1, prefix + ("    " if last else "│   "))

    root_summary = describe(root)
    if root_summary:
        lines[0] += root_summary
    walk(root, 1, "")
    return lines


def _position_table(rows: list[dict], total: int) -> list[str]:
    """Render one `position_rows()` table, shared by the path-segment and token sections."""
    out = ["| Position | Distinct values | Images per value (min/median/max) | Role | Examples |"]
    out.append("| ---: | ---: | :---: | --- | --- |")
    for row in rows:
        examples = ", ".join(f"`{v}`" for v in row["examples"]) or "—"
        out.append(
            f"| {row['position']} | {row['distinct']} | "
            f"{row['min_group']}/{row['median_group']:g}/{row['max_group']} "
            f"| {position_role(row, total)} | {examples} |"
        )
    if not rows:
        out.append("| — | 0 | — | — | — |")
    return out


def render_markdown(result: AuditResult) -> str:
    """Turn an AuditResult into the committed report.

    Sections follow Stage 0 Step 4's numbered list: folder tree, image counts, annotation layout,
    path segments, filename tokens, the annotated-image join, label counts per task, polygons per
    image per task, the (board, package, viewpoint) cell table, joint-task coverage, image
    brightness, and the blank metadata table.
    """
    unique_images = deduplicated_images(result)
    total = len(unique_images)
    duplicates = result.duplicates
    annotated = result.annotations
    tasks = annotations_by_task(result)
    out: list[str] = []
    add = out.append

    add("# Data audit — SolDef_AI")
    add("")
    add(f"Generated by `pcbi audit` on {datetime.now(UTC):%Y-%m-%d %H:%M UTC}.")
    add(f"Root: `{result.root}`")
    add("")
    add(
        f"**{len(result.images)} image files** carrying **{result.distinct_names} distinct "
        f"filenames**, **{len(result.json_files)} JSON files**, "
        f"**{len(annotated)} annotated images**, "
        f"**{sum(result.label_counts.values())} polygons** across "
        f"**{len(result.label_counts)} labels**."
    )
    add("")
    if duplicates:
        add(
            f"> {len(duplicates)} filename(s) appear in more than one folder, so the file count "
            f"above is **not** the number of distinct photographs. See _Image counts_."
        )
        add("")

    # 1. Folder tree ------------------------------------------------------------------------
    add("## 1. Folder tree")
    add("")
    add(f"{TREE_DEPTH} levels deep — deep enough to reach the lighting-setup folders under `V1`.")
    add("")
    add("```")
    out.extend(tree_lines(result.root))
    add("```")
    add("")

    # 2. Image counts -------------------------------------------------------------------------
    add("## 2. Image counts")
    add("")
    add("| Extension | Count |")
    add("| --- | ---: |")
    for ext, count in result.extension_counts.most_common():
        add(f"| `{ext}` | {count} |")
    if not result.extension_counts:
        add("| _none found_ | 0 |")
    add("")
    add(
        f"**{len(result.images)} image files**, **{result.distinct_names} distinct filenames** — "
        f"a filename counted once no matter how many folders hold a copy of it."
    )
    add("")
    if duplicates:
        copies = sum(len(paths) for paths in duplicates.values()) - len(duplicates)
        add(
            f"{len(duplicates)} filename(s) occur in more than one folder, accounting for "
            f"{copies} extra file(s) that are not new data. Grouped by the folders that share them:"
        )
        add("")
        add("| Folders sharing filenames | Shared names | Examples |")
        add("| --- | ---: | --- |")
        mirror_rows = mirrored_folders(result)
        for row in mirror_rows[:MAX_TABLE_ROWS]:
            folders = "<br>".join(f"`{f}`" for f in row["folders"])
            examples = ", ".join(f"`{name}`" for name in row["examples"])
            add(f"| {folders} | {row['shared']} | {examples} |")
        add("")
        if len(mirror_rows) > MAX_TABLE_ROWS:
            add(f"_{len(mirror_rows) - MAX_TABLE_ROWS} further folder group(s) not shown._")
            add("")

        add("**MD5 verification (sampled).** Same name, same bytes means a copy; same name,")
        add("different bytes means two distinct photographs that happen to collide on filename —")
        add("only reading the bytes tells them apart.")
        add("")
        hash_rows = sample_duplicate_hashes(result)
        add("| Name | Folders | Identical bytes |")
        add("| --- | --- | :---: |")
        for row in hash_rows:
            folders = ", ".join(f"`{f}`" for f in row["folders"])
            verdict = (
                "unreadable"
                if row["identical"] is None
                else ("yes" if row["identical"] else "**NO — different photos**")
            )
            add(f"| `{row['name']}` | {folders} | {verdict} |")
        add("")
        group_keys = {row["folders"] for row in mirror_rows}
        sampled_group_keys = {tuple(row["folders"]) for row in hash_rows}
        if len(duplicates) <= len(hash_rows):
            add(f"All {len(hash_rows)} duplicate name(s) were verified.")
        elif group_keys <= sampled_group_keys:
            add(
                f"Sampled {len(hash_rows)} of {len(duplicates)} duplicate name(s), covering "
                f"**every** folder-group above at least once — the pattern each row shows (copy "
                f"or collision) is checked, not assumed, even where only one name was tested."
            )
        else:
            add(
                f"Sampled {len(hash_rows)} of {len(duplicates)} duplicate name(s). "
                f"{len(group_keys - sampled_group_keys)} smaller folder-group(s) above were not "
                f"reached by the sample budget and are unverified."
            )
        add("")
        mismatches = [row for row in hash_rows if row["identical"] is False]  # noqa: E712
        if mismatches:
            add(
                f"> **{len(mismatches)} sampled name(s) are not copies** — same filename, "
                f"different image bytes. Treat these as a genuine collision, not a duplicate."
            )
            add("")
    else:
        add("Every image filename is unique across the tree.")
        add("")

    # 3. Annotation layout --------------------------------------------------------------------
    add("## 3. Annotation layout")
    add("")
    add(f"{len(result.json_files)} JSON files found. Detected layouts:")
    add("")
    add("| Layout | Files |")
    add("| --- | ---: |")
    for layout, count in result.layouts.most_common():
        add(f"| {layout} | {count} |")
    if not result.layouts:
        add("| _none_ | 0 |")
    add("")
    if result.unreadable:
        add(f"{len(result.unreadable)} file(s) could not be parsed:")
        add("")
        for source, reason in result.unreadable:
            add(f"- `{source}` — {reason}")
        add("")

    # 4. Path segments --------------------------------------------------------------------------
    segment_rows, depth_counts = segment_table(unique_images, result.root)
    add("## 4. Path segments")
    add("")
    add("Folder names on the path from the root down to each image, counted by depth. This is")
    add("checked before filenames because in this dataset the metadata lives in the path, not the")
    add("name.")
    add("")
    if duplicates:
        add(
            f"Counted over the {total} distinct filenames rather than all "
            f"{len(result.images)} files, keeping the most deeply nested copy of each."
        )
        add("")
    if depth_counts:
        counts = ", ".join(f"depth {n}: {c} images" for n, c in sorted(depth_counts.items()))
        add(f"Images sit at mixed depths — {counts}.")
        if len(depth_counts) > 1:
            add("")
            add(
                "> Not every image is the same number of folders down, so one depth does not mean "
                "the same thing for every image. Read the folder tree alongside this table."
            )
        add("")
    out.extend(_position_table(segment_rows, total))
    add("")

    # 5. Filename tokens ------------------------------------------------------------------------
    rows, token_counts = token_table(unique_images)
    add("## 5. Filename tokens")
    add("")
    add("Filenames split on `_`, `-`, and `.`, extension dropped. A secondary check: on this")
    add("dataset the path carries the metadata and the filename is a camera timestamp.")
    add("")
    if token_counts:
        counts = ", ".join(f"{n} tokens: {c} files" for n, c in sorted(token_counts.items()))
        add(f"Token counts across filenames — {counts}.")
        if len(token_counts) > 1:
            add("")
            add(
                "> Filenames do not all have the same number of tokens, so a position does not "
                "mean the same thing in every name. Check this before trusting the table below."
            )
        add("")
    out.extend(_position_table(rows, total))
    add("")
    roles = Counter(position_role(row, total) for row in rows)
    inert = roles["constant"] + roles["unique per file"]
    if inert:
        add(
            f"{roles['constant']} position(s) hold the same value in every filename and "
            f"{roles['unique per file']} hold a different value in every filename; "
            f"{inert} of {len(rows)} positions therefore cannot group images at all."
        )
        add("")
    if rows and all(position_role(row, total) != "groups files" for row in rows):
        add(
            "> **No filename position groups images.** Every token is either the same in every "
            "name or different in every name, so filenames alone cannot put two photographs of "
            "one component in the same bucket. Whatever metadata exists is in the path, not the "
            "name."
        )
        add("")
    outliers = token_count_outliers(unique_images)
    if len(token_counts) > 1 and outliers:
        add(f"{len(outliers)} filename(s) have an unusual token count (see above) — these are the")
        add("images to inspect first for hidden metadata such as a component suffix:")
        add("")
        for path in outliers[:MAX_TABLE_ROWS]:
            add(f"- `{result.rel_folder(path)}/{path.name}`")
        if len(outliers) > MAX_TABLE_ROWS:
            add(f"- _{len(outliers) - MAX_TABLE_ROWS} more not shown_")
        add("")

    # 6. Annotated-image join -------------------------------------------------------------------
    add("## 6. Annotated-image join")
    add("")
    add(
        "Each annotated image, traced by filename to its location outside the folder its "
        "annotation sits in. This is what recovers the directory metadata that a flat annotated "
        "copy throws away."
    )
    add("")
    source_folders, ambiguous, missing = annotation_locations(result)
    add("| Source folder | Annotated images |")
    add("| --- | ---: |")
    for folder, count in sorted(source_folders.items(), key=lambda kv: (-kv[1], kv[0])):
        add(f"| `{folder}` | {count} |")
    if not source_folders:
        add("| _none_ | 0 |")
    add("")
    if ambiguous:
        add(
            f"> **{len(ambiguous)} annotated filename(s) match more than one image file**, so the "
            f"folder above is the first match, not a fact. Resolve these before using the folder "
            f"as metadata: "
            + ", ".join(f"`{name}`" for name in ambiguous[:MAX_EXAMPLES])
            + ("…" if len(ambiguous) > MAX_EXAMPLES else "")
        )
        add("")
    if missing:
        add(
            f"> {len(missing)} annotated filename(s) have no matching image file anywhere under "
            f"the root: "
            + ", ".join(f"`{name}`" for name in missing[:MAX_EXAMPLES])
            + ("…" if len(missing) > MAX_EXAMPLES else "")
        )
        add("")
    if not ambiguous and not missing and annotated:
        add(
            f"All {len(annotated)} annotated images matched exactly one file. "
            f"No manual fixup needed."
        )
        add("")

    # 7. Label counts per task ------------------------------------------------------------------
    add("## 7. Label counts per task")
    add("")
    add(
        "The annotations cover two label vocabularies that happen to share the string `good`. "
        "Pooling them would produce a meaningless merged class, so every count here is grouped by "
        "the task its source viewpoint belongs to."
    )
    add("")
    task_titles = {
        "joint": "Joint-quality task (V2 + V2.1, 45 degree view)",
        "placement": "Placement task (V1/Setup*, top view)",
        "unclassified": "Unclassified (viewpoint not resolved)",
    }
    for task in ("joint", "placement", "unclassified"):
        records = tasks.get(task, [])
        if not records and task == "unclassified":
            continue
        add(f"### {task_titles[task]}")
        add("")
        label_counts: Counter[str] = Counter()
        images_with_label: Counter[str] = Counter()
        for record in records:
            label_counts.update(record.labels)
            images_with_label.update(record.labels.keys())
        add(f"{len(records)} images, {sum(label_counts.values())} polygons.")
        add("")
        add("| Label | Polygons | Images containing it |")
        add("| --- | ---: | ---: |")
        for label, count in label_counts.most_common():
            add(f"| `{label}` | {count} | {images_with_label[label]} |")
        if not label_counts:
            add("| _none_ | 0 | 0 |")
        add("")

    # 8. Polygons per image per task ------------------------------------------------------------
    add("## 8. Polygons per image per task")
    add("")
    add("A histogram of how many polygons each annotated image carries, by task. Stage 1's crop")
    add("code derives `joint_side` from the x-order of joints in an image, so it must handle")
    add("whatever counts show up here, not just two.")
    add("")
    for task in ("joint", "placement"):
        records = tasks.get(task, [])
        if not records:
            continue
        histogram = Counter(record.polygon_count for record in records)
        counts = ", ".join(f"{n}: {c} images" for n, c in sorted(histogram.items()))
        add(f"**{task_titles[task]}:** {counts}")
        add("")

    # 9. Cell table -----------------------------------------------------------------------------
    rows9, unclassified_count = cell_table(unique_images, result.root)
    add("## 9. Cell table")
    add("")
    add(
        "Image count for every (board, package, viewpoint folder). A row is **ragged** when its "
        "viewpoint folders don't all hold the same count — evidence of shots missing from the "
        "download, not a difference in meaning."
    )
    add("")
    add("| Board | Package | Scope | Ragged | Counts by viewpoint folder |")
    add("| --- | --- | --- | :---: | --- |")
    for row in rows9:
        scope = "SMD (in scope)" if row["in_scope"] else "other (out of scope)"
        counts = ", ".join(f"{k}: {v}" for k, v in row["counts"].items())
        ragged = "yes" if row["ragged"] else ""
        add(f"| {row['board']} | {row['package']} | {scope} | {ragged} | {counts} |")
    if not rows9:
        add("| _none_ | | | | |")
    add("")
    ragged = [row for row in rows9 if row["ragged"]]
    if ragged:
        cells = ", ".join(f"{row['board']}/{row['package']}" for row in ragged)
        add(f"> {len(ragged)} ragged cell(s): {cells}.")
        add("")
    if unclassified_count:
        add(
            f"{unclassified_count} image(s) have no recognizable viewpoint folder "
            f"(e.g. `Labeled/`) and are excluded from this table."
        )
        add("")

    # 10. Joint-task coverage ---------------------------------------------------------------------
    coverage_cells, coverage_directions = joint_task_coverage(result)
    add("## 10. Joint-task coverage")
    add("")
    add("Where the 45 degree annotations actually are: which (board, package) cells contributed,")
    add("and — where both light directions were annotated for the same cell — whether V2 and V2.1")
    add("cover the same components twice (evidence of leakage risk, not new data).")
    add("")
    add("| Board | Package | Annotated images |")
    add("| --- | --- | ---: |")
    for row in coverage_cells:
        add(f"| {row['board']} | {row['package']} | {row['images']} |")
    if not coverage_cells:
        add("| _none_ | | 0 |")
    add("")
    boards_covered = sorted({row["board"] for row in coverage_cells})
    add(f"Boards covered: {', '.join(boards_covered) if boards_covered else '_none_'}.")
    add("")
    add("**V2 vs. V2.1, per cell:**")
    add("")
    add("| Board | Package | V2 | V2.1 |")
    add("| --- | --- | ---: | ---: |")
    for row in coverage_directions:
        v2 = row["counts"].get("V2", 0)
        v2_1 = row["counts"].get("V2.1", 0)
        add(f"| {row['board']} | {row['package']} | {v2} | {v2_1} |")
    if not coverage_directions:
        add("| _none_ | | 0 | 0 |")
    add("")
    both = [
        row for row in coverage_directions if row["counts"].get("V2") and row["counts"].get("V2.1")
    ]
    if both:
        add(
            f"> {len(both)} cell(s) have annotations under **both** V2 and V2.1 — the same "
            f"components photographed under two light directions, both labeled. A split that "
            f"doesn't group these together leaks."
        )
        add("")

    # 11. Image brightness ------------------------------------------------------------------------
    brightness_rows, pillow_available = brightness_table(unique_images, result.root)
    add("## 11. Image brightness")
    add("")
    add(
        "Mean grayscale brightness per (board, lighting-setup folder) under the top (`V1`) "
        "viewpoint — the only viewpoint with more than one lighting condition per board. This is "
        "direct evidence for which `Setup` is the brighter lighting, which the filesystem alone "
        "cannot answer."
    )
    add("")
    if not pillow_available:
        add("_Pillow is not installed; brightness could not be measured. `uv add pillow`._")
        add("")
    else:
        add("| Board | Setup folder | Images | Mean brightness (0-255) |")
        add("| --- | --- | ---: | ---: |")
        for row in brightness_rows:
            brightness = row["mean_brightness"]
            add(f"| {row['board']} | {row['setup']} | {row['images']} | {brightness:.1f} |")
        if not brightness_rows:
            add("| _none_ | | 0 | — |")
        add("")
        by_board: defaultdict[str, list[dict]] = defaultdict(list)
        for row in brightness_rows:
            by_board[row["board"]].append(row)
        flips = []
        for board, board_rows in sorted(by_board.items()):
            if len(board_rows) < 2:
                continue
            brightest = max(board_rows, key=lambda r: r["mean_brightness"])
            flips.append(f"{board}: `{brightest['setup']}` is brightest")
        if flips:
            add(
                "> " + "; ".join(flips) + ". Confirm against a handful of images before "
                "trusting it."
            )
            add("")

    # 12. Metadata table ----------------------------------------------------------------------
    add("## 12. Metadata table")
    add("")
    add("**To be filled in by hand (Stage 0, Step 6).** For each field, name where it is encoded —")
    add("a path depth from section 4, a filename token position from section 5, or")
    add("`not recoverable` — and say what evidence convinced you.")
    add("")
    add("The paper says: 230 components on 6 PCBs, each photographed 5 times (top view x2")
    add("lightings, 45 degree view x2 lighting directions, axonometric x1) = 1150 images.")
    add("Package types are C0805, R0603, R0805, R1206. Compare that against the counts measured")
    add("above before assuming the download matches the paper. CS7 (through-hole parts) is out")
    add("of scope for this project.")
    add("")
    add("| Field | Expected distinct values | Where encoded | Status | Evidence |")
    add("| --- | --- | --- | --- | --- |")
    add("| Component ID | 230 | | | |")
    add("| Package | 4 (C0805, R0603, R0805, R1206) | | | |")
    add("| Lighting | 2 per viewpoint | | | |")
    add("| Board | 6 | | | |")
    add("| Viewpoint | 3 (top, 45 degree, axonometric) | | | |")
    add("")
    add(
        "> A wrong component ID is worse than a missing one: it silently mixes one component "
        "across splits, and the leakage test will still pass because it only checks the ID it "
        "was given. Write `not recoverable` rather than guessing."
    )
    add("")

    return "\n".join(out) + "\n"


def write_report(root: Path, out: Path) -> AuditResult:
    """Audit `root` and write the markdown report to `out`."""
    result = audit(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(result), encoding="utf-8")
    return result
