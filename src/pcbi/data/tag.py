"""Stage 1 Step 4b: manual pairing of images that show the same physical component.

`pcbi group` tried to construct this grouping automatically and failed — `reports/grouping_*.md`
record the measurement. phash and embed both put the same component *farther* apart than many pairs
of different components, cropping to the component narrowed the gap without closing it, and every
setting that merged the four `_C##` reference pairs had already merged most of the cell with them.
Nothing beat `coarse`, which is 7 groups: too blunt to split on.

So the grouping gets made by hand instead. A person can see in a second what the fingerprints could
not: that this V2 photo and that V2.1 photo are the same joint under different lighting. This module
holds the state that pairing produces, and `tag_server.py` puts the images in front of them.

The state is a list of **groups of dataset paths**, not pairs of indices:

- *Paths*, because indices shift the moment an image is added to `data/raw/` and a saved index file
  would then silently mean something else. A path that no longer exists is reported, not dropped in
  silence — see `unknown_paths`.
- *Groups*, not pairs, because a component photographed three times is one group of three, and
  forcing that into two overlapping pairs invites exactly the under-merge this exists to prevent.
  Two is simply the common size.

The CSV it writes has the same columns as `groups_coarse.csv` and friends, with `method` set to
`manual` and the `threshold`/`crop_tolerance`/`fingerprint` columns empty — a hand-made grouping has
no parameters. Whatever reads a grouping in Step 5 therefore does not need to know which method
produced it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pcbi.data import group as group_mod

METHOD = "manual"

DEFAULT_PAIRS = Path("data/interim/tag_pairs.json")
DEFAULT_OUT = Path("data/interim/groups_manual.csv")

# How much board to show around the component while pairing. Unlike `pcbi group --crop-tolerance`
# this changes nothing about the output — it is a viewing aid only.
#
# The default is wide, which is the opposite of what the fingerprints wanted, and for a reason worth
# recording. `pcbi group` cropped *in* because the shared board was drowning the signal. A person
# needs the shared board: at this tolerance the neighbouring parts and their silkscreen reference
# designators (`R1`, `R2`, `R4`...) are in frame, and that printed designator is the only component
# identity this dataset carries anywhere. Reading a label off the board is a far surer pairing than
# comparing solder textures — so the human view opens wide, and the tight crops stay in the dropdown
# for the cases where two neighbours share a designator's worth of ambiguity and only the fillet
# shape tells them apart.
DEFAULT_TOLERANCE = 1.0
TOLERANCE_CHOICES = (0.0, 0.1, 0.3, 1.0, None)  # None = the whole frame, uncropped

PAIRS_VERSION = 1


@dataclass(frozen=True)
class Cell:
    """One (board, package) cell, split into the viewpoint columns a person pairs across."""

    name: str
    board: str
    package: str
    columns: list[tuple[str, list[int]]]  # (viewpoint, image indices in capture order)

    @property
    def size(self) -> int:
        return sum(len(indices) for _, indices in self.columns)

    @property
    def balanced(self) -> bool:
        """Whether every viewpoint holds the same number of photos.

        Where it is false the two columns cannot line up position by position, which is what makes
        `guess_by_order` untrustworthy there — see its docstring for what that measured.
        """
        sizes = {len(indices) for _, indices in self.columns}
        return len(sizes) == 1

    @property
    def pairable(self) -> int:
        """How many of this cell's images could possibly find a partner, counted as images.

        `2 * (total - largest column)`: the smaller viewpoint sets the number of pairs, and each
        pair accounts for two images. Everything beyond that in the larger viewpoint is surplus with
        nothing to match. A cell with one viewpoint scores 0 — two of this dataset's seven cells
        have no V2.1 photographs at all, so their 10 images can never pair, and the page says so
        rather than letting someone hunt for twins that were never taken.
        """
        sizes = [len(indices) for _, indices in self.columns]
        return 2 * (sum(sizes) - max(sizes, default=0))


@dataclass(frozen=True)
class Summary:
    """Progress, in the terms the person doing the work cares about."""

    images: int
    groups: int
    tagged: int
    untagged: int
    pairable: int
    unknown: int  # saved paths that no longer match an image under `root`

    @property
    def percent(self) -> float:
        return 100.0 * self.tagged / self.pairable if self.pairable else 0.0


def build_cells(images: Sequence[group_mod.ImageRecord]) -> list[Cell]:
    """The cells to page through, each with its viewpoint columns in capture order.

    Filename order is capture order on this dataset (`show.select_cell` documents why), and keeping
    both columns in it is what makes two columns side by side readable at all: the same physical
    board was photographed in roughly the same sequence under both lightings, so a twin is usually
    near the same height on the other side. That is a navigation aid and an assumption, never a
    recorded pairing — `guess_by_order` is the one place it is acted on, and only on request.
    """
    by_cell: dict[str, dict[str, list[int]]] = {}
    for index, image in enumerate(images):
        by_cell.setdefault(image.cell, {}).setdefault(image.viewpoint, []).append(index)

    cells = []
    for name in sorted(by_cell):
        board, _, package = name.partition("/")
        columns = [
            (viewpoint, sorted(indices, key=lambda i: images[i].dataset_path))
            for viewpoint, indices in sorted(by_cell[name].items())
        ]
        cells.append(Cell(name=name, board=board, package=package, columns=columns))
    return cells


# --- the saved state -----------------------------------------------------------------------


def canonical(groups: Sequence[Sequence[str]]) -> list[list[str]]:
    """Sorted, deduplicated, singletons dropped, overlaps merged — the one shape state is kept in.

    Two invariants, both load-bearing:

    - **An image is in at most one group.** Any two groups sharing a member are the same group, so
      they get merged here rather than at every call site. That is what lets an action like
      `pair_by_hint` just append what it knows and stay correct when some of it was already tagged.
    - **The result is byte-stable.** Sorted inside and out, so the JSON diff of a session shows what
      changed and not what order it was clicked in.

    A group of one is dropped: it carries no information, because every untagged image already
    becomes its own group in the CSV.
    """
    merged: list[set[str]] = []
    for group in groups:
        members = set(group)
        if len(members) < 2:
            continue
        for existing in [existing for existing in merged if existing & members]:
            members |= existing
            merged.remove(existing)
        merged.append(members)
    return sorted(sorted(group) for group in merged)


def load_groups(path: Path) -> list[list[str]]:
    """The saved pairing, or an empty one when nothing has been tagged yet."""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return canonical(payload.get("groups", []))


def save_groups(path: Path, groups: Sequence[Sequence[str]]) -> list[list[str]]:
    """Write the pairing, pretty-printed and sorted so a diff shows what a session changed."""
    groups = canonical(groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PAIRS_VERSION,
        "note": (
            "Manual grouping: each list holds dataset paths of images showing the same physical "
            "component. Made by hand with `pcbi tag` because no automatic method could "
            "(see reports/grouping_*.md)."
        ),
        "groups": [list(group) for group in groups],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return groups


def group_of(groups: Sequence[Sequence[str]], path: str) -> list[str] | None:
    """The group containing `path`, or None when it is untagged."""
    for group in groups:
        if path in group:
            return list(group)
    return None


def link(groups: Sequence[Sequence[str]], a: str, b: str) -> list[list[str]]:
    """Put `a` and `b` in one group, absorbing whatever groups they were already in.

    Merging rather than replacing is what makes a third photograph of the same component a
    two-click operation: pair it with either member and all three end up together.
    """
    if a == b:
        return canonical(groups)
    merged = {a, b}
    rest = []
    for group in groups:
        if a in group or b in group:
            merged.update(group)
        else:
            rest.append(list(group))
    return canonical([*rest, sorted(merged)])


def unlink(groups: Sequence[Sequence[str]], path: str) -> list[list[str]]:
    """Take one image out of its group, leaving the rest of that group intact."""
    return canonical([[member for member in group if member != path] for group in groups])


def pair_by_hint(
    images: Sequence[group_mod.ImageRecord],
    cell: Cell,
    groups: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Link this cell's images whose filenames carry the same `_C##` number.

    Unlike `guess_by_order` this is *evidence*, not a heuristic. `WIN_..._Pro_C12.jpg` exists under
    both V2 and V2.1 in CS6/C0805; whoever named those files was recording which component they
    had photographed, and there is nothing to verify. It covers only the 20 files that carry the
    suffix — four components in one cell — but those four come out certain and cost no attention.

    Scoped to the cell because a bare `C7` means "the seventh thing photographed on this board",
    not a globally unique part: the same number on a different board is a different component.
    """
    by_hint: dict[str, list[str]] = {}
    for _, indices in cell.columns:
        for index in indices:
            image = images[index]
            if image.component_hint:
                by_hint.setdefault(image.component_hint, []).append(image.dataset_path)
    found = [sorted(members) for members in by_hint.values() if len(members) > 1]
    return canonical([*[list(group) for group in groups], *found])


def guess_by_order(
    images: Sequence[group_mod.ImageRecord],
    cell: Cell,
    groups: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Pair this cell's first two columns by position, filling only what is still untagged.

    This is the capture-order assumption from `build_cells` acted on deliberately: the nth V2 photo
    is guessed to show the same component as the nth V2.1 photo.

    **It is measurably wrong on the only cell where the answer is known.** CS6/C0805 has 14 V2
    photographs and 7 V2.1, its V2.1 sweep was shot *before* its V2 sweep, and the four `_C##`
    pairs in it say plainly which images go together — this guess matches none of them. It can only
    be plausible in a `balanced` cell, where the two columns at least have the same length, and
    even there nothing confirms it.

    So it is offered as a way to fill a cell and then walk it correcting, not as an answer. Run
    `pair_by_hint` first: it never has to be corrected.

    Existing groups are never overwritten. Running it twice does nothing the second time.
    """
    if len(cell.columns) < 2:
        return canonical(groups)

    tagged = {member for group in groups for member in group}
    (_, left), (_, right) = cell.columns[0], cell.columns[1]
    result = [list(group) for group in groups]
    # strict=False: a cell with 31 V2 and 32 V2.1 photos has one image with no counterpart, and
    # that surplus is real data, not a mismatch to raise on.
    for left_index, right_index in zip(left, right, strict=False):
        left_path = images[left_index].dataset_path
        right_path = images[right_index].dataset_path
        if left_path in tagged or right_path in tagged:
            continue
        result.append([left_path, right_path])
    return canonical(result)


def clear_cell(
    images: Sequence[group_mod.ImageRecord],
    cell: Cell,
    groups: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Drop every group whose members all sit in this cell, leaving other cells untouched."""
    paths = {images[index].dataset_path for _, indices in cell.columns for index in indices}
    return canonical([group for group in groups if not set(group) <= paths])


# --- turning it into the grouping CSV ------------------------------------------------------


def unknown_paths(
    images: Sequence[group_mod.ImageRecord], groups: Sequence[Sequence[str]]
) -> list[str]:
    """Saved paths with no matching image under `root`, in the order they would be dropped.

    Non-empty means the pairing and the dataset have drifted apart — a renamed folder, a partial
    download, a different `--root`. The work is not lost (it is still in the JSON), but the CSV
    written from it is missing those images, so the count gets printed rather than swallowed.
    """
    known = {image.dataset_path for image in images}
    return sorted({member for group in groups for member in group} - known)


def components(
    images: Sequence[group_mod.ImageRecord], groups: Sequence[Sequence[str]]
) -> list[list[int]]:
    """Tagged groups as image indices, plus one singleton per untagged image.

    Untagged means untagged, not "unknown": an image nobody paired becomes its own group. That is
    the under-merge this module's docstring warns about, which is why the CLI prints how many
    singletons the CSV ends up with instead of leaving it to be discovered downstream.
    """
    index_by_path = {image.dataset_path: index for index, image in enumerate(images)}
    result = []
    seen: set[int] = set()
    for group in groups:
        members = sorted(index_by_path[path] for path in group if path in index_by_path)
        if len(members) < 2:
            continue
        result.append(members)
        seen.update(members)
    result.extend([index] for index in range(len(images)) if index not in seen)
    return result


def build_rows(
    images: Sequence[group_mod.ImageRecord], groups: Sequence[Sequence[str]]
) -> list[dict]:
    """One row per image, in the same shape every other grouping method writes.

    `threshold`, `crop_tolerance` and `fingerprint` stay empty on purpose: a hand-made grouping has
    no parameters to record, and writing the tagging tool's viewing tolerance into that column would
    claim the crop affected the result.
    """
    return group_mod.build_rows(images, components(images, groups), {}, METHOD, None, None)


def summarize(
    images: Sequence[group_mod.ImageRecord],
    cells: Sequence[Cell],
    groups: Sequence[Sequence[str]],
) -> Summary:
    known = {image.dataset_path for image in images}
    tagged = {member for group in groups for member in group} & known
    real_groups = [group for group in groups if len(set(group) & known) > 1]
    return Summary(
        images=len(images),
        groups=len(real_groups),
        tagged=len(tagged),
        untagged=len(images) - len(tagged),
        pairable=sum(cell.pairable for cell in cells),
        unknown=len(unknown_paths(images, groups)),
    )


def write_manual_csv(
    images: Sequence[group_mod.ImageRecord], groups: Sequence[Sequence[str]], out: Path
) -> list[dict]:
    """Write `groups_manual.csv` from the current pairing."""
    rows = build_rows(images, groups)
    group_mod.write_rows_csv(rows, out)
    return rows
