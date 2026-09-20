"""Stage 1 Step 4/5: judge a candidate grouping against the little evidence the dataset offers.

`pcbi group` builds groupings; this turns one into a report you can actually read. Three checks
carry most of the weight, and each is only as strong as its evidence:

- **Cell boundaries.** A physical component never moves between boards or packages, so a group
  spanning two (board, package) cells would be a proven false merge. Since grouping now compares
  images only *within* a cell (`group.compare_within_cells`), this can no longer fail by
  construction — it is reported as a guard that the restriction holds, not as evidence of quality.
- **The `_C##` pairs.** Where the same component number appears under both `V2` and `V2.1`, those
  two photographs are almost certainly one physical part under two light directions — the closest
  thing to ground truth here. There are only four such pairs, all in `CS6/C0805`, so treat a score
  out of four as a smoke test, not a measurement.
- **The V2/V2.1 partner rate.** In cells photographed under both light directions, images should
  end up grouped with a partner from the other folder. A low rate means under-merging, the
  direction that leaks. Note the **ceiling**: only annotated twins can pair up, and the two folders
  hold unequal annotated counts, so a correct fine-grained method tops out well below 100%.

The sweep matters more than any single row: it shows whether a method has *any* setting that
separates same-component pairs from different-component pairs, or whether it jumps straight from
all-singletons to one-group-per-cell with nothing useful in between. It runs over two parameters
now — the crop tolerance as well as the threshold — because how much of the board the crop keeps
changes what a threshold even means.

The cropping-stage section exists for the same reason the polygon overlays do: a crop box is a
handful of numbers until you see it drawn on the photograph. If the crop misses the joints or
swallows a neighbouring component, every metric below it is measuring the wrong pixels.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from pcbi.data import group as group_mod
from pcbi.data import qa_polygons as qa_mod
from pcbi.data import show as show_mod
from pcbi.data.group import ImageRecord

# Wide enough to cross the whole range: 0 merges nothing, 256 would merge everything. Dense between
# 80 and 120 because that is where the behaviour actually changes — on the real download, 80 leaves
# 200 singletons and 120 has collapsed to one group per cell, so a coarser grid here would step
# straight over whatever useful settings exist and make the method look more binary than it is.
PHASH_SWEEP = (0, 20, 40, 60, 80, 85, 90, 95, 100, 105, 110, 120, 140, 160)

# Cosine similarity on ImageNet features is compressed into the top of its range — near-duplicate
# photographs of one board rarely fall below 0.7 — and on the real download everything happens
# between 0.85 and 0.92: one group per cell on one side, 200 singletons on the other. The grid is
# dense across that band so a negative verdict rests on having actually looked there.
EMBED_SWEEP = (0.70, 0.80, 0.84, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.94, 1.00)

SWEEPS: dict[str, tuple[float, ...]] = {"phash": PHASH_SWEEP, "embed": EMBED_SWEEP}

GRID_SAMPLE = 20  # random groups to show
GRID_LARGEST = 10  # plus the biggest ones, where over-merging shows up
GRID_MAX_IMAGES = 8  # per grid; a 63-image group makes an unreadable sheet
GRID_SEED = 0  # fixed, so "random" means the same groups on every run
MAX_LISTED = 20  # cap on individually listed violations

CROP_EXAMPLES = 8  # images per crop-example sheet
CROP_SWEEP_EXAMPLES = 4  # fewer per sheet when there is one sheet per swept tolerance
CROP_SEED = 1  # separate from GRID_SEED so the two samples are independent
CROP_BOX_COLOR = (255, 60, 200)  # deliberately not a class colour from qa_polygons


@dataclass
class Metrics:
    """Everything the checks above reduce to, for one grouping."""

    groups: int
    largest: int
    singletons: int
    cell_violations: int
    reference_merged: int
    reference_total: int
    partnered: int
    partner_total: int
    partner_ceiling: int  # most images that *could* be partnered, given the annotated counts

    @property
    def partner_rate(self) -> float:
        return self.partnered / self.partner_total if self.partner_total else 0.0

    @property
    def ceiling_rate(self) -> float:
        return self.partner_ceiling / self.partner_total if self.partner_total else 0.0


def size_histogram(components: Sequence[Sequence[int]]) -> Counter[int]:
    """How many groups hold 1 image, 2 images, and so on."""
    return Counter(len(members) for members in components)


def cell_violations(
    components: Sequence[Sequence[int]], images: Sequence[ImageRecord]
) -> list[list[int]]:
    """Groups spanning more than one (board, package) cell — proven false merges."""
    return [
        list(members)
        for members in components
        if len({images[index].cell for index in members}) > 1
    ]


def reference_pairs(images: Sequence[ImageRecord]) -> list[tuple[int, int]]:
    """Index pairs that are near-certainly the same physical component.

    Same cell, same `_C##` number, and exactly two images carrying it — which in practice means one
    under `V2` and one under `V2.1`.
    """
    by_key: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        if image.component_hint:
            by_key[(image.cell, image.component_hint)].append(index)
    return sorted((members[0], members[1]) for members in by_key.values() if len(members) == 2)


def cross_lit_cells(images: Sequence[ImageRecord]) -> set[str]:
    """Cells photographed under both light directions — the only ones where a partner can exist."""
    viewpoints: defaultdict[str, set[str]] = defaultdict(set)
    for image in images:
        viewpoints[image.cell].add(image.viewpoint)
    return {cell for cell, seen in viewpoints.items() if {"V2", "V2.1"} <= seen}


def group_of_index(components: Sequence[Sequence[int]]) -> dict[int, int]:
    return {index: position for position, members in enumerate(components) for index in members}


def partner_ceiling(images: Sequence[ImageRecord]) -> tuple[int, list[dict]]:
    """The most images that could possibly be partnered, and the per-cell working.

    Only annotated twins can pair up, and the two folders hold unequal annotated counts, so each
    cell can contribute at most `min(V2, V2.1)` pairs — twice that many images. Even this is
    optimistic: it assumes the annotated subsets cover the *same* components, which nothing in the
    data confirms. So the ceiling is an upper bound on an upper bound.
    """
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for image in images:
        counts[image.cell][image.viewpoint] += 1

    rows = []
    for cell in sorted(cross_lit_cells(images)):
        v2, v21 = counts[cell]["V2"], counts[cell]["V2.1"]
        rows.append({"cell": cell, "v2": v2, "v2_1": v21, "pairs": min(v2, v21)})
    return 2 * sum(row["pairs"] for row in rows), rows


def measure(components: Sequence[Sequence[int]], images: Sequence[ImageRecord]) -> Metrics:
    sizes = [len(members) for members in components]
    position_of = group_of_index(components)

    pairs = reference_pairs(images)
    merged = sum(1 for a, b in pairs if position_of[a] == position_of[b])

    eligible = cross_lit_cells(images)
    partnered = total = 0
    for index, image in enumerate(images):
        if image.cell not in eligible:
            continue
        total += 1
        members = components[position_of[index]]
        if any(images[other].viewpoint != image.viewpoint for other in members):
            partnered += 1

    ceiling, _ = partner_ceiling(images)
    return Metrics(
        groups=len(components),
        largest=max(sizes, default=0),
        singletons=sum(1 for size in sizes if size == 1),
        cell_violations=len(cell_violations(components, images)),
        reference_merged=merged,
        reference_total=len(pairs),
        partnered=partnered,
        partner_total=total,
        partner_ceiling=ceiling,
    )


def sweep(
    method: str,
    signatures: dict,
    images: Sequence[ImageRecord],
    thresholds: Sequence[float] | None = None,
) -> list[tuple[float, Metrics]]:
    """Re-group at each threshold, reusing the signatures rather than re-reading any image."""
    thresholds = SWEEPS.get(method, ()) if thresholds is None else thresholds
    return [
        (
            threshold,
            measure(
                group_mod.components_at_threshold(method, signatures, threshold, images),
                images,
            ),
        )
        for threshold in thresholds
    ]


def sweep_at_tolerance(
    root: Path,
    method: str,
    images: Sequence[ImageRecord],
    tolerance: float | None,
    embedder: Callable[[list[Image.Image]], list[list[float]]] | None = None,
) -> list[tuple[float, Metrics]]:
    """Re-crop at one tolerance, re-fingerprint, then run the whole threshold sweep on that."""
    signatures = group_mod.compute_signatures(root, images, method, embedder, tolerance)
    return sweep(method, signatures, images)


def best_row(sweep_rows: Sequence[tuple[float, Metrics]]) -> tuple[float, Metrics] | None:
    """The most promising threshold in one sweep — a judgement, stated so it can be argued with.

    Merging the `_C##` reference pairs comes first, because that is the only thing here resembling
    ground truth. Ties break toward the *smaller largest group*: two settings that both find the
    known pairs are not equally good if one of them also swallowed half a cell. A method that merges
    everything scores 4/4 too, and this ordering is what keeps that from looking like success.
    """
    if not sweep_rows:
        return None
    return min(sweep_rows, key=lambda row: (-row[1].reference_merged, row[1].largest, row[0]))


def crop_example_grid(
    root: Path,
    images: Sequence[ImageRecord],
    tolerance: float | None,
    count: int = CROP_EXAMPLES,
    seed: int = CROP_SEED,
) -> Image.Image:
    """Per sampled image, two tiles: the frame with its boxes drawn, then the resulting crop.

    The sample is seeded independently of `tolerance`, so every tolerance's sheet shows the *same*
    components and the sheets can be read against each other — which is the whole reason to render
    one per swept value rather than one per report.
    """
    picked = random.Random(seed).sample(list(images), k=min(count, len(images)))
    tiles: list[Image.Image] = []
    for record in picked:
        full = group_mod.load_image(root, record, tolerance=None)
        box = (
            group_mod.crop_box(record.joint_box, tolerance or 0.0, full.size)
            if record.joint_box is not None
            else (0, 0, full.width, full.height)
        )

        # The frame is about to be letterboxed into a ~220px tile, an order-of-magnitude shrink, so
        # widths are scaled the way qa_polygons.render_annotated_image scales its own.
        marked = full.copy()
        scale = max(full.width / show_mod.THUMB_SIZE[0], full.height / show_mod.THUMB_SIZE[1], 1.0)
        draw = ImageDraw.Draw(marked)
        if record.joint_box is not None:
            draw.rectangle(
                [round(v) for v in record.joint_box],
                outline=qa_mod.BBOX_COLOR,
                width=max(1, round(qa_mod.BBOX_WIDTH * scale)),
            )
        if tolerance is not None:
            draw.rectangle(
                list(box), outline=CROP_BOX_COLOR, width=max(1, round(qa_mod.OUTLINE_WIDTH * scale))
            )

        cropped = full.crop(box) if tolerance is not None else full
        name = Path(record.dataset_path).stem[-12:]
        tiles.append(show_mod.tile_from_image(marked, f"{record.cell} {name}"))
        tiles.append(show_mod.tile_from_image(cropped, f"crop {cropped.width}x{cropped.height}px"))
    return show_mod.compose_grid(tiles, columns=4)


def write_crop_examples(
    root: Path,
    images: Sequence[ImageRecord],
    tolerance: float | None,
    out_dir: Path,
    name: str,
    count: int = CROP_EXAMPLES,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{name}.png"
    crop_example_grid(root, images, tolerance, count).save(destination)
    return destination


def write_group_grids(
    root: Path,
    images: Sequence[ImageRecord],
    components: Sequence[Sequence[int]],
    group_ids: dict[int, str],
    out_dir: Path,
    max_images: int = GRID_MAX_IMAGES,
) -> dict[str, Path]:
    """One image grid per group, saved as `<group_id>.png`. Returns group id -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for members in components:
        group_id = group_ids[members[0]]
        shown = list(members)[:max_images]
        paths = [root / images[index].dataset_path for index in shown]
        captions = [
            f"{images[index].viewpoint} {images[index].component_hint} "
            f"{images[index].image_name}".strip()
            for index in shown
        ]
        grid = show_mod.build_grid(paths, captions, columns=4)
        destination = out_dir / f"{group_id}.png"
        grid.save(destination)
        written[group_id] = destination
    return written


def choose_groups(
    components: Sequence[Sequence[int]], sample: int = GRID_SAMPLE, largest: int = GRID_LARGEST
) -> tuple[list[int], list[int]]:
    """(positions of the largest groups, positions of a random sample), indexing into `components`.

    The sample is seeded so the same groups come back on every run, which is what makes two reports
    comparable; the largest groups are where over-merging becomes visible.
    """
    by_size = sorted(range(len(components)), key=lambda p: (-len(components[p]), p))
    top = by_size[:largest]
    rest = [position for position in range(len(components)) if position not in set(top)]
    picked = random.Random(GRID_SEED).sample(rest, k=min(sample, len(rest)))
    return top, sorted(picked)


UNITS = {"phash": "max Hamming distance of 256 bits", "embed": "min cosine similarity"}


def crop_text(method: str, tolerance: float | None) -> str:
    """How the header describes the crop stage."""
    if method == "coarse":
        return "n/a (no image is read)"
    if tolerance is None:
        return "**full frame (no crop)**"
    return f"{tolerance:g} × the joint box's longer side, on each side"


def header_lines(
    method: str,
    threshold: float | None,
    images: Sequence[ImageRecord],
    csv_path: Path | None,
    command: str,
    tolerance: float | None = None,
    sweeping_tolerance: bool = False,
) -> list[str]:
    """Title and provenance — the part that doesn't depend on any grouping."""
    if threshold is None:
        threshold_text = "n/a (folder metadata only)" if method == "coarse" else "**not chosen**"
    else:
        threshold_text = f"{threshold:g}"
    out = [
        f"# Grouping report — `{method}`",
        "",
        f"Generated by `{command}` on {datetime.now(UTC):%Y-%m-%d %H:%M UTC}.",
        "",
        f"- **Threshold:** {threshold_text}"
        + (f" ({UNITS[method]})" if method in UNITS and threshold is not None else ""),
        "- **Crop tolerance:** "
        + ("**not chosen** (swept below)" if sweeping_tolerance else crop_text(method, tolerance)),
    ]
    if csv_path is not None:
        out.append(f"- **Groups CSV:** `{csv_path.as_posix()}`")
    out.append(
        f"- **Images:** {len(images)} joint-task images across "
        f"{len({image.cell for image in images})} cells"
    )
    out.append("")
    return out


def ceiling_lines(images: Sequence[ImageRecord]) -> list[str]:
    """The per-cell partner ceiling table — a fact about annotation counts, not about grouping."""
    ceiling_total, rows = partner_ceiling(images)
    out = [
        "**100% is not the target.** Only annotated twins can pair up, and the two folders hold",
        "unequal annotated counts, so each cell contributes at most `min(V2, V2.1)` pairs:",
        "",
        "| Cell | V2 | V2.1 | Max pairs |",
        "| --- | ---: | ---: | ---: |",
    ]
    out += [f"| {row['cell']} | {row['v2']} | {row['v2_1']} | {row['pairs']} |" for row in rows]
    out += [
        f"| **Total** | | | **{sum(row['pairs'] for row in rows)}** ({ceiling_total} images) |",
        "",
        "And even that is optimistic — it assumes the annotated subsets cover the *same*",
        "components, which nothing in the data confirms. A fine-grained method scoring near this",
        "ceiling is doing well; one scoring 100% has merged more than twins.",
        "",
    ]
    return out


def sweep_lines(
    sweep_rows: Sequence[tuple[float, Metrics]],
    threshold: float | None,
    ceiling_total: int,
) -> list[str]:
    """The sweep table, with guidance on how to read it."""
    out = [
        "The same signatures re-grouped at a range of thresholds, always within cells.",
        "",
        "What a *working* fine-grained method looks like: groups settling near the pair count",
        f"({ceiling_total // 2} twins plus unpaired leftovers), largest group around 2, and 4/4",
        "reference pairs merged. What failure looks like: the table stepping straight from",
        "all-singletons to one-group-per-cell, with the reference pairs only merging once the",
        "groups are already too coarse to be worth more than the `coarse` method.",
        "",
        "| Threshold | Groups | Largest | Singletons | Cell violations | Ref. pairs "
        "| Partner rate |",
        "| ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for value, row in sweep_rows:
        marker = " **←**" if threshold is not None and value == threshold else ""
        out.append(
            f"| {value:g}{marker} | {row.groups} | {row.largest} | {row.singletons} "
            f"| {row.cell_violations} | {row.reference_merged}/{row.reference_total} "
            f"| {row.partner_rate:.0%} |"
        )
    out.append("")
    return out


def crop_lines(sheets: Sequence[tuple[float | None, Path]]) -> list[str]:
    """The cropping-stage section: what the crop is, and the sheets that show it happening."""
    out = [
        "Both fine-grained methods fingerprint a **crop around the component**, not the whole",
        "photograph. `pcbi ingest` marks two joint polygons per image, one per lead; the box",
        "containing both is grown by `tolerance × its longer side` on every side and clamped to",
        "the frame. These are macro shots — the component box runs about 1000–1250px across in a",
        "2560×1440 frame — so the crop is removing roughly half the pixels, all of them board that",
        "every image in the cell shares. Past a tolerance of about 0.3 the box clamps against the",
        'frame edge and the "crop" is the full photograph under another name.',
        "",
        "Each sheet pairs two tiles per image: the frame with the joint box (white) and the crop",
        "box (pink) drawn on it, then the crop that actually gets fingerprinted. **Look at these",
        "before reading any number below** — a crop that misses the joints, or that swallows a",
        "neighbouring component, invalidates everything downstream.",
        "",
    ]
    for tolerance, path in sheets:
        label = "full frame (no crop)" if tolerance is None else f"tolerance {tolerance:g}"
        relative = Path("qa") / path.parent.name / path.name  # same convention as the group grids
        out += [f"**{label}**", "", f"![crop examples, {label}]({relative.as_posix()})", ""]
    return out


def tolerance_summary_lines(
    by_tolerance: Sequence[tuple[float, list[tuple[float, Metrics]]]],
    ceiling_total: int,
) -> list[str]:
    """One row per tolerance: its most promising threshold, and what that setting produced."""
    out = [
        "One row per crop tolerance, showing the most promising threshold found for it. "
        "**Most promising**",
        "means: merges the most `_C##` reference pairs, and among settings tied on that, produces",
        "the smallest largest-group. That second half matters — a setting that merges everything",
        "also scores full marks on the reference pairs, and this ordering is what stops that",
        "looking like a result.",
        "",
        "| Crop tolerance | Best threshold | Groups | Largest | Singletons | Ref. pairs "
        "| Partner rate |",
        "| ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for tolerance, rows in by_tolerance:
        best = best_row(rows)
        if best is None:
            out.append(f"| {tolerance:g} | _no sweep_ | | | | | |")
            continue
        value, row = best
        out.append(
            f"| {tolerance:g} | {value:g} | {row.groups} | {row.largest} | {row.singletons} "
            f"| {row.reference_merged}/{row.reference_total} | {row.partner_rate:.0%} |"
        )
    out += [
        "",
        f"What a working setting looks like: groups settling near {ceiling_total // 2} (the twin",
        "count) with a largest group around 2, reference pairs at full marks, and a partner rate",
        "near its ceiling.",
        "",
    ]
    out += verdict_lines(by_tolerance, ceiling_total)
    return out


def verdict_lines(
    by_tolerance: Sequence[tuple[float, list[tuple[float, Metrics]]]],
    ceiling_total: int,
) -> list[str]:
    """State outright whether any swept setting worked, rather than leaving it to be inferred.

    The failure this catches is specific and easy to miss in a table: a method can score full marks
    on the reference pairs at every tolerance while its largest group holds most of a cell. Those
    pairs merged because *everything* merged, not because the method found them.
    """
    best = [(tolerance, best_row(rows)) for tolerance, rows in by_tolerance]
    scored = [(tolerance, row) for tolerance, row in best if row is not None]
    if not scored:
        return []

    tolerance, (threshold, row) = min(scored, key=lambda item: item[1][1].largest)
    if row.largest <= 4:  # twin-scale: 2 images, allowing a little slack for over-merging
        return [
            f"**A setting that behaves like twin-finding exists:** tolerance {tolerance:g} at "
            f"threshold {threshold:g}",
            f"gives {row.groups} groups with a largest of {row.largest} and "
            f"{row.reference_merged}/{row.reference_total} reference pairs. Check its grids before",
            "adopting it — small groups are necessary, not sufficient.",
            "",
        ]
    return [
        "**No swept setting separated one component from its neighbours.** The tightest grouping "
        "found",
        f"anywhere above is tolerance {tolerance:g} at threshold {threshold:g}, and its largest "
        f"group still holds",
        f"{row.largest} images — against a twin count of {ceiling_total // 2}. Read the reference-"
        "pair column with that",
        "in mind: those pairs merged because most of the cell merged with them, not because the",
        "method recognised them. Nothing here beats `coarse`, which reaches the same place from",
        "folder names alone and is honest about being blunt.",
        "",
    ]


def render_sweep_only(
    method: str,
    images: Sequence[ImageRecord],
    by_tolerance: Sequence[tuple[float, list[tuple[float, Metrics]]]],
    crop_sheets: Sequence[tuple[float | None, Path]],
) -> str:
    """The first-pass report: everything you need to *choose* a (tolerance, threshold) pair, and
    nothing that would pretend either had already been chosen."""
    ceiling_total, _ = partner_ceiling(images)
    pairs = reference_pairs(images)
    out = header_lines(
        method,
        None,
        images,
        None,
        f"pcbi group --method {method} --report",
        sweeping_tolerance=True,
    )
    add = out.append

    add("> **Nothing chosen, so no grouping was built.** This pass exists to let you pick a crop")
    add("> tolerance and a threshold. Read the tables below, then re-run with both:")
    add(">")
    add(f"> `pcbi group --method {method} --report --threshold N --crop-tolerance T`")
    add(">")
    add("> That second pass writes the groups CSV and the full report — size histogram, cell")
    add("> check, reference pairs, partner rate and group grids — all describing that one")
    add("> combination.")
    add("")

    add("## 1. Cropping stage")
    add("")
    out.extend(crop_lines(crop_sheets))

    add("## 2. Crop tolerance × threshold")
    add("")
    out.extend(tolerance_summary_lines(by_tolerance, ceiling_total))

    add("## 3. Full sweep at each tolerance")
    add("")
    for tolerance, sweep_rows in by_tolerance:
        add(f"### Crop tolerance {tolerance:g}")
        add("")
        out.extend(sweep_lines(sweep_rows, None, ceiling_total))

    add("## 4. What the partner rate can reach")
    add("")
    out.extend(ceiling_lines(images))

    add("## 5. `_C##` reference pairs available")
    add("")
    add("The near-ground-truth this dataset offers: a component number appearing under both `V2`")
    add("and `V2.1` is almost certainly one physical part under two light directions. The sweep's")
    add("*Ref. pairs* column counts how many of these each threshold merges.")
    add("")
    add("| Component | Cell | Viewpoints |")
    add("| --- | --- | --- |")
    for a, b in pairs:
        add(
            f"| `{images[a].component_hint}` | {images[a].cell} "
            f"| {images[a].viewpoint} + {images[b].viewpoint} |"
        )
    if not pairs:
        add("| _none found_ | | |")
    add("")
    return "\n".join(out) + "\n"


def render_markdown(
    method: str,
    threshold: float | None,
    images: Sequence[ImageRecord],
    components: Sequence[Sequence[int]],
    group_ids: dict[int, str],
    sweep_rows: Sequence[tuple[float, Metrics]],
    grids: dict[str, Path],
    csv_path: Path,
    report_path: Path,
    tolerance: float | None = None,
    crop_sheet: Path | None = None,
) -> str:
    metrics = measure(components, images)
    out = header_lines(
        method,
        threshold,
        images,
        csv_path,
        f"pcbi group --method {method} --report",
        tolerance=tolerance,
    )
    add = out.append

    add(
        f"**{metrics.groups} groups.** Largest holds {metrics.largest} image(s); "
        f"{metrics.singletons} group(s) hold exactly one."
    )
    add("")

    # 1. Cropping ------------------------------------------------------------------------------
    add("## 1. Cropping stage")
    add("")
    if method == "coarse":
        add("`coarse` reads the grouping from the folder tree and never opens an image, so no crop")
        add("is applied.")
        add("")
    elif crop_sheet is None:
        add("_No crop examples were rendered._")
        add("")
    else:
        out.extend(crop_lines([(tolerance, crop_sheet)]))

    # 2. Group sizes ---------------------------------------------------------------------------
    add("## 2. Group size histogram")
    add("")
    add("A grouping that works should sit between the two failure modes: all-singletons means the")
    add("method found nothing (and leaks), one-giant-group means it merged indiscriminately.")
    add("")
    add("| Images per group | Groups | Images covered |")
    add("| ---: | ---: | ---: |")
    histogram = size_histogram(components)
    for size, count in sorted(histogram.items()):
        add(f"| {size} | {count} | {size * count} |")
    add("")

    # 3. Cell violations -----------------------------------------------------------------------
    add("## 3. Groups crossing cell boundaries")
    add("")
    add("A physical component never moves between boards or packages, so any group spanning two")
    add("(board, package) cells would contain a **proven** false merge. Grouping compares images")
    add("only within a cell, so this cannot fail by construction — it is here as a guard that the")
    add("restriction is holding, not as evidence that the method works.")
    add("")
    violations = cell_violations(components, images)
    if not violations:
        add(f"**0 of {metrics.groups} groups** cross a cell boundary, as expected.")
    else:
        add(f"**{len(violations)} of {metrics.groups} groups** cross a cell boundary.")
        add("")
        add("| Group | Images | Cells spanned |")
        add("| --- | ---: | --- |")
        for members in violations[:MAX_LISTED]:
            cells = ", ".join(sorted({images[index].cell for index in members}))
            add(f"| `{group_ids[members[0]]}` | {len(members)} | {cells} |")
        if len(violations) > MAX_LISTED:
            add("")
            add(f"_{len(violations) - MAX_LISTED} further group(s) not listed._")
    add("")

    # 4. Reference pairs -----------------------------------------------------------------------
    add("## 4. `_C##` reference pairs")
    add("")
    add("The only near-ground-truth in the dataset: a component number appearing under both `V2`")
    add("and `V2.1` is almost certainly one physical part photographed under two light directions.")
    add("All of them live in one cell, so this is a smoke test, not a measurement.")
    add("")
    pairs = reference_pairs(images)
    position_of = group_of_index(components)
    add(f"**{metrics.reference_merged} of {metrics.reference_total} known pairs merged.**")
    add("")
    add("| Component | Cell | Viewpoints | Same group? |")
    add("| --- | --- | --- | :---: |")
    for a, b in pairs:
        together = "yes" if position_of[a] == position_of[b] else "**no**"
        viewpoints = f"{images[a].viewpoint} + {images[b].viewpoint}"
        add(f"| `{images[a].component_hint}` | {images[a].cell} | {viewpoints} | {together} |")
    if not pairs:
        add("| _none found_ | | | |")
    add("")

    # 5. Partner rate --------------------------------------------------------------------------
    add("## 5. V2 / V2.1 partner rate")
    add("")
    add("Among images in cells photographed under **both** light directions, the share whose group")
    add("also contains an image from the other folder. Cells shot under only one direction are")
    add("excluded, since no partner could exist for them.")
    add("")
    add(
        f"**{metrics.partnered} of {metrics.partner_total} images "
        f"({metrics.partner_rate:.0%})** are grouped with a partner from the other folder, "
        f"against a ceiling of **{metrics.partner_ceiling} ({metrics.ceiling_rate:.0%})**."
    )
    add("")
    ceiling_total, _ = partner_ceiling(images)
    out.extend(ceiling_lines(images))
    excluded = sorted({image.cell for image in images} - cross_lit_cells(images))
    if excluded:
        add(f"Excluded (one light direction only): {', '.join(excluded)}.")
        add("")

    # 6. Threshold sweep -----------------------------------------------------------------------
    add("## 6. Threshold sweep")
    add("")
    if not sweep_rows:
        add(f"`{method}` reads the grouping straight from folder metadata — there is no threshold")
        add("to sweep.")
        add("")
    else:
        out.extend(sweep_lines(sweep_rows, threshold, ceiling_total))
        add("The marked row is the threshold every other section of this report describes.")
        add("")

    # 7. Grids ---------------------------------------------------------------------------------
    add("## 7. Group grids")
    add("")
    add(
        f"Up to {GRID_MAX_IMAGES} images per group. These are written under `reports/qa/`, which "
        f"is git-ignored — regenerate them with the command at the top of this file."
    )
    add("")
    top, picked = choose_groups(components)

    def grid_section(title: str, positions: Sequence[int], note: str) -> None:
        add(f"### {title}")
        add("")
        add(note)
        add("")
        if not positions:
            add("_No groups to show._")
            add("")
            return
        for position in positions:
            members = components[position]
            group_id = group_ids[members[0]]
            path = grids.get(group_id)
            cells = ", ".join(sorted({images[index].cell for index in members}))
            shown = min(len(members), GRID_MAX_IMAGES)
            add(
                f"**`{group_id}`** — {len(members)} image(s) in {cells}"
                + (f", showing {shown}" if shown < len(members) else "")
            )
            add("")
            if path is not None:
                relative = Path("qa") / path.parent.name / path.name
                add(f"![{group_id}]({relative.as_posix()})")
                add("")

    grid_section(
        f"The {len(top)} largest groups",
        top,
        "Over-merging shows up here first: if these hold visibly different components, the "
        "threshold is too loose.",
    )
    grid_section(
        f"{len(picked)} random groups",
        picked,
        "A seeded sample of everything else, so two reports of the same dataset show the same "
        "groups.",
    )

    return "\n".join(out) + "\n"


def write_report(
    root: Path,
    csv_out: Path,
    report_out: Path,
    method: str,
    threshold: float | None = None,
    taxonomy_path: Path = group_mod.ingest_mod.DEFAULT_TAXONOMY,
    embedder: Callable[[list[Image.Image]], list[list[float]]] | None = None,
    grid_dir: Path | None = None,
    tolerance: float | None = group_mod.DEFAULT_CROP_TOLERANCE,
    tolerance_sweep: Sequence[float] = group_mod.TOLERANCE_SWEEP,
) -> tuple[list[dict], Metrics | None]:
    """Group the images, write the CSV, then write the markdown report and its grids.

    With no `threshold` for a method that needs one, this writes a **sweep-only** report instead:
    every crop tolerance in `tolerance_sweep`, each with its own threshold sweep and its own
    crop-example sheet, and no CSV or group grids. Every other section would otherwise be
    describing a setting nobody chose — and the defaults happen to land on a degenerate grouping,
    which made those sections actively misleading. Returns `(rows, metrics)`, both empty/None in
    that first pass.
    """
    if method not in group_mod.METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {', '.join(group_mod.METHODS)}."
        )

    images = group_mod.load_images(root, taxonomy_path)
    grid_dir = grid_dir or Path("reports/qa") / f"grouping_{method}"
    if method == "coarse":
        tolerance = None  # coarse never opens an image, so no crop was applied

    if method != "coarse" and threshold is None:
        by_tolerance = [
            (value, sweep_at_tolerance(root, method, images, value, embedder))
            for value in tolerance_sweep
        ]
        crop_sheets = [
            (
                value,
                write_crop_examples(
                    root, images, value, grid_dir, f"crop_t{value:g}", CROP_SWEEP_EXAMPLES
                ),
            )
            for value in tolerance_sweep
        ]
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            render_sweep_only(method, images, by_tolerance, crop_sheets), encoding="utf-8"
        )
        return [], None

    threshold = group_mod.resolve_threshold(method, threshold)
    crop_sheet: Path | None = None
    if method == "coarse":
        components = group_mod.coarse_components(images)
        fingerprints: dict[int, str] = {}
        sweep_rows: list[tuple[float, Metrics]] = []
    else:
        signatures = group_mod.compute_signatures(root, images, method, embedder, tolerance)
        components = group_mod.components_at_threshold(method, signatures, threshold, images)
        fingerprints = signatures if method == "phash" else {}
        sweep_rows = sweep(method, signatures, images)
        crop_sheet = write_crop_examples(root, images, tolerance, grid_dir, "crop_examples")

    rows = group_mod.build_rows(images, components, fingerprints, method, threshold, tolerance)
    group_mod.write_rows_csv(rows, csv_out)

    group_ids = group_mod.assign_group_ids(components, images, method)
    top, picked = choose_groups(components)
    wanted = [components[position] for position in (*top, *picked)]
    grids = write_group_grids(root, images, wanted, group_ids, grid_dir)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        render_markdown(
            method,
            threshold,
            images,
            components,
            group_ids,
            sweep_rows,
            grids,
            csv_out,
            report_out,
            tolerance,
            crop_sheet,
        ),
        encoding="utf-8",
    )
    return rows, measure(components, images)
