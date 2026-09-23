"""The `pcbi` command and its subcommands."""

from pathlib import Path

import typer

from pcbi import smoke as smoke_mod
from pcbi.bench import env as env_mod
from pcbi.data import audit as audit_mod
from pcbi.data import crops as crops_mod
from pcbi.data import group as group_mod
from pcbi.data import group_report as group_report_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import qa_polygons as qa_polygons_mod
from pcbi.data import show as show_mod
from pcbi.data import split as split_mod
from pcbi.data import tag as tag_mod
from pcbi.data import tag_server as tag_server_mod

app = typer.Typer(help="PCB solder-joint inspector.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """PCB solder-joint inspector and backbone benchmark."""


@app.command()
def hello(name: str = "world") -> None:
    """Placeholder command; proves the entry point is wired up."""
    typer.echo(f"hello, {name}")


@app.command()
def smoke(
    project: str = typer.Option(smoke_mod.DEFAULT_PROJECT, help="W&B project to log into."),
    steps: int = typer.Option(smoke_mod.DEFAULT_STEPS, help="How many fake steps to log."),
    offline: bool = typer.Option(
        False, "--offline", help="Force offline mode even if a key exists."
    ),
    seed: int = typer.Option(0, help="Seed for the fake loss noise."),
) -> None:
    """Log a tiny fake run to W&B to check the tracking plumbing."""
    mode = smoke_mod.resolve_mode(offline)
    typer.echo(f"W&B mode: {mode}")
    if mode == "offline":
        typer.echo("No API key found — logging locally. Upload later with `wandb sync`.")

    where = smoke_mod.run_smoke(project=project, steps=steps, force_offline=offline, seed=seed)
    typer.echo(f"smoke run finished: {where}")


@app.command()
def audit(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    out: Path = typer.Option(Path("reports/data_audit.md"), help="Where to write the report."),
) -> None:
    """Inventory a raw dataset: counts, labels, filename tokens, image sizes."""
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        typer.echo("Download the dataset from Kaggle and unzip it into data/raw/.", err=True)
        raise typer.Exit(code=1)

    result = audit_mod.write_report(root, out)

    typer.echo(f"{len(result.images)} image files, {len(result.json_files)} JSON files")
    duplicates = result.duplicates
    if duplicates:
        typer.echo(
            f"{result.distinct_names} distinct filenames — {len(duplicates)} appear in more "
            f"than one folder, so the file count double-counts copies."
        )
    typer.echo(f"{len(result.annotations)} annotated images")
    # Two label vocabularies share the string "good" — print them apart, not pooled, or the
    # terminal repeats the exact mistake the report exists to catch.
    tasks = audit_mod.annotations_by_task(result)
    for task in ("joint", "placement", "unclassified"):
        records = tasks.get(task, [])
        if not records:
            continue
        labels: dict[str, int] = {}
        for record in records:
            for label, count in record.labels.items():
                labels[label] = labels.get(label, 0) + count
        typer.echo(f"  {task}: {len(records)} images")
        for label, count in sorted(labels.items(), key=lambda kv: -kv[1]):
            typer.echo(f"    {label}: {count}")
    if result.unreadable:
        typer.echo(f"{len(result.unreadable)} JSON file(s) could not be parsed — see the report.")
    typer.echo(f"wrote {out}")


@app.command()
def show(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    board: str = typer.Option(..., help="Board folder, e.g. CS1."),
    package: str = typer.Option(..., help="Package folder, e.g. R0805."),
    viewpoint: str | None = typer.Option(
        None, help="Viewpoint folder: V1, V2, V2.1, or V3. Required unless --pair."
    ),
    setup: str | None = typer.Option(
        None, help="Lighting-setup folder under V1, e.g. Setup1. Only meaningful with V1."
    ),
    pair: bool = typer.Option(
        False,
        "--pair",
        help="Place this cell's V2 and V2.1 images side by side, matched by timestamp order.",
    ),
    limit: int = typer.Option(
        8, help="Max images (or pairs) to include, so the grid stays a viewable size."
    ),
    out: Path | None = typer.Option(
        None, help="Where to save the grid. Defaults to reports/qa/<board>_<package>_<tag>.png."
    ),
) -> None:
    """Save an image grid for one (board, package, viewpoint) cell — the Step 6 visual check.

    A cell count table proves two folders hold the same number of files; it cannot prove they show
    the same physical part under different light, or say which lighting setup is brighter. This
    puts the pixels in front of you instead.
    """
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        raise typer.Exit(code=1)

    result = audit_mod.audit(root)

    if pair:
        left_all = show_mod.select_cell(result, board, package, "V2")
        right_all = show_mod.select_cell(result, board, package, "V2.1")
        if not left_all or not right_all:
            typer.echo(f"No V2/V2.1 pair found for {board}/{package}.", err=True)
            raise typer.Exit(code=1)
        matched_total = min(len(left_all), len(right_all))
        left, right = left_all[:limit], right_all[:limit]
        grid = show_mod.build_pair_grid(left, right, "V2", "V2.1")
        shown = min(len(left), len(right))
        typer.echo(
            f"{shown} pair(s) shown (of {matched_total} matched; "
            f"{len(left_all)} V2, {len(right_all)} V2.1 total)."
        )
        default_name = f"{board}_{package}_V2_vs_V2.1_pair.png"
    else:
        if viewpoint is None:
            typer.echo("--viewpoint is required unless --pair is given.", err=True)
            raise typer.Exit(code=2)
        images_all = show_mod.select_cell(result, board, package, viewpoint, setup)
        if not images_all:
            cell = f"{board}/{package}/{viewpoint}" + (f"/{setup}" if setup else "")
            typer.echo(f"No images found for {cell}.", err=True)
            raise typer.Exit(code=1)
        images = images_all[:limit]
        grid = show_mod.build_grid(images)
        typer.echo(f"{len(images)} image(s) shown (of {len(images_all)} in this cell).")
        tag = f"{viewpoint}_{setup}" if setup else viewpoint
        default_name = f"{board}_{package}_{tag}.png"

    out = out or Path("reports/qa") / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    typer.echo(f"wrote {out}")


@app.command()
def ingest(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    out: Path = typer.Option(
        Path("data/interim/polygons.csv"), help="Where to write the polygon-level CSV."
    ),
    taxonomy: Path = typer.Option(
        ingest_mod.DEFAULT_TAXONOMY, help="Raw-label -> project-class mapping."
    ),
) -> None:
    """Write one CSV row per joint, merging double-defect joints per notes/s1_three_polygons.md."""
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        raise typer.Exit(code=1)
    if not taxonomy.is_file():
        typer.echo(f"No such taxonomy file: {taxonomy}", err=True)
        raise typer.Exit(code=1)

    try:
        raw_rows, merged_rows = ingest_mod.write_polygons_csv(root, out, taxonomy)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    def class_counts(rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["class"]] = counts.get(row["class"], 0) + 1
        return counts

    raw_images = {row["image_name"] for row in raw_rows}
    merged_images = {row["image_name"] for row in merged_rows}
    merged_count = sum(1 for row in merged_rows if row["merged_from"])

    typer.echo(f"before merge: {len(raw_images)} image(s), {len(raw_rows)} polygon row(s)")
    for class_name, count in sorted(class_counts(raw_rows).items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {class_name}: {count}")

    typer.echo(
        f"after merge (notes/s1_three_polygons.md): {len(merged_images)} image(s), "
        f"{len(merged_rows)} joint row(s) — {merged_count} merged"
    )
    for class_name, count in sorted(class_counts(merged_rows).items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {class_name}: {count}")

    typer.echo(f"wrote {out}")


@app.command()
def group(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    method: str = typer.Option("coarse", help=f"How to group: {', '.join(group_mod.METHODS)}."),
    out: Path | None = typer.Option(
        None, help="Where to write the CSV. Defaults to data/interim/groups_<method>.csv."
    ),
    threshold: float | None = typer.Option(
        None,
        help=(
            "Merge threshold; its meaning depends on --method. phash: max Hamming distance "
            "(default 10 of 256 bits). embed: min cosine similarity (default 0.98). "
            "Ignored for coarse."
        ),
    ),
    crop_tolerance: float = typer.Option(
        group_mod.DEFAULT_CROP_TOLERANCE,
        help=(
            "Fingerprint a crop around the component: the box holding the image's two joint "
            "polygons, grown by this fraction of its longer side on every side. 0 = the joints "
            "alone. Ignored for coarse."
        ),
    ),
    full_frame: bool = typer.Option(
        False,
        "--full-frame",
        help="Fingerprint the whole 2560x1440 photo instead, skipping the crop stage.",
    ),
    taxonomy: Path = typer.Option(
        ingest_mod.DEFAULT_TAXONOMY, help="Raw-label -> project-class mapping."
    ),
    report: bool = typer.Option(
        False,
        "--report",
        help=(
            "Also write reports/grouping_<method>.md. For phash/embed without --threshold this "
            "sweeps crop tolerance against threshold and writes nothing else; re-run with a "
            "chosen --threshold for the full report and the groups CSV."
        ),
    ),
    report_out: Path | None = typer.Option(
        None, help="Where to write the report. Defaults to reports/grouping_<method>.md."
    ),
) -> None:
    """Group joint-task images by which physical component they show, for a leak-free split."""
    if method not in group_mod.METHODS:
        typer.echo(f"method must be one of: {', '.join(group_mod.METHODS)}", err=True)
        raise typer.Exit(code=2)
    if crop_tolerance < 0:
        typer.echo("--crop-tolerance must be 0 or greater.", err=True)
        raise typer.Exit(code=2)
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        raise typer.Exit(code=1)
    if not taxonomy.is_file():
        typer.echo(f"No such taxonomy file: {taxonomy}", err=True)
        raise typer.Exit(code=1)
    if method == "coarse" and threshold is not None:
        typer.echo("--threshold is ignored for --method coarse.")
    if method == "coarse" and full_frame:
        typer.echo("--full-frame is ignored for --method coarse; it never opens an image.")

    tolerance = None if full_frame else crop_tolerance
    out = out or Path(f"data/interim/groups_{method}.csv")
    report_out = report_out or Path(f"reports/grouping_{method}.md")

    sweeping = report and method != "coarse" and threshold is None
    if sweeping and method == "embed":
        typer.echo(
            f"Embedding {len(group_mod.TOLERANCE_SWEEP)} crop tolerances — one CNN pass each, "
            f"so this takes minutes, not seconds."
        )

    try:
        if report:
            rows, metrics = group_report_mod.write_report(
                root, out, report_out, method, threshold, taxonomy, tolerance=tolerance
            )
            if metrics is None:  # nothing chosen: sweep only, nothing to group yet
                typer.echo(f"wrote {report_out} (parameter sweep only — no grouping built)")
                typer.echo("Pick a crop tolerance and a threshold from the sweep, then re-run:")
                typer.echo(
                    f"  pcbi group --method {method} --report --threshold N --crop-tolerance T"
                )
                return
        else:
            rows = group_mod.write_groups_csv(
                root, out, method, threshold, taxonomy, tolerance=tolerance
            )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    sizes: dict[str, int] = {}
    for row in rows:
        sizes[row["group_id"]] = sizes.get(row["group_id"], 0) + 1
    singletons = sum(1 for count in sizes.values() if count == 1)

    typer.echo(f"{len(rows)} image(s) in {len(sizes)} group(s) by {method}")
    if method != "coarse":
        typer.echo(
            "  crop: full frame" if tolerance is None else f"  crop tolerance: {tolerance:g}"
        )
    typer.echo(f"  largest group: {max(sizes.values(), default=0)} image(s)")
    typer.echo(f"  singletons: {singletons} group(s)")
    typer.echo(f"wrote {out}")
    if report:
        typer.echo(f"wrote {report_out}")


@app.command()
def tag(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    pairs: Path = typer.Option(
        tag_mod.DEFAULT_PAIRS, help="Where the pairing is read from and saved back to."
    ),
    out: Path = typer.Option(tag_mod.DEFAULT_OUT, help="Where to write the manual grouping CSV."),
    crop_tolerance: float = typer.Option(
        tag_mod.DEFAULT_TOLERANCE,
        help=(
            "How much board to show around the component while pairing, as a fraction of the "
            "joint box's longer side. A viewing aid only — it changes nothing in the CSV."
        ),
    ),
    full_frame: bool = typer.Option(
        False, "--full-frame", help="Show the whole photo instead of a crop."
    ),
    taxonomy: Path = typer.Option(
        ingest_mod.DEFAULT_TAXONOMY, help="Raw-label -> project-class mapping."
    ),
    port: int = typer.Option(tag_server_mod.DEFAULT_PORT, help="Port for the local tagging page."),
    export: bool = typer.Option(
        False,
        "--export",
        help="Write the CSV from the saved pairing and exit, without serving the page.",
    ),
) -> None:
    """Pair images of the same physical component by hand, in a local browser page.

    Step 4's fingerprints could not recover this grouping — `reports/grouping_*.md` record the
    measurement. A person can see it at a glance, so this puts the images in front of them and
    writes the same CSV every automatic method writes.
    """
    if crop_tolerance < 0:
        typer.echo("--crop-tolerance must be 0 or greater.", err=True)
        raise typer.Exit(code=2)
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        raise typer.Exit(code=1)
    if not taxonomy.is_file():
        typer.echo(f"No such taxonomy file: {taxonomy}", err=True)
        raise typer.Exit(code=1)

    tolerance = None if full_frame else crop_tolerance
    try:
        state = tag_server_mod.build_state(root, pairs, out, tolerance, taxonomy)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    summary = tag_mod.summarize(state.images, state.cells, state.groups)
    typer.echo(f"{summary.images} joint-task image(s) across {len(state.cells)} cell(s)")
    typer.echo(
        f"  {summary.groups} group(s) tagged, covering {summary.tagged} image(s); "
        f"{summary.untagged} untagged ({summary.percent:.0f}% of pairable)"
    )
    # Not an error: the pairing is still in the JSON. But the CSV written from it is missing those
    # images, so saying so beats letting it be discovered downstream.
    if summary.unknown:
        typer.echo(f"  {summary.unknown} saved path(s) no longer exist — see {pairs}")

    if export:
        rows = tag_mod.write_manual_csv(state.images, state.groups, out)
        sizes: dict[str, int] = {}
        for row in rows:
            sizes[row["group_id"]] = sizes.get(row["group_id"], 0) + 1
        singletons = sum(1 for count in sizes.values() if count == 1)
        typer.echo(f"{len(rows)} image(s) in {len(sizes)} group(s) by {tag_mod.METHOD}")
        typer.echo(f"  singletons: {singletons} group(s)")
        typer.echo(f"wrote {out}")
        return

    server = tag_server_mod.make_server(state, port)
    host, bound = server.server_address[0], server.server_address[1]
    typer.echo(f"\ntagging page: http://{host}:{bound}  (ctrl-c to stop)")
    typer.echo(f"every click saves {pairs} and rewrites {out}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nstopped.")
    finally:
        server.server_close()


@app.command()
def split(
    groups: Path = typer.Option(
        split_mod.DEFAULT_GROUPS, help="Chosen groups CSV, from `pcbi group` or the manual tagger."
    ),
    polygons: Path = typer.Option(
        split_mod.DEFAULT_POLYGONS,
        help="Joint-level CSV from `pcbi ingest`; supplies the class labels.",
    ),
    out: Path = typer.Option(split_mod.DEFAULT_OUT, help="Where to write the frozen split."),
    meta_out: Path = typer.Option(
        split_mod.DEFAULT_META, help="Where to write the split's hash and provenance."
    ),
    audit_out: Path = typer.Option(
        split_mod.DEFAULT_AUDIT,
        help="Data-audit report to append the count table to; the section is replaced on re-run.",
    ),
    ratios: tuple[float, float, float] = typer.Option(
        split_mod.DEFAULT_RATIOS,
        help="train val test fractions of the crops, as three numbers summing to 1.",
    ),
    candidates: int = typer.Option(
        split_mod.DEFAULT_CANDIDATES, help="How many candidate splits to generate before choosing."
    ),
    split_seed: int = typer.Option(0, help="Base seed; candidate i uses split_seed + i."),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip the data-audit section; write only the split and metadata."
    ),
) -> None:
    """Freeze a grouped, stratified train/val/test split and fingerprint it.

    Every crop of a group lands in one split, so two photographs of the same physical joint can
    never straddle the train/test line. Candidates are judged on label counts alone — never on how
    a model scores — because a test set chosen to flatter a model is not a test set.
    """
    if candidates < 1:
        typer.echo("--candidates must be at least 1.", err=True)
        raise typer.Exit(code=2)
    try:
        split_mod.fold_counts(ratios)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if not groups.is_file():
        typer.echo(f"No such groups file: {groups}", err=True)
        typer.echo("Run `pcbi group` first, or point --groups at an existing grouping.", err=True)
        raise typer.Exit(code=1)
    if not polygons.is_file():
        typer.echo(f"No such polygons file: {polygons}", err=True)
        typer.echo("Run `pcbi ingest` first.", err=True)
        raise typer.Exit(code=1)

    try:
        crops, choice, hash_hex, meta = split_mod.write_splits_csv(
            groups, polygons, out, ratios, candidates, split_seed, meta_out
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    total_groups = len({crop.group_id for crop in crops})
    typer.echo(f"{total_groups} group(s), {len(crops)} crop(s) from {groups}")
    rejected = choice.considered - choice.accepted
    tally = ", ".join(f"{name} {count}" for name, count in sorted(choice.rejections.items()))
    typer.echo(
        f"  candidate {choice.index + 1} of {choice.considered} (seed {choice.seed}); "
        f"{choice.accepted} passed the gates, {rejected} rejected"
        + (f" — {tally}" if tally else "")
    )

    headers, rows = split_mod.count_table(crops, choice.assignment)
    table = [[str(cell) for cell in row] for row in [headers, *rows]]
    widths = [max(len(row[index]) for row in table) for index in range(len(headers))]
    typer.echo("")
    for row in table:
        typer.echo("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)))
    typer.echo("")

    # What the gates passed by, not just that they passed: "all clear" hides whether the scarcest
    # class made it by one crop or by thirty.
    tightest = meta["gates"]["margins"][0]
    typer.echo(
        f"  tightest gate margin: {tightest['what']} at {tightest['count']} "
        f"(floor {tightest['floor']}, {tightest['slack']:+d})"
    )
    typer.echo(f"  split hash: {hash_hex}")
    typer.echo(f"wrote {out}")
    typer.echo(f"wrote {meta_out}")

    if no_report:
        return
    command = (
        f"pcbi split --groups {groups.as_posix()} --ratios {split_mod.format_ratios(ratios)} "
        f"--candidates {candidates} --split-seed {split_seed}"
    )
    section = split_mod.render_audit_section(crops, choice, meta, command)
    try:
        replaced = split_mod.update_audit_report(audit_out, section)
    except ValueError as exc:
        # The split itself is written and valid; only the write-up is missing. Say so and stop
        # short of failing a command whose real artifact already landed.
        typer.echo(f"  {exc}", err=True)
        return
    typer.echo(f"{'replaced' if replaced else 'appended'} the split section in {audit_out}")


@app.command(name="make-crops")
def make_crops(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    polygons: Path = typer.Option(
        crops_mod.DEFAULT_POLYGONS, help="Joint-level CSV from `pcbi ingest`."
    ),
    splits: Path = typer.Option(
        crops_mod.DEFAULT_SPLITS,
        help="Frozen split from `pcbi split`; supplies group_id and split.",
    ),
    out_dir: Path = typer.Option(
        crops_mod.DEFAULT_OUT_DIR, help="Where to write the crops and manifest.csv."
    ),
    margin: float = typer.Option(
        crops_mod.DEFAULT_MARGIN,
        help=(
            "Context kept around the joint box, as a fraction of its own width (left and right) "
            "and height (top and bottom). 0 = the joint box alone."
        ),
    ),
    sheet_dir: Path = typer.Option(
        crops_mod.DEFAULT_SHEET_DIR, help="Where to write the per-class contact sheets."
    ),
    seed: int = typer.Option(
        crops_mod.SHEET_SEED, help="Seed for the contact-sheet sample when a class exceeds 64."
    ),
    no_sheets: bool = typer.Option(
        False, "--no-sheets", help="Write the crops and manifest only; skip the contact sheets."
    ),
) -> None:
    """Cut one PNG per solder joint, with a manifest and per-class contact sheets.

    Crops are byte-identical when regenerated at the same margin — a frozen split is only worth
    something if the pixels behind it hold still. The sheets are built from train crops only:
    deciding what a class looks like while looking at val or test is how a split leaks.
    """
    if margin < 0:
        typer.echo("--margin must be 0 or greater.", err=True)
        raise typer.Exit(code=2)
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        raise typer.Exit(code=1)
    if not polygons.is_file():
        typer.echo(f"No such polygons file: {polygons}", err=True)
        typer.echo("Run `pcbi ingest` first.", err=True)
        raise typer.Exit(code=1)
    if not splits.is_file():
        typer.echo(f"No such split file: {splits}", err=True)
        typer.echo("Run `pcbi split` first.", err=True)
        raise typer.Exit(code=1)

    try:
        summary = crops_mod.make_crops(
            root, polygons, splits, out_dir, margin, None if no_sheets else sheet_dir, seed
        )
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    width, height = summary.median_size
    typer.echo(f"{summary.crops} crop(s) from {polygons} at margin {summary.margin:g}")
    typer.echo(f"  median crop: {width}x{height} px")
    typer.echo(
        "  "
        + " · ".join(
            f"{name} {count}"
            for name, count in sorted(summary.classes.items(), key=lambda kv: -kv[1])
        )
    )
    typer.echo(
        "  "
        + " · ".join(
            f"{name} {summary.splits[name]}" for name in split_mod.SPLITS if name in summary.splits
        )
    )
    # The column is still worth writing; it just is not what its name suggests on those rows.
    if summary.merged_count:
        typer.echo(
            f"  note: polygon_area on {summary.merged_count} merged joint(s) is a bounding box, "
            f"not a traced outline (notes/s1_three_polygons.md)"
        )
    typer.echo(f"wrote {out_dir} ({summary.crops} png)")
    typer.echo(f"wrote {out_dir / crops_mod.MANIFEST_NAME}")
    if summary.sheets:
        typer.echo(
            f"wrote {len(summary.sheets)} contact sheet(s) to {sheet_dir} (train crops only)"
        )


@app.command(name="qa-polygons")
def qa_polygons(
    root: Path = typer.Option(Path("data/raw"), help="Folder holding the unzipped dataset."),
    out_dir: Path = typer.Option(
        Path("reports/qa"), help="Where to write the overlay sheets and bbox_stats.csv."
    ),
    taxonomy: Path = typer.Option(
        ingest_mod.DEFAULT_TAXONOMY, help="Raw-label -> project-class mapping."
    ),
    seed: int = typer.Option(
        qa_polygons_mod.SAMPLE_SEED, help="Seed for the random 12-image sample."
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        help=(
            "Render just this one joint-task image at full resolution instead of the "
            "sample/3poly sheets. Accepts the dataset-relative path (as in polygons.csv, e.g. "
            "SolDef_AI/Dataset/CS1/R0805/V2/WIN_...Pro.jpg) or a bare filename."
        ),
    ),
) -> None:
    """Draw polygon/bbox overlays for visual QA and write per-class/package bbox stats."""
    if not root.is_dir():
        typer.echo(f"No such folder: {root}", err=True)
        raise typer.Exit(code=1)
    if not taxonomy.is_file():
        typer.echo(f"No such taxonomy file: {taxonomy}", err=True)
        raise typer.Exit(code=1)

    if image is not None:
        try:
            dataset_path, annotated = qa_polygons_mod.render_single(root, image, taxonomy)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(dataset_path).stem}_annotated.png"
        annotated.save(out_path)
        typer.echo(f"{dataset_path}: wrote {out_path}")
        return

    try:
        summary = qa_polygons_mod.write_qa_report(root, out_dir, taxonomy, seed=seed)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{summary['sample_images']} image(s) in overlay_sample.png")
    typer.echo(
        f"{summary['three_poly_images']} three-polygon image(s) across "
        f"{summary['three_poly_sheets']} sheet(s)"
    )
    typer.echo(f"{summary['bbox_stats_rows']} row(s) in bbox_stats.csv")
    typer.echo(f"wrote to {out_dir}")


@app.command()
def env(
    out: Path = typer.Option(Path("reports/latency/env.json"), help="Where to write the profile."),
    power_mode: str = typer.Option(
        "unknown",
        help="Plug state during benchmarking. Laptops throttle on battery; this is not detectable.",
    ),
) -> None:
    """Record the hardware and software profile that latency numbers are measured against."""
    if power_mode not in env_mod.POWER_MODES:
        typer.echo(f"power-mode must be one of: {', '.join(env_mod.POWER_MODES)}", err=True)
        raise typer.Exit(code=2)

    profile = env_mod.write_profile(out, power_mode)
    for line in env_mod.summarize(profile):
        typer.echo(line)
    if power_mode == "unknown":
        typer.echo(
            "\nPower mode is unknown. Benchmark plugged in, then re-run with "
            "`--power-mode plugged-in`.",
        )
    typer.echo(f"\nwrote {out}")
