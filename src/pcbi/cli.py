"""The `pcbi` command and its subcommands."""

from pathlib import Path

import typer

from pcbi import smoke as smoke_mod
from pcbi.bench import env as env_mod
from pcbi.data import audit as audit_mod
from pcbi.data import group as group_mod
from pcbi.data import group_report as group_report_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import qa_polygons as qa_polygons_mod
from pcbi.data import show as show_mod

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
