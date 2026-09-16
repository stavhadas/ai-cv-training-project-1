"""The `pcbi` command and its subcommands."""

from pathlib import Path

import typer

from pcbi import smoke as smoke_mod
from pcbi.bench import env as env_mod
from pcbi.data import audit as audit_mod
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
