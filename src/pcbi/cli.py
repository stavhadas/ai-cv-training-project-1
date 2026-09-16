"""The `pcbi` command and its subcommands."""

import typer

from pcbi import smoke as smoke_mod

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
