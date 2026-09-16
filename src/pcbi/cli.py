"""The `pcbi` command and its subcommands."""

import typer

app = typer.Typer(help="PCB solder-joint inspector.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """PCB solder-joint inspector and backbone benchmark."""


@app.command()
def hello(name: str = "world") -> None:
    """Placeholder command; proves the entry point is wired up."""
    typer.echo(f"hello, {name}")
