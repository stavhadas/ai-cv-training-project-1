from typer.testing import CliRunner

from pcbi.cli import app

runner = CliRunner()


def test_hello_runs():
    result = runner.invoke(app, ["hello", "--name", "Stav"])
    assert result.exit_code == 0
    assert "hello, Stav" in result.stdout


def test_help_lists_hello():
    """`pcbi --help` must list the placeholder command (Stage 0 Step 1 exit criterion)."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "hello" in result.stdout
