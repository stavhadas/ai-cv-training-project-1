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


def test_help_lists_train():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout


def test_train_refuses_a_non_positive_max_steps(tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text("epochs: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(config), "--max-steps", "0"])
    assert result.exit_code == 2


def test_train_reports_an_unknown_config_key(tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text("learning_rate: 0.1\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(config), "--no-wandb"])
    assert result.exit_code == 1
