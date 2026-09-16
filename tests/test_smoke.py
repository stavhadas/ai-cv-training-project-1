"""Tests for the W&B smoke command.

These cover the decision logic only — they never call `wandb.init`, so CI stays fast and does not
depend on the network or on a key being present.
"""

import pytest
from typer.testing import CliRunner

from pcbi import smoke
from pcbi.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known state; the developer's real key must not leak in."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.delenv("NETRC", raising=False)


NETRC_BODY = "machine api.wandb.ai\n  login user\n  password {}\n".format("x" * 40)


def test_help_lists_smoke():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "smoke" in result.stdout


def test_api_key_env_var_means_online(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert smoke.resolve_mode() == "online"


@pytest.mark.parametrize("filename", [".netrc", "_netrc"])
def test_finds_key_in_either_netrc_spelling(filename, tmp_path, monkeypatch):
    """`wandb login` writes `_netrc` on Windows and `.netrc` elsewhere; both must be found."""
    (tmp_path / filename).write_text(NETRC_BODY)
    monkeypatch.setattr(smoke.Path, "home", lambda: tmp_path)
    assert smoke.has_credentials() is True


def test_empty_home_has_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.Path, "home", lambda: tmp_path)
    assert smoke.has_credentials() is False


def test_netrc_env_var_overrides_home(tmp_path, monkeypatch):
    path = tmp_path / "custom-netrc"
    path.write_text(NETRC_BODY)
    monkeypatch.setenv("NETRC", str(path))
    monkeypatch.setattr(smoke.Path, "home", lambda: tmp_path / "nonexistent")
    assert smoke.has_credentials() is True


def test_no_credentials_falls_back_to_offline(monkeypatch):
    monkeypatch.setattr(smoke, "has_credentials", lambda: False)
    assert smoke.resolve_mode() == "offline"


def test_explicit_mode_wins(monkeypatch):
    """A Kaggle cell or CI job that sets WANDB_MODE must not be overridden."""
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert smoke.resolve_mode() == "offline"


def test_force_offline_beats_everything(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setenv("WANDB_MODE", "online")
    assert smoke.resolve_mode(force_offline=True) == "offline"


def test_fake_loss_decreases():
    import random

    rng = random.Random(0)
    losses = [smoke.fake_loss(i, 20, rng) for i in range(20)]
    assert losses[0] > losses[-1]
    # Noise is small enough that the curve should not wander upward overall.
    assert losses[0] - losses[-1] > 1.0
