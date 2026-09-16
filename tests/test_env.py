"""Tests for `pcbi env`.

The values are whatever the running machine happens to be, so these assert on structure and on
the graceful-degradation paths — the parts that must hold on the laptop, in CI, and on Kaggle
alike.
"""

import json

from typer.testing import CliRunner

from pcbi.bench import env as env_mod
from pcbi.cli import app

runner = CliRunner()


def test_profile_has_every_field_step_5_asks_for():
    profile = env_mod.capture()
    assert profile["cpu"]["model"]
    assert profile["cpu"]["physical_cores"] is None or profile["cpu"]["physical_cores"] >= 1
    assert profile["cpu"]["logical_cores"] >= 1
    assert profile["memory"]["total_gb"] > 0
    assert profile["os"]["system"]
    assert profile["python"]["version"]
    assert set(env_mod.TRACKED_PACKAGES) <= set(profile["packages"])
    assert set(env_mod.THREAD_ENV_VARS) == set(profile["thread_env"])
    assert "num_threads" in profile["torch_threads"]
    assert profile["power_mode"] == "unknown"


def test_profile_is_json_serializable():
    json.dumps(env_mod.capture())


def test_torch_absence_is_recorded_not_raised(monkeypatch):
    """torch arrives in Stage 2; until then the command must still work."""
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    threads = env_mod.torch_threads()
    assert threads == {"num_threads": None, "num_interop_threads": None}


def test_unknown_package_version_is_none():
    assert env_mod.package_version("a-package-that-does-not-exist") is None


def test_cpu_model_survives_a_broken_cpuinfo(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "cpuinfo":
            raise RuntimeError("probe failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)
    assert env_mod.cpu_model()  # falls back to platform, never empty


def test_thread_env_vars_are_captured(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    profile = env_mod.capture()
    assert profile["thread_env"]["OMP_NUM_THREADS"] == "4"
    assert profile["thread_env"]["MKL_NUM_THREADS"] is None


def test_cli_writes_valid_json(tmp_path):
    out = tmp_path / "latency" / "env.json"
    result = runner.invoke(app, ["env", "--out", str(out), "--power-mode", "plugged-in"])
    assert result.exit_code == 0
    profile = json.loads(out.read_text(encoding="utf-8"))
    assert profile["power_mode"] == "plugged-in"
    assert profile["schema_version"] == env_mod.SCHEMA_VERSION
    assert "CPU:" in result.stdout


def test_cli_warns_when_power_mode_is_unset(tmp_path):
    result = runner.invoke(app, ["env", "--out", str(tmp_path / "env.json")])
    assert result.exit_code == 0
    assert "Power mode is unknown" in result.stdout


def test_cli_rejects_a_bogus_power_mode(tmp_path):
    result = runner.invoke(app, ["env", "--out", str(tmp_path / "env.json"), "--power-mode", "usb"])
    assert result.exit_code == 2
    assert not (tmp_path / "env.json").exists()
