"""Capture the machine profile that latency numbers are only meaningful against.

Stage 6 reports milliseconds per crop. A millisecond figure without the CPU model, the physical
core count, the thread setting, and whether the laptop was plugged in is not a measurement — it is
a number someone can neither reproduce nor argue with. This module records that profile once, and
the latency harness reuses it.

Every lookup degrades to `None` rather than raising, so the same command works on the laptop, in
CI, and inside a Kaggle session where a different set of packages exists.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil

SCHEMA_VERSION = 1

# Latency changes a lot with these, and they are easy to set once in a shell and forget.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
)

TRACKED_PACKAGES = ("torch", "timm", "numpy", "psutil", "wandb")

POWER_MODES = ("plugged-in", "battery", "unknown")


def package_version(name: str) -> str | None:
    """Installed version of `name`, or None if it is not installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cpu_model() -> str:
    """The most readable CPU name available.

    `platform.processor()` returns something useless on most Linux builds and merely terse on
    Windows, so py-cpuinfo is tried first. It shells out, which is why this is not called twice.
    """
    try:
        from cpuinfo import get_cpu_info

        brand = get_cpu_info().get("brand_raw")
        if brand:
            return str(brand).strip()
    except Exception:  # noqa: BLE001 - a hardware probe must never break the command
        pass

    for fallback in (platform.processor(), platform.machine()):
        if fallback:
            return str(fallback).strip()
    return "unknown"


def torch_threads() -> dict[str, Any]:
    """Torch's current thread settings, or nulls when torch is not installed yet."""
    try:
        import torch
    except ImportError:
        return {"num_threads": None, "num_interop_threads": None}
    return {
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def capture(power_mode: str = "unknown") -> dict[str, Any]:
    """Build the full environment profile."""
    memory = psutil.virtual_memory()
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # Not auto-detected: psutil cannot tell a plugged-in laptop from a desktop, and a wrong
        # value here would silently invalidate every latency number measured under it.
        "power_mode": power_mode,
        "cpu": {
            "model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "architecture": platform.machine(),
        },
        "memory": {
            "total_bytes": memory.total,
            "total_gb": round(memory.total / 1024**3, 2),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {name: package_version(name) for name in TRACKED_PACKAGES},
        "torch_threads": torch_threads(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }


def write_profile(out: Path, power_mode: str = "unknown") -> dict[str, Any]:
    """Capture the profile and write it to `out` as JSON."""
    profile = capture(power_mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def summarize(profile: dict[str, Any]) -> list[str]:
    """Short human-readable lines for the terminal."""
    cpu = profile["cpu"]
    packages = profile["packages"]
    installed = ", ".join(f"{k} {v}" for k, v in packages.items() if v) or "none"
    missing = ", ".join(k for k, v in packages.items() if v is None)

    lines = [
        f"CPU:     {cpu['model']}",
        f"Cores:   {cpu['physical_cores']} physical / {cpu['logical_cores']} logical",
        f"RAM:     {profile['memory']['total_gb']} GB",
        f"OS:      {profile['os']['system']} {profile['os']['release']}",
        f"Python:  {profile['python']['version']}",
        f"Packages: {installed}",
    ]
    if missing:
        lines.append(f"Not installed: {missing}")
    threads = profile["torch_threads"]["num_threads"]
    lines.append(
        f"Torch threads: {threads if threads is not None else 'n/a (torch not installed)'}"
    )
    lines.append(f"Power mode: {profile['power_mode']}")
    return lines
