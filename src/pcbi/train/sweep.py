"""Stage 3: `pcbi sweep` — one config, several seeds, several values of one or more keys.

``--set imbalance.mode=none,class_weighted_loss,weighted_sampler`` with ``--seeds 0,1,2`` is nine
runs, one per (seed, value). Each records *which* option it ran in the W&B config under
``sweep`` (``{"imbalance.mode": "none"}``), and that — not the tags, which a person may edit later
in the UI — is what `pcbi summarize` groups by. The value is also added as a tag, for filtering.

Runs go one after another, never in parallel: a Kaggle GPU holds one at a time, and sequential
runs keep each one's log readable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

DEFAULT_GROUP = "s3_imbalance"


@dataclass(frozen=True)
class Job:
    seed: int
    option: dict[str, str]  # dotted key -> value, e.g. {"imbalance.mode": "none"}

    @property
    def overrides(self) -> list[str]:
        return [f"{key}={value}" for key, value in self.option.items()] + [f"seed={self.seed}"]

    @property
    def label(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.option.items())


def parse_seeds(raw: list[str]) -> list[int]:
    """``["0,1,2"]``, ``["0 1 2"]`` and ``["0", "1", "2"]`` all mean seeds 0, 1 and 2."""
    seeds = []
    for chunk in raw:
        for piece in chunk.replace(",", " ").split():
            try:
                seeds.append(int(piece))
            except ValueError as exc:
                raise ValueError(f"--seeds: {piece!r} is not an integer.") from exc
    if not seeds:
        raise ValueError("--seeds: give at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds: {seeds} repeats a seed; each would be the same run twice.")
    return seeds


def parse_set(raw: str) -> tuple[str, list[str]]:
    """``imbalance.mode=none,weighted_sampler`` -> ``("imbalance.mode", ["none", ...])``."""
    key, sep, values = raw.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"--set {raw!r}: expected key=value1,value2,...")
    if key == "seed":
        raise ValueError("--set seed=...: use --seeds for seeds.")
    options = [v.strip() for v in values.split(",") if v.strip()]
    if not options:
        raise ValueError(f"--set {raw!r}: no values after '='.")
    if len(set(options)) != len(options):
        raise ValueError(f"--set {raw!r}: a value is repeated.")
    return key, options


def expand(seeds: list[int], sets: list[str]) -> list[Job]:
    """Every (seed, value) combination; several --set flags multiply. The option varies fastest.

    So every option runs on seed 0 before any runs on seed 1: a sweep stopped early still compares
    all the options, just on fewer seeds.
    """
    parsed = [parse_set(raw) for raw in sets]
    keys = [key for key, _ in parsed]
    if len(set(keys)) != len(keys):
        raise ValueError(f"--set: {keys} names the same key twice.")
    combos = itertools.product(*(values for _, values in parsed)) if parsed else [()]
    jobs = []
    for combo in combos:
        option = dict(zip(keys, combo, strict=True))
        jobs.extend(Job(seed, option) for seed in seeds)
    # Re-order so all options run for seed 0 before any runs for seed 1.
    return sorted(jobs, key=lambda job: seeds.index(job.seed))
