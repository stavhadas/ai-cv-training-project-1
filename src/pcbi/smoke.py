"""A tiny fake W&B run that proves the experiment-tracking plumbing works.

Nothing here trains anything. The point is to fail *now*, on a five-second run, rather than
forty minutes into a real Kaggle session, if the key, the project name, or the network is wrong.
"""

from __future__ import annotations

import math
import os
import random
from netrc import NetrcParseError, netrc
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_PROJECT = "pcb-inspector"
DEFAULT_STEPS = 20


def netrc_candidates() -> list[Path]:
    """Where a stored W&B key might live.

    Windows writes ``~/_netrc`` while POSIX writes ``~/.netrc``, and the stdlib ``netrc`` module
    only ever looks for the dotted spelling — so checking both is not optional here.
    """
    override = os.environ.get("NETRC")
    if override:
        return [Path(override).expanduser()]
    home = Path.home()
    return [home / ".netrc", home / "_netrc"]


def has_credentials() -> bool:
    """True if W&B can authenticate without prompting.

    Two places hold a key: the ``WANDB_API_KEY`` environment variable (how Kaggle Secrets and CI
    supply it) and the netrc file that ``wandb login`` writes. Neither one lives in the repo,
    which is the whole point.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True

    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    for path in netrc_candidates():
        if not path.is_file():
            continue
        try:
            if netrc(str(path)).authenticators(host) is not None:
                return True
        except (OSError, NetrcParseError):
            # Unreadable or malformed — treat as "no key here" and keep looking.
            continue
    return False


def resolve_mode(force_offline: bool = False) -> str:
    """Pick the W&B mode, preferring anything the caller set explicitly.

    An explicit ``WANDB_MODE`` always wins, so a Kaggle cell or CI job can force a mode without
    this function second-guessing it.
    """
    if force_offline:
        return "offline"
    explicit = os.environ.get("WANDB_MODE")
    if explicit:
        return explicit
    return "online" if has_credentials() else "offline"


def fake_loss(step: int, total: int, rng: random.Random) -> float:
    """A decreasing curve with a little noise, so the chart looks like a real one."""
    decay = math.exp(-3.0 * step / max(total - 1, 1))
    return 2.0 * decay + 0.15 + rng.uniform(-0.03, 0.03)


def run_smoke(
    project: str = DEFAULT_PROJECT,
    steps: int = DEFAULT_STEPS,
    force_offline: bool = False,
    seed: int = 0,
) -> str:
    """Log a short fake loss curve to W&B. Returns the run's URL, or its local dir when offline."""
    import wandb  # imported lazily so `pcbi --help` stays fast and works without a key

    mode = resolve_mode(force_offline)
    rng = random.Random(seed)

    run = wandb.init(
        project=project,
        job_type="smoke",
        mode=mode,
        config={"steps": steps, "seed": seed, "purpose": "plumbing check, not a real run"},
    )
    try:
        for step in range(steps):
            run.log({"loss": fake_loss(step, steps, rng)}, step=step)
        return run.url or run.dir
    finally:
        run.finish()
