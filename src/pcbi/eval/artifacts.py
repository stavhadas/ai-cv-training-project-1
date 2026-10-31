"""Stage 3: the W&B artifact that carries a run's validation predictions.

`pcbi train` uploads ``val_preds.npz`` under a name derived from the run id; `pcbi pull` downloads
it by that same name. Both go through `artifact_name`, so the two cannot drift apart.

Named after the run *id*, not its display name: ids are unique and are what the W&B URL shows,
while display names like ``scary-charm-16`` can be renamed or repeat across projects.
"""

from __future__ import annotations

from pathlib import Path

ARTIFACT_TYPE = "predictions"
PREDS_FILE = "val_preds.npz"


def artifact_name(run_id: str) -> str:
    return f"val_preds-{run_id}"


def pull_predictions(
    run_id: str, out_dir: Path, project: str, entity: str | None = None
) -> Path:
    """Download a run's predictions artifact into ``out_dir/<run_id>/``; return the .npz path.

    Raises:
        ValueError: the artifact does not exist, or holds no ``val_preds.npz``.
    """
    import wandb  # lazily, like smoke.py: `pcbi --help` must not pay for it

    api = wandb.Api()
    entity = entity or api.default_entity
    path = f"{entity}/{project}/{artifact_name(run_id)}:latest"
    try:
        artifact = api.artifact(path, type=ARTIFACT_TYPE)
    except wandb.errors.CommError as exc:
        raise ValueError(f"No predictions artifact {path}: {exc}") from exc

    target = out_dir / run_id
    artifact.download(root=str(target))
    preds = target / PREDS_FILE
    if not preds.is_file():
        raise ValueError(f"{path} downloaded, but it holds no {PREDS_FILE}.")
    return preds
