"""Stage 3: save a validation result to disk and load it back, so evaluation never retrains."""

from pathlib import Path

import numpy as np

from pcbi.train.loop import ValResult


def save_predictions(val_result: ValResult, path: str | Path) -> None:
    """
    Save a validation result as a .npz file.

    Args:
        val_result (ValResult): What `validate` returned: mean_loss, logits [N, 4], labels [N],
            crop_ids (N strings, in the same order as the rows of logits).
        path (str | Path): Where to write it, e.g. run_dir / "val_preds.npz".
    """
    np.savez(
        path,
        mean_loss=np.float64(val_result.mean_loss),
        logits=val_result.logits,
        labels=val_result.labels,
        # dtype=str stores fixed-width unicode, which loads back without pickle. An object array
        # would need allow_pickle=True, and that lets a crafted file run code on load.
        crop_ids=np.array(val_result.crop_ids, dtype=str),
    )


def load_predictions(path: str | Path) -> ValResult:
    """
    Load a validation result written by `save_predictions`.

    Args:
        path (str | Path): The .npz file to read.

    Returns:
        ValResult: The same fields that were saved, with crop_ids as a list of Python str.
    """
    with np.load(path) as data:
        return ValResult(
            mean_loss=float(data["mean_loss"]),
            logits=data["logits"],
            labels=data["labels"],
            # .tolist() turns the array's np.str_ items into plain str.
            crop_ids=data["crop_ids"].tolist(),
        )
