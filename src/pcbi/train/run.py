"""Stage 2: `pcbi train` — wire config, model, data, imbalance correction and the loop together.

Everything here is assembly. The pieces it joins each live in their own module and are tested there;
this one decides only the order they run in, what gets logged, and what is left on disk afterwards.

**Two debugging switches.**

- ``max_steps`` caps the number of optimizer steps across all epochs. The run still validates once
  when the budget runs out, so a three-step check exercises the whole path — data, loss, backward,
  val, the report and the plot — before a long Kaggle run is spent on it.
- ``overfit_batch`` trains repeatedly on the first N training crops and evaluates on *those same*
  crops instead of val. A healthy pipeline drives that loss towards 0 and that accuracy to 1.0; one
  that cannot memorise eight images has a bug, and no amount of tuning will find it.

**The end-of-run printouts are temporary.** A classification report and a confusion matrix, computed
from `validate`'s logits. Stage 3 builds the metrics the product actually needs.

A confusion matrix is a table of counts: row *i*, column *j* is how many crops of true class *i* the
model predicted as class *j*. The diagonal holds the correct predictions; every off-diagonal cell is
a specific kind of mistake (``insufficient`` called ``normal``, say), which is exactly what a single
accuracy number hides. A row normalised by its total is that class's recall; a column normalised by
its total is that prediction's precision.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Subset

from pcbi import smoke
from pcbi.data.datasets import CLASS_ORDER, CropDataset, make_dataloader
from pcbi.data.transforms import build_eval_transform, build_train_transform
from pcbi.train.config import TrainConfig
from pcbi.train.imbalance import build_criterion, build_sampler, class_counts_from_manifest
from pcbi.train.loop import train_one_epoch, validate
from pcbi.train.models import build_model

WANDB_GROUP = "s2_first"
SGD_MOMENTUM = 0.9
SEED = 0
DEFAULT_OUT_DIR = Path("outputs")
DEFAULT_META = Path("data/manifest_meta.json")


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=SGD_MOMENTUM)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def split_hash(meta_path: Path = DEFAULT_META) -> str | None:
    """The frozen split's fingerprint, so a run records which split it was measured on."""
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("split_hash")
    except (OSError, ValueError):
        return None


def format_confusion(matrix: np.ndarray) -> str:
    """The confusion matrix as a labelled text table: rows are true classes, columns predicted."""
    width = max(len(name) for name in CLASS_ORDER) + 2
    header = "true \\ pred".ljust(width) + "".join(name.rjust(width) for name in CLASS_ORDER)
    lines = [header]
    for name, row in zip(CLASS_ORDER, matrix, strict=True):
        lines.append(name.ljust(width) + "".join(str(int(v)).rjust(width) for v in row))
    return "\n".join(lines)


def save_confusion_plot(matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display on Kaggle or CI; must precede pyplot's import
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=list(CLASS_ORDER)).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(logits: np.ndarray, labels: np.ndarray, run_dir: Path, eval_name: str) -> Path:
    """Print the per-class report and the confusion matrix; save them and the plot to `run_dir`."""
    from sklearn.metrics import classification_report, confusion_matrix

    preds = logits.argmax(axis=1)
    class_ids = list(range(len(CLASS_ORDER)))
    text = classification_report(
        labels, preds, labels=class_ids, target_names=list(CLASS_ORDER), digits=3, zero_division=0
    )
    matrix = confusion_matrix(labels, preds, labels=class_ids)
    table = format_confusion(matrix)

    print(f"\nclassification report ({eval_name}, {len(labels)} crops)\n{text}")
    print(f"confusion matrix ({eval_name})\n{table}\n")

    (run_dir / "classification_report.txt").write_text(
        f"{text}\n\nconfusion matrix\n{table}\n", encoding="utf-8"
    )
    plot_path = run_dir / "confusion_matrix.png"
    save_confusion_plot(matrix, plot_path, f"{eval_name}: {len(labels)} crops")
    return plot_path


def run_training(
    cfg: TrainConfig,
    manifest: Path,
    crops_dir: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    use_wandb: bool = True,
    max_steps: int | None = None,
    overfit_batch: int | None = None,
    pretrained: bool = True,
) -> Path:
    """Train `cfg.model` on the train split, validating every epoch. Returns the run folder.

    Args:
        cfg: the loaded experiment config.
        manifest, crops_dir: where `pcbi make-crops` wrote its output (Kaggle mounts it elsewhere).
        out_dir: parent of the run folder.
        use_wandb: ``False`` disables W&B entirely; otherwise online with a key, offline without.
        max_steps: stop after this many optimizer steps in total, then validate once.
        overfit_batch: train and evaluate on the first N training crops only.
        pretrained: load ImageNet weights. Tests pass ``False`` so they need no network.
    """
    import wandb  # lazily, like smoke.py: `pcbi --help` must not pay for it

    torch.manual_seed(SEED)
    device = pick_device()

    model, data_config = build_model(cfg.model, pretrained=pretrained)
    model.to(device)

    train_ds = CropDataset(manifest, crops_dir, "train", build_train_transform(data_config))
    imbalance_cfg = dataclasses.asdict(cfg.imbalance)
    counts = class_counts_from_manifest(train_ds.rows, split="train")
    criterion = build_criterion(imbalance_cfg, counts, device)

    loader_cfg: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if overfit_batch is not None:
        # The same N crops, same order, every epoch — and evaluated on them too, not on val. Built
        # from the eval transform so "train" and "eval" see identical tensors once augmentation
        # lands in the train branch.
        subset = Subset(
            CropDataset(manifest, crops_dir, "train", build_eval_transform(data_config)),
            range(min(overfit_batch, len(train_ds))),
        )
        loader_cfg["batch_size"] = min(len(subset), cfg.batch_size)
        train_loader = make_dataloader(subset, loader_cfg, seed=SEED)
        eval_loader = make_dataloader(subset, loader_cfg, seed=SEED)
        eval_name = f"overfit-{len(subset)}"
    else:
        sampler = build_sampler(imbalance_cfg, train_ds, counts)
        train_loader = make_dataloader(
            train_ds, loader_cfg, seed=SEED, shuffle=sampler is None, sampler=sampler
        )
        val_ds = CropDataset(manifest, crops_dir, "val", build_eval_transform(data_config))
        eval_loader = make_dataloader(val_ds, loader_cfg, seed=SEED)
        eval_name = "val"

    optimizer = build_optimizer(cfg, model)

    run_config = {
        **dataclasses.asdict(cfg),
        "momentum": SGD_MOMENTUM if cfg.optimizer == "sgd" else None,
        "max_steps": max_steps,
        "overfit_batch": overfit_batch,
        "device": device.type,
        "seed": SEED,
        "split_hash": split_hash(),
        "class_counts_train": dict(zip(CLASS_ORDER, counts.tolist(), strict=True)),
    }
    run = wandb.init(
        project=smoke.DEFAULT_PROJECT,
        group=WANDB_GROUP,
        job_type="train",
        mode=smoke.resolve_mode() if use_wandb else "disabled",
        config=run_config,
        tags=["overfit"] if overfit_batch is not None else None,
    )
    try:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = out_dir / (f"{stamp}_{run.id}" if use_wandb and run.id else stamp)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")

        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")

        print(
            f"device {device.type} | train {len(train_loader.dataset)} | "
            f"{eval_name} {len(eval_loader.dataset)} | imbalance {cfg.imbalance.mode}"
        )

        steps_done = 0

        def count_step(step: int, loss: float) -> None:
            nonlocal steps_done
            steps_done += 1

        val = None
        for epoch in range(1, cfg.epochs + 1):
            loader = train_loader
            if max_steps is not None:
                # `train_one_epoch` takes any iterable, so the budget is enforced from out here.
                loader = itertools.islice(train_loader, max_steps - steps_done)
            train = train_one_epoch(model, loader, optimizer, criterion, device, on_step=count_step)

            val = validate(model, eval_loader, criterion, device)
            val_acc = accuracy(val.logits, val.labels)
            print(
                f"epoch {epoch:>3}/{cfg.epochs}  train loss {train.mean_loss:.4f}  "
                f"{eval_name} loss {val.mean_loss:.4f}  {eval_name} acc {val_acc:.3f}  "
                f"({train.seconds:.1f}s)"
            )
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train.mean_loss,
                    "val/loss": val.mean_loss,
                    "val/acc": val_acc,
                }
            )

            if max_steps is not None and steps_done >= max_steps:
                print(f"--max-steps {max_steps} reached; stopping after epoch {epoch}.")
                break

        plot_path = report(val.logits, val.labels, run_dir, eval_name)
        run.log({"val/confusion_matrix": wandb.Image(str(plot_path))})
        return run_dir
    finally:
        run.finish()
