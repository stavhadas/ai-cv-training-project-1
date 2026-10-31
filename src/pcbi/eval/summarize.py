"""Stage 3: `pcbi summarize` — one table comparing the options of a sweep, over its seeds.

Every finished run in a W&B group contributes its ``val_preds.npz``; runs are grouped by the
``sweep`` entry `pcbi sweep` wrote into their config. Per run:

- **macro recall**: the mean of the four per-class recalls, so ``insufficient`` counts as much as
  ``excess`` however few crops it has.
- **normal recall**: the fraction of good joints passed. 1 − this is the argmax false-alarm rate.
- **AUROC** of the defect score (1 − P(normal)), defect as the positive class.
- **FAR @ miss ≤ 0.10**: the false-alarm rate at the *largest* threshold whose miss rate is at most
  0.10, the operating-point rule from the worked example, read off `threshold_sweep`.

Mean ± std is over seeds, with the sample standard deviation (n − 1). With three seeds that std is
itself rough; read it as "how wide is the band", not as a precise number.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcbi.data.datasets import CLASS_ORDER
from pcbi.train.loop import ValResult

MAX_MISS = 0.10
METRICS = ("macro_recall", "normal_recall", "auroc", "far_at_miss")
HEADERS = ("macro recall", "normal recall", "AUROC", f"FAR @ miss ≤ {MAX_MISS:.2f}")
NO_OPTION = "(no sweep option)"


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    option: str
    seed: int | None
    macro_recall: float
    normal_recall: float
    auroc: float
    far_at_miss: float  # nan if no threshold reaches the miss-rate limit


def far_at_max_miss(scores, is_defect, max_miss: float = MAX_MISS) -> float:
    """False-alarm rate at the largest threshold whose miss rate is at most `max_miss`.

    The sweep runs largest τ first, and lowering τ only ever lowers the miss rate, so the first
    row that meets the limit is the largest threshold that does.
    """
    from pcbi.eval.product import threshold_sweep

    for _, miss, far in threshold_sweep(scores, is_defect):
        if miss <= max_miss:
            return far
    return math.nan


def run_metrics(result: ValResult, run_id: str, option: str, seed: int | None) -> RunMetrics:
    import torch
    from sklearn.metrics import recall_score, roc_auc_score

    from pcbi.eval.product import defect_score

    labels = result.labels
    preds = result.logits.argmax(axis=1)
    per_class = recall_score(
        labels, preds, labels=list(range(len(CLASS_ORDER))), average=None, zero_division=0
    )
    scores = defect_score(torch.from_numpy(result.logits))
    is_defect = torch.from_numpy(labels != 0)
    return RunMetrics(
        run_id=run_id,
        option=option,
        seed=seed,
        macro_recall=float(per_class.mean()),
        normal_recall=float(per_class[0]),
        auroc=float(roc_auc_score(is_defect.numpy(), scores.numpy())),
        far_at_miss=far_at_max_miss(scores, is_defect),
    )


def mean_std(values: list[float]) -> str:
    """``0.812 ± 0.034``; a single value has no spread to report; nan runs are dropped and counted."""
    finite = [v for v in values if not math.isnan(v)]
    dropped = len(values) - len(finite)
    note = f" ({dropped} n/a)" if dropped else ""
    if not finite:
        return "n/a"
    if len(finite) == 1:
        return f"{finite[0]:.3f} ± —{note}"
    return f"{np.mean(finite):.3f} ± {np.std(finite, ddof=1):.3f}{note}"


def format_table(rows: list[RunMetrics]) -> str:
    """A markdown table, one line per option, options in sorted order."""
    by_option: defaultdict[str, list[RunMetrics]] = defaultdict(list)
    for row in rows:
        by_option[row.option].append(row)

    lines = [
        "| option | seeds | " + " | ".join(HEADERS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in HEADERS) + " |",
    ]
    for option in sorted(by_option):
        group = by_option[option]
        seeds = sorted(str(r.seed) for r in group if r.seed is not None)
        cells = [mean_std([getattr(r, name) for r in group]) for name in METRICS]
        lines.append(f"| {option} | {len(group)} ({', '.join(seeds)}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def option_of(config: dict) -> str:
    sweep = config.get("sweep") or {}
    if not sweep:
        return NO_OPTION
    return ", ".join(f"{key}={value}" for key, value in sorted(sweep.items()))


@dataclass(frozen=True)
class Summary:
    table: str
    runs: int
    skipped: list[str]  # "<run id>: <reason>" for each finished run that could not be used


def summarize_group(
    group: str, project: str, entity: str | None, out_dir: Path
) -> Summary:
    """Pull every finished run of `group` and build the table.

    Raises:
        ValueError: the group has no finished runs, or none of them has predictions.
    """
    import wandb

    from pcbi.eval.artifacts import PREDS_FILE, pull_predictions
    from pcbi.eval.predictions import load_predictions

    api = wandb.Api()
    entity = entity or api.default_entity
    runs = [r for r in api.runs(f"{entity}/{project}", filters={"group": group})]
    finished = [r for r in runs if r.state == "finished"]
    if not finished:
        raise ValueError(
            f"No finished runs in group {group!r} of {entity}/{project} "
            f"({len(runs)} run(s) in the group in total)."
        )

    rows, skipped = [], []
    for run in finished:
        cached = out_dir / run.id / PREDS_FILE
        try:
            path = cached if cached.is_file() else pull_predictions(run.id, out_dir, project, entity)
        except ValueError as exc:
            skipped.append(f"{run.id}: {exc}")
            continue
        seed = run.config.get("seed")
        rows.append(run_metrics(load_predictions(path), run.id, option_of(run.config), seed))

    if not rows:
        raise ValueError(f"None of the {len(finished)} finished run(s) in {group!r} has predictions.")
    return Summary(format_table(rows), len(rows), skipped)
