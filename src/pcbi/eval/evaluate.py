"""Stage 3: `pcbi eval` — score saved validation predictions the way the factory would.

Reads a ``val_preds.npz``; never touches a model or a GPU. Writes to the output folder:

- ``report.md``: AUROC, AUPRC, per-class recall, the row-normalized confusion matrix, and the
  operating points at a few miss-rate limits;
- ``roc.png`` (drawn by `product.plot_roc`) and ``confusion_matrix.png``;
- ``sweep.csv``: every threshold with its miss and false-alarm rate, for anything the report omits.

**AUROC vs AUPRC.** Both treat *defect* as the positive class. Here defects are the majority (about
four in five crops), so AUPRC starts high even for a weak model — its chance level is the defect
prevalence, not 0.5. The report prints that baseline next to it for that reason.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcbi.data.datasets import CLASS_ORDER

DEFAULT_OUT_DIR = Path("reports/eval")
MISS_LIMITS = (0.10, 0.05, 0.0)


@dataclass(frozen=True)
class Evaluation:
    crops: int
    defects: int
    auroc: float
    auprc: float
    recall: np.ndarray  # per class, CLASS_ORDER
    support: np.ndarray  # crops per true class
    confusion: np.ndarray  # counts, rows true, columns predicted
    sweep: list[tuple[float, float, float]]  # (tau, miss rate, false-alarm rate), largest tau first


def evaluate(logits: np.ndarray, labels: np.ndarray) -> Evaluation:
    import torch
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    from pcbi.eval.product import defect_score, threshold_sweep

    scores = defect_score(torch.from_numpy(logits))
    is_defect = labels != 0
    sweep = threshold_sweep(scores, torch.from_numpy(is_defect))

    class_ids = list(range(len(CLASS_ORDER)))
    matrix = confusion_matrix(labels, logits.argmax(axis=1), labels=class_ids)
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.zeros(len(class_ids)), where=support > 0)

    return Evaluation(
        crops=len(labels),
        defects=int(is_defect.sum()),
        auroc=float(roc_auc_score(is_defect, scores.numpy())),
        auprc=float(average_precision_score(is_defect, scores.numpy())),
        recall=recall,
        support=support,
        confusion=matrix,
        sweep=sweep,
    )


def row_normalized(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)


def operating_point(sweep, max_miss: float) -> tuple[float, float, float] | None:
    """The largest-τ row with miss rate ≤ `max_miss`; the sweep is largest τ first."""
    return next((row for row in sweep if row[1] <= max_miss), None)


def save_confusion_plot(matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(row_normalized(matrix), display_labels=list(CLASS_ORDER)).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".2f"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_report(ev: Evaluation, name: str) -> str:
    prevalence = ev.defects / ev.crops
    normals = ev.crops - ev.defects
    lines = [
        f"# Evaluation: {name}",
        "",
        f"{ev.crops} validation crops: {ev.defects} defect, {normals} normal. "
        f"One normal crop moves the false-alarm rate by {1 / max(normals, 1):.3f}.",
        "",
        "| metric | value | chance level |",
        "| --- | --- | --- |",
        f"| AUROC | {ev.auroc:.3f} | 0.500 |",
        f"| AUPRC | {ev.auprc:.3f} | {prevalence:.3f} (defect prevalence) |",
        "",
        "## Operating points",
        "",
        "The largest threshold τ that meets each miss-rate limit (flagged means score ≥ τ).",
        "",
        "| miss rate ≤ | τ | miss rate | false-alarm rate |",
        "| --- | --- | --- | --- |",
    ]
    for limit in MISS_LIMITS:
        point = operating_point(ev.sweep, limit)
        if point is None:
            lines.append(f"| {limit:.2f} | none reaches it | | |")
        else:
            tau, miss, far = point
            lines.append(f"| {limit:.2f} | {tau:.4f} | {miss:.3f} | {far:.3f} |")

    lines += [
        "",
        "![ROC curve](roc.png)",
        "",
        "## Per-class recall",
        "",
        "| class | crops | recall |",
        "| --- | --- | --- |",
    ]
    for name_, n, r in zip(CLASS_ORDER, ev.support, ev.recall, strict=True):
        lines.append(f"| {name_} | {int(n)} | {r:.3f} |")
    lines.append(f"| **macro** | | **{ev.recall.mean():.3f}** |")

    normalized = row_normalized(ev.confusion)
    lines += [
        "",
        "## Confusion matrix",
        "",
        "Rows are the true class, columns the predicted class; each row sums to 1. Counts in "
        "brackets.",
        "",
        "| true \\ predicted | " + " | ".join(CLASS_ORDER) + " |",
        "| --- | " + " | ".join("---" for _ in CLASS_ORDER) + " |",
    ]
    for i, name_ in enumerate(CLASS_ORDER):
        cells = [f"{normalized[i, j]:.2f} ({ev.confusion[i, j]})" for j in range(len(CLASS_ORDER))]
        lines.append(f"| {name_} | " + " | ".join(cells) + " |")
    lines += ["", "![Confusion matrix](confusion_matrix.png)", ""]
    return "\n".join(lines)


def write_sweep_csv(sweep, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tau", "miss_rate", "false_alarm_rate"])
        for tau, miss, far in sweep:
            writer.writerow([f"{tau:.6f}", f"{miss:.6f}", f"{far:.6f}"])


def run_eval(preds_path: Path, out_dir: Path) -> tuple[Evaluation, Path]:
    """Evaluate `preds_path` into `out_dir`; return the evaluation and the report's path."""
    from pcbi.eval.predictions import load_predictions
    from pcbi.eval.product import plot_roc

    result = load_predictions(preds_path)
    ev = evaluate(result.logits, result.labels)
    if ev.defects in (0, ev.crops):
        raise ValueError(
            f"{preds_path}: every crop is {'normal' if ev.defects == 0 else 'a defect'}; "
            f"AUROC and false-alarm rates need both."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir.name
    plot_roc(ev.sweep, out_dir / "roc.png")
    save_confusion_plot(ev.confusion, out_dir / "confusion_matrix.png", f"{name}: row-normalized")
    write_sweep_csv(ev.sweep, out_dir / "sweep.csv")
    report = out_dir / "report.md"
    report.write_text(render_report(ev, name), encoding="utf-8")
    return ev, report
