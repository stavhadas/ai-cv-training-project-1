"""Tests for `src/pcbi/eval/product.py` and the `pcbi eval` report built on it."""

from fractions import Fraction

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_curve

from pcbi.eval import evaluate as evaluate_mod
from pcbi.eval.predictions import save_predictions
from pcbi.eval.product import defect_score, plot_roc, rates_at, threshold_sweep
from pcbi.train.loop import ValResult

# The Stage 3 worked example: seven crops sorted by defect score, D = defect, N = normal.
WORKED_SCORES = [0.95, 0.80, 0.70, 0.60, 0.40, 0.20, 0.10]
WORKED_TRUTH = "DDNDNDN"
WORKED_TABLE = [  # (tau, miss rate, false-alarm rate), exactly as the page lists them
    (0.95, Fraction(3, 4), Fraction(0, 3)),
    (0.80, Fraction(2, 4), Fraction(0, 3)),
    (0.70, Fraction(2, 4), Fraction(1, 3)),
    (0.60, Fraction(1, 4), Fraction(1, 3)),
    (0.40, Fraction(1, 4), Fraction(2, 3)),
    (0.20, Fraction(0, 4), Fraction(2, 3)),
    (0.10, Fraction(0, 4), Fraction(3, 3)),
]


def worked_example():
    scores = torch.tensor(WORKED_SCORES, dtype=torch.float64)
    is_defect = torch.tensor([c == "D" for c in WORKED_TRUTH])
    return scores, is_defect


def test_the_worked_example_reproduces_the_table_row_for_row():
    sweep = threshold_sweep(*worked_example())
    assert len(sweep) == len(WORKED_TABLE)
    for (tau, miss, far), (want_tau, want_miss, want_far) in zip(sweep, WORKED_TABLE, strict=True):
        assert tau == pytest.approx(want_tau)
        assert miss == pytest.approx(float(want_miss))
        assert far == pytest.approx(float(want_far))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rates_match_sklearn_roc_curve_at_every_threshold(seed):
    rng = np.random.default_rng(seed)
    logits = torch.from_numpy(rng.normal(size=(60, 4)))
    labels = rng.integers(0, 4, size=60)
    labels[:3] = [0, 1, 2]  # both classes present whatever the draw
    scores = defect_score(logits)
    # Rounding forces tied scores, the case where a sloppy sweep and sklearn part ways.
    scores = torch.round(scores * 20) / 20
    is_defect = torch.from_numpy(labels != 0)

    fpr, tpr, thresholds = roc_curve(is_defect.numpy(), scores.numpy(), drop_intermediate=False)
    # sklearn prepends +inf (nothing flagged); every other threshold is a unique score, largest first.
    sweep = threshold_sweep(scores, is_defect)
    assert [t for t, _, _ in sweep] == pytest.approx(list(thresholds[1:]))
    for tau, fa, tp in zip(thresholds[1:], fpr[1:], tpr[1:], strict=True):
        miss, far = rates_at(scores, is_defect, float(tau))
        assert miss == pytest.approx(1 - tp)
        assert far == pytest.approx(fa)


def test_defect_score_is_one_minus_the_normal_probability():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
    assert defect_score(logits).tolist() == pytest.approx([0.75, 3 * np.exp(-10) / (np.exp(0) + 3 * np.exp(-10))], rel=1e-3)


def test_plot_roc_writes_an_image(tmp_path):
    plot_roc(threshold_sweep(*worked_example()), tmp_path / "roc.png")
    assert (tmp_path / "roc.png").stat().st_size > 0


def test_pcbi_eval_writes_the_report_plots_and_sweep(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    logits = rng.normal(size=(20, 4)).astype(np.float32)
    logits[np.arange(20), labels] += 2.0  # a model that is mostly right
    preds = tmp_path / "run1" / "val_preds.npz"
    preds.parent.mkdir()
    save_predictions(ValResult(1.0, logits, labels, [f"c{i}" for i in range(20)]), preds)

    ev, report = evaluate_mod.run_eval(preds, tmp_path / "eval" / "run1")

    for name in ("report.md", "roc.png", "confusion_matrix.png", "sweep.csv"):
        assert (report.parent / name).is_file()
    text = report.read_text(encoding="utf-8")
    assert f"| AUROC | {ev.auroc:.3f} |" in text
    assert "## Per-class recall" in text
    assert ev.support.tolist() == [5, 5, 5, 5]
    assert np.allclose(evaluate_mod.row_normalized(ev.confusion).sum(axis=1), 1.0)


def test_operating_point_takes_the_largest_threshold_that_meets_the_limit():
    sweep = threshold_sweep(*worked_example())
    # Miss rate at most 25%: both 0.60 and 0.40 qualify; the page's rule takes 0.60.
    assert evaluate_mod.operating_point(sweep, 0.25)[0] == pytest.approx(0.60)
    assert evaluate_mod.operating_point(sweep, 0.0)[0] == pytest.approx(0.20)
