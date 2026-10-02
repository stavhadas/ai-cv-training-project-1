"""Stage 2: correcting for class imbalance, in the loss or in the sampler, never both at once.

On this project's frozen train split the classes are not catastrophically imbalanced, but they are
not close either: `excess` 93, `spike` 72, `normal` 39, `insufficient` 34 — a largest:smallest
ratio of about 2.7. Left uncorrected, a model minimising overall cross-entropy has a cheap way to
improve its loss that has nothing to do with getting joints right: predict the common classes more
often. This module gives two independent ways to correct for that, chosen by config rather than
hard-coded, so the correction itself is a comparable, logged experimental choice like any other.

**`mode` picks at most one correction, never a combination of both:**

- ``none`` — no correction. The baseline every other mode is compared against.
- ``class_weighted_loss`` — `build_criterion` returns `nn.CrossEntropyLoss` with a `weight` tensor,
  so a mistake on a rare class costs more than the same mistake on a common one. The batches
  themselves stay at the dataset's natural class proportions.
- ``weighted_sampler`` — `build_sampler` returns a `WeightedRandomSampler` that over-samples rare
  classes so batches arrive closer to balanced, and the loss is left plain.

Combining the two would double-correct: a rare-class example would already be over-represented in
the batch by the sampler, and then weighted again by the loss, compounding one correction on top of
the other rather than applying either cleanly. `build_criterion` and `build_sampler` each read the
same `cfg["mode"]` and are mutually exclusive by construction — only one of them ever does anything
for a given config.

**Both correction modes share one weighting decision**, `weight_scheme`:

- ``inverse_freq`` — a class's weight is inversely proportional to how often it occurs. Simple, and
  treats every example as equally informative regardless of how many others share its class.
- ``effective_number`` — Cui, Jia, Lin, Song & Belongie, "Class-Balanced Loss Based on Effective
  Number of Samples" (CVPR 2019). Assumes examples of an over-represented class overlap in the
  information they carry, and weights by an *effective* count that discounts that overlap rather
  than the raw count. `beta`, in ``[0, 1)``, controls how much overlap is assumed; the paper's
  experiments use values close to 1.

`get_weights` dispatches between them; `inverse_freq_weights` and `effective_number_weights` are
the two functions actually computing the numbers, deliberately left for you to implement.

**Assumed config shape.** `cfg` in `build_criterion` and `build_sampler` is exactly the
``imbalance:`` section itself — ``{"mode": ..., "weight_scheme": ..., "beta": ...}`` — not the
whole training config with an ``imbalance`` key inside it. If your config loader hands the whole
tree to these functions, pass `cfg["imbalance"]` at the call site rather than changing the
functions; keeping them ignorant of where in a larger config they sit is what lets them be tested
with a plain three-key dict, as this module's tests do.

**`CLASS_ORDER` comes from `pcbi.data.datasets`.** It is not redefined here — the classifier's
class order is one project-wide constant, and every weight vector this module produces is ordered
the same way it is, so it lines up with a model's logits without anyone having to remember to
reorder anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from pcbi.data.datasets import CLASS_ORDER, CLASS_TO_INDEX

MODES = ("none", "class_weighted_loss", "weighted_sampler")
WEIGHT_SCHEMES = ("inverse_freq", "effective_number")

DEFAULT_MODE = "none"
DEFAULT_WEIGHT_SCHEME = "inverse_freq"
# Cui et al.'s experiments use beta in [0.9, 0.9999]; 0.999 sits in the middle of that range rather
# than at either extreme, so a config that omits beta doesn't silently pick the least-tested edge.
DEFAULT_BETA = 0.999


class LabeledDataset(Protocol):
    """What `build_sampler` needs from a dataset — not the whole `CropDataset` interface.

    `CropDataset` satisfies this without trying to, since it already exposes `labels` as a
    plain list of `CLASS_ORDER` indices. Naming the requirement here rather than typing the
    parameter as `CropDataset` means a sampler can be built and tested against a handful of
    labels with no manifest, no crop directory and no PNGs on disk.
    """

    labels: Sequence[int]

    def __len__(self) -> int: ...


def class_counts_from_manifest(rows: Sequence[dict[str, str]], split: str = "train") -> np.ndarray:
    """The number of crops per class within one split, ordered like `CLASS_ORDER`.

    `rows` is the same shape `csv.DictReader` produces over `manifest.csv` — and the same shape
    `CropDataset.rows` already holds, so the usual call is
    ``class_counts_from_manifest(dataset.rows, split=dataset.split)`` on a dataset that has already
    validated its labels. Passed raw, unfiltered manifest rows instead, this does its own
    filtering and its own label check, so a bad manifest is caught here rather than three functions
    downstream.

    Args:
        rows: manifest rows, each a dict with at least a ``"label"`` and a ``"split"`` key. Rows
            for splits other than `split` are ignored — the caller does not need to pre-filter.
        split: which split to count. Rows outside it are skipped, not summed in.

    Returns:
        An ``np.ndarray`` of shape ``(len(CLASS_ORDER),)`` and integer dtype: `counts[i]` is the
        number of `split` rows whose label is `CLASS_ORDER[i]`. A class with no examples in this
        split gets a genuine `0`, not a missing entry — every position in `CLASS_ORDER` is present.

    Raises:
        ValueError: a row in `split` has a label outside `CLASS_ORDER`, or no row matches `split`
            at all (an empty split here means a config bug or a manifest built from the wrong
            file, not zero crops to weight).
    """
    counts = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    matched = False
    for row in rows:
        if row["split"] != split:
            continue
        matched = True
        label = row["label"]
        if label not in CLASS_TO_INDEX:
            raise ValueError(
                f"row {row.get('crop_id', '?')!r} has label {label!r}, not in CLASS_ORDER "
                f"({', '.join(CLASS_ORDER)})."
            )
        counts[CLASS_TO_INDEX[label]] += 1

    if not matched:
        raise ValueError(f"no rows with split={split!r} in the given manifest rows.")
    return counts


def inverse_freq_weights(counts: np.ndarray) -> np.ndarray:
    """Per-class weight, inversely proportional to how often each class occurs.

    Args:
        counts: shape ``(num_classes,)``, the number of examples of each class in `CLASS_ORDER`
            order. Every entry must be positive — a class with zero examples has no meaningful
            inverse frequency, and this function does not have to handle that case gracefully.

    Returns:
        An array the same shape as `counts`, one weight per class in the same order.

        **Normalization convention — every caller and every test relies on this exact one:** the
        returned weights must sum to ``len(counts)``, the number of classes. Equivalently, the
        mean weight is exactly ``1``. This is what lets `build_criterion` hand the result straight
        to ``nn.CrossEntropyLoss(weight=...)`` without it silently rescaling the loss's overall
        magnitude relative to an unweighted run, and it is what makes "a balanced `counts` yields
        weights of exactly `1`" a specific number to assert on rather than merely "some equal
        value".

        Rarer classes — a smaller entry in `counts` — must get a **larger** weight than more
        common classes: the weight is a strictly decreasing function of the count.

    A perfectly balanced `counts` (every entry equal) must, after normalization, produce a weights
    array that is all-ones — every class weighted identically, because the classes already are.
    """
    # === YOUR CODE: inverse-frequency class weights, normalized to sum to num_classes ===
    return len(counts) * (1 / counts) / np.sum(1 / counts)
    
    # === END YOUR CODE ===


def effective_number_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    """Per-class weight from the effective number of samples.

    Reference: Cui, Jia, Lin, Song & Belongie, "Class-Balanced Loss Based on Effective Number of
    Samples", CVPR 2019. Where `inverse_freq_weights` treats every example as equally informative,
    this scheme assumes examples of an over-represented class overlap in the information they
    carry, and weights each class by an *effective* count that discounts for that overlap rather
    than its raw count.

    Args:
        counts: shape ``(num_classes,)``, the number of examples of each class, in `CLASS_ORDER`
            order. As with `inverse_freq_weights`, every entry must be positive; a zero count is
            not a case this function has to handle.
        beta: the overlap/redundancy parameter, in ``[0, 1)``. Larger values assume more
            redundancy among examples of a common class.

    Returns:
        An array the same shape as `counts`, one weight per class in the same order, under the
        **same normalization convention as `inverse_freq_weights`**: the weights sum to
        ``len(counts)``. Rarer classes get larger weights, same as the other scheme.

        **`beta=0` is a specific, checkable degenerate case, not just "a small beta":** at
        ``beta=0``, the effective number of samples for *every* class is defined to be `1`,
        regardless of that class's actual count. So at ``beta=0`` every class must receive an
        **equal** weight — the same all-ones-after-normalization result `inverse_freq_weights`
        gives only when `counts` happens to be balanced, except here it must hold for *any*
        `counts`, however imbalanced.

    As `beta` approaches `1`, the effective number approaches the raw count, and this scheme's
    weights approach `inverse_freq_weights`'s.
    """
    # === YOUR CODE: effective-number-of-samples class weights, normalized to sum to num_classes ===
    if beta == 0:
        return np.ones_like(counts, dtype=np.float32)
    elif beta == 1:
        # fill with 0.999 
        return np.full_like(counts, 0.999, dtype=np.float32)
    else:
        effective_num = (1 - np.power(beta, counts)) / (1 - beta)
        weights = 1 / effective_num
        weights = weights * len(counts) / np.sum(weights)
        return weights.astype(np.float32)
    # === END YOUR CODE ===


def get_weights(scheme: str, counts: np.ndarray, beta: float) -> np.ndarray:
    """Route to the configured weighting scheme.

    A thin dispatcher, kept separate from `build_criterion` and `build_sampler` so both of them
    share one place that turns a `weight_scheme` string into a weight vector — neither has to know
    the two schemes exist as distinct functions.
    """
    if scheme == "inverse_freq":
        return inverse_freq_weights(counts)
    if scheme == "effective_number":
        return effective_number_weights(counts, beta)
    raise ValueError(f"weight_scheme must be one of {WEIGHT_SCHEMES}, got {scheme!r}.")


def build_criterion(
    cfg: dict[str, Any], counts: np.ndarray, device: torch.device | str = "cpu"
) -> nn.Module:
    """Build the training loss, weighted per class only when `cfg["mode"] == "class_weighted_loss"`.

    Every other mode — including `"weighted_sampler"`, which corrects the batches instead — returns
    a plain, unweighted `nn.CrossEntropyLoss`. Weighting the loss under `"weighted_sampler"` too
    would double-correct the same imbalance in two places at once; see the module docstring.

    Args:
        cfg: the ``imbalance:`` config section — ``{"mode", "weight_scheme", "beta"}``, all
            optional. Missing keys fall back to `DEFAULT_MODE`, `DEFAULT_WEIGHT_SCHEME` and
            `DEFAULT_BETA`.
        counts: per-class counts from `class_counts_from_manifest`, used only when the mode
            actually weights the loss.
        device: where the weight tensor should live — the same device the model and its logits
            are on, or `nn.CrossEntropyLoss` raises at the first forward pass.

    Returns:
        An `nn.CrossEntropyLoss`. Its `.weight` attribute is `None` unless `mode` is
        `"class_weighted_loss"`, in which case it is a `float32` tensor on `device` holding
        `get_weights`'s result.
    """
    mode = cfg.get("mode", DEFAULT_MODE)
    if mode not in MODES:
        raise ValueError(f"imbalance.mode must be one of {MODES}, got {mode!r}.")
    if mode != "class_weighted_loss":
        return nn.CrossEntropyLoss()

    weights = get_weights(
        cfg.get("weight_scheme", DEFAULT_WEIGHT_SCHEME), counts, cfg.get("beta", DEFAULT_BETA)
    )
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def build_sampler(
    cfg: dict[str, Any], dataset: LabeledDataset, counts: np.ndarray
) -> WeightedRandomSampler | None:
    """Build the sampler that corrects batch composition, or `None` when nothing should.

    Only `cfg["mode"] == "weighted_sampler"` returns a real sampler. Every other mode — including
    `"class_weighted_loss"`, which corrects the loss instead — returns `None`, so the caller wires
    an unweighted, shuffling sampler exactly as if imbalance correction were off.

    Args:
        cfg: the ``imbalance:`` config section, same shape as `build_criterion`'s.
        dataset: anything satisfying `LabeledDataset` — a `.labels` sequence of `CLASS_ORDER`
            indices and a `__len__`. `CropDataset` qualifies without modification.
        counts: per-class counts from `class_counts_from_manifest`, used only under
            `"weighted_sampler"`.

    Returns:
        `None` unless `mode == "weighted_sampler"`, in which case a `WeightedRandomSampler` with
        one weight per *sample* — `dataset`'s per-class weight, looked up by each sample's own
        label — `num_samples == len(dataset)`, so one pass over the sampler is one epoch's worth
        of draws, and `replacement=True`, which is what lets a rare class be drawn more times than
        it physically has examples.
    """
    mode = cfg.get("mode", DEFAULT_MODE)
    if mode not in MODES:
        raise ValueError(f"imbalance.mode must be one of {MODES}, got {mode!r}.")
    if mode != "weighted_sampler":
        return None

    class_weights = get_weights(
        cfg.get("weight_scheme", DEFAULT_WEIGHT_SCHEME), counts, cfg.get("beta", DEFAULT_BETA)
    )
    sample_weights = [float(class_weights[label]) for label in dataset.labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
