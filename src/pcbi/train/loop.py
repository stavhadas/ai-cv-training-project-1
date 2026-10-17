"""Stage 2: the training and validation loops — the code you will debug for the rest of the project.

Two functions, deliberately plain:

- `train_one_epoch` — one pass over the training loader, one optimizer step per batch.
- `validate` — one pass over the validation loader, no gradients, returning the logits for every
  sample so that metrics can be computed *outside* this module.

**What this module does not do.** It computes no metrics: not accuracy, not recall, not a confusion
matrix. `validate` hands back raw logits, labels and crop ids; Stage 3 builds the metrics the
product needs on top of them, and `pcbi train`'s end-of-run printouts are quick and temporary. It
also uses plain full precision with no AMP, no gradient accumulation and no scheduler, so a loss
curve here means exactly one thing.

**Batch shape.** `CropDataset` yields `Sample(image, label, crop_id)` and the default collate turns
a batch into a `Sample` whose ``image`` is ``(B, 3, H, W)`` float32, ``label`` is ``(B,)`` int64,
and ``crop_id`` is a list of ``B`` strings. Move ``image`` and ``label`` to `device`; ``crop_id``
stays on the CPU, it is only a list of strings.

**Averaging.** Both functions report a mean loss that is weighted by batch size — see
`RunningMean` — because the last batch of an epoch is usually smaller than the rest and must not
count as much as a full one.
"""

from __future__ import annotations

import time  # noqa: F401  (for train_one_epoch's timing; unused until that body is written)
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Called once per optimizer step with (step, loss): `step` counts from 0 within the epoch and
# `loss` is that batch's loss as a plain float. Used by `pcbi train` for `--max-steps` and for
# per-step logging; the loop itself needs nothing from it.
OnStep = Callable[[int, float], None]


class RunningMean:
    """A mean over samples, fed one batch at a time.

    `update(value, n)` takes a batch's *mean* value and the number of samples `n` behind it, and
    weights the batch by `n`. That is what makes the final result the mean over **samples** rather
    than over batches: a last batch of 3 samples among batches of 32 pulls the average by 3
    samples' worth, not by 1/Nth of the epoch.

    It holds plain Python floats on purpose. Feed it ``loss.item()`` (or ``float(loss)``), never the
    loss *tensor* — a tensor would keep each step's autograd graph alive for the whole epoch.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        """Add a batch whose mean is `value` and which contained `n` samples."""
        self.total += value * n
        self.count += n

    @property
    def mean(self) -> float:
        """The sample-weighted mean so far. Raises if nothing has been added yet."""
        if self.count == 0:
            raise ValueError("RunningMean has no samples; was the loader empty?")
        return self.total / self.count


@dataclass
class TrainEpochResult:
    mean_loss: float  # mean over samples, not over batches
    samples_seen: int  # how many training samples this epoch consumed
    seconds: float  # wall-clock time for the epoch


@dataclass
class ValResult:
    mean_loss: float  # mean over samples, not over batches
    logits: np.ndarray  # float32, shape (N, 4): one row per sample, in loader order
    labels: np.ndarray  # int64, shape (N,): the true class index of each row of `logits`
    crop_ids: list[str]  # length N: the crop each row came from, in the same order


def progress(loader: Iterable, desc: str) -> Iterator:
    """Wrap `loader` in a tqdm bar; iterate the result exactly as you would the loader.

    The bar clears itself when the epoch ends (``leave=False``) so a long run's log is one line per
    epoch rather than one per batch.
    """
    return tqdm(loader, desc=desc, leave=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    on_step: OnStep | None = None,
) -> TrainEpochResult:
    """Train `model` for one pass over `loader`: one optimizer step per batch.

    Iterate ``progress(loader, "train")`` rather than `loader` itself, so there is a progress bar.

    What the body must do:

    - Put the model in **training mode** (``model.train()``) once, *before* the first batch — and
      do it here, in this function, not once at program start. `validate` switches the model to
      evaluation mode, so unless every epoch re-enters training mode itself, the second epoch would
      silently train with dropout off and batch-norm running on frozen statistics.
    - For each batch: move the images and labels to `device`; clear the gradients left by the
      previous step; run the forward pass; compute ``criterion(logits, labels)``; backpropagate;
      take one optimizer step. That order matters — gradients that are not cleared accumulate
      across steps.
    - Keep a **sample-weighted** running loss with `RunningMean`: update it with the batch's loss
      and the batch's *actual* size (``labels.shape[0]``), not the configured ``batch_size``, so a
      smaller final batch is weighted by how many samples it really contains.
    - Convert the loss to a Python float with ``loss.item()`` for the running mean and for
      `on_step`, and never accumulate the loss *tensor* — that keeps every step's computation
      graph alive in memory. (``.item()`` makes the CPU wait for the GPU, so this is once per
      step, not more.)
    - If `on_step` is not ``None``, call ``on_step(step, loss)`` after each optimizer step, with
      `step` counting from 0 within this epoch.
    - Time the whole epoch with ``time.perf_counter()``. Do not compute any metric here.

    Args:
        model: the network, already on `device`.
        loader: yields `Sample`-style batches (``image``, ``label``, ``crop_id``).
        optimizer: already built over ``model.parameters()``.
        criterion: e.g. `nn.CrossEntropyLoss`, already on `device` if it carries a weight tensor.
        device: where the batches should go; must match where the model is.
        on_step: optional per-step callback, see above.

    Returns:
        `TrainEpochResult` — the sample-weighted ``mean_loss``, ``samples_seen`` (the sum of the
        actual batch sizes), and ``seconds`` for the epoch.
    """
    # === YOUR CODE: one training epoch: train mode, step per batch, weighted mean loss ===
    model.train()
    running_mean = RunningMean()
    start_time = time.perf_counter()  # Start timing the epoch
    # Move the images and labels to the device
    for step, batch in enumerate(progress(loader, "train")):
        image, label, _ = batch
        image, label = image.to(device), label.to(device)
        
        optimizer.zero_grad()  # Clear gradients from the previous step
        
        logits = model(image)  # Forward pass
        loss = criterion(logits, label)  # Compute loss
        
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        
        # Update running mean loss
        running_mean.update(loss.item(), label.size(0))
        
        if on_step is not None:
            on_step(step, loss.item())  # Call the on_step callback if provided
    
    # count the epoch time
    epoch_time = time.perf_counter() - start_time
    return TrainEpochResult(mean_loss=running_mean.mean, samples_seen=running_mean.count, seconds=epoch_time)
          
    # === END YOUR CODE ===


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
) -> ValResult:
    """Evaluate `model` over `loader` without touching its weights, and return every logit.

    Iterate ``progress(loader, "val")`` rather than `loader` itself.

    What the body must do:

    - Put the model in **evaluation mode** (``model.eval()``) before the first batch, so dropout is
      off and batch-norm uses its running statistics instead of the batch's. This function does not
      put the model back in training mode; `train_one_epoch` does that for itself.
    - Run the whole pass under ``torch.inference_mode()``. No gradients are needed, so none are
      recorded: less memory, faster, and it makes it impossible to call ``backward`` by accident.
    - For each batch: move the images and labels to `device`, run the forward pass, compute the
      loss, and keep a **sample-weighted** running loss with `RunningMean` (the batch's loss
      weighted by its actual size, so a smaller final batch counts for what it contains).
    - Collect, for every batch, the logits, the labels and the crop ids — moved back to the CPU,
      as plain numpy / Python values, so nothing here stays tied to the GPU or the autograd graph.
    - Return the logits of **every** sample, **in loader order**: row ``i`` of ``logits``,
      ``labels[i]`` and ``crop_ids[i]`` all describe the same sample, and that sample is the
      ``i``-th one the loader yields. With an unshuffled loader that is dataset order, which is
      what makes two validation runs comparable row by row. Do not reorder, deduplicate or drop
      anything.
    - Compute no metric here — no accuracy, no argmax, no confusion matrix. Logits, not
      predictions.

    Args:
        model: the network, already on `device`.
        loader: yields `Sample`-style batches; it should not shuffle.
        criterion: the same kind of loss used for training, so the two curves are comparable.
        device: where the batches should go; must match where the model is.

    Returns:
        `ValResult` — the sample-weighted ``mean_loss``; ``logits`` as a float32 array of shape
        ``(N, num_classes)``; ``labels`` as an int64 array of shape ``(N,)``; and ``crop_ids`` as
        a list of ``N`` strings, all in loader order.
    """
    # === YOUR CODE: one evaluation pass: eval mode, weighted loss, collect logits ===
    running_mean = RunningMean()
    all_logits = []
    all_labels = []
    all_crop_ids = []
    model.eval()
    with torch.inference_mode():
      for image, label, crop_id in progress(loader, "val"):
          image, label = image.to(device), label.to(device)
          
          logits = model(image)  # Forward pass
          loss = criterion(logits, label)  # Compute loss
          
          running_mean.update(loss.item(), label.size(0))  # Update running mean loss
          
          # Collect logits, labels and crop_ids
          all_logits.append(logits.cpu().numpy())
          all_labels.append(label.cpu().numpy())
          all_crop_ids.extend(crop_id)
    return ValResult(
        mean_loss=running_mean.mean,
        logits=np.concatenate(all_logits, axis=0).astype(np.float32),
        labels=np.concatenate(all_labels, axis=0).astype(np.int64),
        crop_ids=all_crop_ids
    )

    # === END YOUR CODE ===
