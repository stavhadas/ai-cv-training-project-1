"""Stage 2: the preprocessing a crop goes through on its way into a backbone.

Every timm backbone was pretrained with its own resize, normalization and interpolation, and
`timm.data.resolve_data_config(model=...)` hands those back as a dict. Using one model's constants
on another silently mis-normalises the input: the model still trains, the loss still falls, and the
benchmark quietly compares a correctly-fed network against a badly-fed one. So nothing here is
hard-coded — every value comes from the `data_config` the caller passes in.

The crops arrive square, because `pcbi make-crops` squares the box at extraction (see
`crops.crop_box`). That removes the decision this module would otherwise have to make: resizing a
square crop to a square input is a pure scale, with no aspect distortion, no centre-crop discarding
the edges of the joint, and no letterbox bars. Anything here that starts reasoning about aspect
ratio is solving a problem that was already solved upstream.

Two pipelines, deliberately separate:

- **eval** is what val and test are measured through. It must be deterministic — the same crop must
  produce the same tensor on every call, in every process, forever — or a number cannot be compared
  against the one recorded last week.
- **train** may be stochastic. Today it is not: with an empty `aug_cfg` it returns the eval
  pipeline, so the first Stage 2 runs differ from eval only in shuffling. Augmentation arrives in
  Stage 3 and belongs *here*, in the train branch, never in the eval one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchvision.transforms import v2
import torch

# timm names its interpolation modes as strings; torchvision wants an enum member.
INTERPOLATION = {
    "nearest": v2.InterpolationMode.NEAREST,
    "bilinear": v2.InterpolationMode.BILINEAR,
    "bicubic": v2.InterpolationMode.BICUBIC,
    "lanczos": v2.InterpolationMode.LANCZOS,
}
DEFAULT_INTERPOLATION = "bicubic"


def input_size(data_config: dict[str, Any]) -> tuple[int, int]:
    """The (height, width) a backbone expects, from timm's `input_size` triple `(C, H, W)`."""
    _, height, width = data_config["input_size"]
    return (height, width)


def interpolation_mode(data_config: dict[str, Any]) -> v2.InterpolationMode:
    """The resampling filter this backbone was pretrained with."""
    name = data_config.get("interpolation") or DEFAULT_INTERPOLATION
    if name not in INTERPOLATION:
        raise ValueError(
            f"unknown interpolation {name!r}; expected one of {', '.join(sorted(INTERPOLATION))}."
        )
    return INTERPOLATION[name]


def build_eval_transform(data_config: dict[str, Any]) -> Callable:
    """Build the deterministic preprocessing pipeline used for val, test and inference.

    Args:
        data_config: timm's resolved data config for the backbone being fed.

    Returns:
        **One callable**, not a list — ``PIL.Image.Image`` in, ``torch.Tensor`` out, of dtype
        ``torch.float32`` and shape ``(3, H, W)`` with ``(H, W) == input_size(data_config)``.
        ``v2`` provides a container transform that chains a list of transforms into a single
        callable; that container is what this returns.

    What it has to do, in three stages and in this order:

    1. **Resize** to ``input_size(data_config)`` — the helper above returns the ``(H, W)`` you
       need. Pass the resampling filter from ``interpolation_mode(data_config)``, also above;
       leaving it at the library default means every backbone gets resampled the same way
       regardless of how it was pretrained. The crops are already square, so this is a pure scale:
       no aspect-ratio decision, no centre crop, nothing to choose.

    2. **Convert to a float32 tensor scaled to [0, 1].** In ``v2`` this is *two* transforms, not
       one, and this is the step most likely to go wrong. The first wraps the PIL image as an image
       tensor, still 8-bit integers in 0-255. The second casts it to ``torch.float32`` — and that
       cast takes a flag controlling whether values are **rescaled** from 0-255 into 0-1. Left off,
       you get float32 values like ``237.0``, the normalization below then subtracts a mean of
       ``0.485`` from them, and the model receives inputs two orders of magnitude off. It trains.
       The loss falls. The number is quietly wrong. Turn the rescaling on.

       (The older ``transforms.ToTensor`` did the wrap and the scale together, which is why most
       examples you will find show one call where ``v2`` wants two.)

    3. **Normalize** with ``data_config["mean"]`` and ``data_config["std"]`` — per-channel
       sequences of three. This must come *after* stage 2: normalizing 0-255 integers is either an
       error or silently meaningless depending on dtype.

    Two hard requirements, both about the numbers this pipeline is measured through:

    **Deterministic.** Same crop, same tensor — every call, every worker process, every machine,
    indefinitely. No random resize, no random crop, no flip, no jitter. Every transform you reach
    for here should be one whose output depends only on its input. If this pipeline ever becomes
    stochastic, a val score stops being comparable with the one recorded yesterday, and comparing
    scores across runs is the entire point of the frozen split.

    **No hard-coded constants.** Every value comes from ``data_config``, which
    ``timm.data.resolve_data_config(model=...)`` fills in per backbone. Writing the ImageNet mean
    and std as literals works for most models and silently mis-normalises the rest — a bug that
    degrades a model without ever raising, so nothing surfaces it except a benchmark result you
    cannot explain. ``test_normalization_comes_from_the_config_not_from_a_literal`` is the guard:
    it pushes one crop through a ``mean=0, std=1`` config and through an ImageNet one and asserts
    the two tensors differ. Hard-code the constants and both paths produce the same tensor.

    Worth knowing: the returned tensor is **not** confined to ``[0, 1]``. Stage 2 lands it there,
    then stage 3 shifts and scales it away again — with ImageNet constants the range is roughly
    ``[-2.1, 2.6]``. Only a ``mean=0, std=1`` config leaves it in ``[0, 1]``, which is exactly why
    the test uses those values.
    """
    # === YOUR CODE: compose the deterministic resize -> float32 tensor -> normalize pipeline ===
    # Resize
    input_size_tuple = input_size(data_config)
    interpolation = interpolation_mode(data_config)
    resize_transform = v2.Resize(size=input_size_tuple, interpolation=interpolation)

    transform = v2.Compose([
        resize_transform,
        v2.ToImage(),  # Convert to tensor (0-255)
        v2.ConvertImageDtype(torch.float32),  # Convert to float32
        v2.Normalize(mean=data_config["mean"], std=data_config["std"])  # Normalize
    ])
    return transform
    # === END YOUR CODE ===


def build_train_transform(
    data_config: dict[str, Any], aug_cfg: dict[str, Any] | None = None
) -> Callable:
    """The training pipeline. With no augmentation configured, this is exactly the eval pipeline.

    Args:
        data_config: timm's resolved data config, passed through to `build_eval_transform`.
        aug_cfg: augmentation settings. Empty or ``None`` today; Stage 3 work.

    Returns:
        A callable with the same contract as `build_eval_transform` — PIL in, normalized float32
        ``(3, H, W)`` tensor out. Unlike the eval one it is allowed to be stochastic, and once
        augmentation lands it will be.

    Returning eval verbatim when `aug_cfg` is empty is deliberate rather than a placeholder: the
    first Stage 2 runs should differ from evaluation only in shuffling, so that if train and val
    diverge, the cause is the model or the split and not a preprocessing difference nobody chose.
    Augmentation is a change you make on purpose, measure, and keep or discard — not a default.

    **The one rule when Stage 3 arrives.** Whatever goes in the augmented branch stays on the train
    side. An augmentation reaching the eval pipeline makes val and test stochastic, and a
    stochastic val score cannot be compared against one recorded last week — which is the single
    thing the frozen split exists to make possible. Anything added also needs recording in the run
    config, or the score it produces cannot be reproduced.

    ## Third-party pieces worth knowing

    All verified present in this project's pinned versions (torchvision 0.29, timm 1.0.29).

    **Geometric** — `v2.RandomResizedCrop` (scale/ratio jitter; replaces the plain resize rather
    than following it), `v2.RandomHorizontalFlip`, `v2.RandomVerticalFlip`, `v2.RandomRotation`,
    `v2.RandomAffine`.

    Both flips are unusually safe here: a solder joint has no canonical orientation, and the
    dataset already contains each component under two lighting directions. Rotation is less free
    than it looks — the crop is tight, so a large angle pulls background in at the corners.

    **Photometric** — `v2.ColorJitter` (brightness/contrast/saturation/hue), `v2.GaussianBlur`,
    `v2.GaussianNoise`. Relevant because V2 and V2.1 differ precisely in illumination; mild
    brightness and contrast jitter is simulating a real source of variation rather than inventing
    one. Heavy saturation or hue shifts are riskier — solder colour carries class signal.

    **Policies** — `v2.RandAugment`, `v2.TrivialAugmentWide`, `v2.AutoAugment`. Pre-tuned schedules
    that replace a hand-picked stack with one knob. Tuned on ImageNet, so they include operations
    (posterise, solarise) with no physical meaning on a macro shot of solder.

    **Batch-level, not per-sample** — `v2.MixUp`, `v2.CutMix`. These mix *two labelled examples*,
    so they need the labels and belong in the collate function or the training step, not in this
    per-image pipeline. `v2.RandomErasing` is per-sample but must run *after* normalization.

    **Composition** — `v2.RandomApply` (run a sub-pipeline with probability p), `v2.RandomChoice`
    (pick one of several).

    **timm's own** — `timm.data.create_transform` builds a whole train pipeline from a config in
    one call and is the path of least resistance if you want timm's defaults instead of a
    hand-built stack; `timm.data.Mixup` and `timm.data.FastCollateMixup` are its mixup/cutmix
    implementations, designed to sit in the collate; `timm.data.rand_augment_transform` builds a
    RandAugment policy from timm's config string.

    A caution on `create_transform`: it builds its own resize and normalization from the same
    `data_config`, so calling it *and* the eval pipeline would resize twice. It is an alternative
    to this function's body, not an addition to it.
    """
    if aug_cfg:
        # TODO(stage-3): build the augmented pipeline here — random resized crop, horizontal and
        # vertical flip (solder joints have no canonical orientation), light colour jitter for the
        # two lighting directions, and possibly mixup/cutmix at the batch level rather than here.
        #
        # Whatever lands here must stay on the *train* side. An augmentation that reaches the eval
        # pipeline makes val and test stochastic, and a stochastic val score cannot be compared
        # against a previously recorded one — which is the single thing the frozen split exists to
        # make possible. Anything added here also needs recording in the run config, or a score
        # becomes impossible to reproduce.
        raise NotImplementedError(
            f"augmentation is Stage 3 work; got aug_cfg={sorted(aug_cfg)}. "
            f"Pass an empty aug_cfg to train with the eval pipeline."
        )
    return build_eval_transform(data_config)
