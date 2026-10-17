"""Stage 2: one function that turns a model name into a ready-to-train 4-class model.

`build_model` wraps `timm.create_model`. Two details are worth stating because they decide whether
a rerun is comparable to the last one:

- **The name pins the weights.** ``resnet50.a1_in1k`` is one exact set of ImageNet weights (the
  ``a1_in1k`` part is timm's pretrained tag), while a bare ``resnet50`` resolves to whatever timm's
  default tag is *this release*. Always pass the full tag, so every run starts from the same point.
- **Only the head is new.** ``num_classes=4`` swaps ImageNet's 1000-way classifier for a freshly
  initialised 4-way layer; the backbone keeps its pretrained weights.

The second return value, `data_config`, is what `timm.data.resolve_data_config` reads off the model:
input size, mean/std and interpolation. The Stage 1 transform is built from it, so the model and the
preprocessing can't drift apart.
"""

import timm
from timm.data import resolve_data_config
from torch import nn

NUM_CLASSES = 4


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(
    name: str, num_classes: int = NUM_CLASSES, *, pretrained: bool = True
) -> tuple[nn.Module, dict]:
    """Create a timm model with a fresh `num_classes`-way head and report its size.

    Args:
        name: a timm model name, ideally with its pretrained tag, e.g. ``"resnet50.a1_in1k"``.
        num_classes: size of the new classifier head.
        pretrained: load the ImageNet weights named by `name`. Tests pass ``False`` so they need
            no network; real runs leave it on.

    Returns:
        ``(model, data_config)`` — the model, whose output has shape ``[batch, num_classes]``, and
        the dict from `timm.data.resolve_data_config` (``input_size``, ``mean``, ``std``,
        ``interpolation``, ...) that the Stage 1 transform should be built from.
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    data_config = resolve_data_config(model=model)

    total, trainable = count_parameters(model)
    print(f"{name}: {total:,} parameters total, {trainable:,} trainable")
    return model, data_config
