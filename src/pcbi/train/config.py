"""Stage 2: the minimal experiment config, loaded strictly.

The YAML holds only what changes between experiments. It grows in later stages, and only when
something needs a new key.

**Unknown keys fail loudly.** The schema below is an OmegaConf structured config, so merging a YAML
into it raises on any key the schema doesn't declare — a typo like ``learning_rate`` or
``imbalence`` is an error at load time, not a silently ignored setting that costs a Kaggle run.
Values are checked too (an unknown optimizer or imbalance mode, a non-positive lr), so a bad value
fails here rather than three layers into training.

`ImbalanceConfig` is exactly the ``imbalance:`` section `pcbi.train.imbalance` expects, so a caller
passes ``dataclasses.asdict(cfg.imbalance)`` straight to `build_criterion` / `build_sampler`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from pcbi.train.imbalance import MODES, WEIGHT_SCHEMES

OPTIMIZERS = ("sgd", "adam")


@dataclass
class ImbalanceConfig:
    mode: str = "none"
    weight_scheme: str = "inverse_freq"
    beta: float = 0.999


@dataclass
class TrainConfig:
    model: str = "resnet50.a1_in1k"
    optimizer: str = "sgd"
    lr: float = 0.01
    epochs: int = 10
    batch_size: int = 32
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    num_workers: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        # Runs when OmegaConf turns the merged config back into this dataclass (`to_object`), so
        # these checks see the final, YAML-overridden values.
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}.")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}.")
        if not 0 <= self.seed < 2**32:  # NumPy's seed range
            raise ValueError(f"seed must be in [0, 2**32), got {self.seed}.")
        if isinstance(self.imbalance, ImbalanceConfig):
            if self.imbalance.mode not in MODES:
                raise ValueError(
                    f"imbalance.mode must be one of {MODES}, got {self.imbalance.mode!r}."
                )
            if self.imbalance.weight_scheme not in WEIGHT_SCHEMES:
                raise ValueError(
                    f"imbalance.weight_scheme must be one of {WEIGHT_SCHEMES}, "
                    f"got {self.imbalance.weight_scheme!r}."
                )
            if not 0.0 <= self.imbalance.beta < 1.0:
                raise ValueError(f"imbalance.beta must be in [0, 1), got {self.imbalance.beta}.")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> TrainConfig:
    """Load a YAML experiment config, rejecting unknown keys and invalid values.

    `overrides` are dotted ``key=value`` strings applied on top of the YAML, e.g.
    ``imbalance.mode=none``. They go through the same schema, so a mistyped key fails the same way.

    Raises:
        omegaconf.errors.ConfigKeyError: the YAML has a key the schema doesn't declare.
        omegaconf.errors.ValidationError: a value has the wrong type (e.g. ``epochs: many``).
        ValueError: a value is the right type but not an allowed one (e.g. ``optimizer: rmsprop``).
    """
    schema = OmegaConf.structured(TrainConfig)
    merged = OmegaConf.merge(schema, OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(merged)
