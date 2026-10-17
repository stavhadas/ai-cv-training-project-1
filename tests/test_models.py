"""Tests for `pcbi.train.models` and `pcbi.train.config`.

`build_model` is tested with ``pretrained=False`` so nothing needs the network; the one place that
matters for pretrained weights (the exact tag) is a config concern and is covered by the shipped
YAML loading cleanly.
"""

from pathlib import Path

import pytest
import torch
from omegaconf.errors import ConfigKeyError, ValidationError

from pcbi.train.config import TrainConfig, load_config
from pcbi.train.models import build_model, count_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "exp" / "s2_first.yaml"


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- build_model ----------------------------------------------------------------------------------


def test_build_model_outputs_one_logit_per_class():
    model, _ = build_model("resnet18", 4, pretrained=False)
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(3, 3, 64, 64))
    assert out.shape == (3, 4)


def test_build_model_returns_a_data_config_with_what_the_transform_needs():
    _, data_config = build_model("resnet18", 4, pretrained=False)
    for key in ("input_size", "mean", "std", "interpolation"):
        assert key in data_config


def test_build_model_prints_total_and_trainable_parameter_counts(capsys):
    model, _ = build_model("resnet18", 4, pretrained=False)
    total, trainable = count_parameters(model)
    printed = capsys.readouterr().out
    assert f"{total:,}" in printed
    assert f"{trainable:,}" in printed
    assert total == trainable  # nothing is frozen by the factory


def test_a_different_head_size_changes_the_output_width():
    model, _ = build_model("resnet18", 7, pretrained=False)
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 7)


# --- config loader --------------------------------------------------------------------------------


def test_the_shipped_config_loads():
    cfg = load_config(SHIPPED_CONFIG)
    assert isinstance(cfg, TrainConfig)
    assert cfg.model == "resnet50.a1_in1k"
    assert cfg.optimizer in ("sgd", "adam")


def test_omitted_keys_fall_back_to_defaults(tmp_path):
    cfg = load_config(write_yaml(tmp_path, "lr: 0.5\n"))
    assert cfg.lr == 0.5
    assert cfg.imbalance.mode == "none"


def test_an_unknown_top_level_key_fails_loudly(tmp_path):
    with pytest.raises(ConfigKeyError, match="learning_rate"):
        load_config(write_yaml(tmp_path, "learning_rate: 0.1\n"))


def test_an_unknown_imbalance_key_fails_loudly(tmp_path):
    with pytest.raises(ConfigKeyError, match="modee"):
        load_config(write_yaml(tmp_path, "imbalance:\n  modee: none\n"))


def test_a_misspelled_section_name_fails_loudly(tmp_path):
    with pytest.raises(ConfigKeyError, match="imbalence"):
        load_config(write_yaml(tmp_path, "imbalence:\n  mode: none\n"))


def test_a_wrongly_typed_value_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, "epochs: many\n"))


@pytest.mark.parametrize(
    "text, match",
    [
        ("optimizer: rmsprop\n", "optimizer"),
        ("lr: 0\n", "lr"),
        ("epochs: 0\n", "epochs"),
        ("batch_size: 0\n", "batch_size"),
        ("imbalance:\n  mode: oversample\n", "imbalance.mode"),
        ("imbalance:\n  weight_scheme: sqrt\n", "weight_scheme"),
        ("imbalance:\n  beta: 1.0\n", "beta"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_config(write_yaml(tmp_path, text))
