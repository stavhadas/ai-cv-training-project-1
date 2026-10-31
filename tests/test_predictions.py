"""Tests for `src/pcbi/eval/predictions.py`: what `save_predictions` writes, `load_predictions` reads."""

import numpy as np

from pcbi.eval.predictions import load_predictions, save_predictions
from pcbi.train.loop import ValResult


def test_a_saved_result_loads_back_unchanged_with_str_crop_ids(tmp_path):
    original = ValResult(
        mean_loss=1.25,
        logits=np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32),
        labels=np.array([0, 3, 1, 2, 0], dtype=np.int64),
        # Varying lengths and a non-ASCII id: fixed-width unicode must neither truncate nor mangle.
        crop_ids=["CS1_R0805_V2_a_1", "b", "CS6_C0805_V2_long_crop_id_17", "ü_1", "x_2"],
    )
    path = tmp_path / "val_preds.npz"
    save_predictions(original, path)
    loaded = load_predictions(path)

    assert loaded.mean_loss == original.mean_loss
    assert loaded.logits.dtype == np.float32
    np.testing.assert_array_equal(loaded.logits, original.logits)
    np.testing.assert_array_equal(loaded.labels, original.labels)
    assert loaded.crop_ids == original.crop_ids
    # Plain str, not bytes and not np.str_: a later `crop_id == "..."` or dict lookup must just work.
    assert all(type(crop_id) is str for crop_id in loaded.crop_ids)
