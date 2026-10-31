"""Tests for `src/pcbi/eval/gallery.py`: the join, the cell selection, and reproducible images."""

import csv

import numpy as np
import pytest
from PIL import Image

from pcbi.data import crops as crops_mod
from pcbi.data.datasets import CLASS_ORDER
from pcbi.eval import gallery as gallery_mod
from pcbi.eval.gallery import Entry
from pcbi.eval.predictions import save_predictions
from pcbi.train.loop import ValResult

NORMAL, EXCESS, INSUFFICIENT, SPIKE = range(4)


def entry(crop_id, true, pred, score):
    return Entry(crop_id, true, pred, score, "CS1", "R0805", "")


def test_cells_put_the_most_confident_crops_first_and_break_ties_by_crop_id():
    entries = [
        entry("d", NORMAL, EXCESS, 0.60),
        entry("a", NORMAL, EXCESS, 0.90),
        entry("c", NORMAL, EXCESS, 0.60),
        entry("b", NORMAL, EXCESS, 0.70),
        entry("x", SPIKE, NORMAL, 0.40),
        entry("y", SPIKE, NORMAL, 0.10),
    ]
    cells = gallery_mod.build_cells(entries, per_cell=3)

    # Predicted a defect: highest defect score first; "c" beats "d" on the tie.
    assert [e.crop_id for e in cells[(NORMAL, EXCESS)].shown] == ["a", "b", "c"]
    assert cells[(NORMAL, EXCESS)].total == 4
    # Predicted normal: lowest defect score (most confidently normal) first.
    assert [e.crop_id for e in cells[(SPIKE, NORMAL)].shown] == ["y", "x"]
    # Every cell exists, so the grid keeps its shape even where nothing landed.
    assert len(cells) == len(CLASS_ORDER) ** 2
    assert cells[(EXCESS, EXCESS)].total == 0


def test_the_selection_does_not_depend_on_input_order():
    entries = [entry(f"c{i}", INSUFFICIENT, SPIKE, s) for i, s in enumerate([0.3, 0.8, 0.8, 0.5])]
    forward = gallery_mod.build_cells(entries, per_cell=2)
    backward = gallery_mod.build_cells(entries[::-1], per_cell=2)
    assert forward[(INSUFFICIENT, SPIKE)].shown == backward[(INSUFFICIENT, SPIKE)].shown


def test_mistakes_are_off_diagonal_only_largest_first():
    entries = [
        entry("r1", EXCESS, EXCESS, 0.9),
        entry("m1", NORMAL, SPIKE, 0.8),
        entry("m2", INSUFFICIENT, NORMAL, 0.2),
        entry("m3", INSUFFICIENT, NORMAL, 0.3),
    ]
    order = gallery_mod.mistake_order(gallery_mod.build_cells(entries, per_cell=8))
    assert [(c.true, c.pred, c.total) for c in order] == [(INSUFFICIENT, NORMAL, 2), (NORMAL, SPIKE, 1)]


@pytest.fixture
def run(tmp_path):
    """Six val crops on disk, a manifest, a polygons.csv with one merged joint, and predictions."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    specs = [  # (label, logits)
        ("normal", [3.0, 0, 0, 0]),
        ("normal", [0, 2.0, 0, 0]),
        ("excess", [0, 3.0, 0, 0]),
        ("insufficient", [2.0, 0, 0, 0]),
        ("spike", [0, 0, 0, 3.0]),
        ("spike", [0, 0, 2.0, 0]),
    ]
    manifest_rows, polygon_rows, crop_ids = [], [], []
    for k, (label, _) in enumerate(specs):
        dataset_path = f"SolDef_AI/Dataset/CS1/R0805/V2/img_{k}.jpg"
        crop_id = crops_mod.crop_id("CS1", "R0805", "V2", dataset_path, 1)
        crop_ids.append(crop_id)
        Image.new("RGB", (40, 40), (40 * k, 100, 200 - 30 * k)).save(crops_dir / f"{crop_id}.png")
        manifest_rows.append(
            {"crop_id": crop_id, "label": label, "split": "val", "board": "CS1", "package": "R0805"}
        )
        polygon_rows.append(
            {
                "board": "CS1",
                "package": "R0805",
                "viewpoint": "V2",
                "dataset_path": dataset_path,
                "polygon_rank_x": "1",
                "merged_from": "insufficient+spike" if k == 3 else "",
            }
        )
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, manifest_rows)
    polygons = tmp_path / "polygons.csv"
    write_csv(polygons, polygon_rows)

    preds = tmp_path / "abc123" / "val_preds.npz"
    preds.parent.mkdir()
    save_predictions(
        ValResult(
            mean_loss=1.0,
            logits=np.array([logits for _, logits in specs], dtype=np.float32),
            labels=np.array([CLASS_ORDER.index(label) for label, _ in specs]),
            crop_ids=crop_ids,
        ),
        preds,
    )
    return preds, manifest, crops_dir, polygons, crop_ids


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_rerun_writes_byte_identical_images(run, tmp_path):
    preds, manifest, crops_dir, polygons, _ = run
    # Same file name in two folders: the name is drawn into the title, so it must match too.
    first = gallery_mod.make_gallery(preds, manifest, crops_dir, tmp_path / "1" / "g.png", 2, polygons)
    second = gallery_mod.make_gallery(preds, manifest, crops_dir, tmp_path / "2" / "g.png", 2, polygons)

    assert (first.crops, first.wrong, first.merged_known) == (6, 3, True)
    assert first.matrix.read_bytes() == second.matrix.read_bytes()
    assert first.mistakes.name == "g_mistakes.png"
    assert first.mistakes.read_bytes() == second.mistakes.read_bytes()


def test_merged_from_is_joined_from_polygons_csv(run):
    preds, manifest, _, polygons, crop_ids = run
    merged = gallery_mod.load_merged_from(polygons)
    assert merged[crop_ids[3]] == "insufficient+spike"
    assert merged[crop_ids[0]] == ""


def test_a_crop_missing_from_the_manifest_is_refused(run, tmp_path):
    preds, manifest, crops_dir, polygons, _ = run
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))[1:]
    write_csv(manifest, rows)
    with pytest.raises(ValueError, match="not in"):
        gallery_mod.make_gallery(preds, manifest, crops_dir, tmp_path / "g.png", 2, polygons)
