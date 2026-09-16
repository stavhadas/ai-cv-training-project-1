"""Tests for `pcbi show`, the Stage 0 Step 6 visual-check helper."""

from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import audit as audit_mod
from pcbi.data import show as show_mod

runner = CliRunner()


def make_jpeg(path: Path, brightness: int = 128, size: tuple[int, int] = (16, 16)) -> None:
    Image.new("L", size, color=brightness).save(path, format="JPEG")


@pytest.fixture
def cell_dataset(tmp_path):
    """CS1/R0805 with V2 (2 images) and V2.1 (1 image) — a deliberately uneven pair."""

    def place(viewpoint: str, setup: str | None, name: str) -> None:
        parts = [tmp_path, "Dataset", "CS1", "R0805", viewpoint]
        if setup:
            parts.append(setup)
        folder = Path(*parts)
        folder.mkdir(parents=True, exist_ok=True)
        make_jpeg(folder / name)

    place("V1", "Setup1", "a_top1.jpg")
    place("V1", "Setup2", "b_top2.jpg")
    place("V2", None, "c_45_first.jpg")
    place("V2", None, "d_45_second.jpg")
    place("V2.1", None, "e_45alt_first.jpg")
    return tmp_path


def test_select_cell_filters_by_board_package_viewpoint(cell_dataset):
    result = audit_mod.audit(cell_dataset)
    images = show_mod.select_cell(result, "CS1", "R0805", "V2")
    assert [p.name for p in images] == ["c_45_first.jpg", "d_45_second.jpg"]


def test_select_cell_filters_by_setup(cell_dataset):
    result = audit_mod.audit(cell_dataset)
    images = show_mod.select_cell(result, "CS1", "R0805", "V1", setup="Setup2")
    assert [p.name for p in images] == ["b_top2.jpg"]


def test_select_cell_returns_nothing_for_an_unknown_cell(cell_dataset):
    result = audit_mod.audit(cell_dataset)
    assert show_mod.select_cell(result, "CS9", "R0805", "V2") == []


def test_build_grid_produces_one_tile_per_image(cell_dataset):
    result = audit_mod.audit(cell_dataset)
    images = show_mod.select_cell(result, "CS1", "R0805", "V2")
    grid = show_mod.build_grid(images)
    assert grid.width >= show_mod.THUMB_SIZE[0] * len(images)


def test_build_grid_handles_an_empty_list():
    grid = show_mod.build_grid([])
    assert grid.width > 0 and grid.height > 0


def test_build_grid_recovers_from_an_unreadable_file(tmp_path):
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"not a real jpeg")
    real = tmp_path / "real.jpg"
    make_jpeg(real)
    grid = show_mod.build_grid([fake, real])
    assert grid.width >= show_mod.THUMB_SIZE[0] * 2  # both tiles present, one just shows an error


def test_build_pair_grid_stops_at_the_shorter_side(cell_dataset):
    result = audit_mod.audit(cell_dataset)
    v2 = show_mod.select_cell(result, "CS1", "R0805", "V2")
    v21 = show_mod.select_cell(result, "CS1", "R0805", "V2.1")
    grid = show_mod.build_pair_grid(v2, v21, "V2", "V2.1")
    # V2.1 has only one image, so exactly one pair -> two tiles side by side.
    assert grid.width == show_mod.THUMB_SIZE[0] * 2 + show_mod.PADDING * 3


def test_cli_writes_a_grid_for_one_cell(cell_dataset, tmp_path):
    out = tmp_path / "grid.png"
    result = runner.invoke(
        app,
        [
            "show",
            "--root",
            str(cell_dataset),
            "--board",
            "CS1",
            "--package",
            "R0805",
            "--viewpoint",
            "V2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.is_file()
    assert "2 image(s) shown (of 2 in this cell)" in result.stdout


def test_cli_pair_mode_reports_the_shorter_count(cell_dataset, tmp_path):
    out = tmp_path / "pair.png"
    result = runner.invoke(
        app,
        [
            "show",
            "--root",
            str(cell_dataset),
            "--board",
            "CS1",
            "--package",
            "R0805",
            "--pair",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.is_file()
    assert "1 pair(s) shown (of 1 matched; 2 V2, 1 V2.1 total)" in result.stdout


def test_cli_limit_caps_images_without_hiding_the_true_total(cell_dataset, tmp_path):
    out = tmp_path / "grid.png"
    result = runner.invoke(
        app,
        [
            "show",
            "--root",
            str(cell_dataset),
            "--board",
            "CS1",
            "--package",
            "R0805",
            "--viewpoint",
            "V2",
            "--limit",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert "1 image(s) shown (of 2 in this cell)" in result.stdout
    assert Image.open(out).width < show_mod.THUMB_SIZE[0] * 2  # only one tile wide


def test_cli_fails_clearly_when_the_cell_is_empty(cell_dataset):
    result = runner.invoke(
        app,
        [
            "show",
            "--root",
            str(cell_dataset),
            "--board",
            "CS9",
            "--package",
            "R0805",
            "--viewpoint",
            "V2",
        ],
    )
    assert result.exit_code == 1


def test_cli_requires_viewpoint_unless_pair(cell_dataset):
    result = runner.invoke(
        app, ["show", "--root", str(cell_dataset), "--board", "CS1", "--package", "R0805"]
    )
    assert result.exit_code == 2


def test_cli_fails_clearly_when_root_is_missing(tmp_path):
    result = runner.invoke(
        app,
        [
            "show",
            "--root",
            str(tmp_path / "nope"),
            "--board",
            "CS1",
            "--package",
            "R0805",
            "--viewpoint",
            "V2",
        ],
    )
    assert result.exit_code == 1
