"""Tests for `pcbi tag`, the manual pairing tool.

The state model (link, unlink, the guess, the CSV it produces) is pure and tested directly. The
server is tested through a real socket rather than by calling handler methods: the thing that can
actually break is the wiring — a route that never fires, a mutation that answers before it saves —
and only a live request exercises that.
"""

import csv
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from threading import Thread

import pytest
from PIL import Image
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import audit as audit_mod
from pcbi.data import group as group_mod
from pcbi.data import ingest as ingest_mod
from pcbi.data import tag as tag_mod
from pcbi.data import tag_server as tag_server_mod

runner = CliRunner()


def labelme_record(image_name: str) -> dict:
    return {
        "version": "5.0.1",
        "imagePath": image_name,
        "imageWidth": 64,
        "imageHeight": 64,
        "imageData": None,
        "shapes": [
            {"label": label, "shape_type": "polygon", "points": points}
            for label, points in (
                ("spike", [[5, 5], [15, 5], [15, 15], [5, 15]]),
                ("exc_solder", [[40, 5], [55, 5], [55, 15], [40, 15]]),
            )
        ],
    }


@pytest.fixture
def tagging_dataset(tmp_path):
    """Two cells: one with both viewpoints to pair across, one with only V2 to pair nothing in.

    The second cell is not filler — two of the real dataset's seven cells were photographed under
    one lighting only, and `pairable` has to report 0 for them rather than inviting someone to hunt
    for twins that do not exist.
    """
    labeled = tmp_path / "Labeled"
    labeled.mkdir()

    def place(board: str, package: str, viewpoint: str, name: str, shade: int) -> None:
        folder = tmp_path / "Dataset" / board / package / viewpoint
        folder.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (64, 64), color=(shade, shade, shade))
        image.save(folder / name, format="JPEG")
        image.save(labeled / name, format="JPEG")
        (labeled / f"{name.removesuffix('.jpg')}.json").write_text(json.dumps(labelme_record(name)))

    place("CS1", "R0805", "V2", "WIN_20220330_13_11_00_Pro.jpg", 40)
    place("CS1", "R0805", "V2", "WIN_20220330_13_12_00_Pro_C7.jpg", 80)
    place("CS1", "R0805", "V2.1", "WIN_20220330_13_21_00_Pro.jpg", 120)
    place("CS1", "R0805", "V2.1", "WIN_20220330_13_22_00_Pro_C7.jpg", 160)
    place("CS2", "R0603", "V2", "WIN_20220330_16_07_40_Pro.jpg", 200)
    return tmp_path


@pytest.fixture
def images(tagging_dataset):
    result = audit_mod.audit(tagging_dataset)
    return group_mod.collect_images(result, ingest_mod.load_taxonomy())


@pytest.fixture
def cells(images):
    return tag_mod.build_cells(images)


def paths(images, *fragments: str) -> list[str]:
    """Dataset paths of the images whose names contain each fragment, in the order given."""
    found = []
    for fragment in fragments:
        matches = [image.dataset_path for image in images if fragment in image.dataset_path]
        assert len(matches) == 1, f"{fragment!r} matched {len(matches)} images"
        found.append(matches[0])
    return found


# --- cells -----------------------------------------------------------------------------------


def test_cells_split_by_viewpoint_in_capture_order(cells):
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    assert [viewpoint for viewpoint, _ in cell.columns] == ["V2", "V2.1"]
    for _, indices in cell.columns:
        assert indices == sorted(indices)  # capture order, which is dataset_path order


def test_a_cell_with_one_viewpoint_can_pair_nothing(cells):
    lonely = next(cell for cell in cells if cell.name == "CS2/R0603")
    assert lonely.size == 1
    assert lonely.pairable == 0


def test_pairable_counts_images_the_smaller_column_can_absorb(cells):
    both = next(cell for cell in cells if cell.name == "CS1/R0805")
    assert both.pairable == 4  # 2 pairs, so 4 images, all of them
    assert both.balanced


def test_unequal_columns_are_flagged_as_unbalanced(images):
    """Two columns of different lengths cannot line up by position, which `guess_by_order` needs."""
    lopsided = tag_mod.Cell("x", "x", "y", [("V2", [0, 1, 2]), ("V2.1", [3])])
    assert not lopsided.balanced
    assert lopsided.pairable == 2  # one pair; the other two V2 photos have nothing to match


# --- the state model -------------------------------------------------------------------------


def test_link_joins_two_paths():
    assert tag_mod.link([], "b", "a") == [["a", "b"]]


def test_link_merges_the_groups_it_touches():
    groups = tag_mod.link([["a", "b"]], "b", "c")
    assert groups == [["a", "b", "c"]]


def test_link_is_a_no_op_on_itself():
    assert tag_mod.link([["a", "b"]], "a", "a") == [["a", "b"]]


def test_unlink_leaves_the_rest_of_the_group_standing():
    assert tag_mod.unlink([["a", "b", "c"]], "b") == [["a", "c"]]


def test_unlink_dissolves_a_pair_entirely():
    assert tag_mod.unlink([["a", "b"]], "b") == []


def test_canonical_drops_singletons_and_sorts():
    assert tag_mod.canonical([["z"], ["c", "b"], ["b", "c"]]) == [["b", "c"]]


def test_canonical_merges_groups_that_share_an_image():
    """An image belongs to at most one group, so two groups naming it are the same group."""
    assert tag_mod.canonical([["a", "b"], ["b", "c"], ["d", "e"]]) == [["a", "b", "c"], ["d", "e"]]


def test_groups_survive_a_save_and_load(tmp_path):
    path = tmp_path / "pairs.json"
    tag_mod.save_groups(path, [["b", "a"], ["d", "c"]])
    assert tag_mod.load_groups(path) == [["a", "b"], ["c", "d"]]


def test_loading_a_pairing_that_was_never_made_is_empty(tmp_path):
    assert tag_mod.load_groups(tmp_path / "nothing.json") == []


# --- pairing from the filename ---------------------------------------------------------------


def test_the_c_suffix_pairs_across_viewpoints(images, cells):
    """`_C7` under both V2 and V2.1 is a recorded pairing, not a guess."""
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    expected = paths(images, "V2/WIN_20220330_13_12_00", "V2.1/WIN_20220330_13_22_00")
    assert tag_mod.pair_by_hint(images, cell, []) == [sorted(expected)]


def test_the_c_suffix_merges_with_what_is_already_there(images, cells):
    """Running it after some pairing exists must not leave one image in two groups."""
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    hinted_v2, other_v21 = paths(images, "V2/WIN_20220330_13_12_00", "V2.1/WIN_20220330_13_21_00")
    groups = tag_mod.pair_by_hint(images, cell, [[hinted_v2, other_v21]])
    assert len(groups) == 1 and len(groups[0]) == 3


def test_a_cell_with_no_suffixes_gets_nothing(images, cells):
    lonely = next(cell for cell in cells if cell.name == "CS2/R0603")
    assert tag_mod.pair_by_hint(images, lonely, []) == []


# --- the capture-order guess -----------------------------------------------------------------


def test_guess_pairs_the_columns_by_position(images, cells):
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    groups = tag_mod.guess_by_order(images, cell, [])
    first_v2, first_v21 = paths(images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_21_00")
    assert sorted([first_v2, first_v21]) in groups
    assert len(groups) == 2


def test_guess_never_overwrites_work_already_done(images, cells):
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    crossed = paths(images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_22_00")
    groups = tag_mod.guess_by_order(images, cell, [crossed])
    assert sorted(crossed) in groups
    # The two images left over are the ones the guess would have crossed with those — and since
    # each already has a partner, the guess must leave them alone rather than merging four.
    assert all(len(group) == 2 for group in groups)


def test_guess_twice_changes_nothing(images, cells):
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    once = tag_mod.guess_by_order(images, cell, [])
    assert tag_mod.guess_by_order(images, cell, once) == once


def test_guess_does_nothing_in_a_single_viewpoint_cell(images, cells):
    lonely = next(cell for cell in cells if cell.name == "CS2/R0603")
    assert tag_mod.guess_by_order(images, lonely, []) == []


def test_clearing_a_cell_leaves_other_cells_alone(images, cells):
    cell = next(cell for cell in cells if cell.name == "CS1/R0805")
    elsewhere = ["some/other/a.jpg", "some/other/b.jpg"]
    groups = tag_mod.guess_by_order(images, cell, [elsewhere])
    assert tag_mod.clear_cell(images, cell, groups) == [elsewhere]


# --- the CSV ---------------------------------------------------------------------------------


def test_untagged_images_become_singletons(images):
    rows = tag_mod.build_rows(images, [])
    assert len(rows) == len(images)
    assert {row["group_size"] for row in rows} == {1}
    assert {row["method"] for row in rows} == {"manual"}


def test_a_pair_shares_one_group_id(images):
    pair = paths(images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_21_00")
    rows = {row["dataset_path"]: row for row in tag_mod.build_rows(images, [pair])}
    assert rows[pair[0]]["group_id"] == rows[pair[1]]["group_id"]
    assert rows[pair[0]]["group_size"] == 2


def test_the_csv_carries_no_parameters(images):
    """A hand-made grouping has no threshold, tolerance or fingerprint to record."""
    pair = paths(images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_21_00")
    row = tag_mod.build_rows(images, [pair])[0]
    assert row["threshold"] == ""
    assert row["crop_tolerance"] == ""
    assert row["fingerprint"] == ""


def test_the_csv_has_the_same_columns_as_every_other_method(images, tmp_path):
    out = tmp_path / "groups_manual.csv"
    tag_mod.write_manual_csv(images, [], out)
    with out.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == group_mod.CSV_FIELDS


def test_a_saved_path_that_no_longer_exists_is_reported_not_dropped_quietly(images):
    ghost = ["gone/a.jpg", "gone/b.jpg"]
    assert tag_mod.unknown_paths(images, [ghost]) == sorted(ghost)
    # ...and it contributes no row, so the CSV still describes only what is on disk.
    assert len(tag_mod.build_rows(images, [ghost])) == len(images)


def test_summary_counts_progress_against_what_can_be_paired(images, cells):
    pair = paths(images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_21_00")
    summary = tag_mod.summarize(images, cells, [pair])
    assert summary.images == 5
    assert summary.pairable == 4  # only CS1/R0805 has two viewpoints, and all 4 of its can pair
    assert summary.tagged == 2
    assert summary.groups == 1
    assert summary.untagged == 3
    assert summary.percent == 50.0


# --- the server ------------------------------------------------------------------------------


@pytest.fixture
def server(tagging_dataset, tmp_path):
    """A real server on a free loopback port, torn down with the test."""
    state = tag_server_mod.build_state(
        tagging_dataset,
        pairs_path=tmp_path / "pairs.json",
        out_path=tmp_path / "groups_manual.csv",
    )
    httpd = tag_server_mod.make_server(state, port=0)
    Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()
    httpd.server_close()


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def test_the_page_is_served(server):
    base, _ = server
    assert b"pcbi tag" in get(f"{base}/")


def test_the_manifest_describes_every_image_and_cell(server):
    base, state = server
    manifest = json.loads(get(f"{base}/api/state"))
    assert len(manifest["images"]) == len(state.images)
    assert [cell["name"] for cell in manifest["cells"]] == ["CS1/R0805", "CS2/R0603"]
    assert manifest["summary"]["pairable"] == 4


def test_a_tile_is_a_jpeg_of_the_cropped_component(server):
    base, state = server
    tile = Image.open(io.BytesIO(get(f"{base}/img/0")))
    assert tile.format == "JPEG"
    # The joints span (5, 5)-(55, 15) of a 64px frame, so the default tolerance of 1.0 asks for
    # 50px on every side and clamps to the whole frame — the wide view a person pairs from.
    assert state.images[0].joint_box == (5, 5, 55, 15)
    assert tile.size == (64, 64)
    # Asking for a tight one crops: 0.3 adds 15px a side, so height becomes 0..30.
    assert Image.open(io.BytesIO(get(f"{base}/img/0?t=0.3"))).size == (64, 30)


def test_the_whole_frame_is_available_too(server):
    base, _ = server
    full = Image.open(io.BytesIO(get(f"{base}/img/0?t=full")))
    assert full.size == (64, 64)


def test_an_image_index_outside_the_dataset_is_a_404(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(f"{base}/img/9999")
    assert caught.value.code == 404


def test_pairing_saves_both_files_before_it_answers(server, tmp_path):
    base, state = server
    a, b = paths(state.images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_21_00")
    payload = post(f"{base}/api/link", {"a": a, "b": b})

    assert payload["groups"] == [sorted([a, b])]
    assert payload["summary"]["tagged"] == 2
    # Both artifacts exist by the time the response arrives — not at shutdown.
    assert tag_mod.load_groups(tmp_path / "pairs.json") == [sorted([a, b])]
    rows = {
        row["dataset_path"]: row
        for row in csv.DictReader((tmp_path / "groups_manual.csv").open(newline=""))
    }
    assert rows[a]["group_id"] == rows[b]["group_id"]


def test_unpairing_goes_back(server):
    base, state = server
    a, b = paths(state.images, "V2/WIN_20220330_13_11_00", "V2.1/WIN_20220330_13_21_00")
    post(f"{base}/api/link", {"a": a, "b": b})
    payload = post(f"{base}/api/unlink", {"path": a})
    assert payload["groups"] == []
    assert payload["summary"]["tagged"] == 0


def test_a_path_the_dataset_does_not_have_is_rejected(server):
    base, state = server
    (a,) = paths(state.images, "V2/WIN_20220330_13_11_00")
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(f"{base}/api/link", {"a": a, "b": "../../etc/passwd"})
    assert caught.value.code == 400


def test_the_guess_the_hint_and_the_clear_run_over_one_cell(server):
    base, _ = server
    assert post(f"{base}/api/hint", {"cell": "CS1/R0805"})["summary"]["groups"] == 1
    assert post(f"{base}/api/guess", {"cell": "CS1/R0805"})["summary"]["groups"] == 2
    assert post(f"{base}/api/clear", {"cell": "CS1/R0805"})["summary"]["groups"] == 0


def test_an_unknown_cell_is_rejected(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(f"{base}/api/guess", {"cell": "CS9/R9999"})
    assert caught.value.code == 400


# --- the command -----------------------------------------------------------------------------


def test_export_writes_the_csv_without_serving(tagging_dataset, tmp_path):
    pairs = tmp_path / "pairs.json"
    out = tmp_path / "groups_manual.csv"
    a, b = (
        "Dataset/CS1/R0805/V2/WIN_20220330_13_11_00_Pro.jpg",
        ("Dataset/CS1/R0805/V2.1/WIN_20220330_13_21_00_Pro.jpg"),
    )
    tag_mod.save_groups(pairs, [[a, b]])

    result = runner.invoke(
        app,
        [
            "tag",
            "--root",
            str(tagging_dataset),
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--export",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "5 joint-task image(s) across 2 cell(s)" in result.output
    rows = {row["dataset_path"]: row for row in csv.DictReader(out.open(newline=""))}
    assert rows[a]["group_id"] == rows[b]["group_id"]


def test_export_says_when_the_pairing_names_images_that_are_gone(tagging_dataset, tmp_path):
    pairs = tmp_path / "pairs.json"
    tag_mod.save_groups(pairs, [["Dataset/CS1/R0805/V2/renamed.jpg", "Dataset/CS1/gone.jpg"]])
    result = runner.invoke(
        app,
        [
            "tag",
            "--root",
            str(tagging_dataset),
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out.csv"),
            "--export",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 saved path(s) no longer exist" in result.output


def test_a_negative_crop_tolerance_is_refused(tagging_dataset, tmp_path):
    result = runner.invoke(
        app,
        [
            "tag",
            "--root",
            str(tagging_dataset),
            "--crop-tolerance",
            "-1",
            "--out",
            str(tmp_path / "out.csv"),
            "--export",
        ],
    )
    assert result.exit_code == 2
    assert "must be 0 or greater" in result.output


def test_a_missing_dataset_is_refused(tmp_path):
    result = runner.invoke(app, ["tag", "--root", str(tmp_path / "nowhere"), "--export"])
    assert result.exit_code == 1
    assert "No such folder" in result.output


def test_the_page_asset_ships_with_the_package():
    assert Path(tag_server_mod.PAGE).is_file()
