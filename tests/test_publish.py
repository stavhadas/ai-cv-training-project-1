"""Tests for `pcbi publish-crops`, which uploads the crops as a private Kaggle dataset.

**These tests never contact Kaggle.** Everything except the one `subprocess.run` is a pure function,
and that one call is reached through an injected runner, so the whole path can be exercised with no
network, no account and no upload — the same policy `tests/test_smoke.py` states for W&B. An autouse
fixture scrubs the Kaggle environment variables so a developer's real credentials cannot leak into a
run and change what is under test.

The load-bearing test here is `test_no_code_path_ever_asks_kaggle_for_a_public_dataset`. Visibility
is decided by one flag and nothing else; if that assertion ever fails, the dataset is public.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pcbi.cli import app
from pcbi.data import crops as crops_mod
from pcbi.data import publish as publish_mod

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from a known state; the developer's real Kaggle account must not leak in."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)


@pytest.fixture
def published_inputs(tmp_path):
    """A crops folder, a split CSV and a provenance JSON, shaped like the real ones."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()

    rows = []
    for index, (label, split) in enumerate(
        [("normal", "train"), ("excess", "train"), ("spike", "val"), ("normal", "test")]
    ):
        crop_id = f"CS1_R0805_V2_IMG_{index:03d}_1"
        (crops_dir / f"{crop_id}.png").write_bytes(b"fake png bytes " + str(index).encode())
        rows.append(
            {field: "" for field in crops_mod.MANIFEST_FIELDS}
            | {
                "crop_id": crop_id,
                "label": label,
                "split": split,
                "board": "CS1",
                "package": "R0805",
            }
        )

    import csv

    with (crops_dir / crops_mod.MANIFEST_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=crops_mod.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    splits_path = tmp_path / "split_v1.csv"
    splits_path.write_text("crop_id,split\na,train\n", encoding="utf-8")
    meta_path = tmp_path / "manifest_meta.json"
    meta_path.write_text(json.dumps({"split_hash": "abc123", "version": 1}), encoding="utf-8")
    return crops_dir, splits_path, meta_path


class RecordingRunner:
    """Stands in for `subprocess.run`, capturing the argv instead of executing it."""

    def __init__(self, returncode: int = 0, stdout: str = "uploaded", stderr: str = ""):
        self.calls: list[list[str]] = []
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- privacy: the requirement this command exists to satisfy ----------------------------------


def test_no_code_path_ever_asks_kaggle_for_a_public_dataset(tmp_path):
    """The whole privacy guarantee, in one assertion.

    `kaggle datasets create` is private unless `-u/--public` is passed, and `datasets version` has
    no visibility flag at all. So the only way this dataset becomes public is if one of those
    strings reaches the argv. Nothing else in this module — not the metadata JSON, not the slug —
    can change it.
    """
    for argv in (
        publish_mod.argv_create(tmp_path),
        publish_mod.argv_version(tmp_path, "some notes"),
    ):
        assert "--public" not in argv, argv
        assert "-u" not in argv, argv


def test_publishing_end_to_end_never_passes_the_public_flag(
    published_inputs, tmp_path, monkeypatch
):
    """The same check through the real entry point, not just the argv builders."""
    crops_dir, splits_path, meta_path = published_inputs
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    spy = RecordingRunner()

    for notes in (None, "a new version"):
        publish_mod.publish(
            crops_dir,
            splits_path,
            meta_path,
            tmp_path / "staging",
            update_notes=notes,
            runner=spy,
        )

    assert len(spy.calls) == 2
    for argv in spy.calls:
        assert "--public" not in argv and "-u" not in argv, argv


@pytest.mark.parametrize("flag", ["--public", "-u"])
def test_the_cli_refuses_to_accept_a_public_flag(flag, published_inputs, tmp_path, monkeypatch):
    """A flag that cannot be typed cannot be typed by accident.

    Asserting the string is absent from `--help` would be weaker and wrong — the docstring talks
    about `--public` on purpose. What matters is that passing it fails.
    """
    crops_dir, splits_path, meta_path = published_inputs
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    result = runner.invoke(
        app,
        [
            "publish-crops",
            "--crops-dir",
            str(crops_dir),
            "--splits",
            str(splits_path),
            "--meta",
            str(meta_path),
            "--staging",
            str(tmp_path / "staging"),
            "--dry-run",
            flag,
        ],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_an_update_never_deletes_old_versions(tmp_path):
    """Old versions are what let a run that recorded an earlier hash still be reproduced."""
    argv = publish_mod.argv_version(tmp_path, "notes")
    assert "--delete-old-versions" not in argv
    assert "-d" not in argv


def test_the_metadata_records_private_too(published_inputs, tmp_path):
    """Belt-and-braces, not the mechanism — but it should still say what we intend."""
    crops_dir, _, meta_path = published_inputs
    manifest = publish_mod.read_manifest(crops_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = publish_mod.build_metadata("someone", "a-slug", "A title", manifest, meta)
    assert metadata["isPrivate"] is True


# --- the Kaggle command line -------------------------------------------------------------------


def test_directories_are_uploaded_not_skipped(tmp_path):
    """Kaggle's default dir-mode is `skip`, which would upload the CSVs and none of the crops."""
    assert "--dir-mode" in publish_mod.argv_create(tmp_path)
    assert publish_mod.DIR_MODE == "zip"


def test_the_csvs_are_not_rewritten_on_the_way_in(tmp_path):
    """Kaggle converts tabular files by default; split_v1.csv carries a hash of its own contents."""
    assert "--keep-tabular" in publish_mod.argv_create(tmp_path)
    assert "--keep-tabular" in publish_mod.argv_version(tmp_path, "notes")


def test_version_carries_the_message(tmp_path):
    argv = publish_mod.argv_version(tmp_path, "remargined at 0.15")
    assert argv[1:3] == ["datasets", "version"]
    assert "--message" in argv and "remargined at 0.15" in argv


def test_create_is_the_create_subcommand(tmp_path):
    assert publish_mod.argv_create(tmp_path)[1:3] == ["datasets", "create"]


# --- credentials -------------------------------------------------------------------------------


def test_an_explicit_username_wins(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "from-env")
    assert publish_mod.resolve_username("explicit") == "explicit"


def test_the_environment_is_used_when_nothing_explicit(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "from-env")
    assert publish_mod.resolve_username() == "from-env"


def test_the_username_is_read_from_kaggle_json(tmp_path, monkeypatch):
    config = tmp_path / ".kaggle"
    config.mkdir()
    (config / "kaggle.json").write_text(json.dumps({"username": "from-file", "key": "secret"}))
    monkeypatch.setattr(publish_mod.Path, "home", lambda: tmp_path)
    assert publish_mod.resolve_username() == "from-file"


def test_kaggle_config_dir_overrides_home(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "kaggle.json").write_text(json.dumps({"username": "override"}))
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text(json.dumps({"username": "home-user"}))
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(elsewhere))
    monkeypatch.setattr(publish_mod.Path, "home", lambda: home)
    assert publish_mod.resolve_username() == "override"


def test_a_malformed_kaggle_json_is_skipped_not_fatal(tmp_path, monkeypatch):
    config = tmp_path / ".kaggle"
    config.mkdir()
    (config / "kaggle.json").write_text("{not json")
    monkeypatch.setattr(publish_mod.Path, "home", lambda: tmp_path)
    with pytest.raises(ValueError, match="No Kaggle username"):
        publish_mod.resolve_username()


def test_no_credentials_anywhere_names_both_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_mod.Path, "home", lambda: tmp_path)
    with pytest.raises(ValueError, match="KAGGLE_USERNAME"):
        publish_mod.resolve_username()


# --- the slug ------------------------------------------------------------------------------------


@pytest.mark.parametrize("slug", ["pcbi-solder-joint-crops", "abcdef", "a1-b2-c3"])
def test_valid_slugs_are_accepted(slug):
    publish_mod.check_slug(slug)


@pytest.mark.parametrize(
    "slug", ["abcde", "Has-Capitals", "under_scores", "double--hyphen", "-lead"]
)
def test_invalid_slugs_are_refused_before_anything_is_copied(slug):
    with pytest.raises(ValueError):
        publish_mod.check_slug(slug)


def test_the_slug_floor_matches_what_kaggle_actually_enforces():
    """kaggle 2.2.4 raises on a slug under 6 characters — but only after staging the whole upload.

    Guessing 3 here (as this first did) meant a five-character slug sailed through our check and
    failed on Kaggle's, 115MB later. The bound is read from the installed CLI, not invented.
    """
    assert (publish_mod.SLUG_MIN, publish_mod.SLUG_MAX) == (6, 50)
    with pytest.raises(ValueError, match="6-50 characters"):
        publish_mod.check_slug("abcde")


def test_the_generated_subtitle_stays_inside_kaggles_bounds(published_inputs, tmp_path):
    """Kaggle rejects a subtitle outside 20-80 characters, and ours is generated from a count."""
    crops_dir, _, meta_path = published_inputs
    manifest = publish_mod.read_manifest(crops_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    subtitle = publish_mod.build_metadata("someone", "a-slug-x", "A title", manifest, meta)[
        "subtitle"
    ]
    assert publish_mod.SUBTITLE_MIN <= len(subtitle) <= publish_mod.SUBTITLE_MAX


@pytest.mark.parametrize("title", ["short", "x" * 51])
def test_titles_kaggle_would_reject_are_refused_first(title, published_inputs, tmp_path):
    crops_dir, splits_path, meta_path = published_inputs
    with pytest.raises(ValueError, match="title must be"):
        publish_mod.publish(
            crops_dir, splits_path, meta_path, tmp_path / "staging", title=title, dry_run=True
        )


# --- staging -------------------------------------------------------------------------------------


def test_staging_copies_every_file_byte_for_byte(published_inputs, tmp_path):
    crops_dir, splits_path, meta_path = published_inputs
    staging = tmp_path / "staging"
    publish_mod.stage(crops_dir, splits_path, meta_path, staging, {"id": "a/b"})

    for source in crops_dir.glob("*.png"):
        assert digest(staging / "crops" / source.name) == digest(source)
    assert digest(staging / crops_mod.MANIFEST_NAME) == digest(crops_dir / crops_mod.MANIFEST_NAME)
    assert digest(staging / "split_v1.csv") == digest(splits_path)
    assert digest(staging / "manifest_meta.json") == digest(meta_path)
    assert (staging / publish_mod.METADATA_NAME).is_file()


def test_a_stale_crop_from_a_previous_run_is_cleared(published_inputs, tmp_path):
    """A leftover PNG would be uploaded beside a manifest that does not describe it."""
    crops_dir, splits_path, meta_path = published_inputs
    staging = tmp_path / "staging"
    publish_mod.stage(crops_dir, splits_path, meta_path, staging, {"id": "a/b"})
    stale = staging / "crops" / "CS9_LEFTOVER_1.png"
    stale.write_bytes(b"from an older margin")

    publish_mod.stage(crops_dir, splits_path, meta_path, staging, {"id": "a/b"})
    assert not stale.exists()


def test_a_foreign_non_empty_directory_is_refused_not_cleared(tmp_path, published_inputs):
    """A mistyped --staging should fail, not delete someone's documents."""
    crops_dir, splits_path, meta_path = published_inputs
    precious = tmp_path / "documents"
    precious.mkdir()
    (precious / "thesis.txt").write_text("years of work")

    with pytest.raises(ValueError, match="Refusing to clear it"):
        publish_mod.stage(crops_dir, splits_path, meta_path, precious, {"id": "a/b"})
    assert (precious / "thesis.txt").read_text() == "years of work"


def test_verify_catches_a_manifest_row_with_no_crop(published_inputs, tmp_path):
    crops_dir, splits_path, meta_path = published_inputs
    staging = tmp_path / "staging"
    publish_mod.stage(crops_dir, splits_path, meta_path, staging, {"id": "a/b"})
    manifest = publish_mod.read_manifest(crops_dir)
    next((staging / "crops").glob("*.png")).unlink()

    with pytest.raises(ValueError, match="have no crop"):
        publish_mod.verify_staged(staging, manifest)


def test_verify_catches_a_crop_with_no_manifest_row(published_inputs, tmp_path):
    crops_dir, splits_path, meta_path = published_inputs
    staging = tmp_path / "staging"
    publish_mod.stage(crops_dir, splits_path, meta_path, staging, {"id": "a/b"})
    (staging / "crops" / "CS9_UNKNOWN_1.png").write_bytes(b"x")

    with pytest.raises(ValueError, match="not in the manifest"):
        publish_mod.verify_staged(staging, publish_mod.read_manifest(crops_dir))


# --- the metadata --------------------------------------------------------------------------------


def test_the_metadata_names_the_dataset_and_carries_the_split_hash(published_inputs, tmp_path):
    """A run logs the split hash; the dataset page has to say which split it is, or there is no
    way to check the two agree."""
    crops_dir, _, meta_path = published_inputs
    manifest = publish_mod.read_manifest(crops_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = publish_mod.build_metadata("someone", "a-slug", "A title", manifest, meta)

    assert metadata["id"] == "someone/a-slug"
    assert metadata["title"] == "A title"
    assert metadata["licenses"] == [{"name": publish_mod.LICENSE}]
    assert "abc123" in metadata["description"]
    assert "normal 2" in metadata["description"]
    assert publish_mod.SOURCE_DATASET in metadata["description"]


# --- running the CLI tool ------------------------------------------------------------------------


def test_a_failing_kaggle_call_surfaces_its_stderr(tmp_path):
    """Swallowing the error would leave the user with no idea why nothing arrived."""
    spy = RecordingRunner(returncode=1, stdout="", stderr="401 Unauthorized")
    with pytest.raises(ValueError, match="401 Unauthorized"):
        publish_mod.run_kaggle(publish_mod.argv_create(tmp_path), runner=spy)


def test_a_successful_call_returns_its_output(tmp_path):
    spy = RecordingRunner(stdout="Your dataset is being created")
    assert "being created" in publish_mod.run_kaggle(publish_mod.argv_create(tmp_path), runner=spy)


def test_a_dry_run_stages_everything_but_calls_nothing(published_inputs, tmp_path, monkeypatch):
    crops_dir, splits_path, meta_path = published_inputs
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    staging = tmp_path / "staging"
    spy = RecordingRunner()

    report = publish_mod.publish(
        crops_dir, splits_path, meta_path, staging, dry_run=True, runner=spy
    )

    assert spy.calls == []
    assert report.dry_run is True
    assert (staging / publish_mod.METADATA_NAME).is_file()
    assert len(list((staging / "crops").glob("*.png"))) == 4


# --- the CLI --------------------------------------------------------------------------------------


def run_publish(**options):
    args = ["publish-crops"]
    for name, value in options.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return runner.invoke(app, args)


def test_cli_dry_run_reports_the_command_it_would_have_run(published_inputs, tmp_path, monkeypatch):
    crops_dir, splits_path, meta_path = published_inputs
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    result = run_publish(
        crops_dir=crops_dir,
        splits=splits_path,
        meta=meta_path,
        staging=tmp_path / "staging",
        slug="a-test-slug",
        dry_run=True,
    )
    assert result.exit_code == 0, result.output
    assert "someone/a-test-slug (private)" in result.output
    assert "--dry-run: nothing was uploaded" in result.output
    assert "--public" not in result.output


def test_cli_rejects_a_bad_slug(published_inputs, tmp_path):
    crops_dir, splits_path, meta_path = published_inputs
    result = run_publish(
        crops_dir=crops_dir, splits=splits_path, meta=meta_path, slug="Not A Slug", dry_run=True
    )
    assert result.exit_code == 2, result.output
    assert "slug" in result.output


def test_cli_reports_missing_inputs(published_inputs, tmp_path, monkeypatch):
    crops_dir, splits_path, meta_path = published_inputs
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")

    missing_crops = run_publish(
        crops_dir=tmp_path / "nowhere", splits=splits_path, meta=meta_path, dry_run=True
    )
    assert missing_crops.exit_code == 1, missing_crops.output
    assert "No such crops folder" in missing_crops.output

    missing_split = run_publish(
        crops_dir=crops_dir, splits=tmp_path / "no.csv", meta=meta_path, dry_run=True
    )
    assert missing_split.exit_code == 1, missing_split.output
    assert "No such file" in missing_split.output


def test_cli_reports_missing_credentials(published_inputs, tmp_path, monkeypatch):
    """Failing here, before the copy, beats failing after 111MB has been staged."""
    crops_dir, splits_path, meta_path = published_inputs
    monkeypatch.setattr(publish_mod.Path, "home", lambda: tmp_path / "empty-home")
    result = run_publish(
        crops_dir=crops_dir,
        splits=splits_path,
        meta=meta_path,
        staging=tmp_path / "staging",
        dry_run=True,
    )
    assert result.exit_code == 1, result.output
    assert "No Kaggle username" in result.output


# --- the real crops ------------------------------------------------------------------------------

REAL_CROPS = Path("data/crops")
REAL_SPLITS = Path("data/splits/split_v1.csv")
REAL_META = Path("data/manifest_meta.json")


@pytest.mark.skipif(
    not (REAL_CROPS.is_dir() and REAL_SPLITS.is_file() and REAL_META.is_file()),
    reason="requires the generated crops (gitignored)",
)
def test_real_crops_stage_cleanly_and_privately(tmp_path, monkeypatch):
    """Stages the real 400 crops and checks the argv — still without uploading anything."""
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    spy = RecordingRunner()
    report = publish_mod.publish(
        REAL_CROPS, REAL_SPLITS, REAL_META, tmp_path / "staging", dry_run=True, runner=spy
    )

    assert spy.calls == []
    assert report.files == 404  # 400 crops + manifest + split + meta + dataset-metadata
    assert report.dataset_id == f"someone/{publish_mod.DEFAULT_SLUG}"
    assert "--public" not in report.argv and "-u" not in report.argv
