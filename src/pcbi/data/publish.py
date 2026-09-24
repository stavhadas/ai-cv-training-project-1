"""Stage the crops and hand them to the Kaggle CLI as a **private** dataset.

The project's compute is split: the laptop writes code and measures CPU latency, Kaggle trains on
its T4s. Code gets to Kaggle by `git clone` and the raw SolDef_AI photos arrive as a mounted
third-party dataset, but the crops get there by neither — `data/crops/` is gitignored because it is
re-derivable. Re-deriving it on every Kaggle session means running the whole pipeline before a
single training step, and puts the exact pixels a run trains on at the mercy of that re-execution.
Uploading them once, versioned, makes the training input a fixed artifact a notebook mounts.

**How the dataset stays private.** Visibility is decided by exactly one thing: whether `--public`
(`-u`) appears in the Kaggle argv. `kaggle datasets create` is private by default and that flag opts
in; `kaggle datasets version` has no such flag at all, so an update can never change visibility. So
`argv_create` and `argv_version` are pure functions, a test asserts neither flag appears on any
path, and `pcbi publish-crops` deliberately has no `--public` option. Making this dataset public has
to be a deliberate act on the Kaggle website.

`dataset-metadata.json` also carries `isPrivate: true`. That is a record of intent, **not** the
mechanism, and this was checked rather than assumed: in the installed kaggle 2.2.4,
`dataset_create_new` sets `request.is_private = not public` straight from the flag and never reads
`isPrivate` from the metadata at all — that key is only honoured on the *models* API. So do not read
the JSON and conclude the upload is private; read the argv. If a future version starts honouring the
key, `isPrivate: true` already agrees with the flag, which is why it is worth keeping.

**Why private is the right answer, not just the requested one.** These crops are a derivative of
someone else's Kaggle dataset (SolDef_AI, by `mauriziocalabrese`). Private keeps redistribution a
non-question: it is a personal copy of derived data, visible to one account. Before ever flipping it
public, check SolDef_AI's licence and change `LICENSE` below from `other` to whatever that licence
actually permits. If you are the person about to flip that switch, this paragraph is for you.

`notes/s1_publishing_crops.md` records the rest: why `--keep-tabular` and the absence of
`--delete-old-versions` both matter more than they look, and the field-length limits read out of the
installed CLI rather than guessed.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pcbi.data import crops as crops_mod

DEFAULT_CROPS_DIR = Path("data/crops")
DEFAULT_SPLITS = Path("data/splits/split_v1.csv")
DEFAULT_META = Path("data/manifest_meta.json")
DEFAULT_STAGING = Path("data/kaggle/crops")
DEFAULT_SLUG = "pcbi-solder-joint-crops"
DEFAULT_TITLE = "PCB solder-joint crops (pcbi)"

METADATA_NAME = "dataset-metadata.json"
CROPS_SUBDIR = "crops"
KAGGLE_EXE = "kaggle"

# Derivative of a third-party dataset — see the module docstring before changing this.
LICENSE = "other"

# Read out of the installed CLI rather than guessed, because each of these is enforced *after* the
# upload folder has been assembled — kaggle_api_extended.dataset_create_new raises on a 5-character
# slug only once 115MB has already been staged. Checking them first turns a late failure into an
# instant one. (kaggle 2.2.4: slug and title 6-50, subtitle 20-80 when present.)
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SLUG_MIN, SLUG_MAX = 6, 50
TITLE_MIN, TITLE_MAX = 6, 50
SUBTITLE_MIN, SUBTITLE_MAX = 20, 80

# `skip` (the CLI default) silently ignores directories, which would upload the CSVs and none of the
# 400 PNGs. `zip` is what actually sends the crops/ folder.
DIR_MODE = "zip"

SOURCE_DATASET = "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection"


@dataclass(frozen=True)
class Report:
    """What was staged and what was handed to Kaggle."""

    staging: Path
    dataset_id: str
    files: int
    total_bytes: int
    argv: list[str]
    dry_run: bool
    output: str


def kaggle_config_candidates() -> list[Path]:
    """Where a `kaggle.json` may live, most explicit first.

    Same shape as `smoke.netrc_candidates`: an environment variable overrides, then the conventional
    home location. Neither is inside the repo, which is the point.
    """
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    paths = [Path(configured) / "kaggle.json"] if configured else []
    return [*paths, Path.home() / ".kaggle" / "kaggle.json"]


def resolve_username(explicit: str | None = None) -> str:
    """The Kaggle username the dataset will belong to.

    Reads only the username out of `kaggle.json`; the API key in the same file is never read here
    and never printed. The Kaggle CLI picks the key up for itself from the same places.
    """
    if explicit:
        return explicit
    from_env = os.environ.get("KAGGLE_USERNAME")
    if from_env:
        return from_env

    for path in kaggle_config_candidates():
        if not path.is_file():
            continue
        try:
            username = json.loads(path.read_text(encoding="utf-8")).get("username")
        except (OSError, json.JSONDecodeError):
            continue  # unreadable or malformed: treat as "no username here" and keep looking
        if username:
            return str(username)

    # kaggle 2.2.4 tries an access token (KAGGLE_API_TOKEN, ~/.kaggle/access_token, `kaggle auth
    # login`) before falling back to kaggle.json. Someone authenticated that way has a working CLI
    # and no username on disk for us to read, so --username is the documented answer rather than a
    # sign that anything is broken.
    raise ValueError(
        "No Kaggle username found. Pass --username, or set KAGGLE_USERNAME, or put kaggle.json "
        f"in {Path.home() / '.kaggle'} (kaggle.com -> Settings -> API -> Create New Token). "
        "If you signed in with `kaggle auth login`, the CLI is authenticated but stores no "
        "username here — pass --username."
    )


def check_slug(slug: str) -> None:
    """Refuse a slug Kaggle would reject, before anything is copied."""
    if not SLUG_MIN <= len(slug) <= SLUG_MAX:
        raise ValueError(f"slug must be {SLUG_MIN}-{SLUG_MAX} characters, got {len(slug)}.")
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"slug {slug!r} is not a valid Kaggle slug: lowercase letters, digits and single "
            f"hyphens only."
        )


def check_title(title: str) -> None:
    if not TITLE_MIN <= len(title) <= TITLE_MAX:
        raise ValueError(f"title must be {TITLE_MIN}-{TITLE_MAX} characters, got {len(title)}.")


def check_subtitle(subtitle: str) -> None:
    """Kaggle rejects a subtitle outside 20-80 characters, and ours is generated.

    Generated text is exactly the kind that drifts out of range without anyone noticing, so it is
    checked rather than trusted.
    """
    if not SUBTITLE_MIN <= len(subtitle) <= SUBTITLE_MAX:
        raise ValueError(
            f"subtitle must be {SUBTITLE_MIN}-{SUBTITLE_MAX} characters, got {len(subtitle)}: "
            f"{subtitle!r}"
        )


def read_manifest(crops_dir: Path) -> list[dict]:
    path = crops_dir / crops_mod.MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"No such manifest: {path}. Run `pcbi make-crops` first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def describe(manifest: Sequence[dict], meta: dict) -> str:
    """The dataset page's description: what is in here, and which split it belongs to.

    The split hash is the load-bearing line. A training run logs that hash; without it on the
    dataset itself there is no way to check that the two refer to the same partition.
    """
    classes: defaultdict[str, int] = defaultdict(int)
    splits: defaultdict[str, int] = defaultdict(int)
    for row in manifest:
        classes[row["label"]] += 1
        splits[row["split"]] += 1

    lines = [
        f"{len(manifest)} solder-joint crops cut from the SolDef_AI dataset by `pcbi make-crops`, ",
        "one lossless PNG per joint, named by its `crop_id`.",
        "",
        "**Classes:** "
        + ", ".join(f"{k} {v}" for k, v in sorted(classes.items(), key=lambda kv: -kv[1])),
        "**Splits:** "
        + ", ".join(f"{k} {splits[k]}" for k in ("train", "val", "test") if k in splits),
        "",
        f"**Split hash:** `{meta.get('split_hash', 'unknown')}`",
        "",
        "The split is grouped: every crop of one physical component lands on the same side, so no",
        "joint appears in both train and test. `manifest.csv` describes every crop; `split_v1.csv`",
        "and `manifest_meta.json` carry the frozen partition and how it was chosen.",
        "",
        f"Derived from the SolDef_AI dataset (`{SOURCE_DATASET}`). Private: this is a personal",
        "working copy of derived data, not a redistribution.",
    ]
    return "\n".join(lines)


def build_metadata(
    username: str, slug: str, title: str, manifest: Sequence[dict], meta: dict
) -> dict:
    """The `dataset-metadata.json` body Kaggle reads from the staging folder.

    `isPrivate` is recorded for the record's sake. It is not what keeps the dataset private — the
    absence of `--public` in the argv is. See the module docstring.
    """
    subtitle = f"{len(manifest)} solder-joint crops with a frozen train/val/test split"
    check_subtitle(subtitle)
    return {
        "id": f"{username}/{slug}",
        "title": title,
        "isPrivate": True,
        "licenses": [{"name": LICENSE}],
        "subtitle": subtitle,
        "description": describe(manifest, meta),
    }


def looks_like_our_staging(staging: Path) -> bool:
    """Whether this directory is one we previously wrote, and so may be cleared."""
    return (staging / METADATA_NAME).is_file()


def stage(
    crops_dir: Path,
    splits_path: Path,
    meta_path: Path,
    staging: Path,
    metadata: dict,
) -> list[Path]:
    """Copy everything Kaggle should receive into `staging`, and return what was written.

    The folder is cleared first. A crop left over from a previous margin would otherwise be uploaded
    beside a manifest that does not describe it, which is the one kind of staleness that actually
    corrupts a dataset — nothing else in this project clears an output directory.

    Because clearing is destructive, a staging path that already exists, holds files, and is *not*
    one of ours is refused rather than emptied. A mistyped `--staging` should fail, not delete.
    """
    if staging.exists() and any(staging.iterdir()) and not looks_like_our_staging(staging):
        raise ValueError(
            f"{staging} is not empty and does not look like a staging folder "
            f"(no {METADATA_NAME}). Refusing to clear it — point --staging somewhere else."
        )
    if staging.exists():
        shutil.rmtree(staging)

    crop_target = staging / CROPS_SUBDIR
    crop_target.mkdir(parents=True)

    written: list[Path] = []
    for source in sorted(crops_dir.glob("*.png")):
        destination = crop_target / source.name
        shutil.copy2(source, destination)
        written.append(destination)

    for source in (crops_dir / crops_mod.MANIFEST_NAME, splits_path, meta_path):
        destination = staging / source.name
        shutil.copy2(source, destination)
        written.append(destination)

    metadata_path = staging / METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    written.append(metadata_path)
    return written


def verify_staged(staging: Path, manifest: Sequence[dict]) -> None:
    """Every manifest row has its PNG and every PNG has its row.

    Cheap here, expensive on Kaggle: a mismatch discovered after upload means a new version and a
    training run that already read the wrong thing.
    """
    on_disk = {path.stem for path in (staging / CROPS_SUBDIR).glob("*.png")}
    expected = {row["crop_id"] for row in manifest}

    missing = sorted(expected - on_disk)
    extra = sorted(on_disk - expected)
    if missing:
        raise ValueError(
            f"{len(missing)} manifest row(s) have no crop, e.g. {missing[0]}. "
            f"Re-run `pcbi make-crops`."
        )
    if extra:
        raise ValueError(
            f"{len(extra)} crop(s) are not in the manifest, e.g. {extra[0]}. "
            f"The crops folder holds files from an earlier run; re-run `pcbi make-crops`."
        )


def argv_create(staging: Path) -> list[str]:
    """`kaggle datasets create` for a new dataset.

    No `--public`/`-u`: the CLI creates private by default and this is the only thing that decides
    it. `--keep-tabular` stops Kaggle rewriting the CSVs on the way in, which matters because
    `split_v1.csv` carries a hash of its own contents.
    """
    return [
        KAGGLE_EXE,
        "datasets",
        "create",
        "--path",
        str(staging),
        "--dir-mode",
        DIR_MODE,
        "--keep-tabular",
    ]


def argv_version(staging: Path, notes: str) -> list[str]:
    """`kaggle datasets version` for a dataset that already exists.

    `datasets version` has no visibility flag at all, so an update can never make a private dataset
    public. `--delete-old-versions` is deliberately not passed: old versions are what let a training
    run that recorded an old hash still be reproduced.
    """
    return [
        KAGGLE_EXE,
        "datasets",
        "version",
        "--path",
        str(staging),
        "--message",
        notes,
        "--dir-mode",
        DIR_MODE,
        "--keep-tabular",
    ]


def default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def run_kaggle(
    argv: list[str],
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> str:
    """Invoke the Kaggle CLI. The one impure function here.

    `runner` is injected for the same reason `group.embed_all` takes its embedder: so every other
    function in this module can be exercised without a network, an account, or an upload. It is not
    a CLI option — production always gets the real one.
    """
    run = runner or default_runner
    if runner is None and shutil.which(KAGGLE_EXE) is None:
        raise ValueError(
            f"`{KAGGLE_EXE}` is not on PATH. It ships in this project's dev dependencies — "
            f"run `uv sync`, or install it with `pip install kaggle`."
        )

    completed = run(argv)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ValueError(f"kaggle exited {completed.returncode}: {detail}")
    return (completed.stdout or "").strip()


def publish(
    crops_dir: Path = DEFAULT_CROPS_DIR,
    splits_path: Path = DEFAULT_SPLITS,
    meta_path: Path = DEFAULT_META,
    staging: Path = DEFAULT_STAGING,
    slug: str = DEFAULT_SLUG,
    title: str = DEFAULT_TITLE,
    username: str | None = None,
    update_notes: str | None = None,
    dry_run: bool = False,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> Report:
    """Stage the crops and upload them as a private Kaggle dataset.

    `update_notes` switches from creating a dataset to adding a version to the existing one.
    `dry_run` stages everything and reports the argv without calling Kaggle.
    """
    check_slug(slug)
    check_title(title)
    if not crops_dir.is_dir():
        raise ValueError(f"No such crops folder: {crops_dir}. Run `pcbi make-crops` first.")
    for path in (splits_path, meta_path):
        if not path.is_file():
            raise ValueError(f"No such file: {path}. Run `pcbi split` first.")

    manifest = read_manifest(crops_dir)
    if not manifest:
        raise ValueError(f"{crops_dir / crops_mod.MANIFEST_NAME} holds no crops.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = resolve_username(username)

    metadata = build_metadata(resolved, slug, title, manifest, meta)
    written = stage(crops_dir, splits_path, meta_path, staging, metadata)
    verify_staged(staging, manifest)

    argv = argv_create(staging) if update_notes is None else argv_version(staging, update_notes)
    output = "" if dry_run else run_kaggle(argv, runner)

    return Report(
        staging=staging,
        dataset_id=metadata["id"],
        files=len(written),
        total_bytes=sum(path.stat().st_size for path in written),
        argv=argv,
        dry_run=dry_run,
        output=output,
    )
