# pcbi — PCB Solder-Joint Inspector

A bench-top visual-inspection tool for PCB assembly, and the backbone benchmark that decides which
model it ships with.

Given an image of an assembled board plus the known locations of its solder-joint sites, `pcbi`
crops each joint, classifies it as **Normal / Excess / Insufficient / Spike**, makes a pass/fail call
per site, and writes an overlay image marking the failures.

**Hard constraint:** a full board (300 sites assumed) must finish in **under 2 s on CPU only**. No
discrete GPU is available at the factory bench station. That constraint is the point of the project —
it turns "which backbone?" into a product decision that has to be defended with measurements.

Project 1 of Phase 1 of a computer-vision roadmap, with an edge-AI / on-device focus.

## Quickstart

Requires [uv](https://docs.astral.sh/uv).

```bash
uv sync                  # create the environment from uv.lock
uv run pcbi --help       # list available commands
uv run pytest            # run the tests
uv run ruff check .      # lint
```

## Layout

```
configs/                 YAML experiment settings (from Stage 1)
src/pcbi/
  cli.py                 the `pcbi` command and its subcommands
  data/                  audit, crops, splits, datasets
  train/                 training loop, optimizer, EMA, checkpoints
  eval/                  metrics, calibration, error slices
  bench/                 latency and memory measurement
  inspect/               forward hooks and from-scratch attention
  product/               the board-overlay tool
notebooks/               thin Kaggle launchers — real code never lives here
tests/
reports/                 generated results (committed)
notes/                   decisions and reflection answers
```

Code lives in **one** place, this repository. The laptop writes code and measures CPU latency,
GitHub Actions checks every push, Kaggle clones the repo and trains on its GPUs, and W&B collects
results from wherever a run happened. Notebooks are launchers only.

Dependencies are intentionally minimal. `wandb` is here because Stage 0 logs to it; `imagehash`,
`torch` and `timm` arrive in Stage 1 for `pcbi group`, which compares images to guess which ones
show the same physical component, and `scikit-learn` for `pcbi split`, whose `StratifiedGroupKFold`
keeps every photograph of one component on the same side of the train/test line. `torch` is pinned
to the **CPU-only** wheels (see
`[[tool.uv.index]]` in `pyproject.toml`) so CI doesn't pull ~3GB of CUDA libraries on every push —
Stage 2's Kaggle GPU training will need its own torch handling.

## Experiment tracking

Runs log to the W&B project `pcb-inspector`. Authenticate once per machine:

```bash
uv run wandb login          # laptop: the key lands in ~/.netrc, never in the repo
uv run pcbi smoke           # 20 fake steps; proves the plumbing before a real run depends on it
```

On Kaggle the key comes from a Secret named `WANDB_API_KEY`. With no key anywhere, `pcbi smoke`
falls back to offline mode and writes to `wandb/`, which you can upload later with `wandb sync`.
## Compute

### Kaggle

2x Tesla T4 (15360MiB each), driver 580.159.04, CUDA 13.0.

<details>
<summary><code>nvidia-smi</code> output</summary>

```
Wed Sep 16 09:38:10 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   38C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
| N/A   44C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

</details>

Tooling: `torch 2.10.0+cu128` · `timm 1.0.26` · 2 GPUs visible.

### Laptop CPU

| | |
|---|---|
| Model | 13th Gen Intel(R) Core(TM) i7-1360P |
| Physical cores | 12 |
| Logical processors | 16 |

## Stage 0 progress

- [x] Step 1 — repository scaffold
- [x] Step 2 — Kaggle notebook wrapper
- [x] Step 3 — W&B smoke test
- [x] Step 4 — `pcbi audit` dataset inventory
- [x] Step 5 — laptop hardware profile
- [x] Step 6 — interpret the audit, fill the metadata table
- [x] Step 7 — record the compute setup

## Stage 1 progress

- [x] Step 1 — `pcbi ingest`, one CSV row per solder joint
- [x] Step 4 — `pcbi group`, candidate groupings of "same physical component"
- [x] Step 4b — the manual tagger, after every automatic method fell short
- [x] Step 5 — `pcbi split`, the frozen train/val/test partition
- [x] Step 7 — `pcbi make-crops`, the crops a model trains on and their manifest

The split is frozen and fingerprinted. Regenerate it with:

```bash
uv run pcbi split --groups data/interim/groups_manual.csv \
                  --ratios 0.6 0.2 0.2 --candidates 200 --split-seed 0
```

It writes three things: the assignment to `data/splits/split_v1.csv`, its hash and provenance to
`data/manifest_meta.json`, and a count table into section 13 of `reports/data_audit.md` (replaced
in place on re-run). All three are committed — a split that lives only on the laptop that made it
is not frozen, and Kaggle clones this repo to train.

Every training run records the **split hash**, a SHA-256 of the sorted `(crop ID, split)` pairs.
Two runs reporting different hashes did not train on the same data. The hash covers the pairs
alone, not the manifest, so adding a column to the CSV leaves it unchanged.

The crops themselves come from:

```bash
uv run pcbi make-crops --margin 0.1
```

One lossless PNG per solder joint into `data/crops/`, named by its `crop_id`, plus
`data/crops/manifest.csv` joining geometry, folder metadata, label and split into one table, plus
per-class contact sheets in `reports/qa/crops_<class>.png` built from **train crops only** —
deciding what a class looks like while looking at val or test is how a split leaks through a
person. Regenerating at the same margin gives byte-identical files, so the pixels behind the split
hash cannot drift.

Unlike the split, the crops are **not** committed: ~400 PNGs is a few hundred MB and every byte is
re-derivable from the download plus the margin. `--margin` keeps a fraction of the joint box's own
width on the left and right and of its height on the top and bottom, so the crop holds the box's
aspect ratio. (`pcbi group --crop-tolerance` grows by `max(w, h)` on all four sides instead; that
one feeds a fingerprint, not a classifier.)

### Getting the crops to Kaggle

Git is the wrong pipe for 111MB of PNGs, and re-deriving them on Kaggle would mean running the
whole pipeline before every training session — which also puts the exact pixels a run trains on at
the mercy of that re-execution. So they go up once, as a versioned Kaggle dataset:

```bash
uv run pcbi publish-crops --dry-run     # stage and print the command, upload nothing
uv run pcbi publish-crops               # actually create it
uv run pcbi publish-crops --update "remargined at 0.15"   # add a version to an existing one
```

Authenticate the way the Kaggle CLI expects — `~/.kaggle/kaggle.json` (Account → Create New Token)
or `KAGGLE_USERNAME` / `KAGGLE_KEY`. Neither lives in the repo, same as the W&B key. The `kaggle`
CLI is a dev dependency, so `uv sync` provides it.

**The dataset is private, and there is no flag to make it public.** `kaggle datasets create` is
private unless `--public` is passed, `pcbi publish-crops` never passes it, and a test asserts that
for every code path. This matters beyond the preference: the crops are a derivative of a
third-party dataset (SolDef_AI), so a private personal copy is one thing and a public
redistribution is another. Check that licence before ever flipping the switch on kaggle.com.

Cell 5 of `notebooks/kaggle_train.ipynb` checks the mounted dataset every session: 400 PNGs, the
manifest agreeing with them, and the split hash matching `manifest_meta.json`. It also unpacks the
crops archive itself if Kaggle has not. `notes/s1_publishing_crops.md` records the decisions behind
all of this — in particular why `--keep-tabular` is load-bearing rather than cosmetic.
