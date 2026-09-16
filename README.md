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

Dependencies are intentionally minimal. `wandb` is here because Stage 0 logs to it; `torch` and
`timm` arrive in Stage 2, when there is training code that needs them.

## Experiment tracking

Runs log to the W&B project `pcb-inspector`. Authenticate once per machine:

```bash
uv run wandb login          # laptop: the key lands in ~/.netrc, never in the repo
uv run pcbi smoke           # 20 fake steps; proves the plumbing before a real run depends on it
```

On Kaggle the key comes from a Secret named `WANDB_API_KEY`. With no key anywhere, `pcbi smoke`
falls back to offline mode and writes to `wandb/`, which you can upload later with `wandb sync`.

## Stage 0 progress

- [x] Step 1 — repository scaffold
- [x] Step 2 — Kaggle notebook wrapper
- [ ] Step 3 — W&B smoke test
- [ ] Step 4 — `pcbi audit` dataset inventory
- [ ] Step 5 — laptop hardware profile
- [ ] Step 6 — interpret the audit, fill the metadata table
- [ ] Step 7 — record the compute setup
