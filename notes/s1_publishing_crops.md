# Stage 1 — publishing the crops to Kaggle

Decisions and findings behind `pcbi publish-crops` (`src/pcbi/data/publish.py`). Written because
three of them are the kind that only announce themselves as a wrong number much later.

## Why the crops go up at all

`README.md` argues the crops should *not* be committed: 111MB of PNG, every byte re-derivable from
the download plus `--margin`. That still holds for git. It does not hold for Kaggle, where
re-deriving means running ingest, group, split and make-crops before every training session — slow,
and it puts the exact pixels a run trains on at the mercy of that re-execution. So they are uploaded
once, versioned, and mounted as a fixed artifact. Two different questions, two different answers.

## The dataset is private, and the flag is the only thing that makes it so

Verified against the installed CLI rather than assumed. In kaggle 2.2.4,
`kaggle_api_extended.dataset_create_new` does:

```python
request.is_private = not public
```

where `public` comes straight from `-u/--public`. **It never reads `isPrivate` from
`dataset-metadata.json`** — that key is only honoured on the *models* API
(`model_create_new`, around line 7208). `kaggle datasets version` has no visibility flag at all, so
an update cannot change it either.

So the guarantee is: the argv never contains `--public` or `-u`. `argv_create` and `argv_version`
are pure functions, `tests/test_publish.py` asserts the absence on every path, and
`pcbi publish-crops` has no such option to type. `isPrivate: true` is written into the metadata as a
record of intent and as insurance if a future version starts honouring it — **not** as the
mechanism. Anyone auditing this should read the argv, not the JSON.

Private is also the correct answer, not merely the requested one: these crops are a derivative of
someone else's dataset (`mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection`). A private
personal copy of derived data is one thing; a public redistribution is another. Check SolDef_AI's
licence and fix `publish.LICENSE` (currently `other`) before ever flipping that switch.

## `--keep-tabular`, and why it is not a detail

Kaggle **converts tabular files on upload by default**. `-t/--keep-tabular` opts out.

Left at the default, Kaggle would have been free to rewrite `split_v1.csv` and `manifest.csv` on the
way in. `split_v1.csv` carries `split_hash` in every row — a SHA-256 over the sorted
`(crop_id, split)` pairs — and the notebook checks that hash against `manifest_meta.json` on every
session. A silent re-encode would have produced a mismatch *on Kaggle*, pointing at the split logic,
which is exactly the wrong place to look. The bug would have been in the upload.

This is worth recording because it is a concrete instance of the class of drift the hash exists to
catch, and it was caught by reading `kaggle datasets create --help` rather than by the hash firing.
The hash is the backstop; not needing it is better.

## `--delete-old-versions` is deliberately never passed

Same idea one level up. `kaggle datasets version` accepts `-d/--delete-old-versions`. Passing it
would make every earlier version unreachable — and an earlier version is precisely what a training
run that recorded an earlier `split_hash` needs in order to be reproduced. Deleting old versions
turns a recorded hash from a way of finding the data into a way of proving you cannot.

A test asserts the flag's absence, for the same reason the `--public` test exists: the cost of the
mistake is much larger than the cost of the assertion.

## Constraints read out of the CLI, not guessed

`dataset_create_new` validates *after* assembling the upload folder, so a bad slug fails once
115MB has already been staged. These are checked up front instead
(`publish.check_slug` / `check_title` / `check_subtitle`):

| Field | kaggle 2.2.4 | Note |
| :--- | :--- | :--- |
| slug | 6–50 chars | First written as 3–50 from memory. A 5-character slug passed our check and would have failed Kaggle's. |
| title | 6–50 chars | Was not validated at all. |
| subtitle | 20–80 chars, if present | Ours is generated from the crop count, so it can drift without anyone editing it. |

## Credentials

kaggle 2.2.4 tries an access token first (`KAGGLE_API_TOKEN`, `~/.kaggle/access_token`,
`kaggle auth login`) and falls back to `kaggle.json` / `KAGGLE_USERNAME`. `publish.resolve_username`
reads only the *username*, never the key.

Consequence worth knowing: someone authenticated by OAuth has a perfectly working CLI and **no
username on disk for us to read**. That is not a broken setup — pass `--username`. The error message
says so.

## The first real upload — what actually happened

Uploaded as `stavhadas/pcbi-solder-joint-crops` and verified from a Kaggle session by Cell 5 of
`notebooks/kaggle_train.ipynb`:

```
dataset: /kaggle/input/datasets/stavhadas/pcbi-solder-joint-crops
400 crops · 400 manifest rows · crops at /kaggle/input/datasets/stavhadas/pcbi-solder-joint-crops/crops
split hash: 7925ce616931ad52c69a46e94b69e113fc9713d4bb15c5f9163ba0354e4c8b0c
```

- **The mount path is `/kaggle/input/datasets/<owner>/<slug>/`**, not `/kaggle/input/<slug>/`.
  Cell 5 first globbed one level deep and failed on a dataset that was correctly attached. Cell 4's
  own saved output had shown the three-level form all along. It now searches depths 1-4 by
  filename, which covers both layouts, and is bounded rather than `rglob` because SolDef_AI is
  mounted alongside and walking it every session would cost seconds.
- **`--dir-mode zip` arrived expanded.** `crops/` is a browsable folder of 400 PNGs, not an
  archive. The unpack fallback in Cell 5 has therefore never run. Keep it until a second upload
  confirms the behaviour is stable, then delete it rather than carry untested code.
- **The split hash on Kaggle matches the local one exactly**, so the uploaded crops, the uploaded
  `split_v1.csv` and `data/manifest_meta.json` all describe the same partition. `--keep-tabular`
  did its job: the CSVs came through unrewritten.
- Manifest and PNGs agree: 400 rows, 400 files, `crop_id` set equal to the file-stem set.

### Still to record

- [ ] Dataset page reads **Private**. The argv could not have asked for anything else, but this is
      a human eyeball check on what Kaggle did, and nothing above substitutes for it.
- [ ] **Dataset version number**, so a future run can be pinned to it.
