# Stage 1 — merging the 43 three-polygon (double-defect) images

## Observation

- 43 of 200 joint images carry 3 polygons; the other 157 carry 2. Every component has two leads,
  so each image shows two physical joints.
- In most 3-polygon images, the extra polygon is a `spike` that overlaps another polygon on the
  same joint. The two polygons mark two defects annotated on one joint. This is the common case,
  not a rule the code depends on — 2 of the 43 look different (see *Notes from the real
  download*), and the rule handles them the same way regardless.
- The overlapped polygon can be any class: `normal`, `insufficient`, or `excess`.
- `spike` also appears in 2-polygon images as the only label on a joint, so `spike` isn't
  exclusive to the double-annotated case.
- Class counts before the rule (polygons): `excess` 182, `spike` 130, `normal` 74,
  `insufficient` 57. Total 443.

## Why a rule is needed

The model is a 4-class classifier: one crop per joint, one label per crop. A joint annotated with
two defects is multi-label, which this setup can't represent. Keeping both polygons would create
two near-identical crops of the same joint with contradictory labels, and if they landed in
different splits, that would also be leakage. Each double-annotated joint must therefore become
exactly one crop with one label.

## Rule

**Implemented in `pcbi ingest` (`src/pcbi/data/ingest.py`).** Nothing below assumes which classes
are involved or that one of them is `spike` — the rule covers any combination of classes on a
joint, which is what lets it handle every case with no manual review.

1. **Group each image's polygons into physical joints by x-position.** Every component here has
   exactly two leads, so two polygons on the same joint sit close together in x, while polygons on
   different joints are separated by the much larger gap of the component body between them. Sort
   all of an image's polygons by bounding-box center x, find the single largest gap between
   consecutive centers, and split there — always into (at most) two groups. This needs no overlap
   at all (an earlier, overlap-based definition of "same joint" couldn't handle two `spike`
   polygons drawn side by side with no overlap — see *Notes from the real download*), so it also
   catches same-joint polygons that just happen to sit apart.
2. **Within a joint group of more than one polygon, reduce to one row in two phases:**
   - *Same class, same joint:* keep one of the polygons at random and drop the rest — the rule
     doesn't prefer either, so there's nothing to break the tie with.
   - *Different classes, same joint:* keep one label by precedence —
     **insufficient > spike > excess > normal** (a defect always beats normal; among defects,
     insufficient beats spike, and spike beats excess).
3. **The crop box for the merged joint is the union of every merged polygon's bounding box**,
   whatever label is kept.

Real pair-type counts from the download are in *Counts* below.

## Rationale

- **A defect beats normal.** A joint with a spike fails inspection. Labeling it normal would teach
  the model that visible defects can pass, which is the costliest product error. It would also add
  label noise exactly on the normal-vs-defect boundary that the operating point depends on.
- **The order among defects only affects the subtype; the site fails either way.** Precedence
  favors the smaller classes (insufficient 57 < spike 130 < excess 182) to protect them from
  shrinking further.
- **One written precedence** instead of case-by-case choices keeps the rule reproducible and
  covers any combination.

## Implementation checks

- Position-based grouping (not overlap) means every image structurally ends up with exactly two
  joint groups whenever it has ≥ 2 polygons, and the two-phase reduction (same-class, then
  precedence) resolves any combination of classes it finds there. There's no case left that needs
  manual review — `pcbi ingest` always completes and writes a full manifest in one pass.
- After the rule, every image has exactly 2 joints, so `joint_position` = left/right by
  bounding-box center x-order (`polygon_rank_x`/`polygon_count` are recomputed over the merged
  rows too, not left over from the raw per-polygon count).
- The manifest gets a `merged_from` column (for example `spike+excess`; empty for unmerged
  joints), so double-defect joints can be sliced in Stage 8.
- The same-class tie-break is random (`random.choice`), not deterministic — rerunning `pcbi
  ingest` can keep a different one of two same-class duplicates each time. Their bounding boxes
  are near-identical in practice (see `WIN_20220330_16_07_40_Pro.jpg` below), so this doesn't
  change the crop box, only which polygon's `points` happen to be picked before the union.

## Notes from the real download

Two of the 43 merges don't fit the Observation ("the extra polygon is always a spike overlapping a
different-class polygon") — found by running the rule against the real download, not anticipated
in advance. The rule (any class combination, same-class-random / different-class-precedence)
handles both the same way it handles everything else:

- **`WIN_20220330_13_18_32_Pro.jpg`** — the left joint carries `insufficient` and `normal`
  polygons at 99% mutual overlap (effectively duplicate annotations of the same joint), and
  `spike` sits alone on the right joint. No `spike` is involved in the merged pair — precedence
  still applies generically, so `insufficient` beats `normal`.
- **`WIN_20220330_16_07_40_Pro.jpg`** — the right joint carries *two* `spike` polygons (a small
  sliver plus a larger blob) that sit side by side with **zero overlap** between them; `excess`
  sits alone on the left joint. This is the case that broke the original overlap-based definition
  of "same joint" outright — there is no overlapping pair for it to find — which is why the rule
  groups by position instead. The same-class phase applies: one of the two `spike` polygons is
  kept at random.

## Counts

| Pair type | Count |
| --- | ---: |
| spike + excess | 25 |
| insufficient + spike | 8 |
| spike + normal | 8 |
| insufficient + normal | 1 |
| spike + spike (same-class) | 1 |
| **Total merged** | **43** |

| Class | Before | After |
| --- | ---: | ---: |
| excess | 182 | 157 |
| spike | 130 | 121 |
| normal | 74 | 65 |
| insufficient | 57 | 57 |
| **Total** | **443** | **400** |

## Examples

- `WIN_20220330_16_14_05_Pro`: a 3-polygon image. On the right joint, `spike #2` lies inside
  `excess #3`, so the rule keeps `spike`. [image]
- `WIN_20220330_13_11_58_Pro`: a 2-polygon image (`insufficient` + `spike`) on separate joints,
  with no overlap. It shows `spike` as a standalone joint label. [image]
- `WIN_20220330_15_56_06_Pro`: a 2-polygon image (`excess` + `spike`) on separate joints, with no
  overlap. [image]

## Known limitations

- Merged joints really carry two defects, and the kept label hides one of them. This is a known
  label-noise source, to be noted in the Stage 8 error-slice report.
- The spike + normal merges reduce the normal class, which is already small. This is accepted in
  favor of label correctness.
