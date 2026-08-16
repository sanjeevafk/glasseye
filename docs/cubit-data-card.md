# CUBIT downloaded-data audit and data card

Audit date: 2026-08-16.

This card describes the CUBIT archives under `datasets/cubit` and the
decisions made before any derived training data is created. The source
archives are user-provided, gitignored, and never modified by GlassEye
tooling. A repeatable audit is available through:

    .venv/bin/python scripts/audit_cubit_dataset.py

## Provenance

The archives are a CUBIT-Det / CUBIT-InSeg style release referenced by the
CUHK-USR-Group Defect-Dataset repository (crack, spalling, and moisture
classes are described there; see docs/data-card.md licensing notes).
The local download contains crack (CR) and spalling (SP) sequences only;
no moisture sequences are present.

| Archive | SHA-256 (first 12) |
|---|---|
| datasets/cubit/images-001.zip | 895cd6a1dd8f |
| datasets/cubit/val-20260815T184403Z-1-001.zip | 5ea4e0cdb71b |
| datasets/cubit/test-20260815T183921Z-1-001.zip | 99f8f153e0e1 |

## Image inventory

- Train archive `images-001.zip`: 5596 JPG files
  (15.1 GB uncompressed).
- Val archive: 699 images.
- Test archive: 701 images.

Sequence prefixes observed (UAV capture sequences):

| Prefix | Likely defect | Train | Val | Test |
|---|---|---|---|---|
| CR1 | crack | 2535 | 299 | 320 |
| SP0 | spalling | 1555 | 184 | 186 |
| SP1 | spalling | 1506 | 216 | 195 |

## Label format

Labels are YOLO-style segmentation polygons: one annotation per line as
`class x1 y1 x2 y2 ...` with normalized coordinates. The archives contain
**no classes.txt or equivalent authoritative mapping file**.

- Test: 701 label files, 6017 polygons.
- Val: 699 label files, 6135 polygons.
- Class token counts (test): {'0': 5025, '1': 992}.
- Class token counts (val): {'0': 5029, '1': 1106}.

Observed prefix-to-class consistency (not authoritative):

- test: {'CR1': {'observed_class_ids': {'0': 5025}, 'files': 320}, 'SP0': {'observed_class_ids': {'1': 496}, 'files': 186}, 'SP1': {'observed_class_ids': {'1': 496}, 'files': 195}}
- val: {'CR1': {'observed_class_ids': {'0': 5029}, 'files': 299}, 'SP0': {'observed_class_ids': {'1': 521}, 'files': 184}, 'SP1': {'observed_class_ids': {'1': 585}, 'files': 216}}

Every CR1 file contains only class token `0` and every SP0/SP1 file only
class token `1`, which is consistent with `0 = crack`, `1 = spalling`.
Because no mapping file exists in the archives, GlassEye records this as an
**inference** and does not use the two class IDs as documented semantics.
Any derived dataset therefore uses a single binary `defect` class.

## Split integrity and leakage

- Exact filename overlap between train and held-out splits: 0.
- Sequence prefixes are shared across train/val/test, and frame indices
  interleave: the official split is **frame-interleaved, not temporally
  disjoint**. Median index distance from a test frame to the nearest train
  frame is 1 (val: 1);
 701 of 701 test frames are within 20 frames of a
training frame.

**Consequence:** CUBIT held-out benchmarks are inflated by near-duplicate
frames and are not a clean generalization measurement. The BFDD test split
(grouped by capture minute, untouched by any training) remains the primary
apples-to-apples benchmark.

## GlassEye disposition

1. The CUBIT train archive contains images **only** — no CUBIT training
   labels exist in this download.  The only labeled CUBIT data is the val
   (699) and test (701) splits.
2. Use CUBIT val as the CUBIT training source (polygons -> binary boxes)
   in the combined BFDD+CUBIT dataset; validation is therefore BFDD val
   only, so validation images never appear in training.
3. Keep the official CUBIT test split untouched as a secondary evaluation
   set, with the near-duplicate-leakage caveat above.  The BFDD test split
   remains the primary apples-to-apples benchmark.
4. Do not infer the missing moisture class; do not claim a cleanable-surface
   mapping for CUBIT data (no stain/moisture annotations exist in this
   download).
5. Derived labels are generated into a new gitignored directory only; the
   original archives are never modified.

## Visual samples

Rendered annotation samples (polygon outlines in green, derived boxes in
red) are written to `artifacts/cubit-samples/`.
