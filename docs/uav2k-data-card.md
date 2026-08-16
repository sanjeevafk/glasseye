# UAV2K (BFD-UAV2K) extraction and recovery data card

Extraction date: 2026-08-16.

This card documents how the UAV2K archives under `datasets/UAV2K/` were
extracted into `data/external/uav2k/`, what needed repair, and the
validation evidence. UAV2K is not (yet) part of any training dataset; it is
a candidate third source flagged in the handoff follow-ups.

## Provenance and license

- Archives: `datasets/UAV2K/BFD-UAV2K_public_release.part{1,2,3}.rar`
  (~6.8 GB total, RAR5).
- The release ships `LICENSE_SELECTION_REQUIRED.txt` stating that a license
  must be chosen before the repository is made public — **the dataset has no
  confirmed license yet**. Do not redistribute or use for anything beyond
  the local hackathon demo until a license is selected.
- README claims a single `defect` class and a 1,600/200/200 split.

## Class map (authoritative, from the release itself)

`huggingface_dataset/classes.txt` lists `hollow`, `spalling`, `crack`, and
the COCO category IDs are `1 = hollow`, `2 = spalling`, `3 = crack`
(YOLO IDs 0/1/2). The README's "single defect class" claim is **wrong for
this release** — the actual labels and COCO JSONs are 3-class:

| Split | Images | Annotations | hollow | spalling | crack |
|---|---|---|---|---|---|
| train | 1600 | 3988 | 1137 | 1967 | 884 |
| val | 200 | 505 | 135 | 256 | 114 |
| test | 200 | 527 | 119 | 305 | 103 |

Images with annotations: train 1021, val 164, test 95 (the rest are
legitimately empty background frames).

## Extraction problem

The RAR5 parts mix compression methods that no single local tool fully
decodes:

- p7zip (`7z`) extracts images and most small label files but leaves the
  COCO JSONs and metadata as **0 bytes** ("Unsupported Method").
- `unar` decodes the JSONs but leaves 811 small files (mostly label files)
  as **0 bytes**.

No `unrar` or `bsdtar` is available in this environment, so neither tool
alone could produce a complete extraction.

## Recovery

The COCO annotation JSONs are the authoritative annotation source and were
intact in the `unar` extraction, so the 90 damaged label files (71 train /
12 val / 7 test — files that were 0 bytes but whose image *does* have COCO
annotations) were regenerated from COCO:

- Conversion: COCO `bbox` (x, y, w, h) -> YOLO normalized `cx cy w h`,
  `category_id - 1`, 6 decimal places.
- The convention was validated by regenerating labels for the **1,190
  intact** label files: 1,171 were byte-identical to the shipped files; the
  other 19 differ only at the 6th decimal (~1e-6 rounding).

Repeatable recovery + verification:

    .venv/bin/python scripts/recover_uav2k_labels.py
    # -> artifacts/uav2k-recovery-report.json

## Validation evidence

- All 2,000 images present (`images_present: 2000`); a spot check of image
  dimensions matches the COCO metadata (e.g. 4000x2250).
- Every one of the 2,000 images now has a label file (0 missing).
- Authors' `SHA256SUMS.csv` (4,035 files): **3,945 byte-verified**; the
  only 90 mismatches are exactly the regenerated label files (content is
  correct per COCO, but not byte-identical to the authors' originals).

## Status for GlassEye

- **Usable** as a third training source: images + labels + COCO JSONs all
  present and validated, 3 defect classes overlapping BFDD (crack) and
  CUBIT (spalling).
- **Blocked on license**: `LICENSE_SELECTION_REQUIRED.txt` means the
  authors have not committed to a license; keep it local only.
- Not added to any training run. If used later, follow the same discipline:
  binary `defect` mapping, group by building/sequence before splitting, and
  keep held-out sets untouched.
