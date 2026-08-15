# Downloaded-data audit and data card

Audit date: 2026-08-15. The downloaded data are kept outside the application
and ignored by Git. A repeatable local check is available through:

    PYTHONPATH=backend .venv/bin/python scripts/audit_downloaded_datasets.py

## Licensing disposition

| Source | Local finding | Licence / permission status | GlassEye disposition |
|---|---|---|---|
| BFDD | One 528 MiB tar archive containing 838 RGB images and 838 grayscale label masks | [CC BY 4.0 on the primary Mendeley release](https://data.mendeley.com/datasets/9ych7czvyg/1) | Eligible for a future attributed derivative after class-map and split review |
| BFD-UAV2K | Three-part 7.02 GB RAR release; listing shows images, YOLO labels, COCO JSON, and metadata | [The primary release says licence information will be added](https://huggingface.co/datasets/RealUAV-SD/UAV2K) | Prototype/evaluation only; do not redistribute or publish a derived release until explicit reuse permission exists |
| CUBIT-Det / CUBIT-Seg | Downloaded repository has README and sample images, not the training corpus | [Public repository](https://github.com/CUHK-USR-Group/Defect-Dataset) has no LICENSE file | Excluded pending explicit permission and a real data download |
| BD3 | Downloaded repository contains 37 sample classification images and no bounding boxes | [Public repository](https://github.com/Praveenkottari/BD3-Dataset) has no LICENSE file | Excluded pending explicit permission; classification labels are not YOLO annotations |

Public visibility is not treated as a grant of reuse rights.

## Annotation audit

### BFDD

- The archive lists five paired folders: RGB, IR, Label, Label_color, and a
  seven-class label backup. The executable MVP is RGB-only.
- The primary release describes 788 640 × 512 RGB/IR pairs with five semantic
  classes: crack, peeling, hollow area, stain, and erosion. The local archive
  actually contains 838 RGB/mask pairs, so its release count differs from the
  page and must be recorded in a future derivative manifest.
- The inspected Label masks are 640 × 512, grayscale, and contain values 0–5.
  They are structurally valid semantic masks, but the archive does not state
  the authoritative numeric value-to-class mapping. GlassEye does not infer
  that mapping from category ordering.
- The supplied train.txt and test.txt contain image names from the same
  capture dates and near-consecutive timestamp sequences. This is insufficient
  proof against capture-session leakage, so GlassEye rejects that split for
  model evaluation until the source grouping is confirmed and re-split.

### BFD-UAV2K

- Archive listing reports 2,003 image/label entries (including directories),
  three COCO annotation files, and YOLO TXT labels.
- The release metadata and the archive classes require reconciliation: the
  current dataset card describes one defect class, while the archived
  classes.txt lists hollow, spalling, and crack.
- The installed 7-Zip can list but cannot integrity-test several RAR5 methods,
  reporting Unsupported Method. This is explicitly recorded as incomplete
  byte-level validation rather than treated as a clean pass.

### CUBIT and BD3

- CUBIT's downloaded checkout has only five sample composite images; it does
  not contain the claimed detector/segmentation corpus.
- BD3's downloaded checkout has 37 sample JPEGs in seven classification
  folders. It has no per-image box annotations and therefore cannot be
  converted to YOLO labels without new annotation work.

## Executable demo dataset

The checked-in application creates data/glasseye_v1 locally. It contains 256
deterministic, synthetic facade images with two classes:

- 0: cleanable_surface_issue
- 1: structural_issue

Each image and YOLO label are generated together from known geometry. The
manifest records source, source group, split, annotation status, and SHA-256.
Source groups are disjoint across train, validation, and test. The validator
checks label syntax, class IDs, normalized box bounds, image/label pairing,
manifest checksums, and source-group leakage.

This makes the demo reproducible and honest: it is a simulated closed-loop
inspection workflow, not a claim that the model is validated for field use.
Before any real-world use, add 100–300 manually reviewed, project-specific
frames and a held-out capture-session test set.
