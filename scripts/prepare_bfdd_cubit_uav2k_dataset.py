#!/usr/bin/env python3
"""Build the combined BFDD + CUBIT + UAV2K binary defect detection dataset.

Sources
-------
- BFDD: prepared binary dataset at ``data/bfdd_binary_v1`` (mask components
  converted to boxes; splits grouped by capture minute).
- CUBIT: prepared binary dataset at ``data/bfdd_cubit_binary_v1`` — reuse of
  the already-converted binary boxes (CUBIT val as training source, CUBIT
  test kept untouched as the secondary benchmark).
- UAV2K: extracted + repaired release at ``data/external/uav2k``
  (see docs/uav2k-data-card.md). 3 classes (hollow/spalling/crack) collapsed
  to binary ``defect``.

Design decisions (recorded, not hidden)
---------------------------------------
1. Binary ``defect`` taxonomy, matching the BFDD/CUBIT datasets and the
   binary benchmark protocol.  All source class tokens collapse to ``0``.
2. Splits:
   - train = BFDD train + CUBIT val-as-train + UAV2K train
   - val   = BFDD val + UAV2K val   (both untouched by training)
   - test  = BFDD test + CUBIT test + UAV2K test (all untouched; kept for
     benchmarking, never trained on)
3. UAV2K boxes below ``MIN_BOX_AREA`` (512 px^2, mirroring the BFDD
   mask-component and CUBIT polygon protocol) are dropped; the count is
   recorded.  Only 0.7% of UAV2K boxes are affected.
4. BFDD test remains the primary apples-to-apples benchmark; CUBIT test
   numbers carry the documented near-duplicate-leakage caveat; UAV2K test
   is a secondary held-out set (its frames are building-grouped, not
   frame-interleaved with train, so it is cleaner than CUBIT test).
5. Original source data is never modified; derived labels are written only
   into the new gitignored destination directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "bfdd_cubit_uav2k_binary_v1"
BFDD_DATASET = ROOT / "data" / "bfdd_binary_v1"
CUBIT_DATASET = ROOT / "data" / "bfdd_cubit_binary_v1"
UAV2K_DATASET = ROOT / "data" / "external" / "uav2k" / "huggingface_dataset"

MIN_BOX_AREA = 512  # px^2 in original-image pixels, mirrors BFDD protocol
BINARY_CLASS = "0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def add_bfdd(manifest: list[dict[str, object]], split_counts: Counter, box_counts: Counter) -> None:
    """Hardlink BFDD binary images and copy labels from the prepared BFDD dataset."""
    bfdd_manifest = json.loads(
        (BFDD_DATASET / "manifest.json").read_text(encoding="utf-8")
    )
    for entry in bfdd_manifest["manifest"]:
        split = entry["split"]
        stem = Path(entry["file"]).stem
        image_source = BFDD_DATASET / "images" / split / entry["file"]
        label_source = BFDD_DATASET / "labels" / split / f"{stem}.txt"
        if not image_source.is_file() or not label_source.is_file():
            raise FileNotFoundError(f"BFDD entry missing: {entry['file']}")
        link_or_copy(image_source, DESTINATION / "images" / split / entry["file"])
        link_or_copy(label_source, DESTINATION / "labels" / split / f"{stem}.txt")
        split_counts[split] += 1
        box_counts[split] += int(entry["boxes"])
        manifest.append(
            {
                "source": "bfdd",
                "file": entry["file"],
                "group": entry["group"],
                "split": split,
                "boxes": int(entry["boxes"]),
                "class_mapping": "0 -> defect (BFDD mask components)",
            }
        )


def add_cubit(manifest: list[dict[str, object]], split_counts: Counter, box_counts: Counter) -> None:
    """Copy CUBIT entries from the prepared BFDD+CUBIT binary dataset.

    CUBIT val lives in combined train (it was the CUBIT training source);
    CUBIT test lives in combined test (secondary, leaky benchmark).
    """
    cubit_manifest = json.loads(
        (CUBIT_DATASET / "manifest.json").read_text(encoding="utf-8")
    )
    for entry in cubit_manifest["manifest"]:
        if entry["source"] != "cubit":
            continue
        split = entry["split"]
        stem = Path(entry["file"]).stem
        image_source = CUBIT_DATASET / "images" / split / entry["file"]
        label_source = CUBIT_DATASET / "labels" / split / f"{stem}.txt"
        if not image_source.is_file() or not label_source.is_file():
            raise FileNotFoundError(f"CUBIT entry missing: {entry['file']}")
        link_or_copy(image_source, DESTINATION / "images" / split / entry["file"])
        link_or_copy(label_source, DESTINATION / "labels" / split / f"{stem}.txt")
        split_counts[split] += 1
        box_counts[split] += int(entry["boxes"])
        manifest.append(
            {
                "source": "cubit",
                "file": entry["file"],
                "group": entry["group"],
                "split": split,
                "boxes": int(entry["boxes"]),
                "class_mapping": "0 -> defect (CUBIT polygon collapsed; see cubit data card)",
            }
        )


def add_uav2k(
    split: str,
    destination_split: str,
    manifest: list[dict[str, object]],
    split_counts: Counter,
    box_counts: Counter,
    dropped_counts: Counter,
) -> None:
    """Copy a UAV2K split; convert YOLO boxes to binary with a min-area filter."""
    images_dir = UAV2K_DATASET / "images" / split
    labels_dir = UAV2K_DATASET / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"UAV2K {split} split missing. Run scripts/recover_uav2k_labels.py first."
        )
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_candidates = [images_dir / f"{stem}.jpg", images_dir / f"{stem}.JPG"]
        image_source = next((p for p in image_candidates if p.is_file()), None)
        if image_source is None:
            raise FileNotFoundError(f"UAV2K label has no image: {stem}")
        with Image.open(image_source) as img:
            width, height = img.size
        image_target = DESTINATION / "images" / destination_split / image_source.name
        label_target = DESTINATION / "labels" / destination_split / f"{stem}.txt"
        link_or_copy(image_source, image_target)

        boxes: list[str] = []
        dropped = 0
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = line.split()
            if len(tokens) != 5:
                continue
            _, center_x, center_y, box_width, box_height = tokens
            center_x = float(center_x)
            center_y = float(center_y)
            box_width = float(box_width)
            box_height = float(box_height)
            if box_width * box_height * width * height < MIN_BOX_AREA:
                dropped += 1
                continue
            boxes.append(
                f"{BINARY_CLASS} {center_x:.7f} {center_y:.7f} "
                f"{box_width:.7f} {box_height:.7f}"
            )
        label_target.write_text("\n".join(boxes) + ("\n" if boxes else ""), encoding="utf-8")
        split_counts[destination_split] += 1
        box_counts[destination_split] += len(boxes)
        dropped_counts[destination_split] += dropped
        manifest.append(
            {
                "source": "uav2k",
                "file": image_source.name,
                "group": stem.split("_")[0],
                "split": destination_split,
                "boxes": len(boxes),
                "class_mapping": (
                    "0 -> defect (UAV2K hollow/spalling/crack collapsed; "
                    "see uav2k data card)"
                ),
            }
        )


def dataset_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "images").rglob("*")) + sorted(
        (root / "labels").rglob("*")
    )
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def main() -> int:
    for dataset, name in (
        (BFDD_DATASET / "manifest.json", "BFDD binary dataset"),
        (CUBIT_DATASET / "manifest.json", "BFDD+CUBIT binary dataset"),
        (UAV2K_DATASET / "annotations", "UAV2K extraction"),
    ):
        if not dataset.is_file() and not dataset.is_dir():
            raise SystemExit(f"{name} is missing — run its preparation first.")

    for split in ("train", "val", "test"):
        (DESTINATION / "images" / split).mkdir(parents=True, exist_ok=True)
        (DESTINATION / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()

    add_bfdd(manifest, split_counts, box_counts)
    add_cubit(manifest, split_counts, box_counts)
    # UAV2K official splits: train -> train, val -> val, test -> test.
    add_uav2k("train", "train", manifest, split_counts, box_counts, dropped_counts)
    add_uav2k("val", "val", manifest, split_counts, box_counts, dropped_counts)
    add_uav2k("test", "test", manifest, split_counts, box_counts, dropped_counts)

    data_yaml = "\n".join(
        [
            f"path: {DESTINATION}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: defect",
            "",
        ]
    )
    (DESTINATION / "data.yaml").write_text(data_yaml, encoding="utf-8")

    dataset_hash_value = dataset_hash(DESTINATION)
    (DESTINATION / "dataset_hash.txt").write_text(
        dataset_hash_value + "\n", encoding="utf-8"
    )

    report = {
        "destination": str(DESTINATION.relative_to(ROOT)),
        "taxonomy": (
            "binary defect (0) combining BFDD mask boxes, CUBIT polygon boxes "
            "and UAV2K boxes"
        ),
        "min_box_area_pixels": MIN_BOX_AREA,
        "split_counts": dict(sorted(split_counts.items())),
        "annotation_counts": dict(sorted(box_counts.items())),
        "uav2k_boxes_dropped_below_min_area": dict(sorted(dropped_counts.items())),
        "design_notes": [
            "CUBIT val is the CUBIT training source (no CUBIT train labels in download).",
            "Validation = BFDD val + UAV2K val; both untouched by training.",
            (
                "Test = BFDD test + CUBIT test + UAV2K test; all untouched and never "
                "trained on. BFDD test is the primary benchmark; CUBIT test is "
                "near-duplicate-leaky; UAV2K test is a cleaner secondary set."
            ),
            "UAV2K classes hollow/spalling/crack collapse to binary defect.",
        ],
        "source_archives": {
            "bfdd": str((BFDD_DATASET / "manifest.json").relative_to(ROOT)),
            "cubit": str((CUBIT_DATASET / "manifest.json").relative_to(ROOT)),
            "uav2k": "data/external/uav2k (see docs/uav2k-data-card.md)",
        },
        "dataset_hash": dataset_hash_value,
        "manifest": manifest,
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in report.items() if key != "manifest"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
