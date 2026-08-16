#!/usr/bin/env python3
"""Build the combined BFDD + CUBIT binary defect detection dataset.

Sources
-------
- BFDD: the already-prepared binary dataset at ``data/bfdd_binary_v1``
  (mask components converted to boxes; splits grouped by capture minute).
- CUBIT: the user-provided archives under ``datasets/cubit``.  The train
  archive (``images-001.zip``) contains images only; **no CUBIT training
  labels are present in the download**.  Labels exist only for the val and
  test archives (699 + 701 images).  CUBIT labels are YOLO segmentation
  polygons; they are converted to binary boxes here.

Design decisions (recorded, not hidden)
---------------------------------------
1. Binary ``defect`` taxonomy for the combined dataset, matching the BFDD
   binary dataset and the binary benchmark protocol.  The two CUBIT class
   tokens (0/1) are collapsed to ``0``; the crack/spalling reading of those
   tokens is an *observation* (prefix-to-class consistency), not documented
   semantics (see docs/cubit-data-card.md).
2. CUBIT val is used as the CUBIT *training* source because it is the only
   labeled CUBIT split in this download.  Validation is therefore BFDD val
   only, so validation images never appear in training.
3. CUBIT val and test splits are kept as released (no re-splitting); the
   audit shows their frames interleave with each other and with unlabeled
   train frames (median index distance 1), so CUBIT held-out numbers are
   near-duplicate-inflated.  The BFDD test split remains the primary
   apples-to-apples benchmark.
4. Polygons whose shoelace area is below ``MIN_POLYGON_AREA`` (512 px^2, in
   original-image pixels) are dropped, mirroring the BFDD mask-component
   protocol.  The count of dropped polygons is recorded.
5. Original CUBIT archives are never modified; derived labels are written
   only into the new gitignored destination directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "bfdd_cubit_binary_v1"
BFDD_DATASET = ROOT / "data" / "bfdd_binary_v1"
CUBIT_DIR = ROOT / "datasets" / "cubit"
SCRATCH = CUBIT_DIR / "_scratch"

MIN_POLYGON_AREA = 512  # px^2 in original-image pixels, mirrors BFDD protocol
BINARY_CLASS = "0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def ensure_nested_extracted() -> None:
    """Ensure _scratch/{test,val}/{images,labels}.zip exist (nested archives)."""
    for archive_glob, split in (("test-*.zip", "test"), ("val-*.zip", "val")):
        archive = next(CUBIT_DIR.glob(archive_glob), None)
        if archive is None:
            continue
        target = SCRATCH / split
        target.mkdir(parents=True, exist_ok=True)
        for member in ("images.zip", "labels.zip"):
            if not (target / member).is_file():
                with zipfile.ZipFile(archive) as handle:
                    handle.extract(f"{split}/{member}", SCRATCH)
                shutil.move(SCRATCH / split / member, target / member)
                leftover = SCRATCH / split
                if leftover.is_dir() and not any(leftover.iterdir()):
                    leftover.rmdir()


def ensure_labels_extracted() -> None:
    """Extract label text files from the nested label zips into _scratch."""
    for split in ("test", "val"):
        labels_zip = SCRATCH / split / "labels.zip"
        target = SCRATCH / f"{split}_labels"
        if not labels_zip.is_file():
            continue
        if target.is_dir() and any(target.glob("labels/*.txt")):
            continue
        with zipfile.ZipFile(labels_zip) as handle:
            handle.extractall(target)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def polygon_area_normalized(coords: list[float]) -> float:
    xs, ys = coords[0::2], coords[1::2]
    area = 0.0
    for index in range(len(xs)):
        next_index = (index + 1) % len(xs)
        area += xs[index] * ys[next_index] - xs[next_index] * ys[index]
    return abs(area) / 2.0


def polygon_to_boxes(
    label_path: Path, width: int, height: int
) -> tuple[list[str], int]:
    """Convert YOLO polygon lines to binary boxes; returns (boxes, dropped)."""
    boxes: list[str] = []
    dropped = 0
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 5:
            continue
        coords = [float(value) for value in tokens[1:]]
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        if polygon_area_normalized(coords) * width * height < MIN_POLYGON_AREA:
            dropped += 1
            continue
        xs, ys = coords[0::2], coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max <= x_min or y_max <= y_min:
            dropped += 1
            continue
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        box_width = x_max - x_min
        box_height = y_max - y_min
        boxes.append(
            f"{BINARY_CLASS} {center_x:.7f} {center_y:.7f} "
            f"{box_width:.7f} {box_height:.7f}"
        )
    return boxes, dropped


def extract_image_entry(images_zip: zipfile.ZipFile, entry: str, target: Path) -> tuple[int, int]:
    """Write raw JPEG bytes to target; return (width, height) without re-encode."""
    image_bytes = images_zip.read(entry)
    target.write_bytes(image_bytes)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image entry: {entry}")
    return image.shape[1], image.shape[0]


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


def add_cubit_split(
    split: str,
    destination_split: str,
    manifest: list[dict[str, object]],
    split_counts: Counter,
    box_counts: Counter,
    dropped_counts: Counter,
) -> None:
    """Extract a labeled CUBIT split (val or test) into the combined dataset."""
    images_zip_path = SCRATCH / split / "images.zip"
    labels_dir = SCRATCH / f"{split}_labels" / "labels"
    if not images_zip_path.is_file() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"CUBIT {split} split is missing. Run scripts/audit_cubit_dataset.py first."
        )
    with zipfile.ZipFile(images_zip_path) as images_zip:
        for info in images_zip.infolist():
            if not info.filename.endswith(".JPG"):
                continue
            stem = Path(info.filename).stem
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"CUBIT {split} image has no label: {stem}")
            prefix = stem.split("_")[0]
            image_target = DESTINATION / "images" / destination_split / f"{stem}.JPG"
            label_target = DESTINATION / "labels" / destination_split / f"{stem}.txt"
            width, height = extract_image_entry(images_zip, info.filename, image_target)
            boxes, dropped = polygon_to_boxes(label_path, width, height)
            label_target.write_text(
                "\n".join(boxes) + ("\n" if boxes else ""), encoding="utf-8"
            )
            split_counts[destination_split] += 1
            box_counts[destination_split] += len(boxes)
            dropped_counts[destination_split] += dropped
            manifest.append(
                {
                    "source": "cubit",
                    "file": f"{stem}.JPG",
                    "group": prefix,
                    "split": destination_split,
                    "boxes": len(boxes),
                    "class_mapping": (
                        "0 -> defect (CUBIT polygon collapsed; original tokens "
                        "0/1 -> crack/spalling is an observation, see audit)"
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
    ensure_nested_extracted()
    ensure_labels_extracted()
    if not (BFDD_DATASET / "manifest.json").is_file():
        raise SystemExit("BFDD binary dataset is missing. Run prepare_bfdd_binary_dataset.py first.")

    for split in ("train", "val", "test"):
        (DESTINATION / "images" / split).mkdir(parents=True, exist_ok=True)
        (DESTINATION / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()

    add_bfdd(manifest, split_counts, box_counts)
    # CUBIT val -> combined train (the only labeled CUBIT source in this download)
    add_cubit_split("val", "train", manifest, split_counts, box_counts, dropped_counts)
    # CUBIT test -> combined test, kept untouched for the secondary benchmark
    add_cubit_split("test", "test", manifest, split_counts, box_counts, dropped_counts)

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
        "taxonomy": "binary defect (0) combining BFDD mask boxes and CUBIT polygon boxes",
        "min_polygon_area_pixels": MIN_POLYGON_AREA,
        "split_counts": dict(sorted(split_counts.items())),
        "annotation_counts": dict(sorted(box_counts.items())),
        "cubit_polygons_dropped_below_min_area": dict(sorted(dropped_counts.items())),
        "design_notes": [
            (
                "CUBIT train archive (images-001.zip) contains images only; no CUBIT "
                "training labels exist in the download, so CUBIT val is used as the "
                "CUBIT training source."
            ),
            "Validation is BFDD val only; CUBIT val images are never in validation.",
            (
                "CUBIT test remains untouched and is evaluated with a documented "
                "near-duplicate-leakage caveat (see docs/cubit-data-card.md)."
            ),
            "BFDD test split is untouched and remains the primary benchmark.",
        ],
        "source_archives": {
            "bfdd": str((BFDD_DATASET / "manifest.json").relative_to(ROOT)),
            "cubit_train": "datasets/cubit/images-001.zip (images only, no labels)",
            "cubit_val": "datasets/cubit/val-*.zip",
            "cubit_test": "datasets/cubit/test-*.zip",
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
