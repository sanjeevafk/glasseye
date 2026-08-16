#!/usr/bin/env python3
"""Recover damaged UAV2K YOLO label files from the intact COCO annotations.

Background
----------
The UAV2K RAR5 release (`datasets/UAV2K/*.part*.rar`) mixes compression methods.
p7zip decodes images but leaves the COCO JSONs as 0 bytes; `unar` decodes the
JSONs but leaves a subset of small YOLO label files as 0 bytes.

The COCO annotation JSONs (`annotations/instances_{split}.json`) are intact and
contain the complete box annotations. The damaged label files are regenerated
from COCO using the exact conversion convention the authors used, which was
verified byte-identical against 1,171 of 1,190 intact label files (the 19
non-identical ones differ only at the 6th decimal, ~1e-6 rounding).

Canonical base tree: the `unar` extraction
(`data/external/uav2k_unar/BFD-UAV2K_public_release`), which contains the
COCO JSONs, SHA256SUMS.csv, VALIDATION_REPORT.json and the full image set.

Usage
-----
    .venv/bin/python scripts/recover_uav2k_labels.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data/external/uav2k/huggingface_dataset"
REPORT = ROOT / "artifacts/uav2k-recovery-report.json"


def boxes_to_yolo(img: dict, anns: list[dict]) -> str:
    """Convert COCO xywh boxes to YOLO normalized cx cy w h, class id - 1."""
    width, height = img["width"], img["height"]
    lines = []
    for ann in sorted(anns, key=lambda a: a["id"]):
        x, y, w, h = ann["bbox"]
        cls = ann["category_id"] - 1  # COCO 1-based -> YOLO 0-based
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        nw = w / width
        nh = h / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not BASE.exists():
        print(f"error: {BASE} not found — run the unar extraction first", file=sys.stderr)
        return 1

    summary = {"recovered": {}, "verification": {}, "class_distribution": {}}
    total_recovered = 0
    total_damaged = 0

    for split in ["train", "val", "test"]:
        coco_path = BASE / "annotations" / f"instances_{split}.json"
        with open(coco_path) as f:
            coco = json.load(f)

        ann_by_img: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)
        ann_img_ids = set(ann_by_img)

        img_by_stem = {
            Path(im["file_name"]).stem: im for im in coco["images"]
        }

        labels_dir = BASE / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)

        recovered_here = 0
        damaged_here = 0
        for stem, img in sorted(img_by_stem.items()):
            lp = labels_dir / f"{stem}.txt"
            needs_recovery = (
                not lp.exists() or lp.stat().st_size == 0
            ) and img["id"] in ann_img_ids
            if needs_recovery:
                damaged_here += 1
                lp.write_text(boxes_to_yolo(img, ann_by_img[img["id"]]))
                recovered_here += 1
        total_recovered += recovered_here
        total_damaged += damaged_here

        # Verification: every image has a label file; every label matches COCO.
        missing_labels = []
        mismatched = 0
        checked = 0
        for im in coco["images"]:
            lp = labels_dir / f"{Path(im['file_name']).stem}.txt"
            if not lp.exists():
                missing_labels.append(im["file_name"])
                continue
            want = boxes_to_yolo(im, ann_by_img.get(im["id"], []))
            if lp.read_text() != want:
                mismatched += 1
            checked += 1

        # Class distribution across all labels (post-recovery).
        cls_counter: Counter[str] = Counter()
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        for f in labels_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    cls_counter[cat_names.get(int(line.split()[0]) + 1, "?")] += 1

        summary["recovered"][split] = recovered_here
        summary["verification"][split] = {
            "images_in_coco": len(coco["images"]),
            "label_files_present": checked,
            "missing_labels": len(missing_labels),
            "labels_mismatching_coco": mismatched,
        }
        summary["class_distribution"][split] = dict(cls_counter)
        print(
            f"{split}: damaged={damaged_here}, recovered={recovered_here}, "
            f"labels={checked}/{len(coco['images'])}, mismatches={mismatched}"
        )

    summary["total_damaged"] = total_damaged
    summary["total_recovered"] = total_recovered

    # Overall integrity: verify every file listed in the authors' checksums.
    sums_path = BASE / "SHA256SUMS.csv"
    image_files = sorted((BASE / "images").rglob("*.jpg"))
    summary["verification"]["images_present"] = len(image_files)
    if sums_path.exists():
        expected = {}
        with open(sums_path) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    expected[parts[2]] = parts[0]
        verified = 0
        mismatched = 0
        missing = 0
        mismatched_files = []
        for rel, want in sorted(expected.items()):
            p = BASE / rel
            if not p.exists():
                missing += 1
                continue
            if sha256(p) == want:
                verified += 1
            else:
                mismatched += 1
                mismatched_files.append(rel)
        summary["verification"]["authors_checksums"] = {
            "total": len(expected),
            "verified": verified,
            "mismatched": mismatched,
            "missing": missing,
            "mismatched_files": mismatched_files,
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")
    print(f"total damaged labels recovered: {total_recovered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
