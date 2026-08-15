#!/usr/bin/env python3
"""Convert BFDD RGB/mask pairs into a grouped binary YOLO detection dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "bfdd" / "Dataset_1x"
DESTINATION = ROOT / "data" / "bfdd_binary_v1"
SEED = "20260815"
MIN_COMPONENT_AREA = 512


def split_for(image_name: str) -> str:
    # Minute-level groups prevent near-identical consecutive frames leaking between splits.
    group = image_name.split("_")[1][:12]
    value = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def mask_to_yolo(mask_path: Path, width: int, height: int) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read BFDD mask: {mask_path}")
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    labels: list[str] = []
    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]
        if int(area) < MIN_COMPONENT_AREA:
            continue
        center_x = (x + component_width / 2) / width
        center_y = (y + component_height / 2) / height
        normalized_width = component_width / width
        normalized_height = component_height / height
        labels.append(
            f"0 {center_x:.7f} {center_y:.7f} {normalized_width:.7f} {normalized_height:.7f}"
        )
    return labels


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    rgb_directory, mask_directory = SOURCE / "RGB", SOURCE / "Label"
    if not rgb_directory.is_dir() or not mask_directory.is_dir():
        raise SystemExit(
            "BFDD is absent. Extract the source archive before preparing the real-data dataset."
        )
    images = [
        image
        for image in sorted(rgb_directory.glob("*.JPG"))
        if (mask_directory / f"{image.stem}.png").is_file()
    ]
    if not images:
        raise SystemExit("No paired BFDD RGB/mask images found.")
    for split in ("train", "val", "test"):
        (DESTINATION / "images" / split).mkdir(parents=True, exist_ok=True)
        (DESTINATION / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts: Counter[str] = Counter()
    annotation_counts: Counter[str] = Counter()
    manifest: list[dict[str, object]] = []
    for image_path in images:
        split = split_for(image_path.name)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read BFDD image: {image_path}")
        height, width = image.shape[:2]
        labels = mask_to_yolo(mask_directory / f"{image_path.stem}.png", width, height)
        image_destination = DESTINATION / "images" / split / image_path.name
        label_destination = DESTINATION / "labels" / split / f"{image_path.stem}.txt"
        link_or_copy(image_path, image_destination)
        label_destination.write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
        split_counts[split] += 1
        annotation_counts[split] += len(labels)
        manifest.append(
            {
                "file": image_path.name,
                "group": image_path.name.split("_")[1][:12],
                "split": split,
                "boxes": len(labels),
            }
        )

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
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "destination": str(DESTINATION.relative_to(ROOT)),
        "taxonomy": "binary defect from all non-background BFDD mask components",
        "min_component_area_pixels": MIN_COMPONENT_AREA,
        "split_group": "capture timestamp rounded to minute",
        "split_counts": dict(sorted(split_counts.items())),
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "manifest": manifest,
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "manifest"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
