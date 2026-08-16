#!/usr/bin/env python3
"""Evaluate checkpoints on the untouched UAV2K test split.

Protocol mirrors scripts/benchmark_real_data.py (binary defect boxes,
conf 0.20, iou 0.45, imgsz 320) with the same ``max_det=100`` used for the
dense CUBIT split.  Ground truth comes from the derived binary boxes in
``data/bfdd_cubit_uav2k_binary_v1`` restricted to ``source == uav2k`` and
``split == test`` (the same conversion used for training data).

Cleanliness: unlike CUBIT's frame-interleaved split, UAV2K's official split
is building-disjoint — no test/val building appears in train (verified in
the manifest).  UAV2K test is therefore a genuinely clean held-out set.

See docs/uav2k-data-card.md for extraction/recovery provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from benchmark_real_data import (
    Prediction,
    draw_overlay,
    evaluate,
    models_root,
    predict_binary,
)

COMBINED_DATASET = ROOT / "data" / "bfdd_cubit_uav2k_binary_v1"
MAX_DET = 100  # dense split; BFDD protocol uses 20


def load_manifest_entries() -> list[dict[str, object]]:
    manifest = json.loads((COMBINED_DATASET / "manifest.json").read_text(encoding="utf-8"))
    return manifest["manifest"]


def ground_truth_boxes() -> dict[str, list[tuple[float, float, float, float]]]:
    """Convert derived YOLO labels to pixel-space boxes for UAV2K test entries."""
    manifest = load_manifest_entries()
    uav2k_test = {
        entry["file"]: entry
        for entry in manifest
        if entry["source"] == "uav2k" and entry["split"] == "test"
    }
    labels_dir = COMBINED_DATASET / "labels" / "test"
    ground_truth: dict[str, list[tuple[float, float, float, float]]] = {}
    for entry in uav2k_test.values():
        stem = Path(entry["file"]).stem
        label_path = labels_dir / f"{stem}.txt"
        image_path = COMBINED_DATASET / "images" / "test" / entry["file"]
        with Image.open(image_path) as img:
            width, height = img.size
        boxes: list[tuple[float, float, float, float]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if len(tokens) != 5:
                continue
            _, center_x, center_y, width_n, height_n = tokens
            center_x = float(center_x) * width
            center_y = float(center_y) * height
            width_n = float(width_n) * width
            height_n = float(height_n) * height
            x1 = center_x - width_n / 2
            y1 = center_y - height_n / 2
            x2 = center_x + width_n / 2
            y2 = center_y + height_n / 2
            boxes.append((x1, y1, x2, y2))
        ground_truth[stem] = boxes
    return ground_truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=models_root() / "glasseye-yolo-v1" / "best.pt",
    )
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"YOLO checkpoint is missing: {args.checkpoint}")

    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    image_dir = COMBINED_DATASET / "images" / "test"
    ground_truth = ground_truth_boxes()
    stems = sorted(
        path.stem for path in image_dir.glob("*.jpg") if path.stem in ground_truth
    )
    if args.max_images:
        stems = stems[: args.max_images]
    predictions: list[Prediction] = []
    predictions_by_image: dict[str, list[Prediction]] = {}
    predicted_classes: Counter[str] = Counter()
    overlay_inputs: list[tuple[Path, str]] = []
    for index, stem in enumerate(stems):
        image_path = image_dir / f"{stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image_predictions = predict_binary(model, image, stem, max_det=MAX_DET)
        predictions.extend(image_predictions)
        predictions_by_image[stem] = image_predictions
        predicted_classes.update(p.class_name for p in image_predictions)
        if len(overlay_inputs) < args.overlay_count:
            overlay_inputs.append((image_path, stem))

    threshold_reports = [
        evaluate(predictions, ground_truth, threshold)
        for threshold in np.arange(0.5, 1.0, 0.05)
    ]
    at_50 = threshold_reports[0]
    output = (
        ROOT / "artifacts" / "real-benchmark" / "uav2k" / args.checkpoint.parent.name
    )
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for image_path, stem in overlay_inputs:
        image = cv2.imread(str(image_path))
        assert image is not None
        cv2.imwrite(
            str(overlays / f"{stem}.jpg"),
            draw_overlay(image, ground_truth[stem], predictions_by_image[stem]),
        )
    report = {
        "dataset": "UAV2K test (untouched)",
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "prediction_taxonomy": "Both GlassEye classes are collapsed to binary defect.",
            "ground_truth_taxonomy": (
                "UAV2K hollow/spalling/crack boxes collapsed to binary defect "
                "(min box area 512 px^2), same conversion as training data."
            ),
            "cleanliness": (
                "UAV2K official split is building-disjoint: no test/val building "
                "appears in train (verified against the combined-dataset manifest)."
            ),
            "max_det": MAX_DET,
            "imgsz": 320,
            "conf": 0.20,
            "iou": 0.45,
        },
        "images_evaluated": len(stems),
        "ground_truth_boxes": sum(len(v) for v in ground_truth.values()),
        "predictions": len(predictions),
        "predicted_class_counts": dict(sorted(predicted_classes.items())),
        "at_iou_0_50": at_50,
        "mAP50_95": round(
            float(np.mean([e["average_precision"] for e in threshold_reports])), 6
        ),
        "iou_sweep": threshold_reports,
        "overlay_directory": str(overlays.relative_to(ROOT)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
