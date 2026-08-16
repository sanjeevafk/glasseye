#!/usr/bin/env python3
"""Evaluate checkpoints on the untouched CUBIT test split.

The protocol mirrors scripts/benchmark_real_data.py (binary defect boxes,
conf 0.20, iou 0.45, imgsz 320) with two documented differences:

- ``max_det`` is raised to 100 because CUBIT images are far denser than BFDD
  (up to 41 ground-truth boxes per image after the area filter).
- Ground truth comes from the derived polygon-to-box conversion in
  ``data/bfdd_cubit_binary_v1`` (the same conversion used for training),
  restricted to rows whose source is ``cubit`` and split is ``test``.

Important caveat (see docs/cubit-data-card.md): every CUBIT test frame is
within ~2 frame indices of an unlabeled training frame, and CUBIT val (the
CUBIT training source used by the combined model) is also frame-interleaved
with the test split.  CUBIT held-out numbers are therefore inflated by
near-duplicate frames and are NOT a clean generalization measurement; the
BFDD test split remains the primary apples-to-apples benchmark.
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

COMBINED_DATASET = ROOT / "data" / "bfdd_cubit_binary_v1"
MAX_DET = 100  # CUBIT images are dense; BFDD protocol uses 20


def load_manifest_entries() -> list[dict[str, object]]:
    manifest = json.loads((COMBINED_DATASET / "manifest.json").read_text(encoding="utf-8"))
    return manifest["manifest"]


def leak_distance(stem: str, training_stems: set[str]) -> int:
    """Min |index difference| between this test stem and any CUBIT training stem."""
    prefix = stem.split("_")[0]
    index = int(stem.split("_")[1])
    distances = [
        abs(index - int(other.split("_")[1]))
        for other in training_stems
        if other.startswith(prefix + "_")
    ]
    return min(distances) if distances else -1


def ground_truth_boxes() -> dict[str, list[tuple[float, float, float, float]]]:
    """Convert derived YOLO labels to pixel-space boxes.

    YOLO labels store normalized (0..1) center/width/height, so each box is
    denormalized against the actual image dimensions (lazy PIL header read).
    """
    manifest = load_manifest_entries()
    cubit_test = {
        entry["file"]: entry
        for entry in manifest
        if entry["source"] == "cubit" and entry["split"] == "test"
    }
    labels_dir = COMBINED_DATASET / "labels" / "test"
    ground_truth: dict[str, list[tuple[float, float, float, float]]] = {}
    for entry in cubit_test.values():
        stem = Path(entry["file"]).stem
        label_path = labels_dir / f"{stem}.txt"
        image_path = COMBINED_DATASET / "images" / "test" / f"{stem}.JPG"
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
    stems = sorted(path.stem for path in image_dir.glob("*.JPG") if path.stem in ground_truth)
    if args.max_images:
        stems = stems[: args.max_images]
    predictions: list[Prediction] = []
    predictions_by_image: dict[str, list[Prediction]] = {}
    predicted_classes: Counter[str] = Counter()
    overlay_inputs: list[tuple[Path, str]] = []
    for index, stem in enumerate(stems):
        image_path = image_dir / f"{stem}.JPG"
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
    # Less-leaky subset: test frames at least 10 frame indices from any CUBIT
    # training (val-split) frame, i.e. beyond the near-duplicate band.
    training_stems = {
        Path(entry["file"]).stem
        for entry in load_manifest_entries()
        if entry["source"] == "cubit" and entry["split"] == "train"
    }
    distances = {stem: leak_distance(stem, training_stems) for stem in stems}
    far_stems = [stem for stem, distance in distances.items() if distance >= 10]
    far_predictions = [p for p in predictions if p.image_id in set(far_stems)]
    far_ground_truth = {k: v for k, v in ground_truth.items() if k in set(far_stems)}
    far_reports = [
        evaluate(far_predictions, far_ground_truth, threshold)
        for threshold in np.arange(0.5, 1.0, 0.05)
    ]
    far_at_50 = far_reports[0]
    output = (
        ROOT / "artifacts" / "real-benchmark" / "cubit" / args.checkpoint.parent.name
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
        "dataset": "CUBIT test (untouched)",
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "prediction_taxonomy": "Both GlassEye classes are collapsed to binary defect.",
            "ground_truth_taxonomy": (
                "CUBIT polygon annotations converted to binary boxes "
                "(min polygon area 512 px^2), same conversion as training data."
            ),
            "not_a_clean_generalization_measurement": (
                "CUBIT test frames are near-duplicates of training frames "
                "(median index distance to unlabeled train 1; median to CUBIT "
                "val training source 4). Results are inflated by leakage."
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
        "leak_distance_to_cubit_training_frames": {
            "min": min(distances.values()) if distances else None,
            "median": sorted(distances.values())[len(distances) // 2]
            if distances
            else None,
            "max": max(distances.values()) if distances else None,
            "frames_within_10": sum(1 for d in distances.values() if d < 10),
        },
        "less_leaky_subset_index_distance_ge_10": {
            "images_evaluated": len(far_stems),
            "ground_truth_boxes": sum(len(v) for v in far_ground_truth.values()),
            "predictions": len(far_predictions),
            "at_iou_0_50": far_at_50,
            "mAP50_95": round(
                float(np.mean([e["average_precision"] for e in far_reports])), 6
            ),
        },
        "overlay_directory": str(overlays.relative_to(ROOT)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
