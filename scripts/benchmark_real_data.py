#!/usr/bin/env python3
"""Evaluate the current checkpoint on external BFDD real-façade imagery.

This is deliberately a binary defect benchmark. BFDD's semantic masks are
converted into connected-component boxes, while both GlassEye prediction
classes are collapsed to the single label ``defect``. It is therefore a useful
domain-shift check, not an official BFDD segmentation leaderboard result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))


def artifacts_root() -> Path:
    return ROOT / "artifacts"


def models_root() -> Path:
    return ROOT / "models"


@dataclass(frozen=True)
class Prediction:
    image_id: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str


def iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def mask_boxes(
    mask_path: Path, min_component_area: int
) -> list[tuple[float, float, float, float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read annotation mask: {mask_path}")
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    boxes: list[tuple[float, float, float, float]] = []
    for component_index in range(1, component_count):
        x, y, width, height, area = stats[component_index]
        if area >= min_component_area:
            boxes.append((float(x), float(y), float(x + width), float(y + height)))
    return boxes


def average_precision(matches: list[bool], total_ground_truth: int) -> float:
    if total_ground_truth == 0:
        return 0.0
    true_positives = np.cumsum(np.asarray(matches, dtype=np.float64))
    false_positives = np.cumsum(1.0 - np.asarray(matches, dtype=np.float64))
    recall = true_positives / total_ground_truth
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-9)
    levels = np.linspace(0.0, 1.0, 101)
    interpolated = [
        precision[recall >= level].max() if np.any(recall >= level) else 0.0
        for level in levels
    ]
    return float(np.mean(interpolated))


def evaluate(
    predictions: list[Prediction],
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    threshold: float,
) -> dict[str, float | int]:
    matched: dict[str, set[int]] = {image_id: set() for image_id in ground_truth}
    ordered = sorted(predictions, key=lambda prediction: prediction.score, reverse=True)
    matches: list[bool] = []
    for prediction in ordered:
        candidates = ground_truth[prediction.image_id]
        best_index, best_iou = -1, 0.0
        for index, candidate in enumerate(candidates):
            if index in matched[prediction.image_id]:
                continue
            overlap = iou(prediction.bbox_xyxy, candidate)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        is_match = bool(best_index >= 0 and best_iou >= threshold)
        if is_match:
            matched[prediction.image_id].add(best_index)
        matches.append(is_match)
    true_positives = sum(matches)
    false_positives = len(matches) - true_positives
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    false_negatives = total_ground_truth - true_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return {
        "iou_threshold": float(threshold),
        "average_precision": round(average_precision(matches, total_ground_truth), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def draw_overlay(
    image: np.ndarray,
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[Prediction],
) -> np.ndarray:
    overlay = image.copy()
    for x1, y1, x2, y2 in ground_truth:
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction.bbox_xyxy
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"{prediction.class_name} {prediction.score:.2f}",
            (int(x1), max(16, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


sys.path.insert(0, str(ROOT / "backend"))


def predict_binary(
    model: object,
    image: np.ndarray,
    image_id: str,
    max_det: int = 20,
    use_sahi: bool = False,
) -> list[Prediction]:
    try:
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    h, w = image.shape[:2]
    all_raw_predictions: list[tuple[float, tuple[float, float, float, float], str]] = []

    # 1. Global full-frame pass
    result = model.predict(
        source=image,
        conf=0.20,
        iou=0.45,
        imgsz=320,
        device=device,
        max_det=max_det,
        verbose=False,
    )[0]
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            all_raw_predictions.append(
                (
                    float(box.conf.item()),
                    tuple(float(value) for value in box.xyxy[0].tolist()),
                    str(names[int(box.cls.item())]),
                )
            )

    # 2. Sliced pass if SAHI enabled
    if use_sahi and (h > 640 or w > 640):
        from app.detector import _calculate_slices

        slices = _calculate_slices(h, w, slice_size=480, overlap_ratio=0.25)
        for y1, x1, y2, x2 in slices:
            tile = image[y1:y2, x1:x2]
            res_tile = model.predict(
                source=tile,
                conf=0.20,
                iou=0.45,
                imgsz=320,
                device=device,
                max_det=max_det,
                verbose=False,
            )[0]
            if res_tile.boxes is not None:
                for box in res_tile.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    all_raw_predictions.append(
                        (
                            float(box.conf.item()),
                            (
                                float(bx1 + x1),
                                float(by1 + y1),
                                float(bx2 + x1),
                                float(by2 + y1),
                            ),
                            str(names[int(box.cls.item())]),
                        )
                    )

    # 3. Non-Maximum Suppression across merged predictions
    if use_sahi and len(all_raw_predictions) > 1:
        all_raw_predictions.sort(key=lambda item: item[0], reverse=True)
        kept_tuples: list[tuple[float, tuple[float, float, float, float], str]] = []
        for score, bbox, class_name in all_raw_predictions:
            should_keep = True
            for _, k_bbox, k_class in kept_tuples:
                if k_class == class_name and iou(bbox, k_bbox) > 0.45:
                    should_keep = False
                    break
            if should_keep:
                kept_tuples.append((score, bbox, class_name))
        all_raw_predictions = kept_tuples[:max_det]

    return [
        Prediction(
            image_id=image_id,
            score=score,
            bbox_xyxy=bbox,
            class_name=class_name,
        )
        for score, bbox, class_name in all_raw_predictions
    ]


def filter_pairs_for_split(
    pairs: list[tuple[Path, Path]], split: str
) -> list[tuple[Path, Path]]:
    if split == "all":
        return pairs
    manifest_path = ROOT / "data" / "bfdd_binary_v1" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"BFDD split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest"]
    allowed = {entry["file"] for entry in manifest if entry["split"] == split}
    return [pair for pair in pairs if pair[0].name in allowed]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bfdd"], default="bfdd")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Use a deterministic prefix for a smoke run.",
    )
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=models_root() / "glasseye-yolo-v1" / "best.pt",
    )
    parser.add_argument(
        "--bfdd-split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Use the deterministic BFDD binary-dataset split.",
    )
    parser.add_argument(
        "--sahi",
        action="store_true",
        help="Enable Sliced Aided Hyper Inference (SAHI) with cross-tile NMS merging.",
    )
    args = parser.parse_args()
    if args.min_component_area < 1:
        raise ValueError("--min-component-area must be positive")

    root = ROOT / "data" / "external" / "bfdd" / "Dataset_1x"
    rgb_directory, mask_directory = root / "RGB", root / "Label"
    if not rgb_directory.is_dir() or not mask_directory.is_dir():
        raise SystemExit(
            "BFDD is absent. Extract the downloaded BFDD archive to data/external/bfdd first."
        )
    pairs = [
        (image, mask_directory / f"{image.stem}.png")
        for image in sorted(rgb_directory.glob("*.JPG"))
    ]
    pairs = [(image, mask) for image, mask in pairs if mask.is_file()]
    pairs = filter_pairs_for_split(pairs, args.bfdd_split)
    if args.max_images:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise SystemExit("No BFDD RGB/mask pairs were found.")

    if not args.checkpoint.is_file():
        raise SystemExit(f"YOLO checkpoint is missing: {args.checkpoint}")
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    ground_truth: dict[str, list[tuple[float, float, float, float]]] = {}
    predictions: list[Prediction] = []
    predictions_by_image: dict[str, list[Prediction]] = {}
    predicted_classes: Counter[str] = Counter()
    overlay_inputs: list[tuple[Path, str]] = []
    for index, (image_path, mask_path) in enumerate(pairs):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image_id = image_path.stem
        ground_truth[image_id] = mask_boxes(mask_path, args.min_component_area)
        image_predictions = predict_binary(model, image, image_id, use_sahi=args.sahi)
        predictions.extend(image_predictions)
        predictions_by_image[image_id] = image_predictions
        predicted_classes.update(
            prediction.class_name for prediction in image_predictions
        )
        if len(overlay_inputs) < args.overlay_count:
            overlay_inputs.append((image_path, image_id))

    threshold_reports = [
        evaluate(predictions, ground_truth, threshold)
        for threshold in np.arange(0.5, 1.0, 0.05)
    ]
    at_50 = threshold_reports[0]
    output = artifacts_root() / "real-benchmark" / "bfdd" / args.checkpoint.parent.name
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for image_path, image_id in overlay_inputs:
        image = cv2.imread(str(image_path))
        assert image is not None
        cv2.imwrite(
            str(overlays / f"{image_id}.jpg"),
            draw_overlay(image, ground_truth[image_id], predictions_by_image[image_id]),
        )
    report = {
        "dataset": "BFDD",
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "prediction_taxonomy": "Both GlassEye classes are collapsed to binary defect.",
            "ground_truth_taxonomy": "All non-background BFDD mask components are converted to binary boxes.",
            "not_an_official_bfdd_metric": "This is box evaluation against mask-derived boxes, not segmentation IoU.",
            "min_component_area_pixels": args.min_component_area,
            "bfdd_split": args.bfdd_split,
        },
        "images_evaluated": len(pairs),
        "ground_truth_boxes": sum(len(boxes) for boxes in ground_truth.values()),
        "predictions": len(predictions),
        "predicted_class_counts": dict(sorted(predicted_classes.items())),
        "at_iou_0_50": at_50,
        "mAP50_95": round(
            float(np.mean([entry["average_precision"] for entry in threshold_reports])),
            6,
        ),
        "iou_sweep": threshold_reports,
        "overlay_directory": str(overlays.relative_to(ROOT)),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
