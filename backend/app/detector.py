"""Ultralytics YOLO detector adapter behind the frozen GlassEye contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schemas import DefectClass, Detection, DetectorFrame


class DetectorAdapter(Protocol):
    model_version: str

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: str,
        timestamp: float,
        image_id: str,
    ) -> DetectorFrame: ...


_MODEL_CACHE: dict[str, Any] = {}


def _get_cached_yolo(model_path: Path) -> Any:
    key = str(model_path.resolve())
    if key not in _MODEL_CACHE:
        try:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(model_path.parents[2] / ".ultralytics"))
            matplotlib_config = model_path.parents[2] / ".matplotlib"
            matplotlib_config.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
            import torch

            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required; run make setup.") from exc
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def _calculate_slices(
    height: int, width: int, slice_size: int = 480, overlap_ratio: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Generate (ymin, xmin, ymax, xmax) pixel slices with overlap."""
    if height <= slice_size and width <= slice_size:
        return [(0, 0, height, width)]

    stride = max(64, int(slice_size * (1.0 - overlap_ratio)))
    slices: list[tuple[int, int, int, int]] = []

    y_starts = list(range(0, height - slice_size + 1, stride))
    if not y_starts or y_starts[-1] + slice_size < height:
        y_starts.append(max(0, height - slice_size))

    x_starts = list(range(0, width - slice_size + 1, stride))
    if not x_starts or x_starts[-1] + slice_size < width:
        x_starts.append(max(0, width - slice_size))

    for y1 in sorted(set(y_starts)):
        for x1 in sorted(set(x_starts)):
            y2 = min(height, y1 + slice_size)
            x2 = min(width, x1 + slice_size)
            slices.append((y1, x1, y2, x2))

    return slices


def _non_max_suppression(
    detections: list[Detection], iou_threshold: float = 0.45
) -> list[Detection]:
    """Merge overlapping bounding boxes across full-frame and sliced inferences."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    for det in sorted_dets:
        box_a = det.bbox_xyxy
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        should_keep = True

        for kept_det in kept:
            if kept_det.class_name != det.class_name:
                continue
            box_b = kept_det.bbox_xyxy
            area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

            inter_x1 = max(box_a[0], box_b[0])
            inter_y1 = max(box_a[1], box_b[1])
            inter_x2 = min(box_a[2], box_b[2])
            inter_y2 = min(box_a[3], box_b[3])

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            union_area = area_a + area_b - inter_area
            if union_area > 0:
                iou = inter_area / union_area
                if iou > iou_threshold:
                    should_keep = False
                    break
        if should_keep:
            kept.append(det)

    return kept


class YoloDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float = 0.20,
        iou_threshold: float = 0.45,
        image_size: int = 320,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self._model: Any = None
        manifest_path = self.model_path.parent / "model_manifest.json"
        if manifest_path.is_file():
            self.model_version = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "model_version"
            ]
        else:
            self.model_version = self.model_path.parent.name

    def _selected_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO checkpoint is missing at {self.model_path}. Run make train before inference."
            )
        self._model = _get_cached_yolo(self.model_path)

    def _predict_raw(
        self, image_bgr: np.ndarray, offset_x: int = 0, offset_y: int = 0
    ) -> list[Detection]:
        import torch

        with torch.inference_mode():
            result = self._model.predict(
                source=image_bgr,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self._selected_device(),
                verbose=False,
                max_det=20,
            )[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                raw_name = str(names[class_id])
                try:
                    class_name = DefectClass(raw_name)
                except ValueError:
                    class_name = DefectClass.CLEANABLE
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (round(float(value), 3) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=round(confidence, 6),
                        bbox_xyxy=[
                            round(x1 + offset_x, 3),
                            round(y1 + offset_y, 3),
                            round(x2 + offset_x, 3),
                            round(y2 + offset_y, 3),
                        ],
                        mask=None,
                    )
                )
        return detections

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: str,
        timestamp: float,
        image_id: str,
        use_slicing: bool = True,
    ) -> DetectorFrame:
        if self._model is None:
            self.load()
        assert self._model is not None

        h, w = image_bgr.shape[:2]
        all_detections: list[Detection] = []

        # 1. Global full-frame pass (for macro defects)
        all_detections.extend(self._predict_raw(image_bgr))

        # 2. Sliced Aided Hyper Inference (SAHI) on high-resolution images
        if use_slicing and (h > 640 or w > 640):
            slices = _calculate_slices(h, w, slice_size=480, overlap_ratio=0.25)
            for y1, x1, y2, x2 in slices:
                tile = image_bgr[y1:y2, x1:x2]
                all_detections.extend(self._predict_raw(tile, offset_x=x1, offset_y=y1))

        # 3. Non-Maximum Suppression to deduplicate cross-tile boxes
        merged_detections = _non_max_suppression(all_detections, self.iou_threshold)[:20]

        return DetectorFrame(
            frame_id=frame_id,
            timestamp=round(timestamp, 6),
            image_id=image_id,
            model_version=self.model_version,
            detections=merged_detections,
        )
