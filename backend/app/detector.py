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
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required; run make setup.") from exc
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


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
        self._model = None
        self.model_version = self._read_model_version()

    def _read_model_version(self) -> str:
        manifest_path = self.model_path.parent / "model_manifest.json"
        if manifest_path.exists():
            return str(json.loads(manifest_path.read_text(encoding="utf-8")).get("model_version", "glasseye-yolo-v1"))
        return "glasseye-yolo-v1"

    def _selected_device(self) -> str | None:
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

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: str,
        timestamp: float,
        image_id: str,
    ) -> DetectorFrame:
        if self._model is None:
            self.load()
        assert self._model is not None
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
                        bbox_xyxy=[x1, y1, x2, y2],
                        mask=None,
                    )
                )
        return DetectorFrame(
            frame_id=frame_id,
            timestamp=round(timestamp, 6),
            image_id=image_id,
            model_version=self.model_version,
            detections=detections,
        )
