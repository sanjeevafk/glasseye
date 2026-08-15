#!/usr/bin/env python3
"""Run image/video-compatible YOLO inference and save detector-contract JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.detector import YoloDetector
from app.paths import artifacts_root, models_root
from app.perception import infer_video
from app.synthetic import create_demo_media


def main() -> int:
    media = create_demo_media()
    detector = YoloDetector(models_root() / "glasseye-yolo-v1" / "best.pt")
    capture = cv2.VideoCapture(str(media.preinspection_video))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(
            "Could not read the deterministic still-image inference frame."
        )

    image_result = detector.predict_frame(
        image,
        frame_id="preinspection-still-0000",
        timestamp=0.0,
        image_id="preinspection-still",
    )
    video_result = infer_video(
        media.preinspection_video, detector, sample_stride=2
    ).inference
    if not image_result.detections:
        raise RuntimeError("YOLO still-image inference produced no detections.")
    if not video_result.frames or not any(
        frame.detections for frame in video_result.frames
    ):
        raise RuntimeError("YOLO video inference produced no detections.")
    for frame in [image_result, *video_result.frames]:
        for detection in frame.detections:
            if detection.mask is not None:
                raise RuntimeError("The detection-model demo must emit null masks.")
    root = artifacts_root()
    (root / "inference-image-contract.json").write_text(
        image_result.model_dump_json(indent=2), encoding="utf-8"
    )
    (root / "inference-video-contract.json").write_text(
        video_result.model_dump_json(indent=2), encoding="utf-8"
    )
    report = {
        "image_contract": image_result.model_dump(mode="json"),
        "video_contract": video_result.model_dump(mode="json"),
    }
    (root / "inference-contract.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
