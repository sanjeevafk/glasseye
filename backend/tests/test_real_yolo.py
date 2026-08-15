from __future__ import annotations

import cv2
import pytest

from app.detector import YoloDetector
from app.paths import models_root
from app.perception import infer_video
from app.synthetic import create_demo_media

CHECKPOINT = models_root() / "glasseye-yolo-v1" / "best.pt"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run make train to produce the project checkpoint.")
def test_project_checkpoint_loads_and_returns_video_detector_contract():
    media = create_demo_media()
    detector = YoloDetector(CHECKPOINT)
    capture = cv2.VideoCapture(str(media.preinspection_video))
    ok, image = capture.read()
    capture.release()
    assert ok
    still = detector.predict_frame(image, frame_id="still-0000", timestamp=0, image_id="still")
    result = infer_video(media.preinspection_video, detector, sample_stride=4).inference

    assert still.detections
    assert all(detection.mask is None for detection in still.detections)
    assert result.frames
    assert all(frame.frame_id and frame.timestamp >= 0 for frame in result.frames)
    assert all(detection.mask is None for frame in result.frames for detection in frame.detections)
    assert any(frame.detections for frame in result.frames)
