"""Recorded-video inference, tracking, and evidence crop production."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detector import DetectorAdapter
from .schemas import DetectorFrame, EvidenceRecord, VideoInference
from .tracker import IoUTracker


@dataclass
class VideoRun:
    inference: VideoInference
    images: dict[str, np.ndarray]


def infer_video(
    video_path: Path,
    detector: DetectorAdapter,
    *,
    sample_stride: int = 2,
    video_id: str | None = None,
) -> VideoRun:
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least one")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open recorded video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = IoUTracker()
    frames: list[DetectorFrame] = []
    images: dict[str, np.ndarray] = {}
    index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % sample_stride == 0:
                frame_id = f"{video_path.stem}-frame-{index:04d}"
                result = detector.predict_frame(
                    image,
                    frame_id=frame_id,
                    timestamp=index / fps,
                    image_id=f"{video_path.stem}-{index:04d}",
                )
                tracked = tracker.update(result.detections)
                frames.append(result.model_copy(update={"detections": tracked}))
                images[frame_id] = image.copy()
            index += 1
    finally:
        capture.release()
    return VideoRun(
        inference=VideoInference(
            video_id=video_id or video_path.stem,
            fps=round(fps, 6),
            frame_count=frame_count,
            sampled_frames=len(frames),
            model_version=detector.model_version,
            elapsed_seconds=round(time.perf_counter() - started, 6),
            frames=frames,
        ),
        images=images,
    )


def save_evidence_crop(
    *,
    image_bgr: np.ndarray,
    frame: DetectorFrame,
    detection_index: int,
    artifact_root: Path,
    evidence_id: str,
) -> EvidenceRecord:
    detection = frame.detections[detection_index]
    x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
    height, width = image_bgr.shape[:2]
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
    crop = image_bgr[y1:y2, x1:x2]
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = artifact_root / f"{evidence_id}.jpg"
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output, quality=94)
    return EvidenceRecord(
        evidence_id=evidence_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        artifact_ref=output.as_posix(),
        bbox_xyxy=detection.bbox_xyxy,
    )
