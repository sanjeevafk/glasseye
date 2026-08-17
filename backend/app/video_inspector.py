"""Video inspector for user-uploaded drone flight footage."""

from __future__ import annotations

import base64
import gc
import io
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .detector import YoloDetector
from .image_inspector import (
    DetectionItem,
    InspectionRecommendation,
    _classify_defect,
    _get_model_checkpoint,
)
from .localization import locate_bbox
from .schemas import PolicyOutcome, StrictModel


class VideoFrameDetection(StrictModel):
    frame_index: int
    timestamp_seconds: float
    detections_count: int
    detections: list[DetectionItem]
    thumbnail_data_uri: str | None = None
    has_critical_defect: bool = False


class PanelDamageSummary(StrictModel):
    panel_id: str
    row: int
    col: int
    defect_count: int
    max_severity: int
    primary_type: str  # "structural" | "surface" | "nominal"
    status: str  # "CRITICAL" | "ATTENTION" | "NOMINAL"


class VideoInspectionResult(StrictModel):
    inspection_id: str
    filename: str
    video_url: str | None = None
    timestamp: float
    duration_seconds: float
    fps: float
    total_video_frames: int
    sampled_frames_count: int
    model_version: str
    total_detections_count: int
    structural_defect_count: int
    surface_defect_count: int
    health_score: int  # 0 to 100
    health_status: str  # "HEALTHY" | "SURFACE_ATTENTION_NEEDED" | "CRITICAL_STRUCTURAL_ALERT"
    primary_recommendation: InspectionRecommendation
    panel_damage_map: dict[str, PanelDamageSummary]
    frames: list[VideoFrameDetection]
    highlight_frames: list[VideoFrameDetection]


def inspect_video_file(
    video_path: Path,
    filename: str = "drone_flight.mp4",
    sample_fps: float = 1.0,
    confidence: float = 0.20,
    model_choice: str | None = None,
    max_samples: int = 30,
    video_url: str | None = None,
) -> VideoInspectionResult:
    """Process a drone video, extracting frames, running YOLOv8, and accumulating panel damage."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {filename}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if native_fps <= 0:
        native_fps = 24.0
    duration_seconds = total_frames / native_fps if total_frames > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    checkpoint_path, model_version = _get_model_checkpoint(model_choice)
    detector = YoloDetector(checkpoint_path, confidence=confidence, image_size=640)
    detector.load()

    # Determine frame step interval to sample roughly sample_fps frames per second,
    # capped at max_samples to ensure low CPU & RAM overhead.
    ideal_step = max(1, int(round(native_fps / max(0.1, sample_fps))))
    if total_frames > 0 and (total_frames // ideal_step) > max_samples:
        ideal_step = max(1, total_frames // max_samples)

    frame_results: list[VideoFrameDetection] = []
    panel_accumulator: dict[str, list[DetectionItem]] = {}
    total_structural = 0
    total_surface = 0
    total_detections = 0

    frame_idx = 0
    sampled_count = 0

    while cap.isOpened() and sampled_count < max_samples:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        if frame_idx % ideal_step == 0:
            ts = frame_idx / native_fps
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            f_height, f_width = rgb_frame.shape[:2]

            detector_frame = detector.predict_frame(
                rgb_frame,
                frame_id=f"f_{frame_idx}",
                timestamp=ts,
                image_id=f"{filename}_{frame_idx}",
            )

            detections_list: list[DetectionItem] = []
            has_crit = False

            for det in detector_frame.detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                raw_cls = det.class_name.value
                disp_name, sev, ctype = _classify_defect(raw_cls, det.confidence, [x1, y1, x2, y2], f_width, f_height)
                panel_loc = locate_bbox([x1, y1, x2, y2], f_width, f_height)
                area_frac = ((x2 - x1) * (y2 - y1)) / float(f_width * f_height)

                item = DetectionItem(
                    detection_id=f"det_{sampled_count}_{len(detections_list)}",
                    class_name=raw_cls,
                    display_name=disp_name,
                    confidence=round(det.confidence, 3),
                    bbox_xyxy=[round(c, 1) for c in [x1, y1, x2, y2]],
                    normalized_bbox=[round(x1 / f_width, 4), round(y1 / f_height, 4), round(x2 / f_width, 4), round(y2 / f_height, 4)],
                    panel_id=panel_loc.panel_id,
                    area_fraction=round(area_frac, 6),
                    severity_score=sev,
                    classification_type=ctype,
                )
                detections_list.append(item)
                panel_accumulator.setdefault(panel_loc.panel_id, []).append(item)

                if ctype == "structural":
                    total_structural += 1
                    has_crit = True
                else:
                    total_surface += 1
                total_detections += 1

            # Render lightweight annotated thumbnail for key frames
            thumbnail_uri = None
            if len(detections_list) > 0 or sampled_count % 3 == 0:
                annotated_img = Image.fromarray(rgb_frame)
                draw = ImageDraw.Draw(annotated_img)
                for d in detections_list:
                    bx = d.bbox_xyxy
                    color = "#ef4444" if d.classification_type == "structural" else "#f59e0b"
                    draw.rectangle([bx[0], bx[1], bx[2], bx[3]], outline=color, width=3)
                
                # Resize thumbnail to max 320px width for fast JSON delivery
                annotated_img.thumbnail((320, 240), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                annotated_img.save(buf, format="JPEG", quality=75, optimize=True)
                thumbnail_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                buf.close()

            frame_results.append(
                VideoFrameDetection(
                    frame_index=frame_idx,
                    timestamp_seconds=round(ts, 2),
                    detections_count=len(detections_list),
                    detections=detections_list,
                    thumbnail_data_uri=thumbnail_uri,
                    has_critical_defect=has_crit,
                )
            )
            sampled_count += 1

        frame_idx += 1

    cap.release()
    gc.collect()

    # Build 4x3 panel damage map
    panel_damage_map: dict[str, PanelDamageSummary] = {}
    for r in range(4):
        for c in range(3):
            pid = f"P-{r}-{c}"
            items = panel_accumulator.get(pid, [])
            if not items:
                panel_damage_map[pid] = PanelDamageSummary(
                    panel_id=pid,
                    row=r,
                    col=c,
                    defect_count=0,
                    max_severity=0,
                    primary_type="nominal",
                    status="NOMINAL",
                )
            else:
                max_sev = max(it.severity_score for it in items)
                has_s = any(it.classification_type == "structural" for it in items)
                ptype = "structural" if has_s else "surface"
                pstatus = "CRITICAL" if has_s else ("ATTENTION" if max_sev > 40 else "NOMINAL")
                panel_damage_map[pid] = PanelDamageSummary(
                    panel_id=pid,
                    row=r,
                    col=c,
                    defect_count=len(items),
                    max_severity=max_sev,
                    primary_type=ptype,
                    status=pstatus,
                )

    # Compute overall health score and policy recommendation
    if total_structural > 0:
        health_score = max(10, 100 - (total_structural * 25 + total_surface * 5))
        health_status = "CRITICAL_STRUCTURAL_ALERT"
        rec = InspectionRecommendation(
            outcome=PolicyOutcome.ESCALATE,
            urgency="HIGH",
            badge="ESCALATE: STRUCTURAL HAZARD",
            summary=f"Automated drone scan detected {total_structural} structural defects across facade panels. Autonomous wash prohibited.",
            action_steps=[
                "Halt simulated autonomous wash operations immediately.",
                f"Dispatch structural engineering team to inspect flagged panels: {', '.join([p for p, data in panel_damage_map.items() if data.status == 'CRITICAL']) or 'Multiple'}.",
                "Log human-in-the-loop work order for facade crack stabilization.",
            ],
        )
    elif total_surface > 0:
        health_score = max(50, 100 - total_surface * 8)
        health_status = "SURFACE_ATTENTION_NEEDED"
        rec = InspectionRecommendation(
            outcome=PolicyOutcome.CLEAN,
            urgency="MEDIUM",
            badge="APPROVED: SURFACE REMEDIATION",
            summary=f"Detected {total_surface} cleanable surface blemishes (dirt/stains). No structural fractures found.",
            action_steps=[
                "Approve simulated drone pressure wash on affected facade panels.",
                "Schedule routine follow-up aerial survey post-cleaning.",
            ],
        )
    else:
        health_score = 98
        health_status = "HEALTHY"
        rec = InspectionRecommendation(
            outcome=PolicyOutcome.CLEAN,
            urgency="LOW",
            badge="NOMINAL: FACADE INTACT",
            summary="All scanned facade panels are in nominal condition with no structural or surface anomalies.",
            action_steps=["Maintain standard scheduled flight monitoring cycle."],
        )

    # Sort highlight frames by highest detection severity
    highlight_frames = sorted(
        [f for f in frame_results if f.detections_count > 0 and f.thumbnail_data_uri],
        key=lambda f: max((d.severity_score for d in f.detections), default=0),
        reverse=True,
    )[:5]

    return VideoInspectionResult(
        inspection_id=f"vid_insp_{uuid.uuid4().hex[:8]}",
        filename=filename,
        video_url=video_url,
        timestamp=time.time(),
        duration_seconds=round(duration_seconds, 2),
        fps=round(native_fps, 2),
        total_video_frames=total_frames,
        sampled_frames_count=len(frame_results),
        model_version=model_version,
        total_detections_count=total_detections,
        structural_defect_count=total_structural,
        surface_defect_count=total_surface,
        health_score=health_score,
        health_status=health_status,
        primary_recommendation=rec,
        panel_damage_map=panel_damage_map,
        frames=frame_results,
        highlight_frames=highlight_frames,
    )


def inspect_video_bytes(
    content: bytes,
    filename: str = "drone_flight.mp4",
    sample_fps: float = 1.0,
    confidence: float = 0.20,
    model_choice: str | None = None,
    max_samples: int = 30,
) -> VideoInspectionResult:
    """Inspect in-memory video bytes by writing to a temporary file safely."""
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return inspect_video_file(
            tmp_path,
            filename=filename,
            sample_fps=sample_fps,
            confidence=confidence,
            model_choice=model_choice,
            max_samples=max_samples,
        )
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass
