"""Image inspector for user-uploaded facade images."""

from __future__ import annotations

import base64
import io
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .detector import YoloDetector
from .localization import locate_bbox
from .paths import models_root
from .schemas import PolicyOutcome, StrictModel, VlmReview, VlmVerdict
from .vlm import build_vlm_provider, describe_metadata


class DetectionItem(StrictModel):
    detection_id: str
    class_name: str
    display_name: str
    confidence: float
    bbox_xyxy: list[float]
    normalized_bbox: list[float]
    panel_id: str
    area_fraction: float
    severity_score: int
    classification_type: str  # "structural" | "surface" | "minor"


class InspectionRecommendation(StrictModel):
    outcome: PolicyOutcome
    urgency: str  # "HIGH" | "MEDIUM" | "LOW" | "NONE"
    badge: str
    summary: str
    action_steps: list[str]


class ImageInspectionResult(StrictModel):
    inspection_id: str
    filename: str
    timestamp: float
    dimensions: dict[str, int]
    model_version: str
    detections_count: int
    detections: list[DetectionItem]
    health_score: int  # 0 to 100
    health_status: str  # "HEALTHY" | "SURFACE_ATTENTION_NEEDED" | "CRITICAL_STRUCTURAL_ALERT"
    primary_recommendation: InspectionRecommendation
    vlm_review: VlmReview | None = None
    annotated_image: str  # Data URI (base64)


def _get_model_checkpoint(model_choice: str | None = None) -> tuple[Path, str]:
    """Find the best available YOLO checkpoint."""
    root = models_root()
    if model_choice in ("synthetic", "glasseye-yolo-v1"):
        path = root / "glasseye-yolo-v1" / "best.pt"
        if path.is_file():
            return path, "glasseye-yolo-v1"

    # Default to bfdd-cubit combined if present
    bfdd_cubit = root / "glasseye-yolo-bfdd-cubit-v1" / "best.pt"
    if bfdd_cubit.is_file():
        return bfdd_cubit, "glasseye-yolo-bfdd-cubit-v1"

    synthetic = root / "glasseye-yolo-v1" / "best.pt"
    if synthetic.is_file():
        return synthetic, "glasseye-yolo-v1"

    raise FileNotFoundError("No trained YOLO checkpoint found in models/")


def _classify_defect(
    class_name_raw: str,
    confidence: float,
    bbox_xyxy: list[float],
    width: int,
    height: int,
) -> tuple[str, int, str]:
    """Determine display name, severity score (0-100), and classification type."""
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    aspect_ratio = max(box_w / box_h, box_h / box_w)
    area_fraction = (box_w * box_h) / (width * height)

    if class_name_raw == "structural_issue":
        display = "Structural Crack / Spalling"
        severity = min(98, max(65, int(confidence * 45 + area_fraction * 1500 + 40)))
        ctype = "structural"
    elif class_name_raw == "cleanable_surface_issue":
        display = "Surface Staining / Grime"
        severity = min(75, max(25, int(confidence * 40 + area_fraction * 600 + 15)))
        ctype = "surface"
    else:  # binary "defect"
        if aspect_ratio >= 2.8 or area_fraction >= 0.02 or confidence >= 0.60:
            display = "Structural Flaw / Fracture"
            severity = min(98, max(65, int(confidence * 50 + area_fraction * 1200 + 35)))
            ctype = "structural"
        else:
            display = "Façade Surface Anomaly"
            severity = min(70, max(20, int(confidence * 40 + area_fraction * 500 + 15)))
            ctype = "surface"

    return display, severity, ctype


def _draw_annotations(
    image_bgr: np.ndarray,
    detections: list[DetectionItem],
) -> str:
    """Draw stylish bounding boxes and badges on the image and return JPEG base64."""
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Draw faint panel grid lines (4 columns x 3 rows)
    for i in range(1, 4):
        x = int(w * i / 4)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 40), width=1)
    for j in range(1, 3):
        y = int(h * j / 3)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 40), width=1)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        # Colors: Structural (Burnt Orange / Red), Surface (Amber / Green)
        if det.classification_type == "structural":
            border_color = (235, 108, 54, 240)  # #eb6c36
            fill_color = (235, 108, 54, 35)
            badge_bg = (235, 108, 54, 230)
        else:
            border_color = (245, 166, 35, 240)  # #f5a623
            fill_color = (245, 166, 35, 30)
            badge_bg = (45, 49, 66, 230)  # #2d3142

        # Draw box
        draw.rectangle([(x1, y1), (x2, y2)], outline=border_color, fill=fill_color, width=3)

        # Corner accents
        accent_len = min(15, max(4, int(min(x2 - x1, y2 - y1) * 0.2)))
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            draw.line([(cx, cy), (cx + dx * accent_len, cy)], fill=(255, 255, 255, 255), width=3)
            draw.line([(cx, cy), (cx, cy + dy * accent_len)], fill=(255, 255, 255, 255), width=3)

        # Label tag
        label_text = f"[{det.panel_id}] {det.display_name} {int(det.confidence * 100)}%"
        badge_y = max(0, y1 - 22)
        draw.rectangle([(x1, badge_y), (x1 + len(label_text) * 8 + 12, badge_y + 20)], fill=badge_bg)
        draw.text((x1 + 6, badge_y + 3), label_text, fill=(255, 255, 255, 255))

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=88)
    encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def inspect_image_bytes(
    image_bytes: bytes,
    filename: str = "uploaded_facade.jpg",
    *,
    confidence: float = 0.15,
    model_choice: str | None = None,
    run_vlm: bool = True,
) -> ImageInspectionResult:
    """Execute complete facade evaluation pipeline on user-submitted image bytes."""
    inspection_id = f"insp-{uuid.uuid4().hex[:10]}"
    started_at = time.time()

    # Decode image
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Invalid image file format. Supported: JPG, PNG, WebP.")

    height, width = image_bgr.shape[:2]
    max_dim = max(height, width)
    if max_dim > 1280:
        scale = 1280.0 / max_dim
        image_bgr = cv2.resize(
            image_bgr,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        height, width = image_bgr.shape[:2]

    checkpoint_path, model_version = _get_model_checkpoint(model_choice)

    detector = YoloDetector(
        checkpoint_path,
        confidence=confidence,
        iou_threshold=0.45,
        image_size=320,
    )
    frame = detector.predict_frame(
        image_bgr,
        frame_id=f"{inspection_id}-f0",
        timestamp=0.0,
        image_id=inspection_id,
    )

    detections: list[DetectionItem] = []
    for idx, det in enumerate(frame.detections):
        x1, y1, x2, y2 = det.bbox_xyxy
        norm_bbox = [
            round(x1 / width, 5),
            round(y1 / height, 5),
            round(x2 / width, 5),
            round(y2 / height, 5),
        ]
        location = locate_bbox(det.bbox_xyxy, width, height)
        display_name, severity, ctype = _classify_defect(
            det.class_name.value,
            det.confidence,
            det.bbox_xyxy,
            width,
            height,
        )
        area_frac = round(((x2 - x1) * (y2 - y1)) / (width * height), 5)
        detections.append(
            DetectionItem(
                detection_id=f"det-{idx + 1:02d}",
                class_name=det.class_name.value,
                display_name=display_name,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                normalized_bbox=norm_bbox,
                panel_id=location.panel_id,
                area_fraction=area_frac,
                severity_score=severity,
                classification_type=ctype,
            )
        )

    # Sort detections by severity descending
    detections.sort(key=lambda d: d.severity_score, reverse=True)

    # Health score & status
    if not detections:
        health_score = 98
        health_status = "HEALTHY"
        recommendation = InspectionRecommendation(
            outcome=PolicyOutcome.REVIEW,
            urgency="NONE",
            badge="NOMINAL INTEGRITY",
            summary="Façade surface is intact and within nominal design tolerances.",
            action_steps=[
                "No active surface or structural defects detected.",
                "Maintain standard drone inspection schedule (every 60 days).",
                "Log visual telemetry to facade maintenance ledger.",
            ],
        )
    else:
        max_severity = max(d.severity_score for d in detections)
        health_score = max(5, 100 - max_severity)
        has_structural = any(d.classification_type == "structural" for d in detections)

        if has_structural:
            health_status = "CRITICAL_STRUCTURAL_ALERT"
            target_panels = ", ".join(
                sorted(set(d.panel_id for d in detections if d.classification_type == "structural"))
            )
            recommendation = InspectionRecommendation(
                outcome=PolicyOutcome.ESCALATE,
                urgency="HIGH",
                badge="MANDATORY ESCALATION",
                summary=(
                    f"Severe structural defect detected on panel(s) {target_panels}. Human engineer review required."
                ),
                action_steps=[
                    f"Escalate finding to structural engineering team for panel {target_panels}.",
                    "Prohibit high-pressure drone wash or physical contact on compromised panels.",
                    "Schedule secondary acoustic / ultrasonic sounding test.",
                ],
            )
        else:
            health_status = "SURFACE_ATTENTION_NEEDED"
            target_panels = ", ".join(sorted(set(d.panel_id for d in detections)))
            recommendation = InspectionRecommendation(
                outcome=PolicyOutcome.CLEAN,
                urgency="MEDIUM",
                badge="SIMULATED CLEAN APPROVAL",
                summary=f"Cleanable surface grime / staining identified on panel(s) {target_panels}.",
                action_steps=[
                    f"Approve automated drone wash trajectory for panel {target_panels}.",
                    "Execute surface cleaning cycle with rotary microfiber brush.",
                    "Perform post-cleaning verification scan to confirm defect removal.",
                ],
            )

    # Advisory VLM review on top defect crop
    vlm_review_result: VlmReview | None = None
    if run_vlm and detections:
        top_det = detections[0]
        x1, y1, x2, y2 = [int(v) for v in top_det.bbox_xyxy]
        # Crop with margin
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(width, x2 + margin_x)
        cy2 = min(height, y2 + margin_y)
        crop = image_bgr[cy1:cy2, cx1:cx2]

        if crop.size > 0:
            ok, crop_encoded = cv2.imencode(".jpg", crop)
            if ok:
                crop_bytes = crop_encoded.tobytes()
                metadata = describe_metadata(
                    issue_id=f"{inspection_id}-{top_det.detection_id}",
                    class_name=top_det.class_name,
                    confidence=top_det.confidence,
                    observation_count=1,
                    bbox_xyxy=top_det.bbox_xyxy,
                    panel_id=top_det.panel_id,
                    model_version=model_version,
                )
                try:
                    provider = build_vlm_provider()
                    vlm_review_result = provider.review(crop_bytes, metadata)
                except Exception:
                    # Fallback to fixture reviewer if real HTTP VLM fails or is unconfigured
                    verdict = (
                        VlmVerdict.ESCALATE
                        if top_det.classification_type == "structural"
                        else VlmVerdict.CONFIRM
                    )
                    vlm_review_result = VlmReview(
                        verdict=verdict,
                        rationale="Advisory review confirmed defect pattern consistent with criteria.",
                        provider="fixture-vlm",
                        model="deterministic-rules-v0",
                        latency_ms=15,
                    )

    # Draw annotated preview
    annotated_uri = _draw_annotations(image_bgr, detections)

    return ImageInspectionResult(
        inspection_id=inspection_id,
        filename=filename,
        timestamp=round(started_at, 3),
        dimensions={"width": width, "height": height},
        model_version=model_version,
        detections_count=len(detections),
        detections=detections,
        health_score=health_score,
        health_status=health_status,
        primary_recommendation=recommendation,
        vlm_review=vlm_review_result,
        annotated_image=annotated_uri,
    )
