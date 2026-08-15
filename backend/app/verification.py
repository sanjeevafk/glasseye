"""Reinspection verification using detector output, never a UI-only status change."""

from __future__ import annotations

from .schemas import DefectClass, IssueStatus, VideoInference
from .tracker import iou


def issue_visible_after_reinspection(
    *,
    class_name: DefectClass,
    bbox_xyxy: list[float],
    reinspection: VideoInference,
    min_iou: float = 0.10,
) -> bool:
    for frame in reinspection.frames:
        for detection in frame.detections:
            if detection.class_name == class_name and iou(bbox_xyxy, detection.bbox_xyxy) >= min_iou:
                return True
    return False


def verify_cleaning(
    *,
    class_name: DefectClass,
    bbox_xyxy: list[float],
    reinspection: VideoInference,
) -> tuple[IssueStatus, str]:
    if not issue_visible_after_reinspection(
        class_name=class_name,
        bbox_xyxy=bbox_xyxy,
        reinspection=reinspection,
    ):
        return IssueStatus.RESOLVED, "NO_DEFECT_AT_REINSPECTION"
    return IssueStatus.UNRESOLVED, "DEFECT_STILL_VISIBLE_AT_REINSPECTION"
