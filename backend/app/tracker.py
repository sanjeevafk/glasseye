"""A small deterministic IoU tracker for recorded inspection video."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Detection


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    left, top = max(ax1, bx1), max(ay1, by1)
    right, bottom = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox_xyxy: list[float]
    observations: list[Detection] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return max((item.confidence for item in self.observations), default=0.0)


class IoUTracker:
    def __init__(self, match_iou: float = 0.20) -> None:
        self.match_iou = match_iou
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(self, detections: list[Detection]) -> list[Detection]:
        assigned: set[int] = set()
        tracked: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidates = [
                track
                for track in self._tracks.values()
                if track.track_id not in assigned and track.class_name == detection.class_name
            ]
            matching = max(candidates, key=lambda track: iou(track.bbox_xyxy, detection.bbox_xyxy), default=None)
            if matching is None or iou(matching.bbox_xyxy, detection.bbox_xyxy) < self.match_iou:
                matching = Track(
                    track_id=self._next_id,
                    class_name=detection.class_name,
                    bbox_xyxy=list(detection.bbox_xyxy),
                )
                self._tracks[self._next_id] = matching
                self._next_id += 1
            assigned.add(matching.track_id)
            item = detection.model_copy(update={"track_id": matching.track_id})
            matching.bbox_xyxy = list(item.bbox_xyxy)
            matching.observations.append(item)
            tracked.append(item)
        return tracked
