"""Frozen public contracts shared by the detector, mission runner, and dashboard."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DefectClass(StrEnum):
    CLEANABLE = "cleanable_surface_issue"
    STRUCTURAL = "structural_issue"


class PolicyOutcome(StrEnum):
    CLEAN = "CLEAN"
    ESCALATE = "ESCALATE"
    REVIEW = "REVIEW"


class VlmVerdict(StrEnum):
    """Strict structured verdict an advisory VLM may return."""

    CONFIRM = "confirm"
    REJECT = "reject"
    ESCALATE = "escalate"


class IssueStatus(StrEnum):
    IDLE = "IDLE"
    INSPECTING = "INSPECTING"
    DETECTED = "DETECTED"
    EVIDENCE_READY = "EVIDENCE_READY"
    DECIDED = "DECIDED"
    CLEANING = "CLEANING"
    REINSPECTING = "REINSPECTING"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"
    REVIEW = "REVIEW"


class Detection(StrictModel):
    class_name: DefectClass
    class_id: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    mask: list[list[float]] | None = None
    track_id: int | None = Field(default=None, ge=1)


class DetectorFrame(StrictModel):
    """The detector contract from the project specification."""

    frame_id: str
    timestamp: float = Field(ge=0)
    image_id: str
    model_version: str
    detections: list[Detection]


class VideoInference(StrictModel):
    video_id: str
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    sampled_frames: int = Field(ge=0)
    model_version: str
    elapsed_seconds: float = Field(ge=0)
    frames: list[DetectorFrame]


class FacadeLocation(StrictModel):
    panel_id: str
    normalized_centroid: list[float] = Field(min_length=2, max_length=2)


class EvidenceRecord(StrictModel):
    evidence_id: str
    frame_id: str
    timestamp: float = Field(ge=0)
    artifact_ref: str
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)


class PolicyDecision(StrictModel):
    issue_id: str
    outcome: PolicyOutcome
    reason_code: str
    confidence: float = Field(ge=0, le=1)


class VlmReview(StrictModel):
    verdict: VlmVerdict
    rationale: str = Field(min_length=1)
    provider: str
    model: str | None = None
    latency_ms: int = Field(ge=0)


class FacadeIssue(StrictModel):
    issue_id: str
    track_id: int = Field(ge=1)
    class_name: DefectClass
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    location: FacadeLocation
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    decision: PolicyDecision
    status: IssueStatus
    action_taken: str | None = None
    verification_reason: str | None = None
    vlm_review: VlmReview | None = None


class MissionEvent(StrictModel):
    event_id: str
    mission_id: str
    sequence: int = Field(ge=1)
    timestamp: float = Field(ge=0)
    event_type: str
    source: str
    issue_id: str | None = None
    track_id: int | None = Field(default=None, ge=1)
    reason_code: str
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class MissionResult(StrictModel):
    mission_id: str
    scenario_seed: int
    model_version: str
    model_path: str
    state: str
    issues: list[FacadeIssue]
    events: list[MissionEvent]
    preinspection: VideoInference
    reinspection: VideoInference
    event_log_ref: str
    replay_digest: str
    inference_benchmark: dict[str, float]


class ReplayProjection(StrictModel):
    mission_id: str
    event_count: int = Field(ge=0)
    issue_statuses: dict[str, IssueStatus]
    replay_digest: str
