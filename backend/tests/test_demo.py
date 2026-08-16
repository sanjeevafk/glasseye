from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.demo import DemoRunner
from app.replay import replay_log
from app.schemas import DefectClass, Detection, DetectorFrame, IssueStatus, VlmVerdict
from app.vlm import FixtureVlmProvider, VlmFailure


class DeterministicTestDetector:
    model_version = "glasseye-yolo-v1-test"

    def __init__(self, cleanable_confidence: float = 0.93) -> None:
        self.cleanable_confidence = cleanable_confidence

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: str,
        timestamp: float,
        image_id: str,
    ) -> DetectorFrame:
        del image_bgr
        structural = Detection(
            class_name=DefectClass.STRUCTURAL,
            class_id=1,
            confidence=0.94,
            bbox_xyxy=[332, 278, 468, 324],
            mask=None,
        )
        detections = [structural]
        if "preinspection" in frame_id:
            detections.insert(
                0,
                Detection(
                    class_name=DefectClass.CLEANABLE,
                    class_id=0,
                    confidence=self.cleanable_confidence,
                    bbox_xyxy=[174, 146, 302, 220],
                    mask=None,
                ),
            )
        return DetectorFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            image_id=image_id,
            model_version=self.model_version,
            detections=detections,
        )


class FailingVlmProvider:
    def review(self, image_bytes: bytes, metadata: dict) -> VlmVerdict:
        del image_bytes, metadata
        raise VlmFailure("simulated provider outage")


def test_demo_replays_same_closed_loop_outcome(tmp_path):
    runner = DemoRunner(
        detector=DeterministicTestDetector(),
        model_path=Path(tmp_path / "model.pt"),
        output_root=tmp_path / "artifacts" / "demo",
        vlm_provider=FixtureVlmProvider(),
    )
    result = runner.run()
    statuses = {issue.class_name: issue.status for issue in result.issues}
    projection = replay_log(tmp_path / result.event_log_ref)

    assert statuses == {
        DefectClass.CLEANABLE: IssueStatus.RESOLVED,
        DefectClass.STRUCTURAL: IssueStatus.ESCALATED,
    }
    assert projection.replay_digest == result.replay_digest
    for issue in result.issues:
        assert projection.issue_statuses[issue.issue_id] == issue.status


def test_demo_runs_structural_through_advisory_vlm(tmp_path):
    runner = DemoRunner(
        detector=DeterministicTestDetector(),
        model_path=Path(tmp_path / "model.pt"),
        output_root=tmp_path / "artifacts" / "demo",
        vlm_provider=FixtureVlmProvider(),
    )
    result = runner.run()
    structural = next(issue for issue in result.issues if issue.class_name == DefectClass.STRUCTURAL)
    cleanable = next(issue for issue in result.issues if issue.class_name == DefectClass.CLEANABLE)
    assert structural.vlm_review is not None
    assert structural.vlm_review.verdict == VlmVerdict.ESCALATE
    assert structural.vlm_review.provider == "fixture-vlm"
    assert cleanable.vlm_review is None  # high-confidence cleanable never routed
    event_types = [event.event_type for event in result.events]
    assert event_types.count("VLM_REVIEW_REQUESTED") == 1
    assert event_types.count("VLM_REVIEW_RESULT") == 1
    result_event = next(event for event in result.events if event.event_type == "VLM_REVIEW_RESULT")
    assert result_event.payload["verdict"] == "escalate"


def test_demo_vlm_failure_aborts_instead_of_cleaning(tmp_path):
    from app.demo import DemoExecutionError
    from app.events import EventLog

    output_root = tmp_path / "artifacts" / "demo"
    runner = DemoRunner(
        detector=DeterministicTestDetector(cleanable_confidence=0.4),
        model_path=Path(tmp_path / "model.pt"),
        output_root=output_root,
        vlm_provider=FailingVlmProvider(),
    )
    with pytest.raises(DemoExecutionError):
        runner.run()
    events = EventLog.load(output_root / "glasseye-seed-20260815" / "events.jsonl")
    failure_events = [event for event in events if event.event_type == "VLM_REVIEW_RESULT"]
    assert failure_events
    assert all(event.reason_code == "VLM_UNAVAILABLE" for event in failure_events)
    assert not any(event.event_type == "SIMULATED_CLEANING_COMPLETED" for event in events)
