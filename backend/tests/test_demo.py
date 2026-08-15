from __future__ import annotations

from pathlib import Path

import numpy as np

from app.demo import DemoRunner
from app.replay import replay_log
from app.schemas import DefectClass, Detection, DetectorFrame, IssueStatus


class DeterministicTestDetector:
    model_version = "glasseye-yolo-v1-test"

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
                    confidence=0.93,
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


def test_demo_replays_same_closed_loop_outcome(tmp_path):
    runner = DemoRunner(
        detector=DeterministicTestDetector(),
        model_path=Path(tmp_path / "model.pt"),
        output_root=tmp_path / "artifacts" / "demo",
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
