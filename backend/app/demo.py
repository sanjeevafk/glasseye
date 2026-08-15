"""The deterministic GlassEye inspection-to-verification mission runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .detector import DetectorAdapter, YoloDetector
from .events import EventLog
from .localization import locate_bbox
from .paths import artifacts_root, models_root, repo_root
from .perception import VideoRun, infer_video, save_evidence_crop
from .policies import FacadePolicyEngine, PolicyConfig
from .replay import replay_events
from .schemas import (
    DefectClass,
    Detection,
    FacadeIssue,
    IssueStatus,
    MissionResult,
    PolicyOutcome,
)
from .simulator import CleaningSimulator
from .state_machines import IssueStateMachine
from .synthetic import DEMO_SEED, create_demo_media
from .verification import issue_visible_after_reinspection, verify_cleaning

MISSION_ID = "glasseye-seed-20260815"
MODEL_VERSION = "glasseye-yolo-v1"


class DemoExecutionError(RuntimeError):
    pass


def _artifact_reference(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class Candidate:
    class_name: DefectClass
    track_id: int
    observations: list[tuple[int, int, Detection]]

    @property
    def representative(self) -> tuple[int, int, Detection]:
        return max(self.observations, key=lambda item: item[2].confidence)

    @property
    def confidence(self) -> float:
        return max(item[2].confidence for item in self.observations)


def _dominant_candidates(video: VideoRun) -> dict[DefectClass, Candidate]:
    groups: dict[tuple[DefectClass, int], list[tuple[int, int, Detection]]] = {}
    for frame_index, frame in enumerate(video.inference.frames):
        for detection_index, detection in enumerate(frame.detections):
            if detection.track_id is None:
                raise DemoExecutionError("Tracked video inference returned a detection without a track ID.")
            groups.setdefault((detection.class_name, detection.track_id), []).append(
                (frame_index, detection_index, detection)
            )
    candidates = [
        Candidate(class_name=class_name, track_id=track_id, observations=observations)
        for (class_name, track_id), observations in groups.items()
    ]
    result: dict[DefectClass, Candidate] = {}
    for class_name in DefectClass:
        options = [candidate for candidate in candidates if candidate.class_name == class_name]
        if options:
            result[class_name] = max(
                options,
                key=lambda item: (len(item.observations), item.confidence, -item.track_id),
            )
    return result


class DemoRunner:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        detector: DetectorAdapter | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.model_path = model_path or models_root() / MODEL_VERSION / "best.pt"
        self.detector = detector
        self.output_root = output_root or artifacts_root() / "demo"

    def _detector(self) -> DetectorAdapter:
        return self.detector or YoloDetector(
            self.model_path,
            confidence=0.20,
            iou_threshold=0.45,
            image_size=320,
        )

    def _transition(
        self,
        *,
        machine: IssueStateMachine,
        next_state: IssueStatus,
        event_log: EventLog,
        issue_id: str,
        track_id: int,
        reason_code: str,
        evidence_refs: list[str] | None = None,
    ) -> IssueStatus:
        previous, current = machine.transition(next_state)
        event_log.emit(
            "ISSUE_STATUS_CHANGED",
            source="issue_state_machine",
            issue_id=issue_id,
            track_id=track_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            payload={"previous_state": previous.value, "next_state": current.value},
        )
        return current

    def run(self, *, seed: int = DEMO_SEED) -> MissionResult:
        mission_dir = self.output_root / MISSION_ID
        media = create_demo_media(mission_dir / "media", seed=seed)
        event_path = mission_dir / "events.jsonl"
        event_log = EventLog(MISSION_ID, event_path)
        detector = self._detector()
        event_log.emit(
            "MISSION_STARTED",
            source="demo_runner",
            reason_code="SEEDED_DETERMINISTIC_SCENARIO",
            payload={"scenario_seed": seed, "simulation_only": True},
        )

        preinspection = infer_video(media.preinspection_video, detector, sample_stride=2, video_id="preinspection")
        event_log.emit(
            "VIDEO_INFERENCE_COMPLETED",
            source="perception",
            reason_code="PREINSPECTION_VIDEO_PROCESSED",
            payload={
                "video_id": preinspection.inference.video_id,
                "sampled_frames": preinspection.inference.sampled_frames,
                "model_version": preinspection.inference.model_version,
            },
        )
        candidates = _dominant_candidates(preinspection)
        missing = [class_name.value for class_name in DefectClass if class_name not in candidates]
        if missing:
            raise DemoExecutionError(
                f"YOLO inference did not find required seeded defects: {', '.join(missing)}. "
                "The demo refuses to substitute scripted detections."
            )

        policy = FacadePolicyEngine(PolicyConfig.from_file(Path(__file__).with_name("policy_rules.json")))
        simulator = CleaningSimulator()
        issues: list[FacadeIssue] = []
        states: dict[str, IssueStateMachine] = {}

        for class_name in (DefectClass.CLEANABLE, DefectClass.STRUCTURAL):
            candidate = candidates[class_name]
            frame_index, detection_index, representative = candidate.representative
            frame = preinspection.inference.frames[frame_index]
            issue_id = f"issue-{class_name.value}-{candidate.track_id:02d}"
            machine = IssueStateMachine()
            states[issue_id] = machine
            self._transition(
                machine=machine,
                next_state=IssueStatus.INSPECTING,
                event_log=event_log,
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code="RECORDED_VIDEO_INSPECTION_STARTED",
            )
            self._transition(
                machine=machine,
                next_state=IssueStatus.DETECTED,
                event_log=event_log,
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code="YOLO_TRACK_DETECTED",
            )
            event_log.emit(
                "TRACK_STABILIZED",
                source="tracker",
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code="IOU_TRACK_PERSISTED",
                payload={
                    "class_name": class_name.value,
                    "observations": len(candidate.observations),
                    "confidence": candidate.confidence,
                },
            )
            evidence = save_evidence_crop(
                image_bgr=preinspection.images[frame.frame_id],
                frame=frame,
                detection_index=detection_index,
                artifact_root=mission_dir / "evidence",
                evidence_id=f"evidence-{candidate.track_id:02d}",
            )
            evidence = evidence.model_copy(update={"artifact_ref": _artifact_reference(Path(evidence.artifact_ref))})
            self._transition(
                machine=machine,
                next_state=IssueStatus.EVIDENCE_READY,
                event_log=event_log,
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code="EVIDENCE_CROP_SAVED",
                evidence_refs=[evidence.evidence_id, evidence.artifact_ref],
            )
            location = locate_bbox(representative.bbox_xyxy, 640, 384)
            event_log.emit(
                "FACADE_LOCALIZED",
                source="facade_localization",
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code="KNOWN_PANEL_COORDINATE",
                evidence_refs=[evidence.evidence_id],
                payload=location.model_dump(),
            )
            decision = policy.evaluate(
                issue_id=issue_id,
                class_name=class_name,
                confidence=candidate.confidence,
                observation_count=len(candidate.observations),
            )
            self._transition(
                machine=machine,
                next_state=IssueStatus.DECIDED,
                event_log=event_log,
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code="POLICY_EVALUATED",
                evidence_refs=[evidence.evidence_id],
            )
            event_log.emit(
                "POLICY_DECIDED",
                source="policy_engine",
                issue_id=issue_id,
                track_id=candidate.track_id,
                reason_code=decision.reason_code,
                evidence_refs=[evidence.evidence_id],
                payload=decision.model_dump(mode="json"),
            )
            issues.append(
                FacadeIssue(
                    issue_id=issue_id,
                    track_id=candidate.track_id,
                    class_name=class_name,
                    confidence=candidate.confidence,
                    bbox_xyxy=representative.bbox_xyxy,
                    location=location,
                    evidence=[evidence],
                    decision=decision,
                    status=machine.state,
                )
            )

        cleanable_issue = next(issue for issue in issues if issue.class_name == DefectClass.CLEANABLE)
        structural_issue = next(issue for issue in issues if issue.class_name == DefectClass.STRUCTURAL)

        if cleanable_issue.decision.outcome != PolicyOutcome.CLEAN:
            raise DemoExecutionError(
                "Cleanable seeded defect did not meet the configured CLEAN policy gate; "
                "the model confidence or tracking quality must be improved."
            )
        clean_machine = states[cleanable_issue.issue_id]
        self._transition(
            machine=clean_machine,
            next_state=IssueStatus.CLEANING,
            event_log=event_log,
            issue_id=cleanable_issue.issue_id,
            track_id=cleanable_issue.track_id,
            reason_code="POLICY_APPROVED_SIMULATED_CLEANING",
            evidence_refs=[cleanable_issue.evidence[0].evidence_id],
        )
        cleaning = simulator.run(cleanable_issue.issue_id, cleanable_issue.location.panel_id)
        cleanable_issue.action_taken = cleaning.description
        event_log.emit(
            "SIMULATED_CLEANING_COMPLETED",
            source="cleaning_simulator",
            issue_id=cleanable_issue.issue_id,
            track_id=cleanable_issue.track_id,
            reason_code="SIMULATED_ACTION_ONLY",
            evidence_refs=[cleanable_issue.evidence[0].evidence_id],
            payload={"action_id": cleaning.action_id, "status": cleaning.status},
        )
        self._transition(
            machine=clean_machine,
            next_state=IssueStatus.REINSPECTING,
            event_log=event_log,
            issue_id=cleanable_issue.issue_id,
            track_id=cleanable_issue.track_id,
            reason_code="POST_CLEANING_REINSPECTION_REQUIRED",
        )

        if structural_issue.decision.outcome != PolicyOutcome.ESCALATE:
            raise DemoExecutionError("Structural seeded defect did not take the mandatory ESCALATE path.")
        structural_machine = states[structural_issue.issue_id]
        self._transition(
            machine=structural_machine,
            next_state=IssueStatus.ESCALATED,
            event_log=event_log,
            issue_id=structural_issue.issue_id,
            track_id=structural_issue.track_id,
            reason_code="STRUCTURAL_ISSUE_MANDATORY_ESCALATION",
            evidence_refs=[structural_issue.evidence[0].evidence_id],
        )
        structural_issue.status = structural_machine.state
        structural_issue.action_taken = "No cleaning performed; human structural review required."
        structural_issue.verification_reason = "ESCALATED_WITHOUT_CLEANING"
        event_log.emit(
            "ESCALATION_CREATED",
            source="policy_engine",
            issue_id=structural_issue.issue_id,
            track_id=structural_issue.track_id,
            reason_code="ESCALATE_NEVER_CLEAN",
            evidence_refs=[structural_issue.evidence[0].evidence_id],
            payload={"action_taken": structural_issue.action_taken},
        )

        reinspection = infer_video(media.reinspection_video, detector, sample_stride=2, video_id="reinspection")
        event_log.emit(
            "VIDEO_INFERENCE_COMPLETED",
            source="perception",
            reason_code="REINSPECTION_VIDEO_PROCESSED",
            payload={
                "video_id": reinspection.inference.video_id,
                "sampled_frames": reinspection.inference.sampled_frames,
                "model_version": reinspection.inference.model_version,
            },
        )
        final_status, verification_reason = verify_cleaning(
            class_name=cleanable_issue.class_name,
            bbox_xyxy=cleanable_issue.bbox_xyxy,
            reinspection=reinspection.inference,
        )
        self._transition(
            machine=clean_machine,
            next_state=final_status,
            event_log=event_log,
            issue_id=cleanable_issue.issue_id,
            track_id=cleanable_issue.track_id,
            reason_code=verification_reason,
            evidence_refs=[cleanable_issue.evidence[0].evidence_id],
        )
        cleanable_issue.status = clean_machine.state
        cleanable_issue.verification_reason = verification_reason
        event_log.emit(
            "VERIFICATION_COMPLETED",
            source="verification_service",
            issue_id=cleanable_issue.issue_id,
            track_id=cleanable_issue.track_id,
            reason_code=verification_reason,
            evidence_refs=[cleanable_issue.evidence[0].evidence_id],
            payload={"status": final_status.value},
        )
        if final_status == IssueStatus.UNRESOLVED:
            self._transition(
                machine=clean_machine,
                next_state=IssueStatus.ESCALATED,
                event_log=event_log,
                issue_id=cleanable_issue.issue_id,
                track_id=cleanable_issue.track_id,
                reason_code="UNRESOLVED_POST_CLEANING_ESCALATION",
            )
            cleanable_issue.status = clean_machine.state

        if not issue_visible_after_reinspection(
            class_name=DefectClass.STRUCTURAL,
            bbox_xyxy=structural_issue.bbox_xyxy,
            reinspection=reinspection.inference,
        ):
            raise DemoExecutionError(
                "Reinspection did not retain the structural issue. The demo must show a real persistent escalation."
            )
        event_log.emit(
            "REINSPECTION_CONFIRMED_STRUCTURAL_VISIBILITY",
            source="verification_service",
            issue_id=structural_issue.issue_id,
            track_id=structural_issue.track_id,
            reason_code="STRUCTURAL_ISSUE_PERSISTS",
            evidence_refs=[structural_issue.evidence[0].evidence_id],
        )
        event_log.emit(
            "MISSION_COMPLETED",
            source="demo_runner",
            reason_code="CLOSED_LOOP_DEMO_COMPLETE",
            payload={"simulation_only": True},
        )

        projection = replay_events(event_log.events)
        if projection.issue_statuses.get(cleanable_issue.issue_id) != cleanable_issue.status:
            raise DemoExecutionError("Event replay does not reproduce the cleanable issue result.")
        if projection.issue_statuses.get(structural_issue.issue_id) != structural_issue.status:
            raise DemoExecutionError("Event replay does not reproduce the structural issue result.")
        benchmark = {
            "preinspection_seconds": preinspection.inference.elapsed_seconds,
            "reinspection_seconds": reinspection.inference.elapsed_seconds,
            "frames_per_second": round(
                (preinspection.inference.sampled_frames + reinspection.inference.sampled_frames)
                / max(0.000001, preinspection.inference.elapsed_seconds + reinspection.inference.elapsed_seconds),
                3,
            ),
        }
        result = MissionResult(
            mission_id=MISSION_ID,
            scenario_seed=seed,
            model_version=detector.model_version,
            model_path=str(self.model_path.resolve()),
            state="COMPLETE",
            issues=issues,
            events=event_log.events,
            preinspection=preinspection.inference,
            reinspection=reinspection.inference,
            event_log_ref=_artifact_reference(event_path),
            replay_digest=projection.replay_digest,
            inference_benchmark=benchmark,
        )
        result_path = mission_dir / "result.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (self.output_root / "latest.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


def load_latest_demo(output_root: Path | None = None) -> MissionResult | None:
    path = (output_root or artifacts_root() / "demo") / "latest.json"
    return MissionResult.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
