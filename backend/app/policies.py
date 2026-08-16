"""Side-effect-free, configuration-backed remediation decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import DefectClass, PolicyDecision, PolicyOutcome, VlmVerdict


@dataclass(frozen=True)
class PolicyConfig:
    clean_confidence: float = 0.65
    stable_observations: int = 3

    @classmethod
    def from_file(cls, path: Path) -> PolicyConfig:
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            clean_confidence=float(values["clean_confidence"]),
            stable_observations=int(values["stable_observations"]),
        )


class FacadePolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        issue_id: str,
        class_name: DefectClass,
        confidence: float,
        observation_count: int,
        vlm_verdict: VlmVerdict | None = None,
        vlm_available: bool = True,
    ) -> PolicyDecision:
        """Decide CLEAN / ESCALATE / REVIEW.

        ``vlm_verdict`` is the advisory outcome of an optional VLM review of
        the evidence crop.  The VLM never controls an actuator: it can only
        move a cleanable case away from CLEAN (reject -> REVIEW) or toward
        human escalation (escalate -> ESCALATE).  A confirm verdict still
        requires the deterministic confidence/stability gate.  Structural
        issues always ESCALATE regardless of the advisory verdict.
        """
        if class_name == DefectClass.STRUCTURAL:
            return PolicyDecision(
                issue_id=issue_id,
                outcome=PolicyOutcome.ESCALATE,
                reason_code="STRUCTURAL_ISSUE_MANDATORY_ESCALATION",
                confidence=confidence,
            )
        if not vlm_available:
            return PolicyDecision(
                issue_id=issue_id,
                outcome=PolicyOutcome.REVIEW,
                reason_code="VLM_UNAVAILABLE_SAFE_REVIEW",
                confidence=confidence,
            )
        if vlm_verdict == VlmVerdict.REJECT:
            return PolicyDecision(
                issue_id=issue_id,
                outcome=PolicyOutcome.REVIEW,
                reason_code="VLM_REJECTED_DETECTION",
                confidence=confidence,
            )
        if vlm_verdict == VlmVerdict.ESCALATE:
            return PolicyDecision(
                issue_id=issue_id,
                outcome=PolicyOutcome.ESCALATE,
                reason_code="VLM_ADVISED_ESCALATION",
                confidence=confidence,
            )
        if confidence >= self.config.clean_confidence and observation_count >= self.config.stable_observations:
            return PolicyDecision(
                issue_id=issue_id,
                outcome=PolicyOutcome.CLEAN,
                reason_code="HIGH_CONFIDENCE_STABLE_CLEANABLE_TRACK",
                confidence=confidence,
            )
        return PolicyDecision(
            issue_id=issue_id,
            outcome=PolicyOutcome.REVIEW,
            reason_code="INSUFFICIENT_CONFIDENCE_OR_TRACK_STABILITY",
            confidence=confidence,
        )
