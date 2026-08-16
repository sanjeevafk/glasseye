from app.policies import FacadePolicyEngine, PolicyConfig
from app.schemas import DefectClass, PolicyOutcome, VlmVerdict


def test_policy_cleans_only_stable_high_confidence_surface_issue():
    engine = FacadePolicyEngine(PolicyConfig(clean_confidence=0.7, stable_observations=3))

    clean = engine.evaluate(
        issue_id="clean",
        class_name=DefectClass.CLEANABLE,
        confidence=0.88,
        observation_count=5,
    )
    review = engine.evaluate(
        issue_id="review",
        class_name=DefectClass.CLEANABLE,
        confidence=0.6,
        observation_count=5,
    )
    escalate = engine.evaluate(
        issue_id="structural",
        class_name=DefectClass.STRUCTURAL,
        confidence=0.2,
        observation_count=1,
    )

    assert clean.outcome == PolicyOutcome.CLEAN
    assert review.outcome == PolicyOutcome.REVIEW
    assert escalate.outcome == PolicyOutcome.ESCALATE


def test_policy_never_cleans_when_vlm_unavailable():
    engine = FacadePolicyEngine(PolicyConfig(clean_confidence=0.7, stable_observations=3))
    decision = engine.evaluate(
        issue_id="safe",
        class_name=DefectClass.CLEANABLE,
        confidence=0.99,
        observation_count=20,
        vlm_available=False,
    )
    assert decision.outcome == PolicyOutcome.REVIEW
    assert decision.reason_code == "VLM_UNAVAILABLE_SAFE_REVIEW"


def test_policy_review_when_vlm_rejects():
    engine = FacadePolicyEngine(PolicyConfig(clean_confidence=0.7, stable_observations=3))
    decision = engine.evaluate(
        issue_id="noise",
        class_name=DefectClass.CLEANABLE,
        confidence=0.99,
        observation_count=20,
        vlm_verdict=VlmVerdict.REJECT,
    )
    assert decision.outcome == PolicyOutcome.REVIEW
    assert decision.reason_code == "VLM_REJECTED_DETECTION"


def test_policy_escalates_when_vlm_advises_escalation():
    engine = FacadePolicyEngine(PolicyConfig(clean_confidence=0.7, stable_observations=3))
    decision = engine.evaluate(
        issue_id="severe",
        class_name=DefectClass.CLEANABLE,
        confidence=0.99,
        observation_count=20,
        vlm_verdict=VlmVerdict.ESCALATE,
    )
    assert decision.outcome == PolicyOutcome.ESCALATE
    assert decision.reason_code == "VLM_ADVISED_ESCALATION"


def test_policy_vlm_confirm_still_requires_confidence_gate():
    engine = FacadePolicyEngine(PolicyConfig(clean_confidence=0.7, stable_observations=3))
    confirmed_low_confidence = engine.evaluate(
        issue_id="weak",
        class_name=DefectClass.CLEANABLE,
        confidence=0.5,
        observation_count=5,
        vlm_verdict=VlmVerdict.CONFIRM,
    )
    confirmed_high_confidence = engine.evaluate(
        issue_id="strong",
        class_name=DefectClass.CLEANABLE,
        confidence=0.9,
        observation_count=5,
        vlm_verdict=VlmVerdict.CONFIRM,
    )
    assert confirmed_low_confidence.outcome == PolicyOutcome.REVIEW
    assert confirmed_high_confidence.outcome == PolicyOutcome.CLEAN


def test_policy_structural_escalates_even_without_vlm():
    engine = FacadePolicyEngine(PolicyConfig(clean_confidence=0.7, stable_observations=3))
    decision = engine.evaluate(
        issue_id="structural",
        class_name=DefectClass.STRUCTURAL,
        confidence=0.2,
        observation_count=1,
        vlm_available=False,
    )
    assert decision.outcome == PolicyOutcome.ESCALATE
