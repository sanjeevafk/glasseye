from app.policies import FacadePolicyEngine, PolicyConfig
from app.schemas import DefectClass, PolicyOutcome


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
