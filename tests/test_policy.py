import pytest
from pydantic import ValidationError

from kea.claims.models import DamageAssessment, PolicyDecision, ReviewDecision
from kea.claims.policy import decide, settle


def assessment(damage_type="tear", severity=3, confidence=0.9) -> DamageAssessment:
    return DamageAssessment(
        damage_type=damage_type, severity=severity, confidence=confidence, description="test"
    )


def test_small_claim_is_paid_without_review():
    decision = decide(assessment("tear", 3), declared_value_usd=500, prior_claims=0)
    assert decision.payout_usd == 100
    assert not decision.needs_review


def test_payout_is_capped_at_declared_value():
    decision = decide(assessment("crushed", 4), declared_value_usd=120, prior_claims=0)
    assert decision.payout_usd == 120


@pytest.mark.parametrize(
    ("claim", "prior_claims", "reason"),
    [
        (assessment("crushed", 4), 0, "above"),
        (assessment(confidence=0.5), 0, "confidence"),
        (assessment(), 2, "prior claims"),
    ],
)
def test_each_trigger_sends_the_claim_to_review(claim, prior_claims, reason):
    decision = decide(claim, 800, prior_claims)
    assert decision.needs_review
    assert any(reason in r for r in decision.review_reasons)


def test_no_damage_pays_nothing():
    assert decide(assessment("none", 0), 500, 0).payout_usd == 0


def test_settle_applies_the_review():
    policy = PolicyDecision(payout_usd=350, review_reasons=["x"])
    assert settle(policy, None) == 350
    assert settle(policy, ReviewDecision(approved=True)) == 350
    assert settle(policy, ReviewDecision(approved=True, payout_usd=200)) == 200
    assert settle(policy, ReviewDecision(approved=False)) == 0


def test_a_denial_cannot_carry_a_payout():
    with pytest.raises(ValidationError):
        ReviewDecision(approved=False, payout_usd=100)
