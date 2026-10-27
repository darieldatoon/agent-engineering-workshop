"""Payout rules. Plain Python, no model calls: the same claim always gets the same answer."""

from __future__ import annotations

from .models import DamageAssessment, DamageType, PolicyDecision, ReviewDecision

# Payout in USD by severity 0..5. A claim never pays more than the declared bag value.
PAYOUT_TABLE: dict[DamageType, tuple[int, int, int, int, int, int]] = {
    "none": (0, 0, 0, 0, 0, 0),
    "tear": (0, 20, 50, 100, 180, 300),
    "broken_wheel": (0, 25, 50, 80, 120, 200),
    "broken_handle": (0, 15, 40, 70, 100, 150),
    "crushed": (0, 40, 100, 200, 350, 500),
    "cracked_shell": (0, 40, 90, 180, 300, 450),
    "water_damage": (0, 30, 80, 150, 250, 400),
}

REVIEW_ABOVE_USD = 250
REVIEW_BELOW_CONFIDENCE = 0.7
REVIEW_AT_PRIOR_CLAIMS = 2


def payout_for(assessment: DamageAssessment, declared_value_usd: float) -> float:
    base = PAYOUT_TABLE[assessment.damage_type][assessment.severity]
    return min(base, declared_value_usd)


def review_reasons(assessment: DamageAssessment, payout_usd: float, prior_claims: int) -> list[str]:
    reasons = []
    if payout_usd > REVIEW_ABOVE_USD:
        reasons.append(f"payout ${payout_usd:,.0f} is above ${REVIEW_ABOVE_USD}")
    if assessment.confidence < REVIEW_BELOW_CONFIDENCE:
        reasons.append(
            f"model confidence {assessment.confidence:.2f} is below {REVIEW_BELOW_CONFIDENCE}"
        )
    if prior_claims >= REVIEW_AT_PRIOR_CLAIMS:
        reasons.append(f"{prior_claims} prior claims on file")
    return reasons


def decide(
    assessment: DamageAssessment, declared_value_usd: float, prior_claims: int
) -> PolicyDecision:
    payout = payout_for(assessment, declared_value_usd)
    return PolicyDecision(
        payout_usd=payout, review_reasons=review_reasons(assessment, payout, prior_claims)
    )


def settle(policy: PolicyDecision, review: ReviewDecision | None) -> float:
    """The payout after the optional human review: a denial pays nothing, an override wins."""
    if review is None:
        return policy.payout_usd
    if not review.approved:
        return 0.0
    return policy.payout_usd if review.payout_usd is None else review.payout_usd
