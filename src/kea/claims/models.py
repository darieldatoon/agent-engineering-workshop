"""The claim's domain, as types. Anything a node passes along is one of these."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, Field, computed_field, model_validator

DamageType = Literal[
    "none", "tear", "broken_wheel", "broken_handle", "crushed", "cracked_shell", "water_damage"
]
# 0 intact, 5 destroyed. A Literal, not ge/le: Anthropic's structured output rejects min/max.
Severity = Literal[0, 1, 2, 3, 4, 5]


def _unit_interval(value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError("must be between 0 and 1")
    return value


# Checked after parsing: Anthropic's structured output rejects min/max in the schema.
Confidence = Annotated[float, AfterValidator(_unit_interval)]


class ClaimForm(BaseModel):
    """The claim details a passenger types next to their photo in chat."""

    passenger_id: str = Field(
        description="The passenger or loyalty ID, e.g. P-1001; 'guest' if not given."
    )
    flight_number: str = Field(description="The flight, e.g. KA 1182; 'unknown' if not given.")
    declared_value_usd: float | None = Field(
        default=None, description="The bag's value in US dollars, if the passenger states one."
    )


class PhotoCheck(BaseModel):
    """Is this a photo of a piece of luggage at all?"""

    is_luggage: bool
    reason: str = Field(description="One short sentence explaining the call.")


class DamageAssessment(BaseModel):
    """What the vision model sees on the bag."""

    damage_type: DamageType = Field(description="The most significant damage; 'none' if intact.")
    severity: Severity = Field(
        description="0 intact, 1 cosmetic, 3 usable but impaired, 5 destroyed."
    )
    confidence: Confidence = Field(
        description="How sure you are, from 0 to 1. Low if the photo is unclear."
    )
    description: str = Field(description="One or two sentences describing the damage.")


class ClaimRecord(BaseModel):
    """One settled claim, as kept in the Store."""

    flight_number: str
    damage_type: DamageType
    severity: Severity
    payout_usd: float = Field(ge=0)


class PolicyDecision(BaseModel):
    """The rulebook's verdict. Review is derived from the reasons, so the two cannot disagree."""

    payout_usd: float = Field(ge=0)
    review_reasons: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


class ReviewDecision(BaseModel):
    """What the human reviewer sends back when resuming. A denial carries no payout."""

    approved: bool
    payout_usd: float | None = Field(
        default=None, ge=0, description="Overrides the proposed payout."
    )
    note: str = ""

    @model_validator(mode="after")
    def _denial_has_no_payout(self) -> Self:
        if not self.approved and self.payout_usd is not None:
            raise ValueError("a denied claim cannot carry a payout")
        return self
