"""Kea Air damaged-baggage claims: a workflow, not an agent.

    START -> validate_photo -> assess_damage -> check_history -> apply_policy
          -> [human_review] -> draft_letter -> record_claim -> END
    (a photo that is not luggage goes validate_photo -> reject_letter -> END)

The LLM sits inside predefined steps; the code decides which step runs next.
`langgraph dev` serves `graph` and supplies the checkpointer and store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, interrupt

from kea.claims import prompts
from kea.claims.images import attach_image_to_root_run, resolve_photo
from kea.claims.models import (
    ClaimRecord,
    DamageAssessment,
    PhotoCheck,
    PolicyDecision,
    ReviewDecision,
)
from kea.claims.policy import decide, settle
from kea.config import MODEL

Status = Literal["rejected", "paid", "unpaid"]


@dataclass
class Context:
    """Per-run settings. Swap `model` to compare models without touching the graph."""

    model: str = MODEL


class ClaimInput(TypedDict):
    photo: str  # a data URL, or a file name under data/bag-photos
    passenger_id: str
    flight_number: str
    declared_value_usd: float


class ClaimOutput(TypedDict, total=False):
    status: Status
    photo_check: dict[str, Any]
    assessment: dict[str, Any]
    policy: dict[str, Any]
    review: dict[str, Any]
    payout_usd: float
    letter: str


class ClaimState(ClaimInput, ClaimOutput, total=False):
    prior_claims: list[dict[str, Any]]


def _llm(runtime: Runtime[Context]):
    # `runtime.context` is None when the caller passes no context.
    return init_chat_model((runtime.context or Context()).model)


def _photo_messages(system: str, data_url: str) -> list[SystemMessage | HumanMessage]:
    image = {"type": "image_url", "image_url": {"url": data_url}}
    return [SystemMessage(system), HumanMessage([image])]


def validate_photo(
    state: ClaimState, runtime: Runtime[Context], config: RunnableConfig
) -> dict[str, Any]:
    data_url = resolve_photo(state["photo"])
    attach_image_to_root_run(config, data_url)
    check = _llm(runtime).with_structured_output(PhotoCheck)
    result = cast(PhotoCheck, check.invoke(_photo_messages(prompts.PHOTO_CHECK, data_url)))
    return {"photo_check": result.model_dump()}


def reject_letter(state: ClaimState) -> dict[str, Any]:
    reason = state["photo_check"]["reason"]
    letter = (
        "Dear passenger,\n\n"
        "We're sorry to hear your bag had a problem, and thank you for contacting Kea Air "
        "Baggage Claims. We could not process this claim because the photo does not appear to "
        f"show luggage ({reason}), so no payout has been issued. "
        "To submit a new claim, please send a clear photo of the damaged bag.\n\n"
        "Kea Air Baggage Claims"
    )
    return {"letter": letter, "status": "rejected", "payout_usd": 0.0}


def assess_damage(state: ClaimState, runtime: Runtime[Context]) -> dict[str, Any]:
    assess = _llm(runtime).with_structured_output(DamageAssessment)
    messages = _photo_messages(prompts.ASSESS_DAMAGE, resolve_photo(state["photo"]))
    return {"assessment": cast(DamageAssessment, assess.invoke(messages)).model_dump()}


def check_history(state: ClaimState, runtime: Runtime[Context]) -> dict[str, Any]:
    assert runtime.store is not None, "compile the graph with a store"
    items = runtime.store.search(("claims", state["passenger_id"]))
    return {"prior_claims": [item.value for item in items]}


def apply_policy(state: ClaimState) -> dict[str, Any]:
    decision = decide(
        DamageAssessment.model_validate(state["assessment"]),
        state["declared_value_usd"],
        len(state["prior_claims"]),
    )
    return {"policy": decision.model_dump(), "payout_usd": decision.payout_usd}


def human_review(state: ClaimState) -> dict[str, Any]:
    policy = PolicyDecision.model_validate(state["policy"])
    answer = interrupt(
        {
            "claim": {
                "passenger_id": state["passenger_id"],
                "flight_number": state["flight_number"],
                "declared_value_usd": state["declared_value_usd"],
                "assessment": state["assessment"],
            },
            "proposed_payout_usd": policy.payout_usd,
            "reasons": policy.review_reasons,
        }
    )
    review = ReviewDecision.model_validate(answer)
    return {"review": review.model_dump(), "payout_usd": settle(policy, review)}


def draft_letter(state: ClaimState, runtime: Runtime[Context]) -> dict[str, Any]:
    review = state.get("review")
    facts = (
        f"Flight: {state['flight_number']}\n"
        f"Damage found: {state['assessment']['description']}\n"
        f"Payout: ${state['payout_usd']:,.2f}"
        + (f"\nReviewer note: {review['note']}" if review and review["note"] else "")
    )
    letter = _llm(runtime).invoke([SystemMessage(prompts.DRAFT_LETTER), HumanMessage(facts)])
    status: Status = "paid" if state["payout_usd"] > 0 else "unpaid"
    return {"letter": str(letter.text), "status": status}


def record_claim(state: ClaimState, runtime: Runtime[Context]) -> dict[str, Any]:
    assert runtime.store is not None, "compile the graph with a store"
    record = ClaimRecord(
        flight_number=state["flight_number"],
        damage_type=state["assessment"]["damage_type"],
        severity=state["assessment"]["severity"],
        payout_usd=state["payout_usd"],
    )
    # Keyed by flight so replaying a thread (time travel) overwrites instead of duplicating.
    runtime.store.put(
        ("claims", state["passenger_id"]), state["flight_number"], record.model_dump()
    )
    return {}


def route_after_photo_check(state: ClaimState) -> Literal["assess_damage", "reject_letter"]:
    return "assess_damage" if state["photo_check"]["is_luggage"] else "reject_letter"


def route_after_policy(state: ClaimState) -> Literal["human_review", "draft_letter"]:
    return "human_review" if state["policy"]["needs_review"] else "draft_letter"


retry = RetryPolicy(max_attempts=3)

# ty does not recognize plain TypedDicts as LangGraph state; they work at runtime.
builder = StateGraph(
    ClaimState,  # ty: ignore[invalid-argument-type]
    context_schema=Context,
    input_schema=ClaimInput,  # ty: ignore[invalid-argument-type]
    output_schema=ClaimOutput,  # ty: ignore[invalid-argument-type]
)
builder.add_node("validate_photo", validate_photo, retry_policy=retry)  # ty: ignore[invalid-argument-type]
builder.add_node("reject_letter", reject_letter)
builder.add_node("assess_damage", assess_damage, retry_policy=retry)
builder.add_node("check_history", check_history)
builder.add_node("apply_policy", apply_policy)
builder.add_node("human_review", human_review)
builder.add_node("draft_letter", draft_letter, retry_policy=retry)
builder.add_node("record_claim", record_claim)

builder.add_edge(START, "validate_photo")
builder.add_conditional_edges("validate_photo", route_after_photo_check)
builder.add_edge("reject_letter", END)
builder.add_edge("assess_damage", "check_history")
builder.add_edge("check_history", "apply_policy")
builder.add_conditional_edges("apply_policy", route_after_policy)
builder.add_edge("human_review", "draft_letter")
builder.add_edge("draft_letter", "record_claim")
builder.add_edge("record_claim", END)

graph = builder.compile()
