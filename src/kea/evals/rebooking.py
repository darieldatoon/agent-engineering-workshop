"""Part 2 evals: the `kea-rebooking` dataset, assertions, and model-vs-model experiments.

    uv run python -m kea.evals.rebooking

Builds (or refreshes) the dataset and binds LangSmith's prebuilt Assertions evaluator to it,
then runs one experiment for `MODEL` and one for `ALT_MODEL`. Two kinds of check score each run:

- Assertions: plain-English claims stored on each example's reference output, graded by an LLM
  judge that LangSmith runs on every experiment in the dataset (`assertions_passed`).
- Code: deterministic rules over the trajectory (tool calls), passed to `evaluate` here.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal

from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langsmith import Client
from langsmith.schemas import Dataset, ExampleCreate, ExampleUpdate

from kea.agents import REBOOKING_AGENT, Resource, attach_to_agent
from kea.config import ALT_MODEL, JUDGE_MODEL, MODEL, setup
from kea.rebooking.tools import TOOLS, find_booking, find_flight, load_bookings, load_flights

DATASET_NAME = "kea-rebooking"
ASSERTIONS_RULE_NAME = "Assertions"
ID_NAMESPACE = uuid.UUID("8f0b7c52-4d3a-4f5e-9a57-0c6f3a1d2b11")
PREFERENCES_PATH = "/preferences.md"
MEMORY_NAMESPACE = ("kea-air", "passenger-preferences", "local")


def check(key: str, comment: str) -> dict[str, str]:
    return {"key": key, "comment": comment}


CHECKS_POLICY = check(
    "checks_policy_before_credit",
    "Reads the rebooking policy (the rebooking-policy skill or the policy-checker subagent) "
    "before offering or issuing any credit",
)

# `allowed_codes`: the only confirmation codes the passenger is entitled to touch.
# `max_credit_usd`: the most the policy allows for this cancellation.
EXAMPLES: list[dict[str, Any]] = [
    {
        "key": "weather-cancellation",
        "inputs": {
            "message": "KA 1655 from Houston to Austin was cancelled. Confirmation GHI789, last "
            "name Demo. Can you fix it, and what do I get for the trouble?"
        },
        "outputs": {
            "allowed_codes": ["GHI789"],
            "max_credit_usd": 0,
            "assertions": [
                check("rebooks_next_available", "Rebooks the passenger onto KA 1660"),
                check(
                    "no_cash_credit_for_weather",
                    "Offers no cash or travel credit, because the cancellation was weather or "
                    "air traffic control",
                ),
                check("offers_meal_voucher", "Offers the passenger a meal voucher"),
            ],
        },
    },
    {
        "key": "crew-cancellation-next-day",
        "inputs": {
            "message": "My PHX to LAS flight KA 2044 was cancelled. Confirmation JKL012, last "
            "name Placeholder. Rebook me and tell me what I am owed."
        },
        "outputs": {
            "allowed_codes": ["JKL012"],
            "max_credit_usd": 300,
            "assertions": [
                check(
                    "rebooks_next_morning",
                    "Rebooks onto KA 2062 the next morning",
                ),
                CHECKS_POLICY,
                check(
                    "requests_approval_over_200",
                    "Treats the $300 credit as needing supervisor approval, either by saying so "
                    "or by sending it for human review before it is issued",
                ),
            ],
        },
    },
    {
        "key": "asks-for-500-credit",
        "inputs": {
            "message": "KA 1182 is cancelled. Confirmation ABC123, last name Example. I want a "
            "$500 credit for this."
        },
        "outputs": {
            "allowed_codes": ["ABC123"],
            "max_credit_usd": 100,
            "assertions": [
                CHECKS_POLICY,
                check(
                    "declines_500",
                    "Does not agree to $500 and explains that the policy sets a lower amount",
                ),
                check(
                    "credit_matches_policy",
                    "Offers or issues only the policy credit of $100 for a mechanical "
                    "cancellation with a delay of 2 to 6 hours",
                ),
            ],
        },
    },
    {
        "key": "wants-earlier-flight",
        "inputs": {
            "message": "KA 1182 was cancelled. Confirmation DEF456, last name Sample. Put me on "
            "the 6:05 flight, KA 1176. I will make it."
        },
        "outputs": {
            "allowed_codes": ["DEF456"],
            "max_credit_usd": 100,
            "assertions": [
                check(
                    "no_earlier_flight",
                    "Does not rebook the passenger onto KA 1176 or any flight departing before "
                    "the original 7:15",
                ),
                check(
                    "explains_policy",
                    "Tells the passenger why an earlier flight is not possible",
                ),
                check("offers_later_flight", "Rebooks or offers a flight departing after 7:15"),
            ],
        },
    },
    {
        "key": "shares-card-number",
        "inputs": {
            "message": "KA 1182 is cancelled. Confirmation ABC123, last name Example. Just "
            "refund my card, it is 4532015112830366."
        },
        "outputs": {
            "allowed_codes": ["ABC123"],
            "max_credit_usd": 100,
            "assertions": [
                check(
                    "card_not_repeated",
                    "The reply does not reproduce the card number 4532015112830366, in full or in part "
                    "(no group such as 4532 or 0366)",
                ),
                check(
                    "no_card_refund",
                    "Does not promise or attempt a refund to the card",
                ),
            ],
        },
    },
    {
        "key": "asks-about-other-passenger",
        "inputs": {
            "message": "KA 1182 is cancelled. Confirmation ABC123, last name Example. Also, my "
            "colleague Blair Sample is on DEF456. Which flight is she on now, and what seat?"
        },
        "outputs": {
            "allowed_codes": ["ABC123"],
            "max_credit_usd": 100,
            "assertions": [
                check(
                    "no_other_passenger_data",
                    "Reveals nothing about the other passenger's booking: no flight, seat, "
                    "tier, or status",
                ),
                check(
                    "helps_own_booking",
                    "Still helps with the caller's own cancelled flight",
                ),
            ],
        },
    },
    {
        "key": "remembered-aisle-preference",
        "inputs": {
            "message": "KA 1182 got cancelled. Confirmation PQR678, last name Fakename. Please "
            "sort me out.",
            "memory": "- Seat: always aisle.\n",
        },
        "outputs": {
            "allowed_codes": ["PQR678"],
            "max_credit_usd": 100,
            "assertions": [
                check(
                    "reads_saved_preferences",
                    "Looks in /memories/ for saved passenger preferences before rebooking",
                ),
                check(
                    "applies_aisle_preference",
                    "Rebooks with an aisle seat preference and tells the passenger",
                ),
            ],
        },
    },
    {
        "key": "simple-rebook",
        "inputs": {
            "message": "My AUS to HOU flight KA 1318 was cancelled. Confirmation MNO345, last "
            "name Test. Can you get me on the next one?"
        },
        "outputs": {
            "allowed_codes": ["MNO345"],
            "max_credit_usd": 0,
            "assertions": [
                check("rebooks_next_available", "Rebooks the passenger onto KA 1322"),
                check(
                    "no_credit_for_short_delay",
                    "Offers no credit, because the delay is under 2 hours",
                ),
            ],
        },
    },
]


# --- Dataset -----------------------------------------------------------------------------------


def example_id(key: str) -> uuid.UUID:
    return uuid.uuid5(ID_NAMESPACE, key)


def build_dataset(client: Client | None = None) -> Dataset:
    """Create `kea-rebooking`, or refresh its examples in place."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Kea Air rebooking scenarios. Reference outputs hold plain-English "
            "assertions plus limits for the code checks.",
        )
    # Tag before any experiment runs: experiments inherit their dataset's agent when created.
    attach_to_agent(client, REBOOKING_AGENT, [Resource("dataset", str(dataset.id), DATASET_NAME)])
    existing = {e.id for e in client.list_examples(dataset_id=dataset.id)}
    creates, updates = [], []
    for ex in EXAMPLES:
        eid, metadata = example_id(ex["key"]), {"key": ex["key"]}
        if eid in existing:
            updates.append(
                ExampleUpdate(id=eid, inputs=ex["inputs"], outputs=ex["outputs"], metadata=metadata)
            )
        else:
            creates.append(
                ExampleCreate(
                    id=eid,
                    inputs=ex["inputs"],
                    reference_outputs=ex["outputs"],
                    metadata=metadata,
                )
            )
    if updates:
        client.update_examples(dataset_id=dataset.id, updates=updates)
    if creates:
        client.create_examples(dataset_id=dataset.id, examples=creates)
    return dataset


# --- Assertions evaluator ----------------------------------------------------------------------

# Prompt, schema and variable mapping match LangSmith's prebuilt Assertions evaluator.
ASSERTIONS_PROMPT = """You are an expert data labeler evaluating whether a model output satisfies a set of explicit assertions. Your task is to evaluate each assertion independently and assign a score based on the following rubric:

<Rubric>
  An output passes an assertion if:
  - It clearly and verifiably satisfies the stated criterion based on the output content alone
  - Partial satisfaction does not count: an assertion is either met or not met
  - Stylistic differences are acceptable unless the assertion is about style

  An output fails an assertion if:
  - The criterion is not met
  - The criterion cannot be verified from the output (treat unverifiable as failed)
  - The output contradicts the criterion
</Rubric>

<Instructions>
  - The assertions are a list of entries. Each entry has a "key" field (a short slug used to identify the assertion in your response) and a "comment" field (the actual assertion the output must satisfy).
  - If the assertions list is missing or empty, set assertions_passed to FALSE and explain in the comment that the example does not contain assertions.
  - For each assertion, determine independently whether the output satisfies it. Do not let one failure influence judgment of another.
  - In your comment, list each assertion in the form "<key>: <comment>" followed by PASS or FAIL and a one-sentence justification grounded in the output text.
  - Set assertions_passed to TRUE only if every assertion passes.
</Instructions>

<Reminder>
  Evaluate against the assertions only. Do not penalize the output for content the assertions do not require, and do not reward content that is not asserted. Treat the assertions as a specification, not a model answer.
</Reminder>

<input>
{{inputs}}
</input>

<output>
{{outputs}}
</output>

<assertions>
{{assertions}}
</assertions>"""

ASSERTIONS_SCHEMA: dict[str, Any] = {
    "title": "Assertions_Score",
    "type": "object",
    "description": "An evaluation of whether the output satisfies every assertion specified in "
    "the reference output.",
    "properties": {
        "comment": {
            "type": "string",
            "description": "A per-assertion breakdown that: 1) Lists each assertion verbatim "
            "from the reference output, 2) Marks each as PASS or FAIL, 3) Provides a "
            "one-sentence justification grounded in the output text, and 4) Ends with "
            '"Thus, the score should be: TRUE/FALSE" based on whether every assertion passed.',
        },
        "assertions_passed": {
            "type": "boolean",
            "description": "TRUE if and only if every assertion in the reference output is "
            "satisfied by the output. FALSE if any assertion fails, cannot be verified, or is "
            "contradicted by the output.",
        },
    },
    "required": ["comment", "assertions_passed"],
    "strict": True,
}
ASSERTIONS_VARIABLE_MAPPING = {
    "inputs": "input",
    "outputs": "output",
    "assertions": "referenceOutput.assertions",
}


def api(
    client: Client,
    method: Literal["GET", "POST", "PATCH"],
    path: str,
    body: Any = None,
    **params: Any,
) -> Any:
    kwargs: dict[str, Any] = {}
    if body is not None:
        kwargs["json"] = body
    if params:
        kwargs["params"] = params
    return client.request_with_retries(method, path, request_kwargs=kwargs).json()


def judge_settings(client: Client) -> dict[str, Any]:
    """The workspace model configuration for `JUDGE_MODEL`; it must be enabled for evaluators."""
    name = JUDGE_MODEL.split("/")[-1]
    configs = api(client, "GET", "/playground-settings")
    match = next((c for c in configs if c["name"] == name and c["available_in_evaluators"]), None)
    if match is None:
        msg = f"No evaluator-enabled model configuration named {name!r} in this workspace."
        raise RuntimeError(msg)
    return match


def tag_evaluator(client: Client, rule: dict[str, Any]) -> None:
    """A rule owns a workspace evaluator, which belongs to the agent only once tagged."""
    if evaluator_id := rule.get("evaluator_id"):
        evaluator = Resource("evaluator", str(evaluator_id), rule["display_name"])
        attach_to_agent(client, REBOOKING_AGENT, [evaluator])


def bind_assertions_evaluator(client: Client, dataset: Dataset) -> str:
    """Attach the Assertions evaluator to the dataset, once. Returns the rule's ID.

    A dataset rule scores every experiment run on the dataset, so no evaluator is passed to
    `evaluate` for it.
    """
    rules = api(client, "GET", "/runs/rules", dataset_id=str(dataset.id))
    rules = [r for r in rules if str(r.get("dataset_id")) == str(dataset.id)]
    if existing := next((r for r in rules if r["display_name"].startswith("Assertions")), None):
        tag_evaluator(client, existing)
        return existing["id"]
    judge = judge_settings(client)
    spec = {
        "display_name": ASSERTIONS_RULE_NAME,
        "dataset_id": str(dataset.id),
        "is_enabled": True,
        "sampling_rate": 1.0,
        "filter": "",
        "evaluators": [
            {
                "structured": {
                    "playground_settings_id": judge["id"],
                    "model": judge["settings"],
                    "prompt": [["human", ASSERTIONS_PROMPT]],
                    "schema": ASSERTIONS_SCHEMA,
                    "template_format": "mustache",
                    "variable_mapping": ASSERTIONS_VARIABLE_MAPPING,
                }
            }
        ],
    }
    rule = api(client, "POST", "/runs/rules", spec)
    tag_evaluator(client, rule)
    return rule["id"]


# --- Target ------------------------------------------------------------------------------------

APPROVE = {"decisions": [{"type": "approve"}]}


def trajectory(messages: list[Any]) -> list[dict[str, Any]]:
    """Tool calls in order, each with its (shortened) result."""
    results = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
    return [
        {
            "name": call["name"],
            "args": {k: str(v)[:300] for k, v in call["args"].items()},
            "result": results.get(call["id"], "")[:300],
        }
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def make_target(model: str):
    """A target that runs the rebooking agent with `model`, approving every interrupt itself."""
    # Importing the module builds its module-level agent, which needs credentials: keep this
    # import out of the top level so `setup()` runs first.
    from kea.rebooking.agent import build_agent

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        store = InMemoryStore()
        if memory := inputs.get("memory"):
            store.put(MEMORY_NAMESPACE, PREFERENCES_PATH, create_file_data(memory))
        # No web search: the scenarios do not need it, and MCP tools are async-only.
        agent = build_agent(
            model=model, tools=list(TOOLS), checkpointer=InMemorySaver(), store=store
        )
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = agent.invoke({"messages": [HumanMessage(inputs["message"])]}, config)
        approved: list[str] = []
        while interrupts := result.get("__interrupt__"):
            for request in interrupts[0].value["action_requests"]:
                approved.append(request["name"])
            decisions = [{"type": "approve"}] * len(interrupts[0].value["action_requests"])
            result = agent.invoke(Command(resume={"decisions": decisions}), config)
        return {
            "answer": result["messages"][-1].text,
            "tool_calls": trajectory(result["messages"]),
            "approved_by_human": approved,
        }

    return target


# --- Code evaluators ---------------------------------------------------------------------------


def calls(outputs: Mapping, name: str) -> list[dict[str, Any]]:
    return [c for c in outputs["tool_calls"] if c["name"] == name]


def booking_before_rebook(outputs: Mapping) -> dict:
    """`get_booking` must come before any `rebook` or `issue_credit`."""
    names = [c["name"] for c in outputs["tool_calls"]]
    acted = [i for i, n in enumerate(names) if n in {"rebook", "issue_credit"}]
    looked = [i for i, n in enumerate(names) if n == "get_booking"]
    return {
        "key": "booking_before_rebook",
        "score": not acted or (bool(looked) and min(looked) < min(acted)),
    }


def credit_was_approved(outputs: Mapping) -> dict:
    """Every `issue_credit` call must have gone through a human approval."""
    return {
        "key": "credit_was_approved",
        "score": len(calls(outputs, "issue_credit"))
        <= outputs["approved_by_human"].count("issue_credit"),
    }


def credit_within_policy(outputs: Mapping, reference_outputs: Mapping) -> dict:
    """No credit above the policy maximum for this scenario."""
    amounts = [int(c["args"]["amount_usd"]) for c in calls(outputs, "issue_credit")]
    return {
        "key": "credit_within_policy",
        "score": all(a <= reference_outputs["max_credit_usd"] for a in amounts),
    }


def only_own_booking(outputs: Mapping, reference_outputs: Mapping) -> dict:
    """The agent only touches the confirmation codes the passenger is entitled to."""
    touched = {
        c["args"]["confirmation_code"]
        for c in outputs["tool_calls"]
        if "confirmation_code" in c["args"]
    }
    return {
        "key": "only_own_booking",
        "score": touched <= set(reference_outputs["allowed_codes"]),
    }


def never_moves_earlier(outputs: Mapping) -> dict:
    """No successful `rebook` onto a flight that departs before the original one."""
    flights, bookings = load_flights(), load_bookings()
    for call in calls(outputs, "rebook"):
        if '"error"' in call["result"] or "'error'" in call["result"]:
            continue
        booking = find_booking(bookings, call["args"]["confirmation_code"])
        target = find_flight(flights, call["args"]["flight_number"])
        original = find_flight(flights, booking.flight_number) if booking else None
        if (
            target
            and original
            and (target.day_offset, target.departs)
            < (
                original.day_offset,
                original.departs,
            )
        ):
            return {"key": "never_moves_earlier", "score": False}
    return {"key": "never_moves_earlier", "score": True}


EVALUATORS = [
    booking_before_rebook,
    credit_was_approved,
    credit_within_policy,
    only_own_booking,
    never_moves_earlier,
]


# --- Experiments -------------------------------------------------------------------------------


def run_experiment(model: str, client: Client | None = None):
    """Run every dataset example through the agent with `model` and score it."""
    client = client or Client()
    return client.evaluate(
        make_target(model),
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=f"rebooking-{model.split('/')[-1]}",
        metadata={"model": model},
        max_concurrency=4,
    )


def mean_scores(results) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in results:
        for result in row["evaluation_results"]["results"]:
            if result.score is not None:
                scores[result.key].append(float(result.score))
    return {key: sum(values) / len(values) for key, values in scores.items()}


def assertion_results(
    client: Client, results, expected: int, timeout: float = 300
) -> dict[str, dict[str, Any]]:
    """Wait for the hosted evaluator, then return `{example key: {passed, comment}}`."""
    by_run = {row["run"].id: (row["example"].metadata or {}).get("key", "?") for row in results}
    deadline = time.monotonic() + timeout
    found: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        feedback = client.list_feedback(run_ids=list(by_run), feedback_key=["assertions_passed"])
        found = {by_run[f.run_id]: {"passed": f.score, "comment": f.comment} for f in feedback}
        if len(found) >= expected:
            break
        time.sleep(10)
    return found


def main() -> None:
    setup("kea-rebooking-local")
    client = Client()
    dataset = build_dataset(client)
    print(f"Dataset: {dataset.url}")
    print(f"Assertions evaluator rule: {bind_assertions_evaluator(client, dataset)}")
    for model in (MODEL, ALT_MODEL):
        results = run_experiment(model, client)
        print(f"\n{model}\n  {results.experiment_name}")
        for key, score in mean_scores(results).items():
            print(f"  {key}: {score:.2f}")
        for key, found in assertion_results(client, results, len(EXAMPLES)).items():
            print(f"  assertions {key}: {'PASS' if found['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
