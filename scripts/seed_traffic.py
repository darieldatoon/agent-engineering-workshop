"""Send a dozen passenger conversations to the rebooking agent so dashboards have data.

    uv run python scripts/seed_traffic.py --url https://<deployment-url>
    uv run python scripts/seed_traffic.py --url http://127.0.0.1:2024    # `mise run dev`

Credit demands interrupt the agent. Each scenario approves or rejects it through the SDK.
The API key comes from `LANGSMITH_API_KEY` (or `.env`); a local server ignores it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from kea.config import REPO_ROOT

Decision = Literal["approve", "reject"]
MAX_INTERRUPTS_PER_TURN = 3


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[str, ...]
    credit_decision: Decision = "reject"


@dataclass(frozen=True)
class Outcome:
    scenario: str
    thread_id: str
    interrupts: int
    reply: str
    error: str | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "rebook-earliest",
        (
            "My flight KA 1182 from DAL to DEN was cancelled. My confirmation code is ABC123 (last name Example). "
            "Can you put me on the earliest flight tomorrow?",
        ),
    ),
    Scenario(
        "rebook-afternoon",
        (
            "KA 1182 DAL to DEN got cancelled. Confirmation DEF456 (last name Sample). I need to be in Denver by "
            "mid-afternoon tomorrow. What are my options?",
            "Book the best one for me.",
        ),
    ),
    Scenario(
        "rebook-aisle",
        (
            "Confirmation PQR678 (last name Fakename). KA 1182 is cancelled. Please rebook me, and I would like an "
            "aisle seat if possible.",
        ),
    ),
    Scenario(
        "status-only",
        (
            "Is flight KA 1182 from Dallas to Denver still operating? My code is DEF456 (last name Sample).",
        ),
    ),
    Scenario(
        "pii-contact",
        (
            "KA 1182 DAL to DEN is cancelled, code ABC123 (last name Example). Please rebook me. You can reach me at "
            "alex.example@example.com or 555-0142, and my date of birth is 1990-04-12.",
        ),
    ),
    Scenario(
        "pii-payment",
        (
            "Code PQR678 (last name Fakename), KA 1182 is cancelled. Rebook me and charge any difference to my card "
            "4111 1111 1111 1111, exp 12/29, CVV 123. My passport number is X1234567.",
        ),
    ),
    Scenario(
        "credit-approve-1",
        (
            "KA 1182 was cancelled, code ABC123 (last name Example). I want a $500 credit on my account for the "
            "trouble. Issue it now.",
        ),
        "approve",
    ),
    Scenario(
        "credit-reject-1",
        (
            "Code DEF456 (last name Sample), KA 1182 cancelled. I demand $500 compensation as a credit. Do it now.",
        ),
        "reject",
    ),
    Scenario(
        "credit-approve-2",
        (
            "Code PQR678 (last name Fakename). The cancelled KA 1182 made me miss a meeting. Please issue a $500 "
            "travel credit to my account.",
        ),
        "approve",
    ),
    Scenario(
        "credit-reject-2",
        (
            "Confirmation ABC123 (last name Example). I do not want a new flight, I want a $500 credit. Issue it.",
        ),
        "reject",
    ),
    Scenario(
        "multi-turn",
        (
            "KA 1182 DAL to DEN is cancelled, code DEF456 (last name Sample). What can you do for me?",
            "The 9:40 departure works. Please book it.",
            "Thanks. Can I also get a window seat?",
        ),
    ),
    Scenario(
        "denver-weather",
        (
            "KA 1182 is cancelled, code PQR678 (last name Fakename). Rebook me on the earliest flight, and tell me "
            "what the weather in Denver will be like tomorrow.",
        ),
    ),
)


def resume_value(interrupt: Mapping[str, Any], decision: Decision) -> Any:
    """Answer a `HumanInTheLoopMiddleware` interrupt; any other payload gets a bare string."""
    value = interrupt.get("value")
    requests = value.get("action_requests") if isinstance(value, dict) else None
    if not requests:
        return decision
    answer = (
        {"type": "approve"}
        if decision == "approve"
        else {"type": "reject", "message": "Credit not approved by the agent supervisor."}
    )
    return {"decisions": [answer for _ in requests]}


def reply_text(result: Any) -> str:
    messages = result.get("messages") if isinstance(result, dict) else None
    content = (messages[-1].get("content") if messages else "") or ""
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return " ".join(str(content).split())


async def run_scenario(
    client: LangGraphClient, assistant: str, scenario: Scenario, tag: str
) -> Outcome:
    thread = await client.threads.create(metadata={"seed": tag, "scenario": scenario.name})
    thread_id = thread["thread_id"]
    metadata = {"seed": tag, "scenario": scenario.name}
    interrupts, result = 0, {}
    try:
        for turn in scenario.turns:
            result = await client.runs.wait(
                thread_id,
                assistant,
                input={"messages": [{"role": "user", "content": turn}]},
                metadata=metadata,
            )
            for _ in range(MAX_INTERRUPTS_PER_TURN):
                pending = result.get("__interrupt__") if isinstance(result, dict) else None
                if not pending:
                    break
                interrupts += 1
                result = await client.runs.wait(
                    thread_id,
                    assistant,
                    command={"resume": resume_value(pending[0], scenario.credit_decision)},
                    metadata=metadata,
                )
    except Exception as error:
        return Outcome(scenario.name, thread_id, interrupts, "", f"{type(error).__name__}: {error}")
    return Outcome(scenario.name, thread_id, interrupts, reply_text(result))


async def seed(url: str, assistant: str, concurrency: int, tag: str) -> list[Outcome]:
    client = get_client(url=url, api_key=os.environ.get("LANGSMITH_API_KEY"))
    gate = asyncio.Semaphore(concurrency)

    async def guarded(scenario: Scenario) -> Outcome:
        async with gate:
            outcome = await run_scenario(client, assistant, scenario, tag)
        status = outcome.error or f"interrupts={outcome.interrupts}  {outcome.reply[:70]}"
        print(f"{outcome.scenario:18} {status}", flush=True)
        return outcome

    return list(await asyncio.gather(*(guarded(s) for s in SCENARIOS)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="Deployment URL or http://127.0.0.1:2024")
    parser.add_argument("--assistant", default="rebooking", help="Graph id from langgraph.json")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--tag", default="workshop", help="Stored as `seed` metadata on every run")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    outcomes = asyncio.run(seed(args.url, args.assistant, args.concurrency, args.tag))
    failed = [o for o in outcomes if o.error]
    print(f"{len(outcomes) - len(failed)}/{len(outcomes)} scenarios completed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
