"""Helpers the notebooks share, so each notebook shows only what it teaches."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from kea.claims.graph import ClaimInput, Context, builder


def claim(
    photo: str, passenger_id: str, flight_number: str, declared_value_usd: float
) -> ClaimInput:
    """A claim with its fields filled in. `photo` is a file name under data/bag-photos."""
    return {
        "photo": photo,
        "passenger_id": passenger_id,
        "flight_number": flight_number,
        "declared_value_usd": declared_value_usd,
    }


def claims_graph(store: BaseStore | None = None) -> CompiledStateGraph[Any, Context, Any, Any]:
    """The finished claims workflow with an in-memory checkpointer and store."""
    return builder.compile(checkpointer=InMemorySaver(), store=store or InMemoryStore())


async def ask_and_approve(agent: Any, text: str, thread_id: str | None = None) -> dict[str, Any]:
    """Send one message, and approve every action the agent pauses on."""
    config = {"configurable": {"thread_id": thread_id or str(uuid4())}}
    result = await agent.ainvoke({"messages": [HumanMessage(text)]}, config)
    while result.get("__interrupt__"):
        requests = result["__interrupt__"][0].value["action_requests"]
        for request in requests:
            print(f"Paused for approval: {request['name']}({request['args']})")
        decisions = [{"type": "approve"}] * len(requests)
        result = await agent.ainvoke(Command(resume={"decisions": decisions}), config)
    return result


def steps(result: dict[str, Any]) -> None:
    """Print the tools the agent called, in order."""
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []):
            args = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in call["args"].items())
            print(f"  {call['name']}({args})")


def final(result: dict[str, Any]) -> None:
    """Print the agent's last reply."""
    print(result["messages"][-1].text)
