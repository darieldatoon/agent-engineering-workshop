"""Attach LangSmith resources to an agent.

In an agent-based workspace an agent's resources are the ones tagged `Agent=<agent id>`; the UI
under `/a/<agent>/` filters on that tag. Tracing projects named in an agent's environments belong
to it already, but datasets, experiments, evaluators and queues only belong once tagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langsmith import Client
from langsmith.utils import LangSmithError, LangSmithNotFoundError

CLAIMS_AGENT = "kea-claims"
REBOOKING_AGENT = "kea-rebooking"

ResourceType = Literal["dataset", "experiment", "project", "evaluator", "queue", "deployment"]
SourceType = Literal["EXTERNAL", "LSD", "MDA"]


@dataclass(frozen=True)
class Resource:
    type: ResourceType
    id: str
    name: str = ""


def _request(client: Client, method: Literal["GET", "POST"], path: str, body: Any = None) -> Any:
    kwargs = {"json": body} if body is not None else {}
    response = client.request_with_retries(method, path, request_kwargs=kwargs)
    return response.json() if response.content else None


def get_agent(client: Client, agent_id: str) -> dict[str, Any] | None:
    try:
        return _request(client, "GET", f"/api/v1/agents/{agent_id}")
    except LangSmithNotFoundError:
        return None


def ensure_agent(client: Client, agent_id: str, source_type: SourceType = "EXTERNAL") -> dict:
    """Return the agent, creating it (with its four environment projects) when missing."""
    return get_agent(client, agent_id) or _request(
        client,
        "POST",
        "/api/v1/agents",
        {"id": agent_id, "name": agent_id, "source_type": source_type},
    )


def agent_tags(client: Client, resources: list[Resource]) -> dict[str, set[str]]:
    """Current `Agent` tag values per resource ID."""
    if not resources:
        return {}
    body = [{"resource_type": r.type, "resource_id": r.id} for r in resources]
    tags = _request(client, "POST", "/workspaces/current/tags/resources", body)
    return {
        rid: {v["value"] for key in keys if key["key"] == "Agent" for v in key["values"]}
        for rid, keys in tags.items()
    }


def tag_to_agent(client: Client, agent_id: str, resources: list[Resource]) -> list[Resource]:
    """Tag each resource `Agent=<agent_id>`. Idempotent; returns the resources it had to tag.

    Tagging a dataset also tags its existing experiments, and experiments created later inherit it.
    """
    agent = get_agent(client, agent_id)
    if agent is None or not agent.get("tag_value_id"):
        raise LookupError(f"Agent {agent_id!r} does not exist or has no tag value")
    current = agent_tags(client, resources)
    todo = [r for r in resources if agent_id not in current.get(r.id, set())]
    for resource in todo:
        _request(
            client,
            "POST",
            "/workspaces/current/taggings",
            {
                "tag_value_id": agent["tag_value_id"],
                "resource_type": resource.type,
                "resource_id": resource.id,
            },
        )
    return todo


def attach_to_agent(
    client: Client,
    agent_id: str,
    resources: list[Resource],
    source_type: SourceType = "EXTERNAL",
) -> bool:
    """Create the agent if needed and tag `resources` to it. False when the workspace has no agents.

    Never raises on API errors: a project-based workspace must still run the workshop untagged.
    """
    try:
        ensure_agent(client, agent_id, source_type)
        tag_to_agent(client, agent_id, resources)
    except (LangSmithError, LookupError):
        return False
    return True
