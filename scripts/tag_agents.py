"""Attach every workshop resource in LangSmith to its agent. Idempotent.

    uv run python scripts/tag_agents.py            # print the plan
    uv run python scripts/tag_agents.py --apply    # create missing agents, tag the resources

Agents: `kea-claims` (the claims workflow, not deployed) and `kea-rebooking` (the Deep Agent and
its deployment). Resources are tagged `Agent=<id>`, which is what the UI filters on under
`/a/<agent>/`. Resources that do not exist yet are skipped; rerun after creating them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from dotenv import load_dotenv
from langsmith import Client

from kea.agents import (
    CLAIMS_AGENT,
    REBOOKING_AGENT,
    Resource,
    agent_tags,
    ensure_agent,
    get_agent,
    tag_to_agent,
)
from kea.config import REPO_ROOT


@dataclass(frozen=True)
class AgentSpec:
    id: str
    dataset: str
    projects: tuple[str, ...]
    queues: tuple[str, ...] = ()


SPECS = (
    AgentSpec(
        CLAIMS_AGENT,
        dataset="kea-claims",
        projects=("kea-workshop-part1", "kea-claims-workshop"),
    ),
    AgentSpec(
        REBOOKING_AGENT,
        dataset="kea-rebooking",
        projects=(
            "kea-workshop-part2",
            "kea-workshop-part3",
            "kea-rebooking-workshop",
            "kea-deploy-workshop",
        ),
        queues=("kea-rebooking-review",),
    ),
)


def rules(client: Client) -> list[dict]:
    return client.request_with_retries("GET", "/runs/rules").json()


def discover(
    client: Client, spec: AgentSpec, agent: dict | None
) -> tuple[list[Resource], list[str]]:
    """Resources that exist for `spec`, and the names that do not."""
    found: list[Resource] = []
    missing: list[str] = []

    if client.has_dataset(dataset_name=spec.dataset):
        dataset = client.read_dataset(dataset_name=spec.dataset)
        found.append(Resource("dataset", str(dataset.id), spec.dataset))
        found += [
            Resource("experiment", str(p.id), p.name or "")
            for p in client.list_projects(reference_dataset_id=dataset.id)
        ]
        dataset_ids = {str(dataset.id)}
    else:
        missing.append(f"dataset {spec.dataset}")
        dataset_ids = set()

    project_ids: set[str] = set()
    environments = (agent or {}).get("environments", [])
    names = [*spec.projects, *(e["tracer_session_name"] for e in environments)]
    for name in dict.fromkeys(names):
        if client.has_project(project_name=name):
            project_id = str(client.read_project(project_name=name).id)
            found.append(Resource("project", project_id, name))
            project_ids.add(project_id)
        else:
            missing.append(f"project {name}")

    for name in spec.queues:
        queue = next(iter(client.list_annotation_queues(name=name)), None)
        if queue:
            found.append(Resource("queue", str(queue.id), name))
        else:
            missing.append(f"queue {name}")

    for rule in rules(client):
        scoped = (
            str(rule.get("dataset_id")) in dataset_ids or str(rule.get("session_id")) in project_ids
        )
        if scoped and rule.get("evaluator_id"):
            found.append(Resource("evaluator", str(rule["evaluator_id"]), rule["display_name"]))

    for env in environments:
        for binding in env.get("runtime_bindings") or []:
            found.append(Resource("deployment", binding["source_id"], spec.id))

    return list(dict.fromkeys(found)), missing


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="Make the changes (default: dry run)")
    args = parser.parse_args()
    client = Client()

    for spec in SPECS:
        agent = get_agent(client, spec.id)
        if agent is None and args.apply:
            agent = ensure_agent(client, spec.id)
            print(f"created agent {spec.id}")
        if agent is not None:
            agent = client.request_with_retries(
                "GET",
                f"/api/v1/agents/{spec.id}",
                request_kwargs={"params": {"include": "runtime_bindings"}},
            ).json()
        print(
            f"\n== {spec.id}"
            + ("" if agent else "  (agent does not exist yet; --apply creates it)")
        )

        resources, missing = discover(client, spec, agent)
        current = agent_tags(client, resources)
        for resource in resources:
            state = "ok" if spec.id in current.get(resource.id, set()) else "TAG"
            print(f"  {state:4} {resource.type:10} {resource.name}")
        for name in missing:
            print(f"  skip {name} (not found)")

        if args.apply:
            for resource in tag_to_agent(client, spec.id, resources):
                print(f"  tagged {resource.type} {resource.name}")


if __name__ == "__main__":
    main()
