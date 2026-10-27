"""Put the workshop back to its starting state, so every notebook can be rerun from a clean slate.

    uv run python scripts/reset.py                      # local Studio state only
    uv run python scripts/reset.py --langsmith          # plan for LangSmith (prints, changes nothing)
    uv run python scripts/reset.py --langsmith --yes    # apply it
    uv run python scripts/reset.py --deployment --yes   # also clear the deployed agent's threads and memory

LangSmith reset deletes experiments, examples added during the workshop, the review queue, and
the notebooks' tracing projects. It keeps both datasets' starting examples, the deployment, its
tracing project, and the online evaluators.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from urllib.request import urlopen

from dotenv import load_dotenv
from langsmith import Client

from kea.config import REPO_ROOT
from kea.evals import claims, rebooking

LOCAL_STATE = [REPO_ROOT / ".langgraph_api", *REPO_ROOT.glob("notebooks/**/.langgraph_api")]
NOTEBOOK_PROJECTS = [
    "kea-workshop-setup",
    "kea-workshop-part1",
    "kea-workshop-part2",
    "kea-workshop-part3",
    "kea-claims-workshop",
    "kea-rebooking-workshop",
    "kea-deploy-workshop",
    "kea-smoke",
]
REVIEW_QUEUE = "kea-rebooking-review"
# Prefix of `_memories_namespace` in kea.rebooking.agent: every passenger's remembered preferences.
MEMORY_NAMESPACE = ("kea-air", "passenger-preferences")


def server_running(port: int = 2024) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/ok", timeout=2):
            return True
    except OSError:
        return False


def reset_local() -> None:
    # Deleting the checkpoint files under a live server corrupts its state.
    if server_running():
        raise SystemExit("Stop `mise run dev` first (Ctrl-C in its terminal), then rerun reset.")
    for path in LOCAL_STATE:
        if path.exists():
            shutil.rmtree(path)
            print(f"deleted  {path.relative_to(REPO_ROOT)}")
    print("local Studio threads and memory cleared")


def keep_example(dataset_name: str, metadata: dict) -> bool:
    if dataset_name == claims.DATASET_NAME:
        return metadata.get("photo_id") in {t["id"] for t in claims.load_ground_truth()}
    return metadata.get("key") in {ex["key"] for ex in rebooking.EXAMPLES}


def plan_langsmith(client: Client) -> list[tuple[str, str, object]]:
    plan: list[tuple[str, str, object]] = []
    for name in (claims.DATASET_NAME, rebooking.DATASET_NAME):
        if not client.has_dataset(dataset_name=name):
            continue
        dataset = client.read_dataset(dataset_name=name)
        for experiment in client.list_projects(reference_dataset_id=dataset.id):
            plan.append(("experiment", f"{name} / {experiment.name}", experiment.id))
        for example in client.list_examples(dataset_id=dataset.id):
            if not keep_example(name, example.metadata or {}):
                plan.append(("example", f"{name} / {example.id}", example.id))
    for queue in client.list_annotation_queues(name=REVIEW_QUEUE):
        plan.append(("queue", queue.name, queue.id))
    for project in NOTEBOOK_PROJECTS:
        if client.has_project(project):
            plan.append(("project", project, client.read_project(project_name=project).id))
    return plan


def apply_langsmith(client: Client, plan: list[tuple[str, str, object]]) -> None:
    delete = {
        "experiment": lambda i: client.delete_project(project_id=i),
        "example": lambda i: client.delete_example(i),
        "queue": lambda i: client.delete_annotation_queue(i),
        "project": lambda i: client.delete_project(project_id=i),
    }
    for kind, label, item_id in plan:
        delete[kind](item_id)
        print(f"deleted  {kind:<10} {label}")


async def reset_deployment(url: str) -> None:
    from langgraph_sdk import get_client

    remote = get_client(url=url)
    threads = await remote.threads.search(limit=1000)
    for thread in threads:
        await remote.threads.delete(thread["thread_id"])
    items = await remote.store.search_items(MEMORY_NAMESPACE, limit=1000)
    for item in items["items"]:
        await remote.store.delete_item(tuple(item["namespace"]), item["key"])
    print(f"deleted  {len(threads)} deployment threads, {len(items['items'])} memory files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--langsmith", action="store_true", help="Also reset LangSmith")
    parser.add_argument("--deployment", action="store_true", help="Also clear the deployment")
    parser.add_argument("--yes", action="store_true", help="Apply LangSmith/deployment changes")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")

    reset_local()

    if args.langsmith:
        client = Client()
        plan = plan_langsmith(client)
        if not args.yes:
            for kind, label, _ in plan:
                print(f"would delete  {kind:<10} {label}")
            print(f"{len(plan)} items. Rerun with --yes to delete them.")
        else:
            apply_langsmith(client, plan)

    if args.deployment:
        url = os.environ.get("KEA_DEPLOYMENT_URL")
        if not url:
            raise SystemExit("KEA_DEPLOYMENT_URL is not set in .env")
        if not args.yes:
            print(f"would delete every thread and memory file on {url}. Rerun with --yes.")
        else:
            asyncio.run(reset_deployment(url))


if __name__ == "__main__":
    main()
