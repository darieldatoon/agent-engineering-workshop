"""Part 1 evals: the `kea-claims` dataset, evaluators, and model-vs-model experiments.

    uv run python -m kea.evals.claims

Builds (or refreshes) the dataset, then runs one experiment for `MODEL` and one for `ALT_MODEL`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langsmith import Client
from langsmith.schemas import Dataset, ExampleCreate, ExampleUpdate
from openevals.llm import create_llm_as_judge

from kea.agents import CLAIMS_AGENT, Resource, attach_to_agent
from kea.claims import prompts
from kea.claims.graph import Context, builder
from kea.claims.images import IMAGE_ATTACHMENT, PHOTO_DIR, bytes_to_data_url, mime_for
from kea.config import ALT_MODEL, JUDGE_MODEL, MODEL, setup

DATASET_NAME = "kea-claims"
AUTO_APPROVE = {"approved": True, "note": "Auto-approved so the experiment can finish."}


# --- Dataset -----------------------------------------------------------------------------------


def load_ground_truth() -> list[dict[str, Any]]:
    return json.loads((PHOTO_DIR / "ground_truth.json").read_text())


def claim_form(index: int, photo_id: str) -> dict[str, Any]:
    """Deterministic synthetic form fields, so reruns produce the same dataset."""
    return {
        "passenger_id": f"pax-{photo_id}",
        "flight_number": f"KA{100 + index}",
        "declared_value_usd": 150 + 50 * (index % 10),
    }


def build_dataset(client: Client | None = None) -> Dataset:
    """Create `kea-claims`, or refresh it in place. The photo is each example's `image` attachment."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Kea Air baggage-claim photos with a claim form and ground-truth damage.",
        )
    # Tag before any experiment runs: experiments inherit their dataset's agent when created.
    attach_to_agent(client, CLAIMS_AGENT, [Resource("dataset", str(dataset.id), DATASET_NAME)])
    existing = {
        (e.metadata or {}).get("photo_id"): e.id
        for e in client.list_examples(dataset_id=dataset.id)
    }
    creates, updates = [], []
    for index, truth in enumerate(load_ground_truth()):
        fields = {
            "inputs": claim_form(index, truth["id"]),
            "outputs": {k: truth[k] for k in ("is_luggage", "damage_type", "severity")},
            "metadata": {"photo_id": truth["id"], "note": truth["note"]},
            "split": truth["split"],
        }
        if example_id := existing.get(truth["id"]):
            updates.append(ExampleUpdate(id=example_id, **fields))
        else:
            photo = PHOTO_DIR / truth["file"]
            attachments = {IMAGE_ATTACHMENT: (mime_for(photo), photo.read_bytes())}
            creates.append(ExampleCreate(**fields, attachments=attachments))
    if updates:
        client.update_examples(dataset_id=dataset.id, updates=updates)
    if creates:
        client.create_examples(dataset_id=dataset.id, examples=creates)
    return dataset


# --- Target ------------------------------------------------------------------------------------


def make_target(model: str):
    """A target that runs the claims graph with `model`, approving any human review itself."""
    context = Context(model=model)

    def target(inputs: dict[str, Any], attachments: dict[str, Any]) -> dict[str, Any]:
        image = attachments[IMAGE_ATTACHMENT]
        image["reader"].seek(0)
        photo = bytes_to_data_url(image["reader"].read(), image["mime_type"])
        graph = builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        result = graph.invoke({**inputs, "photo": photo}, config, context=context)
        reviewed = "__interrupt__" in result
        if reviewed:
            result = graph.invoke(Command(resume=AUTO_APPROVE), config, context=context)
        # A photo rejected before assessment has no damage: score it as "none".
        assessment = result.get("assessment", {"damage_type": "none", "severity": 0})
        return {
            "is_luggage": result["photo_check"]["is_luggage"],
            "damage_type": assessment["damage_type"],
            "severity": assessment["severity"],
            "confidence": assessment.get("confidence"),
            "reviewed": reviewed,
            "payout_usd": result["payout_usd"],
            "status": result["status"],
            "letter": result["letter"],
        }

    return target


# --- Evaluators --------------------------------------------------------------------------------


def is_luggage_correct(outputs: Mapping, reference_outputs: Mapping) -> dict:
    return {
        "key": "is_luggage_correct",
        "score": outputs["is_luggage"] == reference_outputs["is_luggage"],
    }


def damage_type_correct(outputs: Mapping, reference_outputs: Mapping) -> dict:
    return {
        "key": "damage_type_correct",
        "score": outputs["damage_type"] == reference_outputs["damage_type"],
    }


def severity_within_1(outputs: Mapping, reference_outputs: Mapping) -> dict:
    return {
        "key": "severity_within_1",
        "score": abs(outputs["severity"] - reference_outputs["severity"]) <= 1,
    }


_letter_judge = create_llm_as_judge(
    prompt=prompts.LETTER_JUDGE, model=JUDGE_MODEL, feedback_key="letter_quality"
)


def letter_quality(outputs: Mapping) -> Any:
    """LLM-as-judge: empathetic, states the payout, promises nothing beyond policy."""
    return _letter_judge(outputs={"payout_usd": outputs["payout_usd"], "letter": outputs["letter"]})


EVALUATORS = [is_luggage_correct, damage_type_correct, severity_within_1, letter_quality]


# --- Experiments -------------------------------------------------------------------------------


def run_experiment(model: str, client: Client | None = None):
    """Run every dataset example through the graph with `model` and score it."""
    client = client or Client()
    examples = list(client.list_examples(dataset_name=DATASET_NAME, include_attachments=True))
    return client.evaluate(
        make_target(model),
        data=examples,
        evaluators=EVALUATORS,
        experiment_prefix=f"claims-{model.split('/')[-1]}",
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


def main() -> None:
    setup("kea-claims-local")
    client = Client()
    print(f"Dataset: {build_dataset(client).url}")
    for model in (MODEL, ALT_MODEL):
        results = run_experiment(model, client)
        print(f"\n{model}\n  {results.experiment_name}")
        for key, score in mean_scores(results).items():
            print(f"  {key}: {score:.2f}")


if __name__ == "__main__":
    main()
