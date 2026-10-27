"""Model IDs and environment setup shared by the graphs, notebooks, and evals.

Model strings use the ``langsmith:`` provider, so ``init_chat_model`` routes every call
through the LangSmith LLM Gateway with the same ``LANGSMITH_API_KEY`` used for tracing.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from dotenv import load_dotenv

MODEL = os.environ.get("KEA_MODEL", "langsmith:openai/gpt-5.6-luna")
JUDGE_MODEL = os.environ.get("KEA_JUDGE_MODEL", "langsmith:openai/gpt-5.6-luna")
# The second model in the side-by-side experiments.
ALT_MODEL = os.environ.get("KEA_ALT_MODEL", "langsmith:anthropic/claude-haiku-5-5")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def setup(project: str) -> None:
    """Load `.env`, ask for the LangSmith key if it is missing (Binder), and pick a project."""
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API key: ")
        # `langgraph dev` runs in a terminal and does not inherit this kernel's environment.
        env_file = REPO_ROOT / ".env"
        if not env_file.exists():
            env_file.write_text(
                f"LANGSMITH_API_KEY={os.environ['LANGSMITH_API_KEY']}\nLANGSMITH_TRACING=true\n"
            )
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = project
