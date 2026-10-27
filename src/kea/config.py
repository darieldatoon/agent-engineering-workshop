"""Model IDs and environment setup shared by the graphs, notebooks, and evals.

Model strings use the ``langsmith:`` provider, so ``init_chat_model`` routes every call
through the LangSmith LLM Gateway. It authenticates with ``LANGSMITH_GATEWAY_API_KEY``
(the class's shared key) and falls back to ``LANGSMITH_API_KEY``, which is each attendee's
own key for tracing, datasets, and evals.
"""

from __future__ import annotations

import getpass
import os
import re
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

MODEL = os.environ.get("KEA_MODEL", "langsmith:openai/gpt-6-luna")
JUDGE_MODEL = os.environ.get("KEA_JUDGE_MODEL", "langsmith:openai/gpt-6-luna")
# The second model in the side-by-side experiments.
ALT_MODEL = os.environ.get("KEA_ALT_MODEL", "langsmith:anthropic/claude-haiku-5-5")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

KEY_PROMPTS = {
    "LANGSMITH_API_KEY": "Your LangSmith API key: ",
    "LANGSMITH_GATEWAY_API_KEY": "LangSmith gateway API key (press Enter to use your LangSmith key): ",
}


def with_env_values(text: str, values: Mapping[str, str]) -> str:
    """`.env` text with each key set, replacing an existing or commented-out line, else appended."""
    for name, value in values.items():
        line = f"{name}={value}"
        pattern = re.compile(rf"^#?\s*{re.escape(name)}=.*$", re.MULTILINE)
        text, found = pattern.subn(lambda _, line=line: line, text, count=1)
        if not found:
            text = f"{text.rstrip()}\n{line}\n" if text.strip() else f"{line}\n"
    return text


def setup(project: str) -> None:
    """Load `.env`, ask for the keys on first run (Binder), and pick a project.

    The gateway key is optional: left blank, model calls fall back to `LANGSMITH_API_KEY`.
    """
    env_file = REPO_ROOT / ".env"
    load_dotenv(env_file)
    # Ask only on first run; otherwise a deliberately blank gateway key re-prompts in every notebook.
    first_run = not os.environ.get("LANGSMITH_API_KEY")
    asked = {
        name: getpass.getpass(prompt).strip()
        for name, prompt in KEY_PROMPTS.items()
        if first_run and not os.environ.get(name)
    }
    entered = {name: value for name, value in asked.items() if value}
    if entered:
        os.environ.update(entered)
        # `langgraph dev` runs in a terminal and does not inherit this kernel's environment.
        existing = env_file.read_text() if env_file.exists() else "LANGSMITH_TRACING=true\n"
        env_file.write_text(with_env_values(existing, entered))
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = project
