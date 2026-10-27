"""The Kea Air rebooking assistant: a Deep Agent served by `langgraph dev` as `rebooking`.

`build_agent()` assembles it. The module-level `agent` is what `langgraph.json` points at, so it
takes no checkpointer or store: the Agent Server supplies both. Notebooks and evals call
`build_agent(checkpointer=..., store=...)` to run it in-process.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import ModelCallLimitMiddleware, PIIMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.sessions import Connection
from langgraph.graph.state import CompiledStateGraph
from mcp.types import CallToolResult, TextContent

from kea.config import MODEL
from kea.rebooking.tools import TOOLS

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

EXA_SERVER: dict[str, Connection] = {
    "exa": {"url": "https://mcp.exa.ai/mcp", "transport": "streamable_http"}
}
EXA_TOOL = "web_search_exa"
MAX_RESULTS = 3
MAX_CHARS = 4_000

SYSTEM_PROMPT = "You are the Kea Air rebooking assistant."

POLICY_CHECKER: SubAgent = {
    "name": "policy-checker",
    "description": (
        "Reads Kea Air's rebooking policy and says what a passenger is entitled to. "
        "Give it the cancellation cause, the delay in minutes, and what the passenger asked for."
    ),
    "system_prompt": (
        "Read the rebooking-policy skill, then answer in five lines or fewer: the entitlement, "
        "the amount, whether supervisor approval is needed, and anything to decline. "
        "Do not quote the whole policy."
    ),
    "tools": [],
    "skills": ["/skills/"],
}


MIDDLEWARE: list[Any] = [
    PIIMiddleware("email", strategy="redact", apply_to_output=True, apply_to_tool_results=True),
    PIIMiddleware(
        "credit_card", strategy="redact", apply_to_output=True, apply_to_tool_results=True
    ),
    ModelCallLimitMiddleware(run_limit=40, exit_behavior="end"),
]


def _memories_namespace(rt: Any) -> tuple[str, ...]:
    """Scope `/memories/` to whoever called the deployment; `local` outside one."""
    info = rt.server_info
    identity = info.user.identity if info and info.user else "local"
    return ("kea-air", "passenger-preferences", identity)


def build_backend() -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memory/": FilesystemBackend(root_dir=str(HERE / "memory")),
            "/skills/": FilesystemBackend(root_dir=str(HERE / "skills")),
            "/memories/": StoreBackend(namespace=_memories_namespace),
        },
    )


async def cap_results(
    request: MCPToolCallRequest,
    handler: Any,
) -> MCPToolCallResult:
    """Keep search results small: fewer hits in, truncated text out."""
    result = await handler(request.override(args={**request.args, "numResults": MAX_RESULTS}))
    if not isinstance(result, CallToolResult):
        return result
    text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
    return CallToolResult(
        content=[TextContent(type="text", text=text[:MAX_CHARS])], isError=result.isError
    )


async def _fetch_exa_tools(timeout: float) -> list[BaseTool]:
    client = MultiServerMCPClient(EXA_SERVER, tool_interceptors=[cap_results])
    tools = await asyncio.wait_for(client.get_tools(), timeout)
    return [t for t in tools if t.name == EXA_TOOL]


def mcp_enabled() -> bool:
    return os.environ.get("KEA_ENABLE_MCP", "1").lower() not in {"0", "false", "no", "off"}


def load_exa_tools(timeout: float = 10.0) -> list[BaseTool]:
    """Exa's web search, or no tools at all if MCP is off or the server is unreachable.

    Runs the async client on its own thread so it works with or without a running event loop
    (notebook, `langgraph dev`, script). Do not make this raise: the agent must build offline.
    """
    if not mcp_enabled():
        return []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_fetch_exa_tools(timeout))).result()
    except Exception as exc:
        logger.warning("Exa MCP unavailable, continuing without web search: %s", exc)
        return []


def build_agent(
    *,
    model: str | BaseChatModel = MODEL,
    tools: list[Any] | None = None,
    checkpointer: Any = None,
    store: Any = None,
) -> CompiledStateGraph:
    """The finished agent. `tools=None` means the four rebooking tools plus Exa, if reachable."""
    all_tools = [*TOOLS, *load_exa_tools()] if tools is None else tools
    return create_deep_agent(
        model=model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        memory=["/memory/AGENTS.md"],
        skills=["/skills/"],
        subagents=[POLICY_CHECKER],
        backend=build_backend(),
        # The package is not a scratchpad: the agent may write /memories/ but not its own rules.
        permissions=[
            FilesystemPermission(
                operations=["write"], paths=["/memory/*", "/skills/*"], mode="deny"
            ),
        ],
        interrupt_on={"issue_credit": True},
        middleware=MIDDLEWARE,
        checkpointer=checkpointer,
        store=store,
        name="kea-rebooking",
    )


agent = build_agent()
