# Agent engineering workshop

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/darieldatoon/agent-engineering-workshop/main?urlpath=lab/tree/notebooks/00_setup.ipynb)

Build a LangGraph workflow and a Deep Agent for a fictional airline, Kea Air, then evaluate, deploy, and monitor them in LangSmith.

Start with [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb): it covers LangSmith access, local vs. Binder, and `.env`.

| Part | What you build |
|---|---|
| [1 · LangGraph](notebooks/part1-langgraph/) | A baggage-claims workflow: nodes and edges, checkpointers, interrupts, time travel, stores, datasets, evaluators, experiments |
| [2 · Deep Agents](notebooks/part2-deep-agents/) | A rebooking agent: tools, MCP, backends, memory, skills, subagents, middleware, assertion evals |
| [3 · Deploy and monitor](notebooks/part3-deploy-monitor/) | Deploy the agent, add online evaluators, and turn production failures into test cases |

## Local setup

```sh
mise trust && mise install && mise run setup
cp .env.example .env   # add LANGSMITH_API_KEY
mise run dev           # Agent Server for Studio on :2024
```

Open `notebooks/` in VS Code (Jupyter extension) or run `mise run lab`. `mise run reset` clears local Studio state.

Binder runs the notebooks in your browser with no setup, but without Studio.
