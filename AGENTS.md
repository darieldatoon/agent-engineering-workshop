# Agent instructions

Workshop repo: three notebooks that build a LangGraph workflow, a Deep Agent, and deploy and monitor it in LangSmith, for a fictional airline, **Kea Air**.

## Rules

- **Never name a real customer** (airline or otherwise) in code, notebooks, data, commits, or docs. The airline is Kea Air. `tools/scan_denylist.sh` enforces this from a gitignored `.denylist`.
- Never commit `.env` or print keys. Only `LANGSMITH_API_KEY` is required; model calls go through the LangSmith gateway (`langsmith:` model IDs in `src/kea/config.py`).
- Finished graphs live in `src/kea/` and are served by `langgraph dev` (see `langgraph.json`). Notebooks build up to them in cells, then import the finished version for Studio. Don't use `langsmith-studio-nb`.
- Don't put pedagogy labels ("whole game", "Great Demo", "illustration") in learner-facing text.

## Notebook shape

Each segment is a short demo: show the finished result first, then the fewest cells that produce it, then go deeper. One setup cell (`kea.config.setup(project)`), one `MODEL` constant from `kea.config`. Outputs are stripped on commit.

## Commands

- `mise run setup` installs everything and the git hooks.
- `mise run dev` starts the Agent Server for Studio on :2024.
- `mise run check` runs ruff format, ruff lint, and ty. Run `mise run fix` after editing Python.
- `mise run binder:export` regenerates `.binder/requirements.txt` (the pre-commit hook does it when `uv.lock` changes).
