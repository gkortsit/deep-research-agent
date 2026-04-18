# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install deps (uses the `openai-agents` SDK, which provides the `agents` import):

```
pip install -r requirements.txt
```

Run the research pipeline:

```
python main.py
```

Run non-interactively (uses the hardcoded fallback query in `main.py`):

```
INTERACTIVE_MODE=auto python main.py
```

`.env` is loaded via `python-dotenv` at startup — `OPENAI_API_KEY` must be set there or in the environment.

There is no test suite, linter, or build step configured.

## Architecture

The app is a multi-agent research pipeline orchestrated by `ResearchManager` (`manager.py`). Flow:

1. `planner_agent` (`subagents/planner_agent.py`) — turns the user's query into a `WebSearchPlan` of 5–20 `WebSearchItem`s via structured output (`output_type=WebSearchPlan`).
2. `search_agent` (`subagents/search_agent.py`) — runs once per planned item **concurrently** via `asyncio.create_task` + `as_completed`. Uses the `WebSearchTool` built into the Agents SDK. Individual failures are swallowed (returned as `None`) so one bad search doesn't abort the run.
3. `writer_agent` (`subagents/writer_agent.py`) — consumes all summaries and emits a `ReportData` (short summary + markdown report + follow-ups). Invoked via `Runner.run_streamed` so progress messages can be cycled every ~5s while the model streams.

The whole run is wrapped in `trace("Research trace", trace_id=...)` from the Agents SDK; the generated trace URL is printed so runs can be inspected at `platform.openai.com/traces`. The web-search fan-out is grouped under a `custom_span("Search the web")`.

UI is `rich.live.Live` driven by `printer/main.py`'s `Printer`, which maintains an id-keyed dict of status lines — spinners for in-progress items, ✅ for done. Update items by id with `printer.update_item(id, text, is_done=...)`; `Printer.end()` must be called to stop the `Live` context before plain `print()` output at the end of `manager.run`.

`auto_mode/main.py` provides `input_with_fallback` / `confirm_with_fallback`. They check `INTERACTIVE_MODE=auto` and return the provided fallback/default instead of blocking on stdin. Use these (not bare `input()`) for any new user prompts so the pipeline stays runnable headless.

## Models

Agents currently reference `gpt-5.4` (planner, search) and `gpt-5-mini` (writer), with `ModelSettings(reasoning=Reasoning(effort="medium"))` on the reasoning-capable ones. Change model IDs in the individual `subagents/*.py` files — there is no central config.
