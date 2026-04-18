# openaiagents

A small multi-agent research pipeline built on the [`openai-agents`](https://pypi.org/project/openai-agents/) SDK. Give it a query; it plans web searches, runs them concurrently, and synthesizes a markdown report.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=sk-...
```

## Usage

Interactive:

```bash
python main.py
```

Headless (uses the fallback query hardcoded in `main.py`):

```bash
INTERACTIVE_MODE=auto python main.py
```

The run prints a trace URL (`platform.openai.com/traces/...`) that you can open to inspect each agent step.

## How it works

Three agents, orchestrated by `ResearchManager` in `manager.py`:

1. **Planner** (`subagents/planner_agent.py`) — turns the query into 5–20 search items with structured output.
2. **Search** (`subagents/search_agent.py`) — runs one concurrent task per item using the SDK's `WebSearchTool`, summarizing each in 2–3 paragraphs. Individual failures are skipped, not fatal.
3. **Writer** (`subagents/writer_agent.py`) — streams a long-form markdown report plus a short summary and follow-up questions.

Progress is rendered live with `rich` via `printer/main.py`. The `auto_mode/` helpers (`input_with_fallback`, `confirm_with_fallback`) make the pipeline runnable non-interactively when `INTERACTIVE_MODE=auto`.

## Project layout

```
main.py              # entry point
manager.py           # ResearchManager — orchestrates the pipeline
subagents/           # planner, search, writer agents
printer/             # rich.Live-based status printer
auto_mode/           # non-interactive input/confirm helpers
```

## Configuration

Model IDs live in each `subagents/*.py` file — there is no central config. Planner and search use a reasoning model; writer uses a smaller model with streaming.
