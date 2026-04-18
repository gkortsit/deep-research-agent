# Writer Critique & Rewrite Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a writer→critique→rewrite loop after `writer_agent` so the final report is checked for structure, faithfulness to the research summaries, and coverage of the original query, and re-written up to `WRITER_CRITIQUE_MAX_REVISIONS` times until it passes a score threshold — all env-configurable and disable-able.

**Architecture:** One new structured-output agent (`writer_critic_agent`) with three dimension-typed issue lists, a small revision-input helper on `writer_agent`, three new `config.py` settings, and a new `ResearchManager._write_with_critique` method that wraps the existing `_write_report`. `_write_report` gains an optional `input_override` argument so rewrite calls can pass a richer input string while preserving streaming. Loop is bounded; failures fall back to the last successful report. Planner, plan-critic, search, and research-evaluator behavior is unchanged.

**Tech Stack:** Python, `openai-agents` SDK (`Agent`, `Runner`, `Runner.run_streamed`, `ModelSettings`, `custom_span`), Pydantic for structured outputs, `rich` for TTY progress (existing `Printer`).

---

## File Structure

**Create:**
- `subagents/writer_critic_agent.py` — writer critique agent + `WriterCritique` model

**Modify:**
- `config.py` — three env-overridable settings for the writer-critique loop
- `subagents/writer_agent.py` — add `build_writer_revision_input` helper
- `manager.py` — add `_write_with_critique` + `_format_report_for_critique`, extend `_write_report` with `input_override`, wire into `run`

No test suite exists in this repo (per CLAUDE.md). Verification is manual via `python -c` import checks and `INTERACTIVE_MODE=auto python main.py` runs with env toggles — every task that adds code ends with a verification step and a commit.

---

## Task 1: Add writer-critique settings to `config.py`

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add three env-overridable settings**

Append to `config.py` (after the existing settings, before the end of the file):

```python
ENABLE_WRITER_CRITIQUE = os.getenv("ENABLE_WRITER_CRITIQUE", "1") == "1"
WRITER_CRITIQUE_MAX_REVISIONS = int(os.getenv("WRITER_CRITIQUE_MAX_REVISIONS", "2"))
WRITER_CRITIQUE_SCORE_THRESHOLD = int(os.getenv("WRITER_CRITIQUE_SCORE_THRESHOLD", "8"))
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "import config; print(config.ENABLE_WRITER_CRITIQUE, config.WRITER_CRITIQUE_MAX_REVISIONS, config.WRITER_CRITIQUE_SCORE_THRESHOLD)"`
Expected: `True 2 8`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "Add writer-critique loop tunables to config"
```

---

## Task 2: Add `writer_critic_agent`

**Files:**
- Create: `subagents/writer_critic_agent.py`

- [ ] **Step 1: Create the critic agent**

Write `subagents/writer_critic_agent.py`:

```python
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

from agents import Agent, ModelSettings


PROMPT = (
    "You are a critical reviewer of long-form research reports. Given the original user "
    "query, the research summaries that were available to the writer, and the writer's "
    "current report, assess whether the report is ready to ship.\n\n"
    "Judge three dimensions independently:\n"
    "- Structure & readability: outline coherence, section balance, flow, markdown "
    "hygiene, and the 1000+ word target.\n"
    "- Faithfulness to summaries: every non-trivial claim must be traceable to at least "
    "one summary. Flag invented facts, numbers, dates, or names. Flag strongly "
    "summary-backed findings that the report dropped.\n"
    "- Coverage of the query: the report must materially answer every sub-question "
    "implied by the original query.\n\n"
    "Score the report 1-10 (10 = publish as-is, 8 = minor polish only, <=7 = revise). "
    "Set is_sufficient=true only when the report is clearly ready to ship.\n\n"
    "Populate structure_issues, faithfulness_issues, and coverage_issues with concrete, "
    "specific problems. Populate suggestions with actionable instructions the writer can "
    "use to revise — be concrete (e.g. 'Expand the \"Grid impact\" section to cover "
    "distribution-level effects from summary 3')."
)


class WriterCritique(BaseModel):
    is_sufficient: bool
    """True if the report is good enough to ship as-is."""

    score: int
    """Quality score from 1 (poor) to 10 (excellent)."""

    structure_issues: list[str]
    """Outline, flow, section balance, length, markdown-hygiene problems."""

    faithfulness_issues: list[str]
    """Unsupported claims, or summary-backed findings the report dropped."""

    coverage_issues: list[str]
    """Sub-questions of the original query not answered by the report."""

    suggestions: list[str]
    """Concrete, actionable instructions for the rewriter."""


writer_critic_agent = Agent(
    name="WriterCriticAgent",
    instructions=PROMPT,
    model="gpt-5.4",
    model_settings=ModelSettings(reasoning=Reasoning(effort="high")),
    output_type=WriterCritique,
)
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from subagents.writer_critic_agent import writer_critic_agent, WriterCritique; print(writer_critic_agent.name)"`
Expected: `WriterCriticAgent`

- [ ] **Step 3: Commit**

```bash
git add subagents/writer_critic_agent.py
git commit -m "Add writer_critic_agent with WriterCritique structured output"
```

---

## Task 3: Add `build_writer_revision_input` helper to `writer_agent`

> Named `build_writer_revision_input` (not `build_revision_input`) to avoid a name collision with the existing `build_revision_input` exported by `subagents/planner_agent.py`, which `manager.py` already imports.

**Files:**
- Modify: `subagents/writer_agent.py`

- [ ] **Step 1: Append the helper after the existing agent definition**

Append to the end of `subagents/writer_agent.py`:

```python
def build_writer_revision_input(
    query: str,
    summaries: list[str],
    previous: ReportData,
    structure_issues: list[str],
    faithfulness_issues: list[str],
    coverage_issues: list[str],
    suggestions: list[str],
) -> str:
    """Build the input string for a writer revision call given critic feedback."""

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {s}" for s in items) or "- (none specified)"

    return (
        f"Original query: {query}\n"
        f"Summarized search results: {summaries}\n\n"
        f"Previous report — short summary:\n{previous.short_summary}\n\n"
        f"Previous report — markdown:\n{previous.markdown_report}\n\n"
        f"Critique — structure:\n{bullets(structure_issues)}\n\n"
        f"Critique — faithfulness:\n{bullets(faithfulness_issues)}\n\n"
        f"Critique — coverage:\n{bullets(coverage_issues)}\n\n"
        f"Suggestions:\n{bullets(suggestions)}\n\n"
        "Produce an improved report that addresses the issues above. You may reuse "
        "strong sections; rewrite weak ones. Do not introduce claims that are not "
        "supported by the summaries."
    )
```

- [ ] **Step 2: Verify it imports and runs**

Run:
```
python -c "from subagents.writer_agent import build_writer_revision_input, ReportData; r = ReportData(short_summary='s', markdown_report='m', follow_up_questions=[]); print(build_writer_revision_input('Q', ['sum1'], r, ['si'], ['fi'], ['ci'], ['sg'])[:80])"
```
Expected: Output starts with `Original query: Q`

- [ ] **Step 3: Commit**

```bash
git add subagents/writer_agent.py
git commit -m "Add build_writer_revision_input helper for writer critique-revision loop"
```

---

## Task 4: Add `input_override` support to `_write_report`

**Files:**
- Modify: `manager.py`

- [ ] **Step 1: Extend `_write_report` with an optional `input_override` parameter**

Replace the existing `_write_report` method (currently at the bottom of `manager.py`) with this version. The only changes are (a) the new `input_override` parameter and (b) using it when provided; everything else is preserved including the streaming and the cycling progress messages.

```python
    async def _write_report(
        self,
        query: str,
        search_results: list[str],
        input_override: str | None = None,
    ) -> ReportData:
        self.printer.update_item("writing", "Thinking about report...")
        if input_override is None:
            input = f"Original query: {query}\nSummarized search results: {search_results}"
        else:
            input = input_override
        result = Runner.run_streamed(
            writer_agent,
            input,
        )
        update_messages = [
            "Thinking about report...",
            "Planning report structure...",
            "Writing outline...",
            "Creating sections...",
            "Cleaning up formatting...",
            "Finalizing report...",
            "Finishing report...",
        ]

        last_update = time.time()
        next_message = 0
        async for _ in result.stream_events():
            if time.time() - last_update > 5 and next_message < len(update_messages):
                self.printer.update_item("writing", update_messages[next_message])
                next_message += 1
                last_update = time.time()

        self.printer.mark_item_done("writing")
        return result.final_output_as(ReportData)
```

- [ ] **Step 2: Run end-to-end to confirm no regression**

Run: `INTERACTIVE_MODE=auto python main.py`
Expected: Pipeline completes as before — the `input_override` path is not exercised by anything yet, so behavior is identical to pre-change. A report is saved to `reports/`.

- [ ] **Step 3: Commit**

```bash
git add manager.py
git commit -m "Add input_override parameter to _write_report"
```

---

## Task 5: Wire writer-critique loop into `ResearchManager`

**Files:**
- Modify: `manager.py`

- [ ] **Step 1: Add writer-critic imports**

At the top of `manager.py`, alongside the other `subagents.*` imports, add:

```python
from subagents.writer_critic_agent import WriterCritique, writer_critic_agent
```

Also, change the existing writer import line from:

```python
from subagents.writer_agent import ReportData, writer_agent
```

to:

```python
from subagents.writer_agent import (
    ReportData,
    build_writer_revision_input,
    writer_agent,
)
```

- [ ] **Step 2: Replace the `run()` call-site to use the new method**

In the `run` method, find these two lines:

```python
            search_results = [entry["summary"] for entry in tagged_results]
            report = await self._write_report(query, search_results)
```

Replace them with a single call — the summary list-comprehension moves inside `_write_with_critique`:

```python
            report = await self._write_with_critique(query, tagged_results)
```

- [ ] **Step 3: Add `_write_with_critique` and `_format_report_for_critique`**

Add these two methods to `ResearchManager`. Place them immediately after `_evaluate_and_fill_gaps` / `_format_results_for_evaluation` and before the existing `_write_report` method so related methods stay grouped:

```python
    async def _write_with_critique(
        self, query: str, tagged_results: list[TaggedResult]
    ) -> ReportData:
        summaries = [entry["summary"] for entry in tagged_results]
        report = await self._write_report(query, summaries)

        if not config.ENABLE_WRITER_CRITIQUE:
            return report

        with custom_span("Writer critique"):
            accepted = False
            for revision in range(1, config.WRITER_CRITIQUE_MAX_REVISIONS + 1):
                critique_label = (
                    f"Critiquing report (rev {revision}/{config.WRITER_CRITIQUE_MAX_REVISIONS})"
                )
                self.printer.update_item("reviewing", f"{critique_label}...")
                try:
                    critique_result = await self._run_with_ticker(
                        writer_critic_agent,
                        self._format_report_for_critique(query, summaries, report),
                        "reviewing",
                        critique_label,
                    )
                    critique = critique_result.final_output_as(WriterCritique)
                except Exception as e:
                    self.printer.update_item(
                        "reviewing",
                        f"Critique failed ({type(e).__name__}); keeping current report",
                        is_done=True,
                    )
                    return report

                if (
                    critique.is_sufficient
                    or critique.score >= config.WRITER_CRITIQUE_SCORE_THRESHOLD
                ):
                    self.printer.update_item(
                        "reviewing",
                        f"Report accepted (score {critique.score}/10)",
                        is_done=True,
                    )
                    accepted = True
                    break

                rewrite_label = f"Report score {critique.score}/10, rewriting"
                self.printer.update_item("reviewing", f"{rewrite_label}...")
                try:
                    report = await self._write_report(
                        query,
                        summaries,
                        input_override=build_writer_revision_input(
                            query,
                            summaries,
                            report,
                            critique.structure_issues,
                            critique.faithfulness_issues,
                            critique.coverage_issues,
                            critique.suggestions,
                        ),
                    )
                except Exception as e:
                    self.printer.update_item(
                        "reviewing",
                        f"Rewrite failed ({type(e).__name__}); keeping prior report",
                        is_done=True,
                    )
                    return report

            if not accepted:
                self.printer.update_item(
                    "reviewing",
                    "Revision cap reached; using last report",
                    is_done=True,
                )
            return report

    def _format_report_for_critique(
        self, query: str, summaries: list[str], report: ReportData
    ) -> str:
        lines = [f"Original query: {query}", "", "Research summaries:"]
        for i, summary in enumerate(summaries):
            lines.append(f"[id={i}] {summary}\n")
        lines.append("")
        lines.append("Current report — short summary:")
        lines.append(report.short_summary)
        lines.append("")
        lines.append("Current report — markdown:")
        lines.append(report.markdown_report)
        return "\n".join(lines)
```

- [ ] **Step 4: Run end-to-end headless with writer critique enabled**

Run: `INTERACTIVE_MODE=auto python main.py`
Expected: pipeline completes. Printer shows a `reviewing` item whose terminal state is one of `Report accepted (score X/10)`, `Revision cap reached; using last report`, `Critique failed (...)`, or `Rewrite failed (...)`. Trace URL printed; opening it should show a `Writer critique` span nested under `Research trace`. A report is saved to `reports/`.

- [ ] **Step 5: Run with writer critique disabled, confirm unchanged behavior**

Run: `ENABLE_WRITER_CRITIQUE=0 INTERACTIVE_MODE=auto python main.py`
Expected: no `reviewing` line appears in the printer output. No `Writer critique` span in the trace. Otherwise behavior matches the pre-change pipeline.

- [ ] **Step 6: Run with score threshold forced high to exercise full revision cap**

Run: `WRITER_CRITIQUE_SCORE_THRESHOLD=11 INTERACTIVE_MODE=auto python main.py`
Expected: the critique loop runs `WRITER_CRITIQUE_MAX_REVISIONS` (default 2) iterations without acceptance; terminal state on `reviewing` is `Revision cap reached; using last report`. Pipeline still completes and saves a report.

- [ ] **Step 7: Commit**

```bash
git add manager.py
git commit -m "Wire writer critique + rewrite loop into ResearchManager"
```

---

## Self-Review Checklist (verify after implementation)

- **Spec coverage:**
  - Writer critic agent + `WriterCritique` schema (3 dimension lists): Task 2.
  - Writer revision helper `build_writer_revision_input`: Task 3.
  - `_write_report` gains `input_override`: Task 4.
  - `_write_with_critique` + `_format_report_for_critique` + `run()` wiring: Task 5.
  - `custom_span("Writer critique")` tracing: Task 5, Step 3.
  - Error fallback on critique failure and rewrite failure: Task 5, Step 3.
  - `ENABLE_WRITER_CRITIQUE=0` bypass: Task 5, Step 3 (early return) verified in Step 5.
  - Score-threshold early exit and revision cap: Task 5, Step 3 verified in Steps 4 & 6.
  - "Revision cap reached" terminal state: Task 5, Step 3 (`if not accepted` block).
  - `AGENT_CALL_TIMEOUT_S` coverage via `_run_with_ticker`: Task 5, Step 3 (critic path). The rewrite path reuses `_write_report` which does its own streaming — no per-call timeout applies there, matching the pre-existing treatment of `_write_report`.
  - Config env vars: Task 1.
  - Headless/auto mode (no new stdin): no new prompts introduced; verified in Task 5 Steps 4-6.

- **Placeholders:** none. Every code step contains full code; every verification step is an explicit command with expected output.

- **Names consistent across tasks:** `WriterCritique`, `writer_critic_agent`, `build_writer_revision_input`, `_write_with_critique`, `_format_report_for_critique`, `_write_report`, `input_override`, `ReportData`, `TaggedResult`. The new writer helper is named `build_writer_revision_input` (not `build_revision_input`) to avoid colliding with the existing `build_revision_input` that `manager.py` already imports from `subagents/planner_agent.py`.
