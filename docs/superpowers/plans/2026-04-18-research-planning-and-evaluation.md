# Research Planning & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plan→critique→revise loop to the planning stage and a coverage+quality evaluation loop (with gap-filling searches and low-quality summary dropping) after search execution, all env-configurable and disable-able.

**Architecture:** Two new structured-output agents (`plan_critic_agent`, `research_evaluator_agent`), a single `config.py` for env-overridable tunables, and two new manager methods (`_plan_with_critique`, `_evaluate_and_fill_gaps`) that wrap the existing planning/search phases. Loops are bounded with early-exit; failures fall back to proceeding with current state. All existing agent behavior preserved.

**Tech Stack:** Python, `openai-agents` SDK (`Agent`, `Runner`, `ModelSettings`, `custom_span`), Pydantic for structured outputs, `rich` for TTY progress (existing `Printer`).

---

## File Structure

**Create:**
- `config.py` — env-overridable tunables for both loops
- `subagents/plan_critic_agent.py` — plan critique agent + `PlanCritique` model
- `subagents/research_evaluator_agent.py` — research evaluator agent + `ResearchEvaluation` model

**Modify:**
- `manager.py` — add `_plan_with_critique`, `_evaluate_and_fill_gaps`, wire into `run`; tag search results with ids internally
- `subagents/planner_agent.py` — add a `REVISION_PROMPT` helper/function that produces input text for a revision call (keeps the single planner agent, just a second invocation with richer input)

No test suite exists in this repo (per CLAUDE.md). Verification is manual via `python main.py` runs, which this plan includes as explicit steps.

---

## Task 1: Add `config.py`

**Files:**
- Create: `config.py`

- [ ] **Step 1: Create config.py with env-overridable settings**

```python
import os

PLAN_CRITIQUE_MAX_REVISIONS = int(os.getenv("PLAN_CRITIQUE_MAX_REVISIONS", "2"))
PLAN_CRITIQUE_SCORE_THRESHOLD = int(os.getenv("PLAN_CRITIQUE_SCORE_THRESHOLD", "8"))
EVAL_MAX_EXTRA_ROUNDS = int(os.getenv("EVAL_MAX_EXTRA_ROUNDS", "2"))
EVAL_MAX_GAP_SEARCHES = int(os.getenv("EVAL_MAX_GAP_SEARCHES", "5"))
ENABLE_PLAN_CRITIQUE = os.getenv("ENABLE_PLAN_CRITIQUE", "1") == "1"
ENABLE_RESEARCH_EVAL = os.getenv("ENABLE_RESEARCH_EVAL", "1") == "1"
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "import config; print(config.PLAN_CRITIQUE_MAX_REVISIONS, config.ENABLE_PLAN_CRITIQUE)"`
Expected: `2 True`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "Add config module with env-overridable loop tunables"
```

---

## Task 2: Add `plan_critic_agent`

**Files:**
- Create: `subagents/plan_critic_agent.py`

- [ ] **Step 1: Create the critic agent**

```python
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

from agents import Agent, ModelSettings


PROMPT = (
    "You are a critical reviewer of research search plans. Given an original user query "
    "and a proposed list of web searches, assess whether the plan will actually answer "
    "the query well.\n\n"
    "Evaluate:\n"
    "- Coverage: do the searches address every major sub-question implied by the query?\n"
    "- Redundancy: are any searches near-duplicates?\n"
    "- Specificity: are search terms concrete enough to return high-signal results, or are they vague?\n"
    "- Missing angles: adjacent context, counter-evidence, authoritative sources, recent developments.\n\n"
    "Score the plan 1-10 (10 = excellent). Set is_sufficient=true only if the plan is "
    "clearly good enough to proceed without revision. List concrete issues and actionable "
    "suggestions — the planner will use these to revise."
)


class PlanCritique(BaseModel):
    is_sufficient: bool
    """True if the plan is good enough to execute as-is."""

    score: int
    """Quality score from 1 (poor) to 10 (excellent)."""

    issues: list[str]
    """Specific problems with the current plan."""

    suggestions: list[str]
    """Concrete suggestions for improving the plan."""


plan_critic_agent = Agent(
    name="PlanCriticAgent",
    instructions=PROMPT,
    model="gpt-5.4",
    model_settings=ModelSettings(reasoning=Reasoning(effort="high")),
    output_type=PlanCritique,
)
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from subagents.plan_critic_agent import plan_critic_agent, PlanCritique; print(plan_critic_agent.name)"`
Expected: `PlanCriticAgent`

- [ ] **Step 3: Commit**

```bash
git add subagents/plan_critic_agent.py
git commit -m "Add plan_critic_agent with PlanCritique structured output"
```

---

## Task 3: Add revision-prompt helper to `planner_agent`

**Files:**
- Modify: `subagents/planner_agent.py`

- [ ] **Step 1: Append a `build_revision_input` helper after the existing agent definition**

Add to end of `subagents/planner_agent.py`:

```python
def build_revision_input(
    query: str, previous_plan: WebSearchPlan, issues: list[str], suggestions: list[str]
) -> str:
    """Build the input string for a planner revision call given critic feedback."""
    prev = "\n".join(
        f"- {item.query} — {item.reason}" for item in previous_plan.searches
    )
    issues_text = "\n".join(f"- {i}" for i in issues) or "- (none specified)"
    suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "- (none specified)"
    return (
        f"Query: {query}\n\n"
        f"Previous plan:\n{prev}\n\n"
        f"Critique issues:\n{issues_text}\n\n"
        f"Suggestions:\n{suggestions_text}\n\n"
        "Produce an improved plan addressing the issues and suggestions. "
        "You may reuse good searches from the previous plan, drop weak ones, and add new ones."
    )
```

- [ ] **Step 2: Verify it imports and runs**

Run:
```
python -c "from subagents.planner_agent import build_revision_input, WebSearchPlan, WebSearchItem; p = WebSearchPlan(searches=[WebSearchItem(reason='r', query='q')]); print(build_revision_input('Q', p, ['i1'], ['s1'])[:80])"
```
Expected: Output starts with `Query: Q`

- [ ] **Step 3: Commit**

```bash
git add subagents/planner_agent.py
git commit -m "Add build_revision_input helper for planner critique-revision loop"
```

---

## Task 4: Add `research_evaluator_agent`

**Files:**
- Create: `subagents/research_evaluator_agent.py`

- [ ] **Step 1: Create the evaluator agent**

```python
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

from agents import Agent, ModelSettings

from subagents.planner_agent import WebSearchItem


PROMPT = (
    "You evaluate research progress. Given the original user query and a set of search "
    "summaries (each tagged with an integer id and the search query that produced it), "
    "decide whether the information is sufficient to write a strong report.\n\n"
    "For each summary, judge quality: is it on-topic, specific, and useful? Add the id "
    "to discard_ids if the summary is off-topic, contentless, or clearly low-signal.\n\n"
    "Identify coverage_gaps: sub-questions of the original query that are not yet "
    "answered by the retained summaries.\n\n"
    "If gaps exist, propose additional_searches (WebSearchItem objects) that would close "
    "them. Be concrete — vague terms waste a round. Propose at most 5.\n\n"
    "Set is_sufficient=true only if retained summaries already cover the query well "
    "enough that a reader would find the eventual report complete."
)


class ResearchEvaluation(BaseModel):
    is_sufficient: bool
    """True if no further searches are needed."""

    coverage_gaps: list[str]
    """Sub-questions of the original query not yet answered."""

    discard_ids: list[int]
    """Ids of summaries to drop as low-quality or off-topic."""

    additional_searches: list[WebSearchItem]
    """Follow-up searches to close coverage gaps (capped downstream)."""


research_evaluator_agent = Agent(
    name="ResearchEvaluatorAgent",
    instructions=PROMPT,
    model="gpt-5.4",
    model_settings=ModelSettings(reasoning=Reasoning(effort="high")),
    output_type=ResearchEvaluation,
)
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from subagents.research_evaluator_agent import research_evaluator_agent, ResearchEvaluation; print(research_evaluator_agent.name)"`
Expected: `ResearchEvaluatorAgent`

- [ ] **Step 3: Commit**

```bash
git add subagents/research_evaluator_agent.py
git commit -m "Add research_evaluator_agent with ResearchEvaluation structured output"
```

---

## Task 5: Wire plan-critique loop into `ResearchManager`

**Files:**
- Modify: `manager.py`

- [ ] **Step 1: Update imports at top of `manager.py`**

Replace the existing import block (lines 1-14) with:

```python
from __future__ import annotations

import asyncio
import time

from rich.console import Console

from agents import Runner, custom_span, gen_trace_id, trace

import config
from subagents.planner_agent import (
    WebSearchItem,
    WebSearchPlan,
    build_revision_input,
    planner_agent,
)
from subagents.plan_critic_agent import PlanCritique, plan_critic_agent
from subagents.search_agent import search_agent
from subagents.writer_agent import ReportData, writer_agent
from printer.main import Printer
from report_writer import save_report
```

- [ ] **Step 2: Replace `_plan_searches` with `_plan_with_critique`**

Replace the existing `_plan_searches` method (currently at lines 58-69) with:

```python
    async def _plan_with_critique(self, query: str) -> WebSearchPlan:
        with custom_span("Plan critique"):
            self.printer.update_item("planning", "Planning searches...")
            result = await Runner.run(planner_agent, f"Query: {query}")
            plan = result.final_output_as(WebSearchPlan)

            if not config.ENABLE_PLAN_CRITIQUE:
                self.printer.update_item(
                    "planning",
                    f"Will perform {len(plan.searches)} searches",
                    is_done=True,
                )
                return plan

            for revision in range(1, config.PLAN_CRITIQUE_MAX_REVISIONS + 1):
                self.printer.update_item(
                    "planning",
                    f"Critiquing plan (rev {revision}/{config.PLAN_CRITIQUE_MAX_REVISIONS})...",
                )
                try:
                    critique_result = await Runner.run(
                        plan_critic_agent,
                        self._format_plan_for_critique(query, plan),
                    )
                    critique = critique_result.final_output_as(PlanCritique)
                except Exception as e:
                    self.printer.update_item(
                        "planning",
                        f"Critique failed ({type(e).__name__}); proceeding with current plan",
                    )
                    break

                if (
                    critique.is_sufficient
                    or critique.score >= config.PLAN_CRITIQUE_SCORE_THRESHOLD
                ):
                    self.printer.update_item(
                        "planning",
                        f"Plan accepted (score {critique.score}/10)",
                    )
                    break

                self.printer.update_item(
                    "planning",
                    f"Plan score {critique.score}/10, revising...",
                )
                try:
                    revised = await Runner.run(
                        planner_agent,
                        build_revision_input(
                            query, plan, critique.issues, critique.suggestions
                        ),
                    )
                    plan = revised.final_output_as(WebSearchPlan)
                except Exception as e:
                    self.printer.update_item(
                        "planning",
                        f"Revision failed ({type(e).__name__}); keeping prior plan",
                    )
                    break

            self.printer.update_item(
                "planning",
                f"Will perform {len(plan.searches)} searches",
                is_done=True,
            )
            return plan

    def _format_plan_for_critique(self, query: str, plan: WebSearchPlan) -> str:
        lines = [f"Query: {query}", "", "Proposed plan:"]
        for i, item in enumerate(plan.searches, 1):
            lines.append(f"{i}. {item.query} — {item.reason}")
        return "\n".join(lines)
```

- [ ] **Step 3: Update `run` to call the new method**

In the `run` method body, change:

```python
            search_plan = await self._plan_searches(query)
```

to:

```python
            search_plan = await self._plan_with_critique(query)
```

- [ ] **Step 4: Run end-to-end headless with critique enabled**

Run: `INTERACTIVE_MODE=auto python main.py`
Expected: pipeline completes; printer shows `Critiquing plan...` and either `Plan accepted (score X/10)` or `Will perform N searches` as final line on the planning item. Trace URL printed.

- [ ] **Step 5: Run with critique disabled, confirm unchanged behavior**

Run: `ENABLE_PLAN_CRITIQUE=0 INTERACTIVE_MODE=auto python main.py`
Expected: no critique lines shown; `Will perform N searches` printed directly after planning.

- [ ] **Step 6: Commit**

```bash
git add manager.py
git commit -m "Wire plan-critique loop into ResearchManager"
```

---

## Task 6: Wire research-evaluation + gap-fill loop into `ResearchManager`

**Files:**
- Modify: `manager.py`

- [ ] **Step 1: Add evaluator imports**

At the top of `manager.py` alongside the other `subagents.*` imports added in Task 5, add:

```python
from subagents.research_evaluator_agent import (
    ResearchEvaluation,
    research_evaluator_agent,
)
```

- [ ] **Step 2: Change `_perform_searches` to return tagged results**

Replace the `_perform_searches` method (currently lines 71-100) so it returns `list[dict]` with `id`/`query`/`summary` entries, using an `id_offset` so gap-fill rounds can extend ids without collision:

```python
    async def _perform_searches(
        self, items: list[WebSearchItem], id_offset: int = 0, label: str = "searching"
    ) -> list[dict]:
        with custom_span("Search the web"):
            self.printer.update_item(label, "Searching...")
            num_completed = 0
            num_succeeded = 0
            num_failed = 0
            tasks = [
                asyncio.create_task(self._search_indexed(i + id_offset, item))
                for i, item in enumerate(items)
            ]
            results: list[dict] = []
            for task in asyncio.as_completed(tasks):
                entry = await task
                if entry is not None:
                    results.append(entry)
                    num_succeeded += 1
                else:
                    num_failed += 1
                num_completed += 1
                status = f"Searching... {num_completed}/{len(tasks)} finished"
                if num_failed:
                    status += f" ({num_succeeded} succeeded, {num_failed} failed)"
                self.printer.update_item(label, status)
            summary = f"Searches finished: {num_succeeded}/{len(tasks)} succeeded"
            if num_failed:
                summary += f", {num_failed} failed"
            self.printer.update_item(label, summary, is_done=True)
            return results

    async def _search_indexed(self, id_: int, item: WebSearchItem) -> dict | None:
        summary = await self._search(item)
        if summary is None:
            return None
        return {"id": id_, "query": item.query, "summary": summary}
```

- [ ] **Step 3: Update the `run` method to call the new signatures**

Replace in `run`:

```python
            search_results = await self._perform_searches(search_plan)
            report = await self._write_report(query, search_results)
```

with:

```python
            tagged_results = await self._perform_searches(search_plan.searches)
            tagged_results = await self._evaluate_and_fill_gaps(query, tagged_results)
            search_results = [entry["summary"] for entry in tagged_results]
            report = await self._write_report(query, search_results)
```

- [ ] **Step 4: Add `_evaluate_and_fill_gaps`**

Add this method to `ResearchManager` (place after `_perform_searches`):

```python
    async def _evaluate_and_fill_gaps(
        self, query: str, tagged_results: list[dict]
    ) -> list[dict]:
        if not config.ENABLE_RESEARCH_EVAL:
            return tagged_results

        with custom_span("Research evaluation"):
            for round_num in range(1, config.EVAL_MAX_EXTRA_ROUNDS + 1):
                self.printer.update_item(
                    "evaluating",
                    f"Evaluating coverage (round {round_num}/{config.EVAL_MAX_EXTRA_ROUNDS})...",
                )
                try:
                    eval_result = await Runner.run(
                        research_evaluator_agent,
                        self._format_results_for_evaluation(query, tagged_results),
                    )
                    evaluation = eval_result.final_output_as(ResearchEvaluation)
                except Exception as e:
                    self.printer.update_item(
                        "evaluating",
                        f"Evaluation failed ({type(e).__name__}); proceeding with current results",
                        is_done=True,
                    )
                    return tagged_results

                if evaluation.discard_ids:
                    discard = set(evaluation.discard_ids)
                    before = len(tagged_results)
                    tagged_results = [
                        r for r in tagged_results if r["id"] not in discard
                    ]
                    dropped = before - len(tagged_results)
                    self.printer.update_item(
                        "evaluating",
                        f"Round {round_num}: dropped {dropped} low-quality summaries",
                    )

                if evaluation.is_sufficient or not evaluation.additional_searches:
                    self.printer.update_item(
                        "evaluating",
                        f"Coverage sufficient after round {round_num}",
                        is_done=True,
                    )
                    return tagged_results

                gap_items = evaluation.additional_searches[: config.EVAL_MAX_GAP_SEARCHES]
                next_id = (max((r["id"] for r in tagged_results), default=-1)) + 1
                self.printer.update_item(
                    "evaluating",
                    f"Round {round_num}: running {len(gap_items)} gap-fill searches",
                )
                new_results = await self._perform_searches(
                    gap_items, id_offset=next_id, label=f"gapfill_{round_num}"
                )
                tagged_results.extend(new_results)

            self.printer.update_item(
                "evaluating",
                f"Evaluation rounds exhausted; proceeding with {len(tagged_results)} summaries",
                is_done=True,
            )
            return tagged_results

    def _format_results_for_evaluation(
        self, query: str, tagged_results: list[dict]
    ) -> str:
        lines = [f"Original query: {query}", "", "Current research summaries:"]
        for entry in tagged_results:
            lines.append(
                f"[id={entry['id']}] query={entry['query']}\nsummary: {entry['summary']}\n"
            )
        return "\n".join(lines)
```

- [ ] **Step 5: Run end-to-end headless with evaluation enabled**

Run: `INTERACTIVE_MODE=auto python main.py`
Expected: pipeline completes; printer shows `Evaluating coverage (round 1/2)...` and either `Coverage sufficient after round N` or `Evaluation rounds exhausted`. Trace shows a `Research evaluation` span.

- [ ] **Step 6: Run with evaluation disabled**

Run: `ENABLE_RESEARCH_EVAL=0 INTERACTIVE_MODE=auto python main.py`
Expected: no evaluating lines appear; behavior matches pre-change flow after search stage.

- [ ] **Step 7: Run with both loops disabled to confirm full fallback**

Run: `ENABLE_PLAN_CRITIQUE=0 ENABLE_RESEARCH_EVAL=0 INTERACTIVE_MODE=auto python main.py`
Expected: pipeline behavior is identical to pre-change baseline (no critique, no evaluation, unchanged printer output for planning/searching).

- [ ] **Step 8: Run with threshold forced high to exercise full revision cap**

Run: `PLAN_CRITIQUE_SCORE_THRESHOLD=11 INTERACTIVE_MODE=auto python main.py`
Expected: planner is revised up to 2 times before proceeding; final line still shows `Will perform N searches`.

- [ ] **Step 9: Commit**

```bash
git add manager.py
git commit -m "Wire research evaluation + gap-fill loop into ResearchManager"
```

---

## Self-Review Checklist (verify after implementation)

- Spec coverage: plan-critique loop (Task 5), research eval loop (Task 6), config module (Task 1), critic + evaluator agents (Tasks 2, 4), planner revision helper (Task 3), tracing spans (Tasks 5 & 6), error fallback (Tasks 5 & 6), headless mode (all verification steps use `INTERACTIVE_MODE=auto`). All spec sections covered.
- Names consistent across tasks: `PlanCritique`, `ResearchEvaluation`, `WebSearchItem`, `WebSearchPlan`, `build_revision_input`, `_plan_with_critique`, `_evaluate_and_fill_gaps`, `_perform_searches`, `_search_indexed`.
- No placeholders — every code step contains full code; every test step is an explicit run command with expected output.
