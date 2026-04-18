# Writer Critique & Rewrite Loop — Design

**Date:** 2026-04-18
**Status:** Draft — awaiting user review

## Problem

The research pipeline has feedback loops at the planning stage (`plan_critic_agent` → planner revision) and after searches (`research_evaluator_agent` → gap-fill searches), but the final writer stage is still single-shot. Whatever `writer_agent` emits is saved verbatim. There is no check that the report actually answers the user's query, faithfully reflects the research summaries, or holds together structurally. Long-form reports fail in distinctive ways (dropped findings, unsupported claims, off-topic tangents, thin coverage of a query sub-question) that are cheap for a reviewer to catch and often fixable by re-invoking the writer with targeted feedback.

## Goals

1. After the writer emits `ReportData`, run a critic that judges three dimensions: **structure & readability**, **faithfulness to the research summaries**, and **coverage of the original query**.
2. If the critic isn't satisfied, re-invoke `writer_agent` with the original query, the same summaries, the previous report, and the critique as additional input — bounded by `WRITER_CRITIQUE_MAX_REVISIONS` (default 2) with early-exit when `is_sufficient=true` or `score >= WRITER_CRITIQUE_SCORE_THRESHOLD` (default 8).
3. Keep the pipeline headless-runnable, cost-bounded, and backward compatible: `ENABLE_WRITER_CRITIQUE=0` must make the behavior identical to today.

## Non-goals

- Targeted section-level patches. Each revision is a full regeneration of the report; the writer uses the previous report as context.
- Interactive user approval between revisions. All knobs are env vars; no new stdin prompts.
- Source citations / URLs. The critic will flag unsupported claims, but it cannot synthesize citations that the upstream search summaries never carried.
- Changes to planner, plan-critic, search, or research-evaluator behavior.
- Keeping the highest-scoring report across revisions. V1 returns the last successful `ReportData` (which may be the pre-critique report if the first rewrite fails).
- Saving intermediate report drafts to disk. Only the final accepted `ReportData` reaches `save_report`.

## Architecture

```
query
 └─> planner → plan-critic loop → searches → research-eval loop      (existing)
 └─> writer_agent                                ──┐
       └─> writer_critic_agent                     │  Writer critique loop (NEW)
             └─> writer_agent (rewrite)            │
                   (≤2 revisions; early-exit on sufficient or score ≥ threshold)
 └─> save_report                                                     (existing)
```

**New files**

- `subagents/writer_critic_agent.py` — defines `writer_critic_agent` (Agent) and `WriterCritique` (Pydantic `BaseModel`, structured-output type).

**Modified files**

- `manager.py` — add `_write_with_critique` method; `run()` calls it in place of the direct `_write_report` call; add `_format_report_for_critique` helper. `_write_report` gains an optional `input_override: str | None = None` parameter so rewrites can pass a richer input string while preserving streaming.
- `subagents/writer_agent.py` — add `build_revision_input(query, summaries, previous, structure_issues, faithfulness_issues, coverage_issues, suggestions) -> str` helper (mirrors `planner_agent.build_revision_input`).
- `config.py` — add three env-overridable settings for the new loop.

**Where it sits in the flow.** Critique happens after `research_evaluator_agent` has already settled on the summary set. The critic sees the same frozen summaries that the writer consumed. `save_report(report)` stays where it is, *after* the loop — only the final accepted report is saved.

**Separation from `research_evaluator_agent`.** Research-evaluator asks *"do the summaries cover the query?"* (input-side). Writer-critic asks *"does the report faithfully reflect the summaries and answer the query?"* (output-side). Different artifacts, different failure modes — no shared state between the two loops.

**Tracing.** Wrap the critique loop body in `custom_span("Writer critique")`, nested under the existing top-level `trace("Research trace", ...)`.

## Components

### `writer_critic_agent`

- Model: `gpt-5.4` with `ModelSettings(reasoning=Reasoning(effort="high"))`. Matches `plan_critic_agent`; judging a long-form report against many summaries is the most reasoning-heavy step in the pipeline, so effort is spent on the judge, not on repeat rewrites.
- Input: original query, research summaries in the same tagged-id format used by `research_evaluator_agent` (`[id=N] query=... summary: ...`), and the current report (`short_summary` plus `markdown_report`; `follow_up_questions` is omitted — judging follow-ups is out of scope and they are cheap to regenerate).
- Output (Pydantic structured output):

```python
class WriterCritique(BaseModel):
    is_sufficient: bool                     # True if the report is good enough to ship
    score: int                              # 1-10 (10 = publish as-is)
    structure_issues: list[str]             # outline, flow, section balance, length, markdown hygiene
    faithfulness_issues: list[str]          # unsupported claims; summary-backed findings dropped
    coverage_issues: list[str]              # sub-questions of the query not answered
    suggestions: list[str]                  # concrete, actionable instructions to the rewriter
```

Three separate issue-lists (vs. one flat `issues[]`) because the critic is asked to judge three distinct dimensions — keeping them separate forces the model to consider each in turn, and gives the revision prompt category labels that tend to produce more targeted rewrites.

- Prompt emphasizes:
  - **Structure**: outline coherence, section balance, 1000+ word target, markdown hygiene.
  - **Faithfulness**: every non-trivial claim traceable to at least one summary; no invented facts, numbers, dates, names; no strongly summary-backed findings dropped.
  - **Coverage**: the report materially answers every sub-question implied by the original query.
  - Score rubric: 10 = publish as-is, 8 = minor polish only, ≤7 = revise.
  - `is_sufficient=true` only when the report is clearly ready to ship.

### Writer revision helper

New function in `subagents/writer_agent.py`:

```python
def build_revision_input(
    query: str,
    summaries: list[str],
    previous: ReportData,
    structure_issues: list[str],
    faithfulness_issues: list[str],
    coverage_issues: list[str],
    suggestions: list[str],
) -> str:
    """Build the input string for a writer revision call given critic feedback."""
```

Produced input is shaped like:

```
Original query: ...
Summarized search results: [...]
Previous report — short summary: ...
Previous report — markdown:
...
Critique — structure:
- ...
Critique — faithfulness:
- ...
Critique — coverage:
- ...
Suggestions:
- ...

Produce an improved report that addresses the issues above. You may reuse
strong sections; rewrite weak ones. Do not introduce claims that are not
supported by the summaries.
```

Same `writer_agent` instance, same `ReportData` output type; only the input string is richer on revision calls. Streaming via `Runner.run_streamed` is preserved (see `_write_report` changes below).

### `config.py` additions

```python
ENABLE_WRITER_CRITIQUE          = os.getenv("ENABLE_WRITER_CRITIQUE", "1") == "1"
WRITER_CRITIQUE_MAX_REVISIONS   = int(os.getenv("WRITER_CRITIQUE_MAX_REVISIONS", "2"))
WRITER_CRITIQUE_SCORE_THRESHOLD = int(os.getenv("WRITER_CRITIQUE_SCORE_THRESHOLD", "8"))
```

## Control flow

### `_write_report` change

`_write_report` gains an optional `input_override: str | None = None` parameter. When `None`, it builds the existing default input (`f"Original query: {query}\nSummarized search results: {search_results}"`); when provided, `input_override` is passed to the writer verbatim and the `search_results` argument is ignored for input-building purposes (the caller is responsible for including whatever summaries the revision needs). Streaming (`Runner.run_streamed`) and the cycling progress messages are unchanged.

### `_write_with_critique(query: str, tagged_summaries: list[TaggedResult]) -> ReportData`

1. `summaries = [r["summary"] for r in tagged_summaries]`.
2. `report = await self._write_report(query, summaries)`.
3. If `not config.ENABLE_WRITER_CRITIQUE`, return `report`.
4. `with custom_span("Writer critique"):` for `revision` in `1..config.WRITER_CRITIQUE_MAX_REVISIONS`:
   - Printer: update `reviewing` item → `Critiquing report (rev N/M)...`; elapsed-time ticker via `_run_with_ticker`.
   - Run `writer_critic_agent` on `self._format_report_for_critique(query, summaries, report)`.
   - On exception: printer logs `Critique failed ({TypeName}); keeping current report`; break; return last successful `report`.
   - If `critique.is_sufficient` or `critique.score >= config.WRITER_CRITIQUE_SCORE_THRESHOLD`: printer logs `Report accepted (score X/10)`; break.
   - Otherwise: printer logs `Report score X/10, rewriting...`; call `_write_report(query, summaries, input_override=writer_agent.build_revision_input(...))`. On exception: printer logs `Rewrite failed ({TypeName}); keeping prior report`; break; return last successful `report`.
5. If the loop exits by exhausting `WRITER_CRITIQUE_MAX_REVISIONS` without ever hitting the score threshold or `is_sufficient`, printer logs `Revision cap reached; using last report` as the terminal state on the `reviewing` item.
6. Return the final `report` (the last successful one — may be pre-critique if the very first rewrite failed).

### `_format_report_for_critique(query, summaries, report) -> str`

Produces input of the form:

```
Original query: ...

Research summaries:
[id=0] ...
[id=1] ...
...

Current report — short summary:
...

Current report — markdown:
...
```

### `run()` wiring change

Replace `report = await self._write_report(query, search_results)` with:

```python
report = await self._write_with_critique(query, tagged_results)
```

This also removes the existing `search_results = [entry["summary"] for entry in tagged_results]` line from `run()`; the list-comprehension moves inside `_write_with_critique`.

## Error handling

- `writer_critic_agent` failure → log to printer, break, return last successful report. Same shape as `plan_critic_agent` failures.
- `writer_agent` rewrite failure → log to printer, break, return the report from before the failed rewrite.
- `ENABLE_WRITER_CRITIQUE=0` → skip the loop entirely.
- `AGENT_CALL_TIMEOUT_S` already applies via `_run_with_ticker`.

## Headless / auto mode

No new stdin prompts. `INTERACTIVE_MODE=auto python main.py` continues to work unchanged. All knobs are env vars.

## Tracing & printer

- `custom_span("Writer critique")` wraps the loop body.
- New printer item keyed `reviewing`. State transitions:
  - `Critiquing report (rev 1/2)...` (spinner, ticker)
  - `Report score 7/10, rewriting...` (spinner, ticker) — rewrite reuses the existing `writing` printer item's streaming-progress messages
  - `Report accepted (score 9/10)` (✅) — terminal success
  - `Critique failed (TimeoutError); keeping current report` (✅) — terminal fallback
  - `Revision cap reached; using last report` (✅) — terminal fallback when max revisions is exhausted without acceptance

## Testing

No automated test suite in repo. Manual verification:

- `INTERACTIVE_MODE=auto python main.py` with defaults → pipeline completes; trace shows `Writer critique` span; printer shows `reviewing` item transitioning to a terminal state.
- `ENABLE_WRITER_CRITIQUE=0 INTERACTIVE_MODE=auto python main.py` → identical behavior to the current pipeline; no `reviewing` item appears; no `Writer critique` span in the trace.
- `WRITER_CRITIQUE_SCORE_THRESHOLD=11 INTERACTIVE_MODE=auto python main.py` → loop runs the full `WRITER_CRITIQUE_MAX_REVISIONS` iterations; final state is `Revision cap reached; using last report`.
- Deliberately narrow query (e.g., one that the writer tends to under-cover) → visible rewrite happens in at least one revision.

## Risks & mitigations

- **Cost / latency increase.** Reports are the most expensive stage. Mitigated by default `WRITER_CRITIQUE_MAX_REVISIONS=2`, `WRITER_CRITIQUE_SCORE_THRESHOLD=8`, and the `ENABLE_WRITER_CRITIQUE` disable flag.
- **Critic score-inflation / perpetual 7.** If a critic always scores in the 6–7 band we hit the revision cap every run. Acceptable: the cap is a hard ceiling and the cost is still bounded.
- **Rewrite drift** (a later revision worse than an earlier one). V1 returns the *last* successful report, not the highest-scoring. Acceptable for a first version; keeping the best-scoring version across revisions is a possible future improvement but adds state and isn't justified by evidence yet.
- **Structured-output drift.** Pydantic `output_type=` validation is already enforced by the SDK.
