# Research Planning & Evaluation — Design

**Date:** 2026-04-18
**Status:** Approved for implementation planning

## Problem

The current pipeline is single-shot at every stage: the planner produces one plan and we commit to it, and the writer consumes whatever summaries come back without any check that coverage is adequate. Models produce better results with more time to think and with the ability to correct their own work, but today there are no feedback loops.

## Goals

1. Give the planner a chance to self-correct via a plan→critique→revise loop.
2. After searches complete, evaluate coverage and summary quality; run gap-filling searches and drop low-quality summaries before writing.
3. Keep the pipeline headless-runnable, cost-bounded, and backward compatible (loops can be disabled via env vars).

## Non-goals

- Post-write report critique / rewrite loop (explicitly out of scope).
- Reworking `search_agent` or `writer_agent` prompts/models.
- Any central model/config refactor beyond what this feature needs.

## Architecture

```
query
  └─> planner_agent         ──┐
        └─> plan_critic_agent ─┤  plan critique loop
              └─> planner_agent (revise with critique)
                    (≤2 revisions; early-exit on sufficient or score ≥ threshold)
  └─> perform_searches (concurrent, existing)
  └─> research_evaluator_agent ──┐ research eval loop
        ├─> drop low-quality summaries by id
        └─> gap-fill searches (≤5 per round) ─> perform_searches ─> re-evaluate
              (≤2 extra rounds; early-exit on sufficient)
  └─> writer_agent (existing)
```

New files:

- `subagents/plan_critic_agent.py`
- `subagents/research_evaluator_agent.py`
- `config.py`

Modified:

- `manager.py` — new `_plan_with_critique` and `_evaluate_and_fill_gaps` methods; existing methods unchanged in signature.
- `subagents/planner_agent.py` — add a small revision-prompt helper or second agent instance that accepts `(query, previous_plan, critique)` context.

## Components

### `plan_critic_agent`

- Model: `gpt-5.4`, `ModelSettings(reasoning=Reasoning(effort="high"))`.
- Input: original query plus current `WebSearchPlan` serialized as JSON-like text.
- Output (structured):

```python
class PlanCritique(BaseModel):
    is_sufficient: bool
    score: int  # 1-10
    issues: list[str]
    suggestions: list[str]
```

- Prompt emphasizes: coverage of the query's sub-questions, redundancy, specificity of search terms, missing angles.

### Planner revision

- When the critic is not satisfied, the planner is re-invoked with input:
  `Query: ...\nPrevious plan: ...\nCritique issues: ...\nSuggestions: ...\nProduce a revised plan.`
- Same `planner_agent` instance — no new model, just a second call with richer input.

### `research_evaluator_agent`

- Model: `gpt-5.4`, `ModelSettings(reasoning=Reasoning(effort="high"))`.
- Input: original query plus list of `{id: int, query: str, summary: str}`.
- Output (structured):

```python
class ResearchEvaluation(BaseModel):
    is_sufficient: bool
    coverage_gaps: list[str]
    discard_ids: list[int]
    additional_searches: list[WebSearchItem]  # capped downstream to EVAL_MAX_GAP_SEARCHES
```

- Prompt emphasizes: which query sub-questions are not yet answered, which summaries are off-topic / low-signal / should be discarded, and specific follow-up searches to close the gap.

### `config.py`

Single source of truth for tunables, all env-overridable with sensible defaults:

```python
PLAN_CRITIQUE_MAX_REVISIONS = int(os.getenv("PLAN_CRITIQUE_MAX_REVISIONS", "2"))
PLAN_CRITIQUE_SCORE_THRESHOLD = int(os.getenv("PLAN_CRITIQUE_SCORE_THRESHOLD", "8"))
EVAL_MAX_EXTRA_ROUNDS = int(os.getenv("EVAL_MAX_EXTRA_ROUNDS", "2"))
EVAL_MAX_GAP_SEARCHES = int(os.getenv("EVAL_MAX_GAP_SEARCHES", "5"))
ENABLE_PLAN_CRITIQUE = os.getenv("ENABLE_PLAN_CRITIQUE", "1") == "1"
ENABLE_RESEARCH_EVAL = os.getenv("ENABLE_RESEARCH_EVAL", "1") == "1"
```

## Control flow

### `_plan_with_critique(query) -> WebSearchPlan`

1. Call planner → `plan`.
2. If `ENABLE_PLAN_CRITIQUE` is false, return `plan`.
3. For up to `PLAN_CRITIQUE_MAX_REVISIONS` iterations:
   - Call critic with `query` + `plan`.
   - If `critique.is_sufficient` or `critique.score >= PLAN_CRITIQUE_SCORE_THRESHOLD`, break.
   - Re-run planner with `(query, plan, critique)` → updated `plan`.
4. Return final `plan`.
5. Printer states on the `planning` item: `Planning searches…` → `Plan rev 1/2 — critique score 7/10, revising…` → `Plan ready (score 9/10)`.

### `_evaluate_and_fill_gaps(query, initial_results) -> list[str]`

1. Summaries are tagged with `id` starting at 0.
2. If `ENABLE_RESEARCH_EVAL` is false, return summaries unchanged.
3. For up to `EVAL_MAX_EXTRA_ROUNDS` iterations:
   - Call evaluator with `query` + current tagged summaries.
   - Drop entries whose `id` is in `discard_ids`.
   - If `is_sufficient`, break.
   - Truncate `additional_searches` to `EVAL_MAX_GAP_SEARCHES`.
   - Run those searches concurrently (reuse `_search`), append successful results with new ids.
4. Return untagged summary strings.
5. Printer: new `evaluating` item, and the existing `searching` item is reused for gap rounds with labels like `Gap-fill round 1: 3/4 finished`.

### Tracing

- Wrap `_plan_with_critique` body in `custom_span("Plan critique")`.
- Wrap `_evaluate_and_fill_gaps` body in `custom_span("Research evaluation")`.
- All nested within the existing top-level `trace("Research trace", ...)`.

## Error handling

- Any exception from `plan_critic_agent` → log to printer and skip remaining critique iterations; proceed with current plan.
- Any exception from `research_evaluator_agent` → log to printer and proceed with summaries gathered so far.
- Gap-fill searches reuse `_search`, which already swallows per-search failures.
- `ENABLE_*` flags let operators disable either loop entirely.

## Headless / auto mode

No new stdin prompts. All knobs are env vars, so `INTERACTIVE_MODE=auto python main.py` continues to work unchanged.

## Testing

No automated test suite in repo. Manual verification:

- Run `INTERACTIVE_MODE=auto python main.py` with defaults; confirm trace shows plan-critique and research-evaluation spans.
- Run with `ENABLE_PLAN_CRITIQUE=0 ENABLE_RESEARCH_EVAL=0`; confirm behavior is identical to the pre-change pipeline.
- Run with `PLAN_CRITIQUE_SCORE_THRESHOLD=11` to force max revisions; confirm cap is respected.
- Run a query that will obviously be under-covered by the initial plan (e.g., deliberately narrow) and confirm gap-fill runs.

## Risks & mitigations

- **Cost / latency increase.** Mitigated by caps, early-exit thresholds, and disable flags.
- **Evaluator hallucinates gaps indefinitely.** Hard cap on rounds; per-round search cap.
- **Structured-output drift.** Pydantic `output_type=` validation already enforced by the SDK.
