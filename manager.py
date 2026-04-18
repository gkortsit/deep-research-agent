from __future__ import annotations

import asyncio
import time
from typing import TypedDict

from rich.console import Console

from agents import Agent, Runner, custom_span, gen_trace_id, trace

import config
from subagents.planner_agent import (
    WebSearchItem,
    WebSearchPlan,
    build_revision_input,
    planner_agent,
)
from subagents.plan_critic_agent import PlanCritique, plan_critic_agent
from subagents.research_evaluator_agent import (
    ResearchEvaluation,
    research_evaluator_agent,
)
from subagents.search_agent import search_agent
from subagents.writer_agent import ReportData, writer_agent
from printer.main import Printer
from report_writer import save_report


class TaggedResult(TypedDict):
    id: int
    query: str
    summary: str


class ResearchManager:
    def __init__(self):
        self.console = Console()
        self.printer = Printer(self.console)

    async def run(self, query: str) -> None:
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            self.printer.update_item(
                "trace_id",
                f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}",
                is_done=True,
                hide_checkmark=True,
            )

            self.printer.update_item(
                "starting",
                "Starting research...",
                is_done=True,
                hide_checkmark=True,
            )
            search_plan = await self._plan_with_critique(query)
            tagged_results = await self._perform_searches(search_plan.searches)
            tagged_results = await self._evaluate_and_fill_gaps(query, tagged_results)
            search_results = [entry["summary"] for entry in tagged_results]
            report = await self._write_report(query, search_results)

            final_report = f"Report summary\n\n{report.short_summary}"
            self.printer.update_item("final_report", final_report, is_done=True)

            saved_path = save_report(report)
            self.printer.update_item(
                "save", f"Saved report to {saved_path}", is_done=True
            )

            self.printer.end()

        print("\n\n=====REPORT=====\n\n")
        print(f"Report: {report.markdown_report}")
        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        follow_up_questions = "\n".join(report.follow_up_questions)
        print(f"Follow up questions: {follow_up_questions}")

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
                critique_label = (
                    f"Critiquing plan (rev {revision}/{config.PLAN_CRITIQUE_MAX_REVISIONS})"
                )
                self.printer.update_item("planning", f"{critique_label}...")
                try:
                    critique_result = await self._run_with_ticker(
                        plan_critic_agent,
                        self._format_plan_for_critique(query, plan),
                        "planning",
                        critique_label,
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

                revise_label = f"Plan score {critique.score}/10, revising"
                self.printer.update_item("planning", f"{revise_label}...")
                try:
                    revised = await self._run_with_ticker(
                        planner_agent,
                        build_revision_input(
                            query, plan, critique.issues, critique.suggestions
                        ),
                        "planning",
                        revise_label,
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

    async def _run_with_ticker(
        self, agent: Agent, input_text: str, printer_key: str, label: str
    ):
        """Run an agent while ticking elapsed-time updates on a printer item.

        Applies AGENT_CALL_TIMEOUT_S so calls cannot hang forever.
        """
        task = asyncio.create_task(
            asyncio.wait_for(
                Runner.run(agent, input_text), timeout=config.AGENT_CALL_TIMEOUT_S
            )
        )
        started = time.time()
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except asyncio.TimeoutError:
                elapsed = int(time.time() - started)
                self.printer.update_item(printer_key, f"{label} ({elapsed}s)")
        return await task

    def _format_plan_for_critique(self, query: str, plan: WebSearchPlan) -> str:
        lines = [f"Query: {query}", "", "Proposed plan:"]
        for i, item in enumerate(plan.searches, 1):
            lines.append(f"{i}. {item.query} — {item.reason}")
        return "\n".join(lines)

    async def _perform_searches(
        self, items: list[WebSearchItem], id_offset: int = 0, label: str = "searching"
    ) -> list[TaggedResult]:
        with custom_span("Search the web"):
            self.printer.update_item(label, "Searching...")
            num_completed = 0
            num_succeeded = 0
            num_failed = 0
            tasks = [
                asyncio.create_task(self._search_indexed(i + id_offset, item))
                for i, item in enumerate(items)
            ]
            results: list[TaggedResult] = []
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

    async def _search_indexed(self, id_: int, item: WebSearchItem) -> TaggedResult | None:
        summary = await self._search(item)
        if summary is None:
            return None
        return {"id": id_, "query": item.query, "summary": summary}

    async def _search(self, item: WebSearchItem) -> str | None:
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def _evaluate_and_fill_gaps(
        self, query: str, tagged_results: list[TaggedResult]
    ) -> list[TaggedResult]:
        if not config.ENABLE_RESEARCH_EVAL:
            return tagged_results

        with custom_span("Research evaluation"):
            for round_num in range(1, config.EVAL_MAX_EXTRA_ROUNDS + 1):
                eval_label = (
                    f"Evaluating coverage (round {round_num}/{config.EVAL_MAX_EXTRA_ROUNDS})"
                )
                self.printer.update_item("evaluating", f"{eval_label}...")
                try:
                    eval_result = await self._run_with_ticker(
                        research_evaluator_agent,
                        self._format_results_for_evaluation(query, tagged_results),
                        "evaluating",
                        eval_label,
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
        self, query: str, tagged_results: list[TaggedResult]
    ) -> str:
        lines = [f"Original query: {query}", "", "Current research summaries:"]
        for entry in tagged_results:
            lines.append(
                f"[id={entry['id']}] query={entry['query']}\nsummary: {entry['summary']}\n"
            )
        return "\n".join(lines)

    async def _write_report(self, query: str, search_results: list[str]) -> ReportData:
        self.printer.update_item("writing", "Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
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
