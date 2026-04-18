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
