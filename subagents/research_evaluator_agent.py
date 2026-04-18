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
