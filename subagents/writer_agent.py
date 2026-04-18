# Agent used to synthesize a final report from the individual summaries.
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

from agents import Agent, ModelSettings

PROMPT = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research "
    "assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)


class ReportData(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary of the findings."""

    markdown_report: str
    """The final report"""

    follow_up_questions: list[str]
    """Suggested topics to research further"""


writer_agent = Agent(
    name="WriterAgent",
    instructions=PROMPT,
    model="gpt-5-mini",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=ReportData,
)


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
