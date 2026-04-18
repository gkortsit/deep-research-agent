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
