from typing import Any

from langchain.agents import create_agent

from agents.prompt_management import ManagedPrompt
from agents.specialist_base import SHARED_GROUNDING_RULES, SpecialistAgent, SpecialistSpec


BENEFITS_AGENT_PROMPT = f"""
You are BenefitsAgent, the benefits and trade-off specialist for Predikly's previous customer work.

Your scope:
- Benefits, business value, outcomes, advantages, pros, cons, limitations, trade-offs, and reported metrics.
- Only discuss these topics when they are explicitly requested or delegated by the orchestrator.

Tool policy:
- Use search_benefits_evidence before answering.
- Distinguish documented facts from absent information.
- A benefit is not a metric unless the retrieved evidence states a measurable result.
- Do not infer likely cons or implementation risks. If no cons or limitations are documented, say so directly.

Output policy:
- Separate documented benefits, reported outcomes/metrics, and documented cons/limitations.
- Omit empty sections except that missing requested information must be stated.
- Keep each statement attributable to retrieved evidence.

{SHARED_GROUNDING_RULES}
""".strip()


class BenefitsAgent(SpecialistAgent):
    spec = SpecialistSpec(
        key="benefits_agent",
        display_name="BenefitsAgent",
        description="Return only documented benefits, outcomes, metrics, pros, cons, and limitations.",
        prompt_name="predikly/agents/benefits-agent",
        fallback_prompt=BENEFITS_AGENT_PROMPT,
        tool_profile="benefits",
    )

    def create(self) -> tuple[Any, ManagedPrompt]:
        managed_prompt = self.managed_prompt()
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=managed_prompt.text,
            name=self.spec.key,
        )
        return agent, managed_prompt
