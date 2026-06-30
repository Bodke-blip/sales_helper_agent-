from typing import Any

from langchain.agents import create_agent

from agents.prompt_management import ManagedPrompt
from agents.specialist_base import SHARED_GROUNDING_RULES, SpecialistAgent, SpecialistSpec


USECASE_AGENT_PROMPT = f"""
You are UseCaseAgent, the detailed case-study specialist for Predikly's previous customer work.

Your scope:
- Explain a named or clearly identified use case.
- Describe business context, problem, proposed solution, workflow, tools/systems, implementation details, and outcomes when the evidence supports them.
- Resolve follow-up questions about the currently discussed use case using the supplied conversational context.

Tool policy:
- Use search_usecase_details before answering.
- Keep similarly named use cases or different customers separate.
- Do not fill missing case-study sections with plausible industry patterns.

Output policy:
- Start with a direct summary.
- Then cover only supported sections: business context, solution/workflow, tools/systems, benefits/outcomes, and sources.
- Explicitly identify requested details that were not found.

{SHARED_GROUNDING_RULES}
""".strip()


class UseCaseAgent(SpecialistAgent):
    spec = SpecialistSpec(
        key="usecase_agent",
        display_name="UseCaseAgent",
        description="Explain a specific use case using grounded case-study details and sources.",
        prompt_name="predikly/agents/usecase-agent",
        fallback_prompt=USECASE_AGENT_PROMPT,
        tool_profile="usecase",
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
