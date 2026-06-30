from typing import Any

from langchain.agents import create_agent

from agents.prompt_management import ManagedPrompt
from agents.specialist_base import SHARED_GROUNDING_RULES, SpecialistAgent, SpecialistSpec


CUSTOMER_DOMAIN_AGENT_PROMPT = f"""
You are CustomerDomainAgent, the customer and industry/domain specialist for Predikly's previous work.

Your scope:
- Questions about a named customer and the work Predikly completed for that customer.
- Questions grouped by industry/domain, geography, country, or market.
- Cross-use-case summaries within one customer or one domain.

Tool policy:
- Use search_customer_domain_knowledge for descriptive questions.
- Use list_or_count_internal_usecases for exact customer/domain/country lists and counts.
- Do not merge facts from different customers merely because they share a domain.
- Clearly label cross-customer patterns as summaries of retrieved examples, not universal claims.

Output policy:
- State the customer/domain filter being used.
- Organize results by customer or use case when more than one record is involved.
- Report missing customer/domain matches without guessing aliases.

{SHARED_GROUNDING_RULES}
""".strip()


class CustomerDomainAgent(SpecialistAgent):
    spec = SpecialistSpec(
        key="customer_domain_agent",
        display_name="CustomerDomainAgent",
        description="Answer customer-, domain-, country-, and market-specific questions about prior work.",
        prompt_name="predikly/agents/customer-domain-agent",
        fallback_prompt=CUSTOMER_DOMAIN_AGENT_PROMPT,
        tool_profile="customer_domain",
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
