"""Registry for separately defined specialist agent classes."""

from typing import Any, Callable

from agents.benefits_agent import BenefitsAgent
from agents.customer_domain_agent import CustomerDomainAgent
from agents.list_of_agent import ListOfAgent
from agents.prompt_management import ManagedPrompt
from agents.specialist_base import SpecialistAgent, SpecialistSpec
from agents.usecase_agent import UseCaseAgent


SPECIALIST_AGENT_CLASSES = {
    ListOfAgent.spec.key: ListOfAgent,
    BenefitsAgent.spec.key: BenefitsAgent,
    UseCaseAgent.spec.key: UseCaseAgent,
    CustomerDomainAgent.spec.key: CustomerDomainAgent,
}

SPECIALIST_SPECS = {
    key: agent_class.spec
    for key, agent_class in SPECIALIST_AGENT_CLASSES.items()
}


def create_specialist_agent(
    *,
    model: Any,
    spec: SpecialistSpec,
    tools: list[Callable],
) -> tuple[Any, ManagedPrompt]:
    agent_class = SPECIALIST_AGENT_CLASSES.get(spec.key)

    if agent_class is None:
        raise ValueError(f"No agent class is registered for specialist '{spec.key}'.")

    return agent_class(model=model, tools=tools).create()


__all__ = [
    "BenefitsAgent",
    "CustomerDomainAgent",
    "ListOfAgent",
    "SPECIALIST_AGENT_CLASSES",
    "SPECIALIST_SPECS",
    "SpecialistAgent",
    "SpecialistSpec",
    "UseCaseAgent",
    "create_specialist_agent",
]
