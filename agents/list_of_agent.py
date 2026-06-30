from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Callable

from langchain.agents import create_agent

from agents.prompt_management import ManagedPrompt
from agents.specialist_base import SHARED_GROUNDING_RULES, SpecialistAgent, SpecialistSpec


ALL_PREDIKLY_USECASES_PATH = (
    Path(__file__).resolve().parent / "catalogs" / "predikly_usecases.txt"
)


@lru_cache(maxsize=1)
def load_all_predikly_usecases() -> tuple[tuple[str, str], ...]:
    entries = []

    for line_number, raw_line in enumerate(
        ALL_PREDIKLY_USECASES_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        usecase_name, separator, company_or_source = line.partition("|")

        if not separator or not usecase_name.strip() or not company_or_source.strip():
            raise ValueError(
                f"Invalid use-case catalog entry at line {line_number}: {raw_line}"
            )

        entries.append((usecase_name.strip(), company_or_source.strip()))

    if not entries:
        raise ValueError(f"Use-case catalog is empty: {ALL_PREDIKLY_USECASES_PATH}")

    return tuple(entries)


def is_all_predikly_usecases_request(query: str) -> bool:
    """Match only explicit requests for the complete unfiltered use-case catalog."""
    normalized = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
    exact_requests = {
        "list all use cases",
        "list all the use cases",
        "show all use cases",
        "show all the use cases",
        "list every use case",
        "show every use case",
    }

    if normalized in exact_requests:
        return True

    patterns = (
        r"\b(?:list|show|give me)\s+(?:all|every)\s+(?:the\s+)?use\s*cases?\s+ever\s+uploaded(?:\s+(?:about|for)\s+predikly)?\b",
        r"\b(?:all|every)\s+(?:the\s+)?use\s*cases?\s+(?:that\s+)?predikly\s+has\s+(?:ever\s+)?(?:done|worked on|uploaded)\b",
        r"\bcomplete\s+list\s+of\s+(?:all\s+)?predikly\s+use\s*cases?\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def build_all_predikly_usecases_answer() -> str:
    usecases = load_all_predikly_usecases()
    lines = [
        f"Predikly's complete uploaded use-case snapshot contains {len(usecases)} entries:",
        "",
    ]
    lines.extend(
        f"{index}. {usecase_name} — {company_or_source}"
        for index, (usecase_name, company_or_source) in enumerate(usecases, start=1)
    )
    return "\n".join(lines)


LIST_AGENT_PROMPT = f"""
You are ListOfAgent, the listing and catalog specialist for Predikly's previous customer work.

Your scope:
- Lists, names, examples, catalogs, and counts of use cases, customers, domains, countries, tools, benefits, or other fields explicitly requested by the user.
- Filtered lists such as use cases for a company, domain, country, or market.

Tool policy:
- For an explicit request for all/every use case ever uploaded about Predikly, return the fixed complete snapshot exactly as supplied by the deterministic ListOfAgent path.
- Use list_or_count_internal_usecases for exact use-case lists and counts.
- Use search_internal_knowledge when the requested list is not represented by the catalog tool, such as tools, benefits, workflows, or outcomes.
- Preserve exact names and counts returned by tools. Deduplicate only clearly identical records.
- Do not turn a list request into an unsolicited deep explanation.

Output policy:
- Give the requested list or count first.
- Mention applied filters and missing requested fields.
- Do not include benefits, recommendations, or analysis unless the user explicitly asked for them.

{SHARED_GROUNDING_RULES}
""".strip()


class ListOfAgent(SpecialistAgent):
    spec = SpecialistSpec(
        key="list_of_agent",
        display_name="ListOfAgent",
        description="List or count internal use cases and return requested catalogs or enumerations.",
        prompt_name="predikly/agents/list-of-agent",
        fallback_prompt=LIST_AGENT_PROMPT,
        tool_profile="list",
    )

    @staticmethod
    def matches_complete_catalog_request(query: str) -> bool:
        return is_all_predikly_usecases_request(query)

    @staticmethod
    def complete_catalog_answer() -> str:
        return build_all_predikly_usecases_answer()

    def create(self) -> tuple[Any, ManagedPrompt]:
        managed_prompt = self.managed_prompt()
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=managed_prompt.text,
            name=self.spec.key,
        )
        return agent, managed_prompt
