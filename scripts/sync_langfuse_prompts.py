"""Create a new Langfuse version for each local agent prompt.

Run this explicitly when bootstrapping or intentionally publishing local prompt changes.
"""

from agents.prompt_management import LANGFUSE_PROMPT_LABEL, prompt_client
from agents.main_orchestrator_agent import MAIN_ORCHESTRATOR_PROMPT, ORCHESTRATOR_PROMPT_NAME
from agents.specialist_agents import SPECIALIST_SPECS


def main() -> None:
    if prompt_client is None:
        raise RuntimeError("Langfuse prompt credentials are not configured.")

    prompts = {
        ORCHESTRATOR_PROMPT_NAME: MAIN_ORCHESTRATOR_PROMPT,
        **{
            spec.prompt_name: spec.fallback_prompt
            for spec in SPECIALIST_SPECS.values()
        },
    }

    for name, prompt in prompts.items():
        created = prompt_client.create_prompt(
            name=name,
            type="text",
            prompt=prompt,
            labels=[LANGFUSE_PROMPT_LABEL],
        )
        print(f"Published {name} version {created.version}")


if __name__ == "__main__":
    main()
