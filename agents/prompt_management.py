import os
from dataclasses import dataclass

from agents.tracing import langfuse_client


LANGFUSE_PROMPT_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "production")


def _load_prompt_client():
    if os.getenv("ENABLE_LANGFUSE_PROMPTS", "true").lower() not in {"1", "true", "yes"}:
        return None

    if langfuse_client is not None:
        return langfuse_client

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


prompt_client = _load_prompt_client()


@dataclass(frozen=True)
class ManagedPrompt:
    text: str
    name: str
    version: int | None
    source: str


def load_managed_prompt(name: str, fallback: str) -> ManagedPrompt:
    """Load a production prompt from Langfuse without making it a runtime dependency."""
    if prompt_client is None:
        return ManagedPrompt(fallback, name, None, "local_fallback")

    try:
        prompt = prompt_client.get_prompt(
            name,
            label=LANGFUSE_PROMPT_LABEL,
            fallback=fallback,
            max_retries=1,
            fetch_timeout_seconds=3,
        )
        compiled = prompt.compile()

        if not isinstance(compiled, str) or not compiled.strip():
            raise ValueError("Langfuse returned an empty or non-text prompt.")

        return ManagedPrompt(
            text=compiled.strip(),
            name=name,
            version=None if getattr(prompt, "is_fallback", False) else getattr(prompt, "version", None),
            source="local_fallback" if getattr(prompt, "is_fallback", False) else "langfuse",
        )
    except Exception:
        return ManagedPrompt(fallback, name, None, "local_fallback")
