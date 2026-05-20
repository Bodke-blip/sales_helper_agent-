from agents.state import SalesHelperState


PROMPT_ATTACK_PATTERNS = (
    "reveal system prompt",
    "show system prompt",
    "ignore previous instructions",
    "ignore your instructions",
    "hidden chain-of-thought",
    "show chain of thought",
    "show hidden reasoning",
    "developer message",
)

SELF_HARM_ABUSE_PATTERNS = (
    "make my colleague suicidal",
    "make someone suicidal",
    "convince someone to kill",
    "push someone to suicide",
    "drive someone to suicide",
    "encourage suicide",
    "bully someone",
    "harass my colleague",
    "blackmail",
    "threaten my colleague",
)

VIOLENCE_PATTERNS = (
    "hurt my colleague",
    "harm my colleague",
    "attack my colleague",
    "kill my colleague",
    "how to poison",
    "build a weapon",
)

CREDENTIAL_PATTERNS = (
    "show api key",
    "reveal api key",
    "print token",
    "steal credentials",
    "extract password",
)

def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def requests_self_harm_inducement(text: str) -> bool:
    if "suicid" not in text:
        return False

    inducement_terms = (
        "make",
        "cause",
        "push",
        "drive",
        "convince",
        "encourage",
        "pressure",
        "force",
        "bully",
        "harass",
        "manipulate",
    )
    return any(term in text for term in inducement_terms)


def input_guardrail(state: SalesHelperState) -> SalesHelperState:
    query = state.get("user_query", "").strip()

    if not query:
        return {
            **state,
            "input_guardrail_status": "needs_clarification",
            "input_guardrail_reason": "The user query is empty.",
        }

    lowered = query.lower()

    if contains_any(lowered, PROMPT_ATTACK_PATTERNS):
        return {
            **state,
            "input_guardrail_status": "blocked",
            "input_guardrail_reason": "The request attempts to reveal protected instructions or hidden reasoning.",
        }

    if contains_any(lowered, SELF_HARM_ABUSE_PATTERNS) or requests_self_harm_inducement(lowered):
        return {
            **state,
            "input_guardrail_status": "blocked",
            "input_guardrail_reason": "The request asks for help harming, harassing, or inducing self-harm in another person.",
        }

    if contains_any(lowered, VIOLENCE_PATTERNS):
        return {
            **state,
            "input_guardrail_status": "blocked",
            "input_guardrail_reason": "The request asks for violent or physically harmful assistance.",
        }

    if contains_any(lowered, CREDENTIAL_PATTERNS):
        return {
            **state,
            "input_guardrail_status": "blocked",
            "input_guardrail_reason": "The request asks for credential or secret extraction.",
        }

    return {
        **state,
        "input_guardrail_status": "safe",
        "input_guardrail_reason": "Input is safe for processing.",
    }


def output_guardrail(state: SalesHelperState) -> SalesHelperState:
    final_response = state.get("final_response", {})

    if not final_response:
        return {
            **state,
            "eval_status": "needs_revision",
            "fallback_status": "required",
            "fallback_reason": "Final response package is empty.",
        }

    return state
