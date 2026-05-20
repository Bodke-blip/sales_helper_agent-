from agents.state import SalesHelperState
from agents.tools import EVAL_TOOLS


AGENT_NAME = "eval"
TOOLS = EVAL_TOOLS


def eval_agent(state: SalesHelperState) -> SalesHelperState:
    has_sources = bool(state.get("qdrant_sources"))
    has_content = bool(state.get("internal_context"))

    status = "approved" if has_sources and has_content else "needs_revision"
    model = "rule_based_eval"

    if has_sources and has_content:
        notes = "Response has grounded internal context and sources."
    elif has_content:
        notes = "Internal context was found, but source records were missing."
    else:
        notes = "No grounded internal context was found."

    return {
        **state,
        "eval_status": status,
        "eval_notes": notes,
        "eval_llm_model": model,
    }
