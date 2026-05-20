from agents.state import SalesHelperState


def fallback_handler(state: SalesHelperState) -> SalesHelperState:
    if state.get("input_guardrail_status") == "needs_clarification":
        return {
            **state,
            "fallback_status": "clarification_required",
            "final_response": {
                "message": "Please provide more detail so I can help with the sales request.",
                "sources": [],
                "evaluation": state.get("evaluations", []),
                "fallback_status": "clarification_required",
                "trace_id": state.get("trace_id"),
                "workflow_timings": state.get("workflow_timings", []),
            },
        }

    if state.get("input_guardrail_status") == "blocked":
        return {
            **state,
            "fallback_status": "blocked",
            "final_response": {
                "message": (
                    "I cannot help with that request. "
                    f"Reason: {state.get('input_guardrail_reason', 'The request was blocked by safety guardrails.')}"
                ),
                "sources": [],
                "evaluation": state.get("evaluations", []),
                "fallback_status": "blocked",
                "trace_id": state.get("trace_id"),
                "workflow_timings": state.get("workflow_timings", []),
            },
        }

    return {
        **state,
        "fallback_status": "insufficient_context",
        "fallback_reason": "One or more agent evaluations failed.",
        "final_response": {
            "message": "I could not find enough grounded internal context to answer reliably.",
            "sources": state.get("qdrant_sources", []),
            "evaluation": state.get("evaluations", []),
            "fallback_status": "insufficient_context",
            "trace_id": state.get("trace_id"),
            "workflow_timings": state.get("workflow_timings", []),
        },
    }
