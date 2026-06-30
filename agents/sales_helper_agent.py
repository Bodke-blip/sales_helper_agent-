import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.evaluation import evaluate_eval, evaluate_knowledge_retrieval, has_failed_evaluation
from agents.eval_agent import eval_agent
from agents.knowledge_retrieval_agent import knowledge_retrieval_agent
from agents.llm import (
    LLMGatewayError,
    PRIMARY_LLM_MODEL,
    SECONDARY_LLM_MODEL,
    gemini_enabled,
    get_llm_provider_status,
    get_primary_llm,
    get_secondary_llm,
    invoke_llm,
)
from agents.main_orchestrator_agent import (
    MAIN_ORCHESTRATOR_PROMPT,
    MainOrchestratorAgent,
    ORCHESTRATOR_PROMPT_NAME,
)
from agents.response_helpers import (
    build_capabilities_answer,
    build_grounded_fallback_answer,
    sanitize_runtime_error,
    strip_markdown_markers,
)
from agents.specialist_agents import (
    SPECIALIST_SPECS,
    ListOfAgent,
    SpecialistSpec,
    create_specialist_agent,
)
from agents.state import SalesHelperState
from agents.tracing import agent_observation, sanitize_for_trace, trace_event


SALES_HELPER_SYSTEM_PROMPT = MAIN_ORCHESTRATOR_PROMPT
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "10"))


def compact_json(
    data: Any,
    *,
    max_text_length: int = 700,
    max_list_items: int = 6,
) -> str:
    def compact_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: compact_value(item) for key, item in value.items()}

        if isinstance(value, list):
            return [compact_value(item) for item in value[:max_list_items]]

        if isinstance(value, str):
            cleaned = re.sub(r"\s+", " ", value).strip()
            return cleaned[:max_text_length]

        return value

    return json.dumps(compact_value(data), ensure_ascii=False)


def compact_context_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer_name": item.get("customer_name") or item.get("company_name") or "",
        "usecase_name": item.get("usecase_name") or item.get("use_case_name") or "",
        "customer_domain": item.get("customer_domain", ""),
        "ppt_name": item.get("ppt_name", ""),
        "slide_number": item.get("slide_number"),
        "text": item.get("text", ""),
    }


def compact_source_item(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer_name": source.get("customer_name", ""),
        "usecase_name": source.get("usecase_name", ""),
        "ppt_name": source.get("ppt_name", ""),
        "slide_number": source.get("slide_number"),
    }


def compact_chat_history_for_agent(history: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, str]]:
    compact_history = []

    for message in history[-limit:]:
        role = str(message.get("role") or "")
        content = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()

        if not content:
            continue

        max_length = 500 if role == "assistant" else 300
        compact_history.append(
            {
                "role": role,
                "content": content[:max_length],
            }
        )

    return compact_history


def build_tool_response(tool_state: SalesHelperState, *, max_items: int = 6) -> dict[str, Any]:
    return {
        "internal_context": [
            compact_context_item(item)
            for item in tool_state.get("internal_context", [])[:max_items]
        ],
        "sources": [
            compact_source_item(source)
            for source in tool_state.get("qdrant_sources", [])[:max_items]
        ],
        "retrieval_collection": tool_state.get("retrieval_collection", ""),
        "retrieval_cache_status": tool_state.get("retrieval_cache_status", "not_used"),
        "retrieval_error": tool_state.get("retrieval_error", ""),
        "eval_status": tool_state.get("eval_status", ""),
        "eval_notes": tool_state.get("eval_notes", ""),
    }


def run_retrieval_flow(
    base_state: SalesHelperState,
    *,
    tool_name: str,
    tool_input: str,
) -> SalesHelperState:
    retrieval_state: SalesHelperState = {
        **base_state,
        "orchestrator_tool": tool_name,
        "orchestrator_tool_input": tool_input,
        "orchestrator_reason": "The create_agent sales helper selected this callable tool.",
        "selected_agents": ["knowledge_retrieval", "eval"],
    }
    retrieval_state = knowledge_retrieval_agent(retrieval_state)
    retrieval_state = evaluate_knowledge_retrieval(retrieval_state)

    if not has_failed_evaluation(retrieval_state):
        retrieval_state = eval_agent(retrieval_state)
        retrieval_state = evaluate_eval(retrieval_state)

    return retrieval_state


def extract_agent_answer(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get("messages", [])

    for message in reversed(messages):
        content = getattr(message, "content", "")

        if not content:
            continue

        if isinstance(content, list):
            parts = [
                str(part.get("text") if isinstance(part, dict) else part)
                for part in content
            ]
            return "\n".join(part for part in parts if part).strip()

        return str(content).strip()

    return ""


def build_final_response(
    state: SalesHelperState,
    *,
    answer: str,
    answer_model: str,
    fallback_status: str = "not_used",
    answer_composer_error: str = "",
) -> SalesHelperState:
    answer = strip_markdown_markers(answer)

    if not answer:
        answer = "I could not find enough grounded internal context to answer reliably."
        answer_model = answer_model or "unavailable"

    return {
        **state,
        "answer_composer_error": answer_composer_error,
        "fallback_status": fallback_status,
        "final_response": {
            "answer": answer,
            "reasoning_summary": "The create_agent sales helper selected callable tools and composed the final answer from allowed outputs.",
            "sources": [*state.get("qdrant_sources", [])],
            "evaluation": state.get("evaluations", []),
            "llm_models": {
                "orchestrator": state.get("orchestrator_llm_model"),
                "eval": state.get("eval_llm_model"),
                "answer_composer": answer_model,
            },
            "fallback_status": fallback_status,
            "orchestrator_error": state.get("orchestrator_error", ""),
            "answer_composer_error": answer_composer_error,
            "trace_id": state.get("trace_id"),
            "workflow_timings": state.get("workflow_timings", []),
        },
    }


def _specialist_retrieval_state(
    base_state: SalesHelperState,
    collector: dict[str, SalesHelperState],
    *,
    tool_name: str,
    tool_input: str,
) -> SalesHelperState:
    current_state = collector.get("state", base_state)
    trace_event(
        current_state,
        name=f"tool.{tool_name}.start",
        input_data={"tool_input": tool_input},
        output_data={"status": "started"},
        metadata={
            "node_type": "knowledge_retriever",
            "tool_name": tool_name,
            "observation_type": "retriever",
        },
    )
    retrieval_base = {
        **current_state,
        "contextual_query": tool_input if tool_name == "hybrid_retrieval" else current_state.get("contextual_query", ""),
    }
    tool_state = run_retrieval_flow(
        retrieval_base,
        tool_name=tool_name,
        tool_input=tool_input,
    )
    collector["state"] = tool_state
    trace_event(
        tool_state,
        name=f"tool.{tool_name}.complete",
        input_data={"tool_input": tool_input},
        output_data={
            "status": "completed" if not tool_state.get("retrieval_error") else "completed_with_error",
            "retrieval_collection": tool_state.get("retrieval_collection", ""),
            "context_count": len(tool_state.get("internal_context", [])),
            "sources_count": len(tool_state.get("qdrant_sources", [])),
            "eval_status": tool_state.get("eval_status", ""),
        },
        metadata={
            "node_type": "knowledge_retriever",
            "tool_name": tool_name,
            "observation_type": "retriever",
        },
    )
    return tool_state


LIST_INTERNAL_KNOWLEDGE_DEFAULT_TOP_K = 6
LIST_INTERNAL_KNOWLEDGE_MAX_TOP_K = int(os.getenv("LIST_INTERNAL_KNOWLEDGE_MAX_TOP_K", "20"))


def infer_list_internal_knowledge_top_k(query: str) -> int:
    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    top_k = LIST_INTERNAL_KNOWLEDGE_DEFAULT_TOP_K

    if not normalized_query:
        return top_k

    numeric_match = re.search(
        r"\b(?:top|first|next|show|list|give me|display)\s+(\d+)\b",
        normalized_query,
    )
    if not numeric_match:
        numeric_match = re.search(
            r"\b(\d+)\s+(?:items?|entries?|examples?|results?|records?|use\s*cases?|customers?|tools?|benefits?|workflows?|outcomes?)\b",
            normalized_query,
        )

    if numeric_match:
        return max(
            LIST_INTERNAL_KNOWLEDGE_DEFAULT_TOP_K,
            min(LIST_INTERNAL_KNOWLEDGE_MAX_TOP_K, int(numeric_match.group(1))),
        )

    if re.search(r"\b(all|every|complete|full|entire|everything)\b", normalized_query):
        return LIST_INTERNAL_KNOWLEDGE_MAX_TOP_K

    list_signals = sum(
        1
        for pattern in (
            r"\blist\b",
            r"\bshow\b",
            r"\bgive me\b",
            r"\bdisplay\b",
            r"\benumerate\b",
        )
        if re.search(pattern, normalized_query)
    )
    connector_signals = normalized_query.count(",") + normalized_query.count(" and ") + normalized_query.count(" or ")

    top_k += min(8, list_signals * 2 + connector_signals)

    return max(
        LIST_INTERNAL_KNOWLEDGE_DEFAULT_TOP_K,
        min(LIST_INTERNAL_KNOWLEDGE_MAX_TOP_K, top_k),
    )


def build_specialist_tools(
    base_state: SalesHelperState,
    spec: SpecialistSpec,
    collector: dict[str, SalesHelperState],
) -> list[Any]:
    @tool
    def search_internal_knowledge(query: str) -> str:
        """Search Predikly's internal Qdrant knowledge and return grounded context with sources.

        Args:
            query: Complete standalone search query for the delegated request.
        """
        tool_state = _specialist_retrieval_state(
            base_state,
            collector,
            tool_name="hybrid_retrieval",
            tool_input=query,
        )
        return compact_json(build_tool_response(tool_state))

    @tool
    def search_list_internal_knowledge(query: str) -> str:
        """Search Predikly's internal knowledge for list-style requests with a dynamic result window.

        Args:
            query: Complete standalone list-style query for the delegated request.
        """
        top_k = infer_list_internal_knowledge_top_k(query)
        tool_state = _specialist_retrieval_state(
            base_state,
            collector,
            tool_name="list_internal_knowledge",
            tool_input=json.dumps(
                {
                    "query": query,
                    "top_k": top_k,
                }
            ),
        )
        return compact_json(
            build_tool_response(tool_state, max_items=top_k),
            max_list_items=top_k,
        )

    @tool
    def list_or_count_internal_usecases(
        action: str = "list",
        company: str = "",
        domain: str = "",
        country: str = "",
        request: str = "",
        top_k: int | None = None,
    ) -> str:
        """List or count Qdrant use cases with optional company, domain, and country filters.

        Args:
            action: Either list or count.
            company: Exact or partial customer/company filter.
            domain: Industry or business-domain filter.
            country: Country or market filter.
            request: Full list/count request so the tool can size result windows dynamically.
            top_k: Optional explicit result window for list requests.
        """
        request_text = request.strip() or " ".join(
            value for value in (action, company, domain, country) if value
        )
        resolved_top_k = (
            max(
                LIST_INTERNAL_KNOWLEDGE_DEFAULT_TOP_K,
                min(LIST_INTERNAL_KNOWLEDGE_MAX_TOP_K, int(top_k)),
            )
            if top_k is not None
            else infer_list_internal_knowledge_top_k(request_text)
        )
        request = {
            "action": "count" if str(action).lower() == "count" else "list",
            "company": company,
            "domain": domain,
            "country": country,
            "top_k": resolved_top_k,
        }
        tool_state = _specialist_retrieval_state(
            base_state,
            collector,
            tool_name="usecase_catalog",
            tool_input=json.dumps(request),
        )
        return compact_json(
            build_tool_response(tool_state, max_items=resolved_top_k),
            max_list_items=resolved_top_k,
        )

    @tool
    def search_benefits_evidence(query: str) -> str:
        """Retrieve evidence for documented benefits, outcomes, metrics, pros, cons, or limitations.

        Args:
            query: Complete standalone benefits question including the target use case or customer.
        """
        tool_state = _specialist_retrieval_state(
            base_state,
            collector,
            tool_name="hybrid_retrieval",
            tool_input=query,
        )
        return compact_json(build_tool_response(tool_state))

    @tool
    def search_usecase_details(query: str) -> str:
        """Retrieve grounded business context, solution, workflow, tools, outcomes, and sources for a use case.

        Args:
            query: Complete standalone query naming the use case and requested details.
        """
        tool_state = _specialist_retrieval_state(
            base_state,
            collector,
            tool_name="hybrid_retrieval",
            tool_input=query,
        )
        return compact_json(build_tool_response(tool_state))

    @tool
    def search_customer_domain_knowledge(query: str) -> str:
        """Retrieve grounded prior work for a customer, industry/domain, country, geography, or market.

        Args:
            query: Complete standalone customer/domain question with all known filters.
        """
        tool_state = _specialist_retrieval_state(
            base_state,
            collector,
            tool_name="hybrid_retrieval",
            tool_input=query,
        )
        return compact_json(build_tool_response(tool_state))

    tools = [search_internal_knowledge]

    if spec.tool_profile == "list":
        tools = [search_list_internal_knowledge, list_or_count_internal_usecases]
    elif spec.tool_profile == "benefits":
        tools.append(search_benefits_evidence)
    elif spec.tool_profile == "usecase":
        tools.append(search_usecase_details)
    elif spec.tool_profile == "customer_domain":
        tools.extend([search_customer_domain_knowledge, list_or_count_internal_usecases])

    return tools


def invoke_specialist_agent(
    *,
    base_state: SalesHelperState,
    spec: SpecialistSpec,
    model: Any,
    model_name: str,
    request: str,
) -> tuple[SalesHelperState, str, dict[str, Any]]:
    user_query = base_state.get("user_query", "")

    if spec.key == "list_of_agent" and (
        ListOfAgent.matches_complete_catalog_request(request)
        or ListOfAgent.matches_complete_catalog_request(user_query)
    ):
        answer = ListOfAgent.complete_catalog_answer()
        agent_run = {
            "agent": spec.key,
            "display_name": spec.display_name,
            "status": "hardcoded_catalog",
            "model": "deterministic",
            "prompt_name": spec.prompt_name,
            "prompt_version": None,
            "prompt_source": "class_level_hardcoded_catalog",
            "retrieval_collection": "hardcoded_predikly_usecase_catalog",
            "sources_count": 1,
        }
        hardcoded_state: SalesHelperState = {
            **base_state,
            "hardcoded_all_usecases_answer": answer,
            "internal_context": [
                {
                    "rank": 1,
                    "score": 1.0,
                    "text": answer,
                    "content_type": "hardcoded_usecase_catalog",
                }
            ],
            "qdrant_sources": [
                {
                    "collection": "hardcoded_predikly_usecase_catalog",
                    "customer_name": "Predikly",
                    "usecase_name": "Complete uploaded use-case snapshot",
                    "ppt_name": "Class-level hard-coded catalog",
                    "score": 1.0,
                }
            ],
            "retrieval_collection": "hardcoded_predikly_usecase_catalog",
            "retrieval_cache_status": "hardcoded_catalog",
            "selected_agents": list(dict.fromkeys([
                *base_state.get("selected_agents", []),
                spec.key,
            ])),
            "agent_runs": [*base_state.get("agent_runs", []), agent_run],
        }
        trace_event(
            hardcoded_state,
            name=spec.display_name,
            input_data={"delegated_request": request},
            output_data=agent_run,
            metadata={
                "node_type": "specialist_agent",
                "agent_name": spec.display_name,
                "observation_type": "agent",
            },
        )
        return (
            hardcoded_state,
            compact_json(
                {
                    "agent": spec.display_name,
                    "status": "hardcoded_catalog",
                    "answer": "The complete hard-coded use-case snapshot is loaded and will be returned verbatim.",
                }
            ),
            agent_run,
        )

    retrieval_collector: dict[str, SalesHelperState] = {}
    specialist_tools = build_specialist_tools(base_state, spec, retrieval_collector)
    specialist, managed_prompt = create_specialist_agent(
        model=model,
        spec=spec,
        tools=specialist_tools,
    )
    compact_history = compact_chat_history_for_agent(base_state.get("chat_history", []))
    specialist_input = (
        f"Delegated request from MainOrchestratorAgent:\n{request}\n\n"
        f"Current user query:\n{base_state.get('user_query', '')}\n\n"
        f"Relevant recent chat context:\n{compact_history}\n\n"
        "Stay within your assigned scope, retrieve evidence, and return a grounded specialist response."
    )

    with agent_observation(
        base_state,
        name=spec.display_name,
        input_data={"delegated_request": request},
        metadata={
            "node_type": "specialist_agent",
            "agent_name": spec.display_name,
            "prompt_name": managed_prompt.name,
            "prompt_version": managed_prompt.version,
            "prompt_source": managed_prompt.source,
            "model": model_name,
        },
        model=model_name,
    ) as observation:
        result = specialist.invoke(
            {"messages": [HumanMessage(content=specialist_input)]},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
        answer = extract_agent_answer(result)
        specialist_state = retrieval_collector.get("state", base_state)
        grounded = bool(
            specialist_state.get("internal_context")
            or specialist_state.get("qdrant_sources")
        )
        agent_run = {
            "agent": spec.key,
            "display_name": spec.display_name,
            "status": "grounded" if grounded else "no_grounded_context",
            "model": model_name,
            "prompt_name": managed_prompt.name,
            "prompt_version": managed_prompt.version,
            "prompt_source": managed_prompt.source,
            "retrieval_collection": specialist_state.get("retrieval_collection", ""),
            "sources_count": len(specialist_state.get("qdrant_sources", [])),
        }

        if observation is not None:
            observation.update(output=sanitize_for_trace(agent_run))
    payload = {
        "agent": spec.display_name,
        "status": agent_run["status"],
        "answer": answer if grounded else "No grounded internal context was found for this delegated request.",
        "sources": [
            compact_source_item(source)
            for source in specialist_state.get("qdrant_sources", [])[:6]
        ],
        "missing_information": [] if grounded else ["No relevant internal context was retrieved."],
    }
    selected_agents = list(dict.fromkeys([
        *base_state.get("selected_agents", []),
        spec.key,
        *specialist_state.get("selected_agents", []),
    ]))
    specialist_state = {
        **specialist_state,
        "selected_agents": selected_agents,
        "agent_runs": [*base_state.get("agent_runs", []), agent_run],
    }
    return specialist_state, compact_json(payload, max_text_length=2500), agent_run


def deterministic_retrieval_fallback(state: SalesHelperState, error: Exception | None = None) -> SalesHelperState:
    fallback_state = run_retrieval_flow(
        state,
        tool_name="hybrid_retrieval",
        tool_input=state.get("contextual_query") or state.get("user_query", ""),
    )
    error_text = sanitize_runtime_error(error) if error else ""
    compact_context = compact_json(build_tool_response(fallback_state))

    try:
        answer, answer_model = invoke_llm(
            system_prompt=(
                "You are the final answer agent for Predikly Sales Helper. "
                "The tool-calling agent was unavailable, but retrieval has already completed. "
                "Answer the user using only the provided retrieved internal context and sources. "
                "Match the user's requested format. For detailed use-case questions, include business context, "
                "solution/workflow, tools/systems used, benefits/outcomes, and source grounding where available. "
                "Do not invent facts outside the retrieved context."
            ),
            user_prompt=(
                f"User query:\n{state.get('user_query', '')}\n\n"
                f"Retrieval query:\n{state.get('contextual_query') or state.get('user_query', '')}\n\n"
                f"Retrieved internal context and sources:\n{compact_context}"
            ),
        )
        fallback_status = "agent_tool_call_unavailable"
        answer_composer_error = error_text
    except Exception as composer_error:
        answer = build_grounded_fallback_answer(fallback_state)
        answer_model = "grounded_fallback_answer"
        fallback_status = fallback_state.get("fallback_status") or "llm_unavailable"
        answer_composer_error = sanitize_runtime_error(composer_error) if composer_error else error_text

    return build_final_response(
        {
            **fallback_state,
            "orchestrator_error": error_text,
            "orchestrator_llm_model": "create_agent_unavailable",
        },
        answer=answer,
        answer_model=answer_model,
        fallback_status=fallback_status,
        answer_composer_error=answer_composer_error,
    )


def sales_helper_agent_node(state: SalesHelperState) -> SalesHelperState:
    model = get_primary_llm() if gemini_enabled() else None

    if model is None:
        return deterministic_retrieval_fallback(
            state,
            LLMGatewayError("Gemini is not available. Configure GEMINI_API_KEY."),
        )

    compact_history = compact_chat_history_for_agent(state.get("chat_history", []))
    prompt = (
        f"Recent chat history for reference only:\n{compact_history}\n\n"
        f"Current user query:\n{state.get('user_query', '')}\n\n"
        f"Retrieval query candidate:\n{state.get('contextual_query') or state.get('user_query', '')}\n\n"
        "Use the available tools according to the system instructions, then provide the final answer."
    )

    agent_result = None
    agent_model_name = PRIMARY_LLM_MODEL
    agent_errors = []
    completed_state: SalesHelperState = state
    completed_tools_called: list[str] = []
    completed_prompt = None

    for candidate_model, candidate_name in (
        (model, PRIMARY_LLM_MODEL),
        (get_secondary_llm() if gemini_enabled() else None, SECONDARY_LLM_MODEL),
    ):
        if candidate_model is None:
            continue

        state_holder: dict[str, SalesHelperState] = {"state": state}
        tools_called: list[str] = []

        def delegate_to_specialist(spec_key: str, request: str) -> str:
            current_state = state_holder["state"]
            isolated_state: SalesHelperState = {
                **current_state,
                "internal_context": [],
                "qdrant_sources": [],
                "evaluations": [],
            }
            specialist_state, response, _ = invoke_specialist_agent(
                base_state=isolated_state,
                spec=SPECIALIST_SPECS[spec_key],
                model=candidate_model,
                model_name=candidate_name,
                request=request,
            )
            state_holder["state"] = {
                **specialist_state,
                "internal_context": [
                    *current_state.get("internal_context", []),
                    *specialist_state.get("internal_context", []),
                ],
                "qdrant_sources": [
                    *current_state.get("qdrant_sources", []),
                    *specialist_state.get("qdrant_sources", []),
                ],
                "evaluations": [
                    *current_state.get("evaluations", []),
                    *specialist_state.get("evaluations", []),
                ],
                "selected_agents": list(dict.fromkeys([
                    *current_state.get("selected_agents", []),
                    *specialist_state.get("selected_agents", []),
                ])),
            }
            tools_called.append(spec_key)
            return response

        @tool
        def ask_list_of_agent(request: str) -> str:
            """Delegate list, catalog, enumeration, example, and count requests to ListOfAgent.

            Args:
                request: Complete standalone list/count instruction for the specialist, including how many results to show when relevant.
            """
            return delegate_to_specialist("list_of_agent", request)

        @tool
        def ask_benefits_agent(request: str) -> str:
            """Delegate requested benefits, outcomes, metrics, pros, cons, limitations, and trade-offs.

            Args:
                request: Complete standalone benefits instruction for the specialist.
            """
            return delegate_to_specialist("benefits_agent", request)

        @tool
        def ask_usecase_agent(request: str) -> str:
            """Delegate detailed use-case, problem, solution, workflow, implementation, and tools questions.

            Args:
                request: Complete standalone use-case instruction for the specialist.
            """
            return delegate_to_specialist("usecase_agent", request)

        @tool
        def ask_customer_domain_agent(request: str) -> str:
            """Delegate customer-, industry-, domain-, country-, geography-, and market-specific questions.

            Args:
                request: Complete standalone customer/domain instruction for the specialist.
            """
            return delegate_to_specialist("customer_domain_agent", request)

        @tool
        def search_internal_knowledge(query: str) -> str:
            """Grounded fallback retrieval for factual requests that do not fit any specialist.

            Args:
                query: Complete standalone internal-knowledge query.
            """
            current_state = state_holder["state"]
            retrieval_state = run_retrieval_flow(
                {**current_state, "contextual_query": query},
                tool_name="hybrid_retrieval",
                tool_input=query,
            )
            state_holder["state"] = retrieval_state
            tools_called.append("hybrid_retrieval")
            return compact_json(build_tool_response(retrieval_state))

        @tool
        def explain_capabilities() -> str:
            """Explain what the Predikly Sales Helper can do and its grounded retrieval limits."""
            tools_called.append("explain_capabilities")
            return build_capabilities_answer()

        agent, managed_orchestrator_prompt = MainOrchestratorAgent(
            model=candidate_model,
            tools=[
                ask_list_of_agent,
                ask_benefits_agent,
                ask_usecase_agent,
                ask_customer_domain_agent,
                search_internal_knowledge,
                explain_capabilities,
            ],
        ).create()

        try:
            with agent_observation(
                state_holder["state"],
                name=MainOrchestratorAgent.display_name,
                input_data={
                    "user_query": state.get("user_query", ""),
                    "contextual_query": state.get("contextual_query", ""),
                },
                metadata={
                    "node_type": "orchestrator_agent",
                    "prompt_name": managed_orchestrator_prompt.name,
                    "prompt_version": managed_orchestrator_prompt.version,
                    "prompt_source": managed_orchestrator_prompt.source,
                },
                model=candidate_name,
            ) as observation:
                agent_result = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": AGENT_RECURSION_LIMIT},
                )

                if observation is not None:
                    observation.update(
                        output=sanitize_for_trace(
                            {
                                "tools_called": tools_called,
                                "selected_agents": state_holder["state"].get(
                                    "selected_agents",
                                    [],
                                ),
                            }
                        )
                    )
            agent_model_name = candidate_name
            completed_state = state_holder["state"]
            completed_tools_called = tools_called
            completed_prompt = managed_orchestrator_prompt
            break
        except Exception as error:
            agent_errors.append(f"{candidate_name}: {sanitize_runtime_error(error)}")

    if agent_result is None:
        return deterministic_retrieval_fallback(state, LLMGatewayError(" | ".join(agent_errors)))

    selected_tool = (
        "multi_specialist"
        if len([name for name in completed_tools_called if name in SPECIALIST_SPECS]) > 1
        else completed_tools_called[-1] if completed_tools_called else "direct_response"
    )
    answer = completed_state.get("hardcoded_all_usecases_answer") or extract_agent_answer(agent_result)

    return build_final_response(
        {
            **completed_state,
            "intent": selected_tool,
            "orchestrator_tool": selected_tool,
            "orchestrator_tool_input": state.get("contextual_query") or state.get("user_query", ""),
            "orchestrator_reason": "MainOrchestratorAgent routed the request to create_agent specialists.",
            "orchestrator_decision": {"tools_called": completed_tools_called},
            "orchestrator_error": "",
            "llm_provider_status": get_llm_provider_status(),
            "orchestrator_llm_model": agent_model_name,
            "orchestrator_prompt_name": getattr(completed_prompt, "name", ORCHESTRATOR_PROMPT_NAME),
            "orchestrator_prompt_version": getattr(completed_prompt, "version", None),
            "orchestrator_prompt_source": getattr(completed_prompt, "source", "local_fallback"),
        },
        answer=answer,
        answer_model=agent_model_name,
    )
