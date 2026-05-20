from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable | None = None


ORCHESTRATOR_TOOLS = [
    ToolSpec("intent_classification", "Classifies user intent."),
    ToolSpec("agent_selection", "Chooses the next agent dynamically."),
    ToolSpec("task_planning", "Creates short execution plans."),
    ToolSpec("state_management", "Reads and updates graph state."),
    ToolSpec("evaluation_aggregator", "Aggregates per-agent evaluations."),
    ToolSpec("fallback_router", "Routes failed outputs to fallback handling."),
    ToolSpec("final_response_composer", "Composes safe final response package."),
    ToolSpec("trace_logging", "Records trace ids and agent path summaries."),
]

KNOWLEDGE_RETRIEVAL_TOOLS = [
    ToolSpec("qdrant_retrieval", "Retrieves internal chunks from Qdrant."),
    ToolSpec("metadata_filter", "Applies Qdrant metadata filters."),
    ToolSpec("context_builder", "Builds grounded internal context."),
    ToolSpec("source_ranking", "Ranks retrieved sources."),
    ToolSpec("source_citation", "Formats source citations."),
]

EVAL_TOOLS = [
    ToolSpec("hallucination_check", "Checks hallucination risk."),
    ToolSpec("source_grounding", "Checks source grounding."),
    ToolSpec("confidentiality_check", "Checks confidentiality risk."),
    ToolSpec("sales_claim_validation", "Checks sales claims."),
    ToolSpec("tone_brand_safety", "Checks tone and brand safety."),
    ToolSpec("completeness_check", "Checks completeness."),
    ToolSpec("final_approval", "Returns final eval status."),
]
