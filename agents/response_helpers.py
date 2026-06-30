import re
from typing import Any

from agents.state import SalesHelperState

FALLBACK_STOPWORDS = {
    "a",
    "about",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "case",
    "cases",
    "detail",
    "detailed",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "one",
    "please",
    "specific",
    "summarise",
    "summarize",
    "tell",
    "the",
    "this",
    "to",
    "use",
    "usecase",
    "usecases",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
}

FALLBACK_SECTION_LABELS = [
    "Business Context",
    "Use Case",
    "Use Case Category",
    "Solution Proposed",
    "Workflow/Image Summary",
    "Tools Used",
    "Benefits",
]


def strip_markdown_markers(answer: str) -> str:
    cleaned_lines = []

    for line in answer.splitlines():
        cleaned_line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        cleaned_line = re.sub(r"^(\s*)[*+]\s+", r"\1- ", cleaned_line)
        cleaned_line = re.sub(r"(\s)[*+]\s+", r"\1- ", cleaned_line)
        cleaned_line = cleaned_line.replace("**", "")
        cleaned_line = cleaned_line.replace("__", "")
        cleaned_line = re.sub(r"(?<!\w)\*(?!\w)", "", cleaned_line)
        cleaned_lines.append(cleaned_line.rstrip())

    return "\n".join(cleaned_lines).strip()


def build_capabilities_answer() -> str:
    return "\n".join(
        [
            "I am the Predikly Sales Helper for internal case-study and use-case knowledge.",
            "",
            "I can help with:",
            "- Search Predikly's internal case-study/use-case data through hybrid Qdrant retrieval.",
            "- List or count use cases from Qdrant metadata, optionally filtered by company, domain, or country.",
            "- Use dense semantic retrieval plus sparse BM25-style retrieval fused with RRF.",
            "- Switch from the main collection to the fallback collection when the main collection has no usable context.",
            "- Answer questions about customers, use cases, tools, benefits, domains, counts, and source slides when the data is retrieved.",
            "- Use the current chat only for clear follow-up questions, while allowing new topics in the same chat.",
            "- Draft grounded sales content from retrieved internal context.",
            "",
            "Limits:",
            "- I answer from retrieved internal context, not from general web or model memory.",
            "- If retrieval does not provide enough context, I should say that instead of inventing an answer.",
        ]
    )


def normalize_lookup_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def query_tokens(query: str) -> set[str]:
    return {
        token
        for token in normalize_lookup_text(query).split()
        if len(token) >= 3 and token not in FALLBACK_STOPWORDS
    }


def is_capability_query(query: str) -> bool:
    normalized_query = normalize_lookup_text(query)
    patterns = [
        r"\bwhat\s+(?:do|can)\s+you\s+do\b",
        r"\bwho\s+are\s+you\b",
        r"\bhow\s+can\s+you\s+help\b",
        r"\bwhat\s+are\s+your\s+capabilities\b",
        r"\bcapabilit(?:y|ies)\b",
        r"\bwhat\s+tools?\s+do\s+you\s+have\b",
    ]
    return any(re.search(pattern, normalized_query) for pattern in patterns)


def infer_fallback_intent(query: str) -> str:
    normalized_query = normalize_lookup_text(query)

    if is_capability_query(query):
        return "capabilities"

    if re.search(r"\b(count|how many|number)\b", normalized_query) and re.search(
        r"\b(usecase|usecases|use case|use cases)\b",
        normalized_query,
    ):
        return "count_usecases"

    if re.search(r"\b(list|name|names|show|all|enumerate)\b", normalized_query) and re.search(
        r"\b(usecase|usecases|use case|use cases)\b",
        normalized_query,
    ):
        return "list_usecases"

    if re.search(r"\b(explain|detail|detailed|deep|tell me about|walk through|describe)\b", normalized_query):
        return "explain_specific_usecase"

    return "grounded_answer"


def sanitize_runtime_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"\b(?:AQ|AIza)[A-Za-z0-9._-]+", "[redacted_api_key]", message)
    message = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [redacted_token]", message, flags=re.IGNORECASE)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{type(error).__name__}: {message[:300]}"


def item_label_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in (
            item.get("customer_name"),
            item.get("company_name"),
            item.get("usecase_name"),
            item.get("use_case_name"),
            item.get("customer_domain"),
            item.get("use_case_category"),
            item.get("ppt_name"),
            item.get("slide_title"),
            item.get("text"),
        )
    )


def score_context_item(item: dict[str, Any], tokens: set[str]) -> int:
    if not tokens:
        return 0

    metadata_text = normalize_lookup_text(
        " ".join(
            str(value or "")
            for value in (
                item.get("customer_name"),
                item.get("company_name"),
                item.get("usecase_name"),
                item.get("use_case_name"),
                item.get("ppt_name"),
                item.get("slide_title"),
            )
        )
    )
    full_text = normalize_lookup_text(item_label_text(item))
    score = 0

    for token in tokens:
        if re.search(rf"\b{re.escape(token)}\b", metadata_text):
            score += 3
        elif re.search(rf"\b{re.escape(token)}\b", full_text):
            score += 1

    return score


def select_fallback_context(context: list[dict[str, Any]], query: str) -> tuple[list[dict[str, Any]], bool]:
    tokens = query_tokens(query)

    if not context:
        return [], False

    scored_items = [
        (score_context_item(item, tokens), index, item)
        for index, item in enumerate(context)
    ]
    scored_items.sort(key=lambda item: (-item[0], item[1]))
    best_score = scored_items[0][0]

    if tokens and best_score == 0:
        return [], False

    selected = [
        item
        for score, _, item in scored_items
        if score == best_score or (best_score > 1 and score >= max(1, best_score - 2))
    ]
    return selected[:6], True


def source_label(source: dict[str, Any]) -> str:
    ppt_name = source.get("ppt_name") or source.get("source") or "Unknown source"
    slide_number = source.get("slide_number") or source.get("page")

    if slide_number:
        return f"{ppt_name}, slide {slide_number}"

    return str(ppt_name)


def matching_sources_for_context(
    selected_context: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> list[str]:
    selected_keys = {
        (
            item.get("drive_id") or item.get("document_id"),
            item.get("customer_name") or item.get("company_name"),
            item.get("usecase_name") or item.get("use_case_name"),
        )
        for item in selected_context
    }
    labels = []

    for source in sources:
        source_key = (
            source.get("drive_id") or source.get("document_id"),
            source.get("customer_name") or source.get("company_name"),
            source.get("usecase_name") or source.get("use_case_name"),
        )

        if selected_keys and source_key not in selected_keys:
            continue

        label = source_label(source)

        if label not in labels:
            labels.append(label)

    if labels:
        return labels[:5]

    for source in sources[:5]:
        label = source_label(source)

        if label not in labels:
            labels.append(label)

    return labels


def extract_labeled_sections(text: str) -> dict[str, str]:
    compact_text = re.sub(r"\s+", " ", str(text or "")).strip()
    sections: dict[str, str] = {}

    if not compact_text:
        return sections

    label_pattern = "|".join(re.escape(label) for label in FALLBACK_SECTION_LABELS)
    pattern = rf"({label_pattern})\s*:\s*(.*?)(?=\s+(?:{label_pattern})\s*:|$)"

    for match in re.finditer(pattern, compact_text, flags=re.IGNORECASE):
        label = next(
            (
                section_label
                for section_label in FALLBACK_SECTION_LABELS
                if section_label.lower() == match.group(1).lower()
            ),
            match.group(1),
        )
        value = match.group(2).strip(" .")

        if value and value not in sections.get(label, ""):
            sections[label] = " ".join(part for part in (sections.get(label, ""), value) if part)

    return sections


def truncate_sentence(value: str, max_length: int = 800) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()

    if len(cleaned) <= max_length:
        return cleaned

    truncated = cleaned[:max_length].rsplit(" ", 1)[0].rstrip(" .,;:")
    return f"{truncated}..."


def merge_context_sections(context: list[dict[str, Any]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {label: [] for label in FALLBACK_SECTION_LABELS}
    merged["Retrieved Details"] = []

    for item in context:
        sections = extract_labeled_sections(str(item.get("text") or ""))

        if sections:
            for label, value in sections.items():
                snippet = truncate_sentence(value, 900)

                if snippet and snippet not in merged[label]:
                    merged[label].append(snippet)
            continue

        text = truncate_sentence(str(item.get("text") or ""), 900)

        if text and text not in merged["Retrieved Details"]:
            merged["Retrieved Details"].append(text)

    return merged


def primary_context_title(context: list[dict[str, Any]]) -> tuple[str, str]:
    first_item = context[0] if context else {}
    customer = (
        first_item.get("customer_name")
        or first_item.get("company_name")
        or "Unknown company"
    )
    usecase = (
        first_item.get("usecase_name")
        or first_item.get("use_case_name")
        or "Use case not named"
    )
    return str(customer), str(usecase)


def build_list_fallback_answer(
    state: SalesHelperState,
    selected_context: list[dict[str, Any]],
    intent: str,
) -> str:
    seen = set()
    entries = []

    for item in selected_context:
        customer = item.get("customer_name") or item.get("company_name") or "Unknown company"
        usecase = item.get("usecase_name") or item.get("use_case_name") or "Use case not named"
        key = (normalize_lookup_text(customer), normalize_lookup_text(usecase))

        if key in seen:
            continue

        seen.add(key)
        entries.append(f"- {customer}: {usecase}")

    if intent == "count_usecases":
        lines = [f"I found {len(entries)} matching use case(s) in the retrieved internal context."]
    else:
        lines = ["Matching use cases from the retrieved internal context:"]

    lines.extend(entries or ["- No named use cases were present in the retrieved context."])
    lines.extend(["", "Sources used:"])
    lines.extend(f"- {label}" for label in matching_sources_for_context(selected_context, state.get("qdrant_sources", [])))
    return "\n".join(lines)


def build_grounded_fallback_answer(state: SalesHelperState) -> str:
    sources = state.get("qdrant_sources", [])
    context = state.get("internal_context", [])
    user_query = state.get("user_query", "")
    intent = infer_fallback_intent(user_query)

    if not sources and not context:
        return ""

    if intent == "capabilities":
        return build_capabilities_answer()

    selected_context, matched_query = select_fallback_context(context, user_query)

    if not matched_query:
        source_lines = [f"- {source_label(source)}" for source in sources[:5]]
        lines = [
            "I could not find retrieved internal context that clearly matches this exact question.",
        ]

        if source_lines:
            lines.extend(["", "Closest sources retrieved:"])
            lines.extend(source_lines)

        return "\n".join(lines)

    if intent in {"list_usecases", "count_usecases"}:
        return build_list_fallback_answer(state, selected_context, intent)

    customer, usecase = primary_context_title(selected_context)
    merged_sections = merge_context_sections(selected_context)
    lines = [
        "Answer based on retrieved internal context.",
        "",
        f"Customer: {customer}",
        f"Use case: {usecase}",
    ]

    section_map = [
        ("Business Context", "Business context"),
        ("Use Case", "Use case details"),
        ("Use Case Category", "Use case category"),
        ("Solution Proposed", "Solution/workflow"),
        ("Workflow/Image Summary", "Workflow/image summary"),
        ("Tools Used", "Tools/systems used"),
        ("Benefits", "Benefits/outcomes"),
        ("Retrieved Details", "Retrieved details"),
    ]

    for source_label_name, output_label in section_map:
        values = merged_sections.get(source_label_name, [])

        if not values:
            continue

        lines.extend(["", f"{output_label}:"])

        for value in values[:3]:
            lines.append(f"- {value}")

    lines.append("")
    lines.append("Sources used:")
    lines.extend(f"- {label}" for label in matching_sources_for_context(selected_context, sources))

    return "\n".join(lines)

