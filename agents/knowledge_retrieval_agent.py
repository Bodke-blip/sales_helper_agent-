import os
import json
import re
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType

from ingestion import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
)
from agents.state import SalesHelperState
from agents.tools import KNOWLEDGE_RETRIEVAL_TOOLS


AGENT_NAME = "knowledge_retrieval"
TOOLS = KNOWLEDGE_RETRIEVAL_TOOLS
DEFAULT_TOP_K = 5
CUSTOMER_TOP_K = int(os.getenv("QDRANT_CUSTOMER_TOP_K", "40"))
MIN_RETRIEVAL_RESULTS = 1
CUSTOMER_MANIFEST_PATH = Path(os.getenv("CUSTOMER_MANIFEST_PATH", "data/customer_manifest.json"))
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "300"))
RETRIEVAL_CACHE_MAX_ENTRIES = int(os.getenv("RETRIEVAL_CACHE_MAX_ENTRIES", "256"))
retrieval_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
retrieval_cache_lock = Lock()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be configured.")

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def ensure_payload_indexes() -> None:
    client = get_qdrant_client()
    indexes = {
        "customer_name_normalized": PayloadSchemaType.KEYWORD,
        "company_name": PayloadSchemaType.TEXT,
        "use_case_name": PayloadSchemaType.TEXT,
        "source_type": PayloadSchemaType.KEYWORD,
        "is_internal": PayloadSchemaType.BOOL,
        "content_type": PayloadSchemaType.KEYWORD,
    }

    for field_name, field_schema in indexes.items():
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as error:
            if "already exists" not in str(error).lower():
                print(f"Payload index unavailable for {field_name}: {error}")


def get_fallback_collections() -> list[str]:
    if os.getenv("ENABLE_QDRANT_FALLBACK_COLLECTIONS", "").lower() not in {"1", "true", "yes"}:
        return []

    raw_value = os.getenv("QDRANT_FALLBACK_COLLECTIONS", "")
    return [
        collection.strip()
        for collection in raw_value.split(",")
        if collection.strip()
    ]


@lru_cache(maxsize=1)
def load_customer_manifest() -> list[dict]:
    if not CUSTOMER_MANIFEST_PATH.exists():
        manifest = build_customer_manifest_from_qdrant(QDRANT_COLLECTION_NAME)

        if manifest:
            try:
                CUSTOMER_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
                CUSTOMER_MANIFEST_PATH.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass

        return manifest

    try:
        return json.loads(CUSTOMER_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def build_customer_manifest_from_qdrant(collection_name: str) -> list[dict]:
    customers = {}

    for payload in scroll_collection_payloads(collection_name):
        metadata = get_payload_metadata(payload)
        normalized_name = str(metadata.get("customer_name_normalized") or "").strip()

        if not normalized_name:
            continue

        customer = customers.setdefault(
            normalized_name,
            {
                "company_name": metadata.get("company_name") or metadata.get("customer_name", ""),
                "customer_name_normalized": normalized_name,
                "customer_domain": metadata.get("customer_domain", ""),
                "use_cases": {},
            },
        )
        use_case_name = metadata.get("use_case_name") or metadata.get("usecase_name")

        if use_case_name:
            customer["use_cases"].setdefault(
                str(use_case_name),
                {
                    "use_case_name": str(use_case_name),
                    "customer_domain": metadata.get("customer_domain", ""),
                    "ppt_names": set(),
                    "slide_numbers": set(),
                },
            )
            use_case = customer["use_cases"][str(use_case_name)]

            if metadata.get("ppt_name"):
                use_case["ppt_names"].add(metadata["ppt_name"])

            if metadata.get("slide_number") is not None:
                use_case["slide_numbers"].add(metadata["slide_number"])

    manifest = []

    for customer in customers.values():
        use_cases = []

        for use_case in customer["use_cases"].values():
            use_cases.append(
                {
                    **use_case,
                    "ppt_names": sorted(use_case["ppt_names"]),
                    "slide_numbers": sorted(use_case["slide_numbers"]),
                }
            )

        manifest.append(
            {
                **customer,
                "use_cases": sorted(
                    use_cases,
                    key=lambda item: item["use_case_name"].lower(),
                ),
                "use_case_count": len(use_cases),
            }
        )

    return sorted(manifest, key=lambda item: item["customer_name_normalized"])


def retrieve_from_collection(
    query: str,
    *,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    customer_name_normalized: str = "",
) -> list[tuple]:
    client = get_qdrant_client()
    query_vector = get_embeddings().embed_query(query)
    query_filter = None

    if customer_name_normalized:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="customer_name_normalized",
                    match=MatchValue(value=customer_name_normalized),
                )
            ]
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    results = []

    for point in response.points:
        payload = get_point_payload(point)
        document = Document(
            page_content=get_payload_text(payload),
            metadata=get_payload_metadata(payload),
        )
        results.append((document, float(point.score or 0.0)))

    return results


def normalize_lookup_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_query_for_retrieval_cache(query: str) -> str:
    return " ".join(query.lower().strip().split())


def build_retrieval_cache_key(
    *,
    query: str,
    collections: list[str],
) -> tuple[Any, ...]:
    return (
        normalize_query_for_retrieval_cache(query),
        tuple(collections),
        QDRANT_COLLECTION_NAME,
        CUSTOMER_TOP_K,
        DEFAULT_TOP_K,
    )


def get_cached_retrieval(cache_key: tuple[Any, ...]) -> dict[str, Any] | None:
    if RETRIEVAL_CACHE_TTL_SECONDS <= 0:
        return None

    now = time.monotonic()

    with retrieval_cache_lock:
        cached_item = retrieval_cache.get(cache_key)

        if not cached_item:
            return None

        if now - cached_item["created_at"] > RETRIEVAL_CACHE_TTL_SECONDS:
            retrieval_cache.pop(cache_key, None)
            return None

        return deepcopy(cached_item["payload"])


def set_cached_retrieval(cache_key: tuple[Any, ...], payload: dict[str, Any]) -> None:
    if RETRIEVAL_CACHE_TTL_SECONDS <= 0:
        return

    with retrieval_cache_lock:
        if len(retrieval_cache) >= RETRIEVAL_CACHE_MAX_ENTRIES:
            oldest_key = min(
                retrieval_cache,
                key=lambda key: retrieval_cache[key]["created_at"],
            )
            retrieval_cache.pop(oldest_key, None)

        retrieval_cache[cache_key] = {
            "created_at": time.monotonic(),
            "payload": deepcopy(payload),
        }


def retrieval_cache_status() -> dict[str, Any]:
    with retrieval_cache_lock:
        return {
            "enabled": RETRIEVAL_CACHE_TTL_SECONDS > 0,
            "entries": len(retrieval_cache),
            "ttl_seconds": RETRIEVAL_CACHE_TTL_SECONDS,
            "max_entries": RETRIEVAL_CACHE_MAX_ENTRIES,
            "scope": "knowledge_retrieval",
        }


def clear_retrieval_cache() -> dict[str, Any]:
    with retrieval_cache_lock:
        cleared_entries = len(retrieval_cache)
        retrieval_cache.clear()

    return {
        "status": "cleared",
        "entries_cleared": cleared_entries,
        "scope": "knowledge_retrieval",
    }


def lookup_tokens(value: str) -> list[str]:
    return [
        token
        for token in normalize_lookup_value(value).split()
        if len(token) > 1
    ]


def customer_name_matches_query(customer_name: str, query: str) -> bool:
    normalized_customer_name = normalize_lookup_value(customer_name)
    normalized_query = normalize_lookup_value(query)

    if not normalized_customer_name:
        return False

    if normalized_customer_name in normalized_query:
        return True

    customer_tokens = lookup_tokens(customer_name)
    query_tokens = set(lookup_tokens(query))

    if len(customer_tokens) == 1:
        return customer_tokens[0] in query_tokens

    return all(token in query_tokens for token in customer_tokens[:2])


def get_point_payload(point: Any) -> dict:
    return dict(point.payload or {})


def get_payload_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata")

    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {
            key: payload.get(key, "")
            for key in (
                "vector_point_id",
                "document_id",
                "ppt_name",
                "slide_number",
                "slide_title",
                "company_name",
                "customer_name_normalized",
                "customer_domain",
                "use_case_name",
                "use_case_category",
                "solution_proposed",
                "workflow_image_summary",
                "tools_used",
                "benefits",
                "chunk_type",
                "content_type",
                "source_type",
                "is_internal",
                "usecase_name",
                "customer_name",
                "drive_id",
                "match_key",
            )
        }

    company_name = metadata.get("company_name") or metadata.get("customer_name") or ""
    use_case_name = metadata.get("use_case_name") or metadata.get("usecase_name") or ""
    document_id = metadata.get("document_id") or metadata.get("drive_id") or ""

    metadata["company_name"] = metadata.get("company_name") or company_name
    metadata["customer_name"] = metadata.get("customer_name") or company_name
    metadata["customer_name_normalized"] = (
        metadata.get("customer_name_normalized")
        or normalize_lookup_value(str(company_name))
    )
    metadata["use_case_name"] = metadata.get("use_case_name") or use_case_name
    metadata["usecase_name"] = metadata.get("usecase_name") or use_case_name
    metadata["document_id"] = metadata.get("document_id") or document_id
    metadata["drive_id"] = metadata.get("drive_id") or document_id

    return metadata


def get_payload_text(payload: dict) -> str:
    return str(
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or ""
    )


def scroll_collection_payloads(collection_name: str) -> list[dict]:
    client = get_qdrant_client()
    offset = None
    payloads = []

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(get_point_payload(point) for point in points)

        if offset is None:
            break

    return payloads


def find_requested_customer_name(query: str, collection_name: str) -> str:
    del collection_name
    manifest = load_customer_manifest()
    matching_customers = []

    for customer in manifest:
        company_name = str(customer.get("company_name") or "").strip()
        normalized_name = str(customer.get("customer_name_normalized") or "").strip()

        if customer_name_matches_query(company_name, query) or customer_name_matches_query(normalized_name, query):
            matching_customers.append(customer)

    if not matching_customers:
        return ""

    best_match = max(
        matching_customers,
        key=lambda item: len(str(item.get("customer_name_normalized", ""))),
    )
    return str(best_match.get("customer_name_normalized") or "")


def get_manifest_customer(customer_name_normalized: str) -> dict:
    normalized = normalize_lookup_value(customer_name_normalized)

    for customer in load_customer_manifest():
        if normalize_lookup_value(str(customer.get("customer_name_normalized", ""))) == normalized:
            return customer

    return {}


def build_exact_customer_context_and_sources(
    *,
    query: str,
    collection_name: str,
    customer_name: str,
) -> tuple[list[dict], list[dict]]:
    results = retrieve_from_collection(
        query,
        collection_name=collection_name,
        top_k=CUSTOMER_TOP_K,
        customer_name_normalized=normalize_lookup_value(customer_name),
    )
    return build_context_and_sources(results, collection_name)


def extract_customer_name_from_count_query(query: str) -> str:
    match = re.search(
        r"(?:of|for|with|from|about)\s+(.+?)\s+(?:has|have|did|worked|cases|use cases)",
        query,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).strip(" ?.,")

    return ""


def is_count_query(query: str) -> bool:
    lowered = query.lower()
    return bool(re.search(r"\b(how many|count|number of)\b", lowered))


def count_unique_use_cases_for_customer(
    customer_name: str,
    *,
    collection_name: str,
) -> int | None:
    if not customer_name:
        return None

    del collection_name
    customer = get_manifest_customer(customer_name)

    if customer:
        return int(customer.get("use_case_count", len(customer.get("use_cases", []))))

    return None


def build_manifest_context_and_sources(
    *,
    customer_name_normalized: str,
    collection_name: str,
) -> tuple[list[dict], list[dict]]:
    customer = get_manifest_customer(customer_name_normalized)

    if not customer:
        return [], []

    company_name = customer.get("company_name") or customer_name_normalized
    use_cases = customer.get("use_cases", [])
    case_count = int(customer.get("use_case_count", len(use_cases)))
    internal_context = [
        {
            "customer_name": company_name,
            "customer_name_normalized": customer_name_normalized,
            "customer_domain": customer.get("customer_domain", ""),
            "case_count": case_count,
            "text": (
                f"{company_name} maps to {case_count} unique use case(s) "
                f"in Qdrant collection {collection_name}."
            ),
        }
    ]
    qdrant_sources = []

    for use_case in use_cases:
        ppt_names = use_case.get("ppt_names", [])
        slide_numbers = use_case.get("slide_numbers", [])
        qdrant_sources.append(
            {
                "collection": collection_name,
                "customer_name": company_name,
                "usecase_name": use_case.get("use_case_name", ""),
                "customer_domain": use_case.get("customer_domain") or customer.get("customer_domain", ""),
                "drive_id": "",
                "match_key": "",
                "ppt_name": ppt_names[0] if ppt_names else "",
                "slide_number": slide_numbers[0] if slide_numbers else None,
                "chunk_type": "manifest_summary",
                "content_type": "customer_manifest",
                "score": 1.0,
            }
        )

    return internal_context, qdrant_sources


def build_manifest_use_case_matches(
    *,
    query: str,
    collection_name: str,
    limit: int = DEFAULT_TOP_K,
) -> tuple[list[dict], list[dict]]:
    query_tokens = set(lookup_tokens(query))
    matches = []

    for customer in load_customer_manifest():
        company_name = customer.get("company_name") or customer.get("customer_name_normalized", "")

        for use_case in customer.get("use_cases", []):
            use_case_name = str(use_case.get("use_case_name", ""))
            use_case_tokens = set(lookup_tokens(use_case_name))

            if not use_case_tokens:
                continue

            overlap = use_case_tokens & query_tokens
            exact_phrase = normalize_lookup_value(use_case_name) in normalize_lookup_value(query)
            strong_match = exact_phrase or (
                len(overlap) >= min(2, len(use_case_tokens))
                and len(overlap) / len(use_case_tokens) >= 0.6
            )

            if not strong_match:
                continue

            matches.append(
                {
                    "score": 1.0 if exact_phrase else len(overlap) / len(use_case_tokens),
                    "customer": customer,
                    "use_case": use_case,
                    "company_name": company_name,
                }
            )

    matches = sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]
    internal_context = []
    qdrant_sources = []

    for rank, match in enumerate(matches, start=1):
        customer = match["customer"]
        use_case = match["use_case"]
        ppt_names = use_case.get("ppt_names", [])
        slide_numbers = use_case.get("slide_numbers", [])
        use_case_name = use_case.get("use_case_name", "")
        company_name = match["company_name"]

        internal_context.append(
            {
                "rank": rank,
                "score": match["score"],
                "customer_name": company_name,
                "customer_name_normalized": customer.get("customer_name_normalized", ""),
                "customer_domain": use_case.get("customer_domain") or customer.get("customer_domain", ""),
                "use_case_name": use_case_name,
                "usecase_name": use_case_name,
                "ppt_name": ppt_names[0] if ppt_names else "",
                "slide_number": slide_numbers[0] if slide_numbers else None,
                "text": f"{company_name}: {use_case_name}",
            }
        )
        qdrant_sources.append(
            {
                "collection": collection_name,
                "customer_name": company_name,
                "usecase_name": use_case_name,
                "customer_domain": use_case.get("customer_domain") or customer.get("customer_domain", ""),
                "drive_id": "",
                "match_key": "",
                "ppt_name": ppt_names[0] if ppt_names else "",
                "slide_number": slide_numbers[0] if slide_numbers else None,
                "chunk_type": "manifest_summary",
                "content_type": "customer_manifest",
                "score": match["score"],
            }
        )

    return internal_context, qdrant_sources


def build_context_and_sources(results: list[tuple], collection_name: str) -> tuple[list[dict], list[dict]]:
    internal_context = []
    qdrant_sources = []
    seen_sources = set()

    for rank, (document, score) in enumerate(results, start=1):
        metadata = dict(document.metadata or {})
        context_item = {
            "rank": rank,
            "score": float(score),
            "text": document.page_content,
            **metadata,
        }
        internal_context.append(context_item)

        source_key = (
            metadata.get("drive_id") or metadata.get("document_id"),
            metadata.get("customer_name") or metadata.get("company_name"),
            metadata.get("usecase_name") or metadata.get("use_case_name"),
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        qdrant_sources.append(
            {
                "collection": collection_name,
                "customer_name": metadata.get("customer_name") or metadata.get("company_name", ""),
                "usecase_name": metadata.get("usecase_name") or metadata.get("use_case_name", ""),
                "customer_domain": metadata.get("customer_domain", ""),
                "drive_id": metadata.get("drive_id") or metadata.get("document_id", ""),
                "match_key": metadata.get("match_key", ""),
                "ppt_name": metadata.get("ppt_name", ""),
                "slide_number": metadata.get("slide_number"),
                "chunk_type": metadata.get("chunk_type", ""),
                "content_type": metadata.get("content_type", ""),
                "score": float(score),
            }
        )

    return internal_context, qdrant_sources


def knowledge_retrieval_agent(state: SalesHelperState) -> SalesHelperState:
    query = state.get("user_query", "")
    collections = [QDRANT_COLLECTION_NAME, *get_fallback_collections()]
    cache_key = build_retrieval_cache_key(query=query, collections=collections)
    cached_retrieval = get_cached_retrieval(cache_key)

    if cached_retrieval:
        return {
            **state,
            "internal_context": [
                *state.get("internal_context", []),
                *cached_retrieval.get("retrieved_context", []),
            ],
            "qdrant_sources": [
                *state.get("qdrant_sources", []),
                *cached_retrieval.get("retrieved_sources", []),
            ],
            "retrieval_collection": cached_retrieval.get("used_collection", QDRANT_COLLECTION_NAME),
            "retrieval_error": cached_retrieval.get("retrieval_error", ""),
            "retrieval_customer_filter": cached_retrieval.get("requested_customer_name", ""),
            "retrieval_cache_status": "hit",
        }

    retrieval_error = ""
    results = []
    used_collection = QDRANT_COLLECTION_NAME
    retrieved_context = []
    retrieved_sources = []
    requested_customer_name = ""

    try:
        for collection_name in collections:
            requested_customer_name = find_requested_customer_name(query, collection_name)

            if requested_customer_name and is_count_query(query):
                retrieved_context, retrieved_sources = build_manifest_context_and_sources(
                    customer_name_normalized=requested_customer_name,
                    collection_name=collection_name,
                )

                if retrieved_context:
                    used_collection = collection_name
                    break

            if not requested_customer_name:
                retrieved_context, retrieved_sources = build_manifest_use_case_matches(
                    query=query,
                    collection_name=collection_name,
                )

                if len(retrieved_sources) >= MIN_RETRIEVAL_RESULTS:
                    used_collection = collection_name
                    break

            if requested_customer_name:
                retrieved_context, retrieved_sources = build_exact_customer_context_and_sources(
                    query=query,
                    collection_name=collection_name,
                    customer_name=requested_customer_name,
                )

                if len(retrieved_sources) >= MIN_RETRIEVAL_RESULTS:
                    used_collection = collection_name
                    break

            results = retrieve_from_collection(query, collection_name=collection_name)

            if len(results) >= MIN_RETRIEVAL_RESULTS:
                used_collection = collection_name
                retrieved_context, retrieved_sources = build_context_and_sources(results, used_collection)
                break
    except Exception as error:
        retrieval_error = str(error)
        results = []

    if not retrieved_context and results:
        retrieved_context, retrieved_sources = build_context_and_sources(results, used_collection)

    internal_context = [*state.get("internal_context", []), *retrieved_context]
    qdrant_sources = [*state.get("qdrant_sources", []), *retrieved_sources]

    if is_count_query(query) and internal_context and not any(
        item.get("case_count") is not None for item in internal_context
    ):
        customer_name = requested_customer_name or extract_customer_name_from_count_query(query)

        if not customer_name:
            customer_name = str(internal_context[0].get("customer_name", ""))

        try:
            case_count = count_unique_use_cases_for_customer(
                customer_name,
                collection_name=used_collection,
            )
        except Exception:
            case_count = None

        if case_count is not None:
            internal_context.insert(
                0,
                {
                    "customer_name": customer_name,
                    "case_count": case_count,
                    "text": (
                        f"{customer_name} maps to {case_count} unique use case(s) "
                        f"in Qdrant collection {used_collection}."
                    ),
                },
            )

    cache_payload = {
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "used_collection": used_collection,
        "retrieval_error": retrieval_error,
        "requested_customer_name": requested_customer_name,
    }

    if retrieved_context or retrieved_sources:
        set_cached_retrieval(cache_key, cache_payload)

    return {
        **state,
        "internal_context": internal_context,
        "qdrant_sources": qdrant_sources,
        "retrieval_collection": used_collection,
        "retrieval_error": retrieval_error,
        "retrieval_customer_filter": requested_customer_name,
        "retrieval_cache_status": "miss",
    }
