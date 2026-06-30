from __future__ import annotations

import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

READINESS_TIMEOUT_SECONDS = float(os.getenv("READINESS_TIMEOUT_SECONDS", "8"))
QDRANT_RECOVERY_COOLDOWN_SECONDS = float(
    os.getenv("QDRANT_RECOVERY_COOLDOWN_SECONDS", "60")
)
QDRANT_RECOVERY_RETRIES = max(1, int(os.getenv("QDRANT_RECOVERY_RETRIES", "8")))
QDRANT_RECOVERY_RETRY_DELAY_SECONDS = float(
    os.getenv("QDRANT_RECOVERY_RETRY_DELAY_SECONDS", "4")
)

_recovery_lock = Lock()
_last_recovery_attempt = 0.0


def _result(
    check_id: str,
    label: str,
    status: str,
    message: str,
    *,
    critical: bool,
    started_at: float,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": check_id,
        "label": label,
        "status": status,
        "message": message,
        "critical": critical,
        "duration_ms": round((time.perf_counter() - started_at) * 1000),
    }

    if recovery:
        payload["recovery"] = recovery

    return payload


def check_backend() -> dict[str, Any]:
    started_at = time.perf_counter()
    return _result(
        "backend",
        "Application server",
        "passed",
        "FastAPI is online and responding.",
        critical=True,
        started_at=started_at,
    )


def _probe_qdrant() -> tuple[list[str], str]:
    from agents.knowledge_retrieval_agent import get_qdrant_client

    client = get_qdrant_client()
    response = client.get_collections()
    collection_names = sorted(collection.name for collection in response.collections)
    cluster_kind = "Qdrant Cloud" if ".cloud.qdrant.io" in (os.getenv("QDRANT_URL") or "") else "Qdrant"
    return collection_names, cluster_kind


def _run_qdrant_recovery() -> dict[str, Any]:
    global _last_recovery_attempt

    with _recovery_lock:
        now = time.monotonic()
        elapsed = now - _last_recovery_attempt

        if _last_recovery_attempt and elapsed < QDRANT_RECOVERY_COOLDOWN_SECONDS:
            return {
                "attempted": False,
                "method": "cooldown",
                "message": "A recovery attempt was made recently; waiting before another restart.",
            }

        _last_recovery_attempt = now
        management_key = (
            os.getenv("QDRANT_CLOUD_MANAGEMENT_KEY")
            or os.getenv("QDRANT_CLOUD_API_KEY")
            or ""
        ).strip()
        account_id = (os.getenv("QDRANT_CLOUD_ACCOUNT_ID") or "").strip()
        cluster_id = (os.getenv("QDRANT_CLOUD_CLUSTER_ID") or "").strip()
        webhook_url = (os.getenv("QDRANT_RECOVERY_WEBHOOK_URL") or "").strip()
        recovery_command = (os.getenv("QDRANT_RESTART_COMMAND") or "").strip()

        if management_key and account_id and cluster_id:
            api_base_url = (
                os.getenv("QDRANT_CLOUD_API_BASE_URL")
                or "https://api.cloud.qdrant.io"
            ).rstrip("/")
            cluster_url = (
                f"{api_base_url}/api/cluster/v1/accounts/{quote(account_id, safe='')}"
                f"/clusters/{quote(cluster_id, safe='')}"
            )
            headers = {"Authorization": f"apikey {management_key}"}

            try:
                cluster_response = requests.get(
                    cluster_url,
                    headers=headers,
                    timeout=READINESS_TIMEOUT_SECONDS,
                )
                cluster_response.raise_for_status()
                cluster = cluster_response.json().get("cluster", {})
                phase = str(cluster.get("state", {}).get("phase", "")).upper()
                action = "unsuspend" if "SUSPEND" in phase else "restart"
                recovery_response = requests.post(
                    f"{cluster_url}/{action}",
                    headers=headers,
                    timeout=READINESS_TIMEOUT_SECONDS,
                )
                recovery_response.raise_for_status()
                return {
                    "attempted": True,
                    "method": f"qdrant_cloud_{action}",
                    "message": (
                        "Qdrant Cloud accepted the cluster resume request."
                        if action == "unsuspend"
                        else "Qdrant Cloud accepted the cluster restart request."
                    ),
                }
            except (requests.RequestException, ValueError, TypeError):
                return {
                    "attempted": True,
                    "method": "qdrant_cloud",
                    "message": "Qdrant Cloud did not accept the automatic recovery request.",
                }

        if webhook_url:
            headers = {"Content-Type": "application/json"}
            token = (os.getenv("QDRANT_RECOVERY_WEBHOOK_TOKEN") or "").strip()

            if token:
                headers["Authorization"] = f"Bearer {token}"

            try:
                response = requests.post(
                    webhook_url,
                    headers=headers,
                    json={"action": "restart", "service": "qdrant"},
                    timeout=READINESS_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return {
                    "attempted": True,
                    "method": "webhook",
                    "message": "The configured Qdrant recovery webhook accepted the restart request.",
                }
            except requests.RequestException:
                return {
                    "attempted": True,
                    "method": "webhook",
                    "message": "The configured Qdrant recovery webhook did not accept the request.",
                }

        if recovery_command:
            try:
                completed = subprocess.run(
                    shlex.split(recovery_command),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=READINESS_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                return {
                    "attempted": True,
                    "method": "command",
                    "message": "The configured Qdrant restart command could not be executed.",
                }

            return {
                "attempted": True,
                "method": "command",
                "message": (
                    "The configured Qdrant restart command completed."
                    if completed.returncode == 0
                    else "The configured Qdrant restart command returned an error."
                ),
            }

        return {
            "attempted": False,
            "method": "retry",
            "message": (
                "No cluster-management hook is configured; the app retried the Qdrant endpoint "
                "to wake or reconnect it."
            ),
        }


def _retry_qdrant_probe() -> tuple[list[str], str] | None:
    for attempt in range(QDRANT_RECOVERY_RETRIES):
        if attempt:
            time.sleep(QDRANT_RECOVERY_RETRY_DELAY_SECONDS)

        try:
            return _probe_qdrant()
        except Exception:
            continue

    return None


def check_qdrant() -> list[dict[str, Any]]:
    from hybrid_retrieval import HYBRID_COLLECTION_NAME, HYBRID_FALLBACK_COLLECTION_NAME

    started_at = time.perf_counter()
    recovery = None

    try:
        collection_names, cluster_kind = _probe_qdrant()
    except Exception:
        recovery = _run_qdrant_recovery()
        recovered_probe = _retry_qdrant_probe()

        if recovered_probe is None:
            failed = _result(
                "qdrant",
                "Qdrant connection",
                "failed",
                "Qdrant is unavailable after automatic recovery and retry attempts.",
                critical=True,
                started_at=started_at,
                recovery=recovery,
            )
            collections = _result(
                "qdrant_collections",
                "Knowledge collections",
                "failed",
                "Collections could not be verified because Qdrant is unavailable.",
                critical=True,
                started_at=started_at,
            )
            return [failed, collections]

        collection_names, cluster_kind = recovered_probe

    connection = _result(
        "qdrant",
        "Qdrant connection",
        "passed",
        f"{cluster_kind} is reachable.",
        critical=True,
        started_at=started_at,
        recovery=recovery,
    )
    available = set(collection_names)
    main_available = HYBRID_COLLECTION_NAME in available
    fallback_available = HYBRID_FALLBACK_COLLECTION_NAME in available

    if main_available:
        collection_status = "passed"
        collection_message = f"Primary collection '{HYBRID_COLLECTION_NAME}' is available."
    elif fallback_available:
        collection_status = "warning"
        collection_message = (
            f"Primary collection is missing; fallback '{HYBRID_FALLBACK_COLLECTION_NAME}' is available."
        )
    else:
        collection_status = "failed"
        collection_message = "Neither the primary nor fallback knowledge collection is available."

    collections = _result(
        "qdrant_collections",
        "Knowledge collections",
        collection_status,
        collection_message,
        critical=True,
        started_at=started_at,
    )
    return [connection, collections]


def check_gemini() -> dict[str, Any]:
    from agents.llm import PRIMARY_LLM_MODEL

    started_at = time.perf_counter()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    if not api_key:
        return _result(
            "gemini",
            "Gemini model",
            "failed",
            "GEMINI_API_KEY is not configured.",
            critical=True,
            started_at=started_at,
        )

    model_path = quote(PRIMARY_LLM_MODEL, safe="")

    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}",
            headers={"x-goog-api-key": api_key},
            timeout=READINESS_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            raise requests.RequestException(f"status={response.status_code}")
    except requests.RequestException:
        return _result(
            "gemini",
            "Gemini model",
            "failed",
            f"Gemini model '{PRIMARY_LLM_MODEL}' could not be verified.",
            critical=True,
            started_at=started_at,
        )

    return _result(
        "gemini",
        "Gemini model",
        "passed",
        f"Gemini model '{PRIMARY_LLM_MODEL}' is accessible.",
        critical=True,
        started_at=started_at,
    )


def check_embeddings() -> dict[str, Any]:
    from agents.embedding_client import EMBEDDING_MODEL, get_embeddings

    started_at = time.perf_counter()

    try:
        embedding = get_embeddings().embed_query("Predikly readiness check")
    except Exception:
        return _result(
            "embeddings",
            "Embedding service",
            "failed",
            f"Hosted embedding model '{EMBEDDING_MODEL}' is unavailable.",
            critical=True,
            started_at=started_at,
        )

    return _result(
        "embeddings",
        "Embedding service",
        "passed",
        f"Hosted embeddings are ready ({len(embedding)} dimensions).",
        critical=True,
        started_at=started_at,
    )


def check_chat_storage() -> dict[str, Any]:
    from backend.chat_memory import get_postgres_connection, postgres_enabled

    started_at = time.perf_counter()

    if not postgres_enabled():
        return _result(
            "chat_storage",
            "Conversation storage",
            "warning",
            "Using in-memory chat storage; conversations reset when the server restarts.",
            critical=False,
            started_at=started_at,
        )

    try:
        with get_postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select 1")
                cursor.fetchone()
    except Exception:
        return _result(
            "chat_storage",
            "Conversation storage",
            "warning",
            "PostgreSQL is configured but could not be reached.",
            critical=False,
            started_at=started_at,
        )

    return _result(
        "chat_storage",
        "Conversation storage",
        "passed",
        "PostgreSQL conversation storage is reachable.",
        critical=False,
        started_at=started_at,
    )


def check_langfuse() -> dict[str, Any]:
    from agents.tracing import langfuse_client, langfuse_enabled

    started_at = time.perf_counter()
    tracing_requested = (os.getenv("ENABLE_LANGFUSE_TRACING") or "").lower() in {
        "1",
        "true",
        "yes",
    }

    if not tracing_requested:
        return _result(
            "langfuse",
            "Langfuse observability",
            "warning",
            "Langfuse tracing is disabled.",
            critical=False,
            started_at=started_at,
        )

    if not langfuse_enabled() or langfuse_client is None:
        return _result(
            "langfuse",
            "Langfuse observability",
            "warning",
            "Langfuse is enabled but its client could not be initialized.",
            critical=False,
            started_at=started_at,
        )

    try:
        auth_check = getattr(langfuse_client, "auth_check", None)
        authenticated = bool(auth_check()) if callable(auth_check) else True
    except Exception:
        authenticated = False

    return _result(
        "langfuse",
        "Langfuse observability",
        "passed" if authenticated else "warning",
        "Langfuse is connected." if authenticated else "Langfuse credentials could not be verified.",
        critical=False,
        started_at=started_at,
    )


def check_local_knowledge() -> dict[str, Any]:
    from agents.list_of_agent import ALL_PREDIKLY_USECASES_PATH, load_all_predikly_usecases
    from hybrid_retrieval import BM25_STATE_PATH

    started_at = time.perf_counter()

    try:
        usecases = load_all_predikly_usecases()
        sparse_state_ready = Path(BM25_STATE_PATH).is_file()
    except Exception:
        usecases = ()
        sparse_state_ready = False

    if not ALL_PREDIKLY_USECASES_PATH.is_file() or not usecases or not sparse_state_ready:
        return _result(
            "local_knowledge",
            "Local knowledge assets",
            "failed",
            "The use-case catalog or sparse retrieval state is missing.",
            critical=True,
            started_at=started_at,
        )

    return _result(
        "local_knowledge",
        "Local knowledge assets",
        "passed",
        f"Catalog and sparse retrieval state are ready ({len(usecases)} use cases).",
        critical=True,
        started_at=started_at,
    )


def run_readiness_checks() -> dict[str, Any]:
    started_at = time.perf_counter()
    checks: list[dict[str, Any]] = [check_backend()]
    independent_checks: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("qdrant", check_qdrant),
        ("gemini", check_gemini),
        ("embeddings", check_embeddings),
        ("chat_storage", check_chat_storage),
        ("langfuse", check_langfuse),
        ("local_knowledge", check_local_knowledge),
    )

    with ThreadPoolExecutor(max_workers=len(independent_checks)) as executor:
        futures = {executor.submit(check): check_id for check_id, check in independent_checks}

        for future in as_completed(futures):
            try:
                result = future.result()
                checks.extend(result if isinstance(result, list) else [result])
            except Exception:
                checks.append(
                    _result(
                        futures[future],
                        "Technology check",
                        "failed",
                        "An unexpected readiness check error occurred.",
                        critical=True,
                        started_at=started_at,
                    )
                )

    order = {
        "backend": 0,
        "qdrant": 1,
        "qdrant_collections": 2,
        "gemini": 3,
        "embeddings": 4,
        "local_knowledge": 5,
        "chat_storage": 6,
        "langfuse": 7,
    }
    checks.sort(key=lambda item: order.get(item["id"], 99))
    blocking_failures = [
        check for check in checks if check["critical"] and check["status"] == "failed"
    ]
    warnings = [check for check in checks if check["status"] == "warning"]

    return {
        "status": "ready" if not blocking_failures else "blocked",
        "can_continue": not blocking_failures,
        "checks": checks,
        "summary": (
            "All critical systems are ready."
            if not blocking_failures
            else f"{len(blocking_failures)} critical system check(s) failed."
        ),
        "warning_count": len(warnings),
        "duration_ms": round((time.perf_counter() - started_at) * 1000),
    }
