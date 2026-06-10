import os
from functools import lru_cache
from typing import Any

import requests


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL") or os.getenv(
    "EMBEDDING_MODEL",
    DEFAULT_EMBEDDING_MODEL,
)
HF_EMBEDDING_API_URL = os.getenv(
    "HF_EMBEDDING_API_URL",
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction",
)
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "20"))


class EmbeddingGatewayError(RuntimeError):
    pass


def clean_env_value(value: str | None) -> str:
    return (value or "").strip().strip("\"'")


def get_hf_token() -> str:
    for env_name in (
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_HUB_TOKEN1",
        "HF_HUB_TOKEN2",
        "HF_HUB_TOKEN3",
    ):
        token = clean_env_value(os.getenv(env_name))

        if token:
            return token

    return ""


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int | float) for item in value)


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []

    dimensions = len(vectors[0])
    if dimensions == 0:
        return []

    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]


def parse_embedding_response(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        if "error" in payload:
            raise EmbeddingGatewayError(str(payload["error"]))
        if "embedding" in payload:
            return parse_embedding_response(payload["embedding"])
        if "embeddings" in payload:
            return parse_embedding_response(payload["embeddings"])

    if _is_number_list(payload):
        return [float(value) for value in payload]

    if isinstance(payload, list) and payload:
        if len(payload) == 1:
            return parse_embedding_response(payload[0])

        if all(_is_number_list(item) for item in payload):
            return _mean_pool([[float(value) for value in item] for item in payload])

    raise EmbeddingGatewayError("Hosted embedding response did not contain a vector.")


class HostedMiniLMEmbeddings:
    def __init__(
        self,
        *,
        api_url: str = HF_EMBEDDING_API_URL,
        token: str | None = None,
        timeout_seconds: float = EMBEDDING_TIMEOUT_SECONDS,
    ) -> None:
        self.api_url = api_url
        self.token = clean_env_value(token) or get_hf_token()
        self.timeout_seconds = timeout_seconds

    def embed_query(self, text: str) -> list[float]:
        if not self.token:
            raise EmbeddingGatewayError(
                "HF_TOKEN or HF_HUB_TOKEN must be configured for hosted embeddings."
            )

        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "inputs": text,
                    "options": {"wait_for_model": True},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise EmbeddingGatewayError(f"Hosted embedding request failed: {error}") from error

        embedding = parse_embedding_response(response.json())

        if not embedding:
            raise EmbeddingGatewayError("Hosted embedding response returned an empty vector.")

        return embedding


@lru_cache(maxsize=1)
def get_embeddings() -> HostedMiniLMEmbeddings:
    return HostedMiniLMEmbeddings()
