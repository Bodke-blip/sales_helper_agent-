# Predikly Sales Helper Agent

An internal agentic AI sales assistant for finding and reusing Predikly customer use cases and case-study knowledge. The app lets a user ask sales questions, retrieve grounded internal context from Qdrant, and generate concise answers or sales drafts through a controlled LangGraph workflow.

## What It Does

- Answers questions about previous Predikly customer use cases.
- Retrieves internal case-study context from a Qdrant vector database.
- Supports count-style queries such as how many use cases exist for a customer.
- Drafts short sales content using retrieved internal context.
- Applies input guardrails for prompt injection, unsafe requests, and credential extraction.
- Evaluates retrieved context before returning a final response.
- Returns source metadata, trace IDs, cache status, model details, and workflow timings.

## Architecture

The runtime flow is:

```text
User / Web UI / API
  -> FastAPI app
  -> LangGraph workflow
  -> Input guardrail
  -> Main orchestrator
  -> Knowledge retrieval agent
  -> Evaluation agent
  -> Response composer or fallback handler
  -> Output guardrail
  -> Final response
```

The ingestion flow is:

```text
Google Drive + reference Excel/PPT files
  -> metadata extraction
  -> dense embeddings with sentence-transformers/all-MiniLM-L6-v2
  -> BM25 sparse vectors
  -> Qdrant hybrid collection
  -> customer manifest
```

## Tech Stack

- Python
- FastAPI
- LangGraph
- LangChain
- Qdrant
- Google Gemini
- Ollama fallback
- HuggingFace sentence-transformers
- Google Drive API
- Optional Langfuse tracing

## Project Structure

```text
.
├── app.py                         # FastAPI app, UI, and API endpoints
├── requirements.txt               # Python dependencies
├── agents/
│   ├── graph.py                   # LangGraph workflow
│   ├── orchestrator_agent.py      # Intent routing and final response composition
│   ├── knowledge_retrieval_agent.py
│   ├── eval_agent.py
│   ├── evaluation.py
│   ├── guardrails.py
│   ├── fallback.py
│   ├── llm.py
│   ├── state.py
│   └── tracing.py
└── data/                          # Local generated data, ignored by Git
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
touch .env
```

Add the required configuration:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
HYBRID_QDRANT_COLLECTION_NAME=predikly_hybrid_search_data_v2
HYBRID_QDRANT_FALLBACK_COLLECTION_NAME=predikly_hybrid_serch_data

GEMINI_API_KEY=your_gemini_api_key

OLLAMA_BASE_URL=http://127.0.0.1:11434
REQUEST_FALLBACK_AFTER_SECONDS=999
PRIMARY_LLM_TIMEOUT_SECONDS=12
OLLAMA_TIMEOUT_SECONDS=10

VISION_MODEL=qwen2.5vl
VISION_PROVIDER=ollama
VISION_MAX_TOKENS=700
VISION_TIMEOUT_SECONDS=120
VISION_MAX_IMAGE_SIDE=1024
VISION_IMAGE_JPEG_QUALITY=82

ENABLE_LANGFUSE_TRACING=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

GOOGLE_CLIENT_SECRET_FILE=client_secret.json
REFERENCE_EXCEL_NAME=reference_points.xlsx
```

Optional retrieval settings:

```env
QDRANT_TOP_K=15
QDRANT_HYBRID_PREFETCH_LIMIT=50
QDRANT_DOCUMENT_EXPANSION_SOURCE_LIMIT=3
QDRANT_DOCUMENT_EXPANSION_CHUNK_LIMIT=40
QDRANT_MAX_CONTEXT_ITEMS=30
QDRANT_TIMEOUT_SECONDS=8
ENSURE_PAYLOAD_INDEXES_ON_QUERY=false
MIN_RETRIEVAL_RESULTS=1
QDRANT_UPSERT_BATCH_SIZE=50
BM25_STATE_PATH=data/bm25_sparse_encoder.json
RETRIEVAL_CACHE_TTL_SECONDS=300
RETRIEVAL_CACHE_MAX_ENTRIES=256
MAX_CHAT_HISTORY_TURNS=6
CHAT_HISTORY_LIMIT=40
CHAT_DB_URL=postgresql://USER:PASSWORD@HOST:5432/predikly_sales_helper?sslmode=require
```

## Secrets and Local Files

Do not commit local secrets or OAuth files. The `.gitignore` excludes:

- `.env`
- `google_token.json`
- `client_secret_*.json`
- `.venv/`
- `__pycache__/`
- `data/customer_manifest.json`

For Google Drive ingestion, keep the OAuth client secret JSON locally in the project root and set `GOOGLE_CLIENT_SECRET_FILE` to its filename. The first ingestion run may create `google_token.json`.

## Run the App

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open the web UI:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Query API

Example request:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List the relevant case studies for a healthcare client.",
    "use_gemini_llm": true,
    "use_local_llm": true,
    "verbose": true
  }'
```

Useful options:

- `use_gemini_llm`: enables Gemini.
- `use_local_llm`: enables local Ollama fallback.
- `verbose`: includes agent trace details.

## Cache Endpoints

Check retrieval cache:

```bash
curl http://127.0.0.1:8000/cache/status
```

Clear retrieval cache:

```bash
curl -X DELETE http://127.0.0.1:8000/cache
```

## Ingestion and Qdrant Upload

Ingestion, upload, and evaluation dataset tooling is local-only and intentionally excluded from this public repository because it can reference internal customer material. The hosted app expects Qdrant collections and sparse encoder state to be prepared through the private/local ingestion flow.

## Local Ollama Fallback

The fallback model is configured as:

```text
llama3.2:3b
```

Install and run Ollama, then pull the model:

```bash
ollama pull llama3.2:3b
```

The app checks Ollama at:

```text
http://127.0.0.1:11434
```

## Notes

- The default primary model is `gemini-2.5-flash`.
- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- The default hybrid Qdrant collection is `predikly_hybrid_search_data_v2`.
- The default fallback hybrid Qdrant collection is `predikly_hybrid_serch_data`.
- Retrieval checks the main hybrid collection first and only tries the fallback collection when the main collection returns fewer than `MIN_RETRIEVAL_RESULTS`.
- This project is currently shaped for internal development/pilot use. Add authentication, production secret management, monitoring, CI checks, and a formal security review before production deployment.
