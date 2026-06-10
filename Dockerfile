FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

COPY app.py hybrid_retrieval.py ./
COPY data/bm25_sparse_encoder.json ./data/bm25_sparse_encoder.json
COPY agents ./agents
COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
