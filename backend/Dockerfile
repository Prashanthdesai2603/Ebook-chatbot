FROM python:3.11-slim AS base

# System deps required by torch / sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first — avoids resolver edge-cases on slim images
RUN pip install --no-cache-dir --upgrade pip

# Deps first — maximises layer cache re-use
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY backend/ ./backend/
COPY backend/ai/       ./ai/

# PYTHONPATH ensures `from ai.rag_pipeline import ...` resolves without any
# sys.path manipulation in application code.
ENV PYTHONPATH=/app

# data/ is NOT baked in — mounted as a named volume at runtime.
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────────────────
FROM base AS dev
# --reload watches both backend/ and ai/ for changes
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/backend", \
     "--reload-dir", "/app/ai"]

# ─────────────────────────────────────────────────────────────────────────────
FROM base AS prod
# 2 workers: intentionally conservative for low-RAM VPS.
# Increase to (2 * CPU cores + 1) if the host has headroom.
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2"]
