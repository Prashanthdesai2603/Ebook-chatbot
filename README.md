# Offline eBook Chatbot (RAG + LoRA)

A strictly offline, privacy-first AI chatbot that answers questions based **only** on a provided PDF ebook.
It uses **Retrieval Augmented Generation (RAG)** for factual accuracy and **LoRA (Low-Rank Adaptation)** for response styling.

---

## System Requirements

| | Requirement |
|---|---|
| **OS** | Windows 11 / Linux (Ubuntu 22.04+ recommended for VPS) |
| **CPU** | Modern multi-core (16 GB RAM recommended) |
| **GPU** | Not required — runs on CPU |
| **Python** | 3.10+ |
| **Node.js** | 20.19+ |
| **Docker** | Docker Engine + Compose plugin (for containerised setup) |

---

## Repository Structure

```
EBOOKCHATBOT/
├── .dockerignore
├── .gitignore
├── docker-compose.dev.yml     # Dev stack (hot-reload)
├── docker-compose.prod.yml    # Prod stack (detached, port 8088)
├── Makefile                   # Convenience commands
│
├── backend/
│   ├── Dockerfile             # Multi-stage: dev + prod targets
│   ├── .env                   # Secrets — never commit this
│   ├── .env.example           # Template — safe to commit
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       └── ingest.py
│
├── ai/                        # RAG pipeline — imported by backend
│
├── data/                      # Persisted on host, mounted as Docker volume
│   ├── ebooks/                # Place your PDF here
│   └── vectorstore/
│       └── chroma.sqlite3
│
├── frontend/
│   ├── Dockerfile             # Multi-stage: dev + builder + prod (Nginx)
│   ├── vite.config.js         # /api proxy for dev
│   └── src/
│
├── docker/
│   └── nginx/
│       ├── nginx.conf                # Container Nginx config (prod)
│       └── vps-hestia.nginx.conf     # VPS host Nginx reference
│
└── scripts/
    ├── dev.sh
    └── prod.sh
```

---

## Environment Variables

Copy the template and fill in your values before starting:

```bash
cp backend/.env.example backend/.env
```

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `VECTOR_DB_PATH` | ChromaDB SQLite path inside container | `/app/data/vectorstore/chroma.sqlite3` |
| `TOP_K` | Number of chunks retrieved per query | `10` |

> **Never commit `backend/.env`** — it is listed in `.gitignore`.

---

## Data Preparation

### 1. Add your PDF

Place your ebook inside `data/ebooks/`.

### 2. Run ingestion

```bash
# Without Docker
cd backend
python app/ingest.py

# With Docker (dev stack running)
docker compose -f docker-compose.dev.yml exec backend python backend/app/ingest.py
```

This creates the ChromaDB vectorstore at `data/vectorstore/chroma.sqlite3`.

---

## LoRA Training *(Optional)*

LoRA customises the **style** of responses — not the facts (those come from RAG).

```bash
# 1. Prepare dataset
python lora/data_prep.py
# Edit the generated dataset.jsonl with your stylistic examples

# 2. Train the adapter
python lora/train.py
```

The adapter loads automatically on the next backend start.

---

## Running — Docker *(Recommended)*

### Development

Hot-reload for both frontend (Vite HMR) and backend (uvicorn `--reload`).
No rebuild needed for code changes.

```bash
# Start
make dev

# Stop
make down-dev

# Follow logs
make logs-dev
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

> On first start, Docker downloads the base images and installs all pip deps (including torch).
> This can take **10–20 minutes**. Subsequent starts use layer cache and are much faster.

---

### Production

Builds frozen images, no source mounts. Container Nginx listens on port `8088`;
your host-level Nginx reverse proxies to it.

```bash
# Build and start (detached)
make prod

# Stop
make down-prod

# Follow logs
make logs-prod
```

**VPS / Hestia setup** — set the reverse proxy URL in the panel to:

```
http://127.0.0.1:8088
```

Then block port 8088 from public access:

```bash
sudo ufw deny 8088
sudo ufw allow 80,443
```

| Service | URL |
|---|---|
| Application | https://chatbot.symphonytech.com |
| Container Nginx | http://127.0.0.1:8088 (VPS localhost only) |

---

### Makefile Reference

| Command | Action |
|---|---|
| `make dev` | Build and start dev stack |
| `make down-dev` | Stop dev stack |
| `make logs-dev` | Follow dev logs |
| `make prod` | Build and start prod stack (detached) |
| `make down-prod` | Stop prod stack |
| `make logs-prod` | Follow prod logs |
| `make ps` | Show running containers |
| `make clean` | Prune stopped containers and dangling images |

---

## Running — Without Docker

### Backend

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

Server starts at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Client starts at `http://localhost:5173`.

---

## Usage

- Open the frontend URL.
- Toggle between **Short** (concise) and **Detailed** (structured) response modes.
- Ask questions about the ebook content.
- If the answer isn't in the book, the bot responds: *"I don't know based on the ebook."*

---

## Changes from Original Setup

| Area | Change |
|---|---|
| `backend/app/main.py` | Removed `sys.path` manipulation — resolved via `PYTHONPATH=/app` in Docker |
| `frontend/src/App.jsx` | Replaced `VITE_BACKEND_URL` + hardcoded localhost with relative `/api/chat` |
| `frontend/vite.config.js` | Added `host: 0.0.0.0`, port, and `/api` proxy with 503 fallback |
| `docker/nginx/nginx.conf` | Container Nginx: 900s timeouts, real IP forwarding, JSON error fallback |
| Proxy routing | All FE→BE requests go through `/api` — stripped before hitting FastAPI in both dev and prod |
| Port | Prod exposed on `8088` for VPS reverse proxy compatibility |
| Vectorstore | Mounted as a named volume bound to `./data` — survives container rebuilds |
