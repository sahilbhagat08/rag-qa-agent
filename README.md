# RAG QA Agent

A production-ready **Document Q&A system** combining Retrieval-Augmented Generation (RAG) with agent-based tool use. Upload documents, ask natural language questions, and receive accurate, cited answers streamed in real time.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-purple.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker](#docker)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Running Tests](#running-tests)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## Overview

`rag-qa-agent` lets you ingest a corpus of documents (PDF, TXT, Markdown) and query them through a conversational agent. The agent uses **Anthropic Claude** as the reasoning engine and decides autonomously when to retrieve, summarize, or compute — rather than blindly retrieving on every turn.

```
User question
    → AgentExecutor (Claude + tools)
        → retrieve_documents  → ChromaDB (SentenceTransformer embeddings)
        → summarize_document  → Claude (direct)
        → calculator          → safe AST eval (local)
    → Redis (session memory)
    → Streamed answer with citations
```

---

## Features

- **Streaming responses** — token-by-token output via Server-Sent Events (SSE)
- **Tool-based RAG** — agent decides when to retrieve, enabling multi-hop and conversational queries
- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2`, no external embedding API required
- **Redis session memory** — per-session conversation history with configurable TTL
- **Persistent vector store** — ChromaDB persisted to disk, survives restarts
- **Safe calculator** — AST-based math evaluator with a strict whitelist (no `eval`)
- **Document ingestion** — PDF, TXT, and Markdown via REST upload or CLI bulk script
- **Structured logging** — `structlog` JSON logs throughout
- **Typed throughout** — Pydantic v2 request/response models
- **Fully containerised** — Docker + docker-compose for app and Redis

---

## Architecture

Full C4 architecture documentation lives in [`docs/`](docs/architecture.md):

| Level | Diagram |
|---|---|
| L1 — System Context | [c1_system_context.md](docs/diagrams/c1_system_context.md) |
| L2 — Container | [c2_container.md](docs/diagrams/c2_container.md) |
| L3 — Component | [c3_component.md](docs/diagrams/c3_component.md) |
| L4 — Code | [c4_code.md](docs/diagrams/c4_code.md) |

Architecture decisions are documented in [docs/adr/001-architecture.md](docs/adr/001-architecture.md).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-5`) |
| Orchestration | LangChain `AgentExecutor` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (embedded, persisted) |
| Session Memory | Redis 7 + `RedisChatMessageHistory` |
| API | FastAPI + uvicorn |
| Logging | structlog (JSON) |
| Config | pydantic-settings |
| Containerisation | Docker + docker-compose |

---

## Project Structure

```
rag-qa-agent/
├── app/
│   ├── main.py                      # FastAPI app factory + lifespan
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py              # POST /chat, POST /chat/stream (SSE)
│   │   │   └── documents.py         # POST /documents/ingest, GET /documents
│   │   └── schemas.py               # Pydantic v2 request/response models
│   ├── agent/
│   │   ├── agent.py                 # create_agent() factory
│   │   ├── memory.py                # Redis-backed session memory
│   │   ├── prompts.py               # System prompt templates
│   │   └── tools/
│   │       ├── retriever.py         # retrieve_documents tool
│   │       ├── summarizer.py        # summarize_document tool
│   │       └── calculator.py        # calculator tool (safe AST eval)
│   ├── rag/
│   │   ├── ingestion.py             # Load → chunk → embed → store
│   │   ├── retriever.py             # ChromaDB wrapper
│   │   └── embeddings.py            # SentenceTransformer adapter
│   └── core/
│       ├── config.py                # pydantic-settings env config
│       └── logging.py               # structlog setup
├── tests/
│   ├── unit/                        # Tool, ingestion, and agent unit tests
│   └── integration/                 # FastAPI endpoint tests
├── scripts/
│   └── ingest_docs.py               # CLI bulk document ingestion
├── docs/
│   ├── architecture.md
│   ├── diagrams/                    # C1–C4 Mermaid diagrams
│   └── adr/                         # Architecture Decision Records
├── data/                            # Drop source documents here
├── chroma_db/                       # Persisted vector store (gitignored)
├── agents.md                        # Git workflow guidelines
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & docker-compose (recommended)
- An [Anthropic API key](https://console.anthropic.com/)

### Local Development

```bash
# 1. Clone the repo
git clone <repo-url>
cd rag-qa-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 5. Start Redis (requires Docker)
docker run -d -p 6379:6379 redis:7-alpine

# 6. Run the API server
uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 2. Start all services (app + Redis)
docker-compose up --build

# App available at: http://localhost:8000
# API docs at:      http://localhost:8000/docs
```

### Ingest Documents

```bash
# Single file
python scripts/ingest_docs.py --file ./data/report.pdf

# Entire directory
python scripts/ingest_docs.py --dir ./data

# Filtered by pattern
python scripts/ingest_docs.py --dir ./data --pattern "*.pdf"
```

Supported formats: `.pdf`, `.txt`, `.md`, `.mdx`, `.rst`

---

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Claude model ID |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_SESSION_TTL_SECONDS` | `86400` | Session memory TTL (24h) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Chroma persistence directory |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHUNK_SIZE` | `512` | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (tokens) |
| `TOP_K_RESULTS` | `5` | Retrieval results per query |
| `MEMORY_WINDOW_K` | `10` | Conversation turns kept in memory |
| `AGENT_MAX_ITERATIONS` | `10` | Max agent tool-call iterations |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## API Reference

Interactive docs available at `http://localhost:8000/docs` when the server is running.

### `POST /chat/stream`
Stream the agent response token-by-token (SSE).

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings?", "session_id": "user-123"}'
```

**Response:** `text/event-stream`
```
data: {"type": "token", "content": "The", "session_id": "user-123"}
data: {"type": "token", "content": " key", "session_id": "user-123"}
...
data: {"type": "done", "content": "", "session_id": "user-123"}
```

### `POST /chat`
Non-streaming — wait for the full answer.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise the report", "session_id": "user-123"}'
```

### `POST /documents/ingest`
Upload and ingest a document.

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@./data/report.pdf"
```

**Response:**
```json
{
  "status": "success",
  "source": "report.pdf",
  "doc_id": "a1b2c3d4",
  "chunks_stored": 42,
  "pages_loaded": 10
}
```

### `GET /documents`
List all ingested document sources.

```bash
curl http://localhost:8000/documents
```

### `GET /health`
Service health check.

```bash
curl http://localhost:8000/health
```

---

## Usage Examples

### Python client (streaming)

```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/chat/stream",
    json={"message": "What is the report about?", "session_id": "demo"},
    timeout=60,
) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            import json
            event = json.loads(line[6:])
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
```

### Agent tools in action

The agent autonomously selects tools based on the question:

| User question | Tool(s) called |
|---|---|
| "What does section 3 say about risk?" | `retrieve_documents` |
| "Summarise the methodology" | `retrieve_documents` → `summarize_document` |
| "What is 15% of the $4.2M budget?" | `calculator` |
| "Hello, how are you?" | *(none — direct response)* |

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest --cov=app --cov-report=term-missing
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System overview and data flow |
| [docs/diagrams/c1_system_context.md](docs/diagrams/c1_system_context.md) | C4 Level 1 — System Context |
| [docs/diagrams/c2_container.md](docs/diagrams/c2_container.md) | C4 Level 2 — Containers |
| [docs/diagrams/c3_component.md](docs/diagrams/c3_component.md) | C4 Level 3 — Components |
| [docs/diagrams/c4_code.md](docs/diagrams/c4_code.md) | C4 Level 4 — Code (class + sequence diagrams) |
| [docs/adr/001-architecture.md](docs/adr/001-architecture.md) | Architecture Decision Records |
| [agents.md](agents.md) | Git workflow guidelines |

---

## Contributing

1. Follow the Git workflow in [agents.md](agents.md)
2. Branch naming: `feature/`, `bugfix/`, `refactor/`
3. Run tests and lint before committing
4. Use conventional commit messages (`feat:`, `fix:`, `refactor:`, etc.)
5. Never push without approval

---

## License

MIT License. See [LICENSE](LICENSE) for details.
