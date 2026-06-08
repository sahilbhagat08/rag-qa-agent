# RAG QA Agent — Architecture Documentation

## Overview

`rag-qa-agent` is a production-ready Document Q&A system that combines Retrieval-Augmented Generation (RAG) with agent-based tool use. Users can upload documents, ask natural language questions, and receive accurate, cited answers streamed in real time.

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-5`) | Reasoning, tool calling, answer generation |
| Orchestration | LangChain `AgentExecutor` | Tool dispatch, agent loop, memory integration |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local vector embeddings (no external API) |
| Vector Store | ChromaDB (embedded, persisted) | Similarity search over document chunks |
| Session Memory | Redis 7 + `RedisChatMessageHistory` | Per-session conversation history with TTL |
| API | FastAPI + uvicorn | REST + SSE streaming endpoints |
| Containerisation | Docker + docker-compose | App + Redis services |

## C4 Diagram Levels

| Level | File | Scope |
|---|---|---|
| L1 — System Context | [c1_system_context.md](diagrams/c1_system_context.md) | System and external actors |
| L2 — Container | [c2_container.md](diagrams/c2_container.md) | Deployable units |
| L3 — Component | [c3_component.md](diagrams/c3_component.md) | Internals of the FastAPI app |
| L4 — Code | [c4_code.md](diagrams/c4_code.md) | Key class interactions and data flow |

## Architecture Decisions

See [adr/001-architecture.md](adr/001-architecture.md) for the full decision record.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Synchronous Q&A (full response) |
| `POST` | `/chat/stream` | Streaming Q&A via SSE |
| `POST` | `/documents/ingest` | Upload and ingest a document |
| `GET` | `/documents` | List ingested document sources |
| `GET` | `/health` | Service health check |

## Data Flow

```
User → POST /chat/stream
         │
         ▼
    AgentExecutor (Claude + tools)
         │
         ├─► retrieve_documents ──► ChromaDB ──► SentenceTransformer embeddings
         ├─► summarize_document ──► Anthropic API (direct)
         └─► calculator         ──► safe AST eval (local)
         │
         ▼
    Redis (read/write session history)
         │
         ▼
    SSE stream → client
```

## Getting Started

```bash
# 1. Copy and fill in env vars
cp .env.example .env

# 2. Start services
docker-compose up --build

# 3. Ingest documents
python scripts/ingest_docs.py --dir ./data

# 4. Query
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise the key findings", "session_id": "demo-1"}'
```
