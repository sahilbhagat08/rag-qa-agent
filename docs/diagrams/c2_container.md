# C4 Level 2 — Container Diagram

This diagram shows the deployable containers within the `rag-qa-agent` system boundary.

```mermaid
C4Container
  title Container Diagram — RAG QA Agent

  Person(user, "End User", "Sends queries and uploads documents.")

  System_Boundary(sys, "RAG QA Agent System") {
    Container(api, "FastAPI Application", "Python 3.11 / uvicorn", "Hosts REST + SSE endpoints. Orchestrates the LangChain agent, RAG pipeline, and streaming.")
    ContainerDb(chroma, "Chroma Vector Store", "chromadb (embedded)", "Persists document chunk embeddings to disk. Queried via similarity search.")
    Container(embedder, "SentenceTransformer", "sentence-transformers (in-process)", "Encodes text into dense vectors using all-MiniLM-L6-v2. Loaded once at startup.")
    ContainerDb(redisDb, "Redis", "Redis 7 / Docker or managed", "Stores per-session conversation history. Keys expire after 24h TTL.")
  }

  System_Ext(anthropicApi, "Anthropic API", "External LLM service (Claude).")
  SystemDb_Ext(fileStorage, "File Storage", "Local disk / object store for source documents.")

  Rel(user, api, "REST/SSE requests", "HTTPS :8000")
  Rel(api, chroma, "Similarity search + upsert", "In-process function calls")
  Rel(api, embedder, "Embed queries and documents", "In-process function calls")
  Rel(api, redisDb, "Read/write chat history", "TCP :6379")
  Rel(api, anthropicApi, "LLM completion requests", "HTTPS")
  Rel(api, fileStorage, "Read documents at ingest time", "File I/O")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Container Descriptions

| Container | Technology | Responsibility |
|---|---|---|
| **FastAPI Application** | Python 3.11, uvicorn | REST API, agent orchestration, document ingestion, SSE streaming |
| **Chroma Vector Store** | chromadb (embedded) | Persist and query document chunk embeddings |
| **SentenceTransformer** | sentence-transformers | In-process embedding model; no external API key required |
| **Redis** | Redis 7 | Session-scoped conversation memory; auto-expires after TTL |

## Deployment Notes

- **Development**: all containers run via `docker-compose up`. Chroma data and the Redis append-only log are persisted via named volumes.
- **Production**: The FastAPI app can be scaled horizontally (multiple replicas) because all state lives in Redis (session memory) and Chroma (persisted to a shared volume or migrated to a managed vector DB like Qdrant).
