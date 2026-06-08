# C4 Level 1 — System Context

This diagram shows the `rag-qa-agent` system and its relationships with external actors and systems.

```mermaid
C4Context
  title System Context — RAG QA Agent

  Person(user, "End User", "Asks questions about documents via REST API or a frontend client.")

  System(ragAgent, "RAG QA Agent", "Document Q&A system. Ingests documents, answers questions with citations, streams responses.")

  System_Ext(anthropicApi, "Anthropic API", "Claude LLM (claude-sonnet-4-5). Provides reasoning, tool calling, and summarization.")
  System_Ext(redis, "Redis", "Managed or self-hosted Redis instance. Stores per-session conversation history with TTL.")
  SystemDb_Ext(fileStorage, "File Storage", "Local filesystem or object storage (S3/GCS). Source documents: PDF, TXT, Markdown.")

  Rel(user, ragAgent, "Sends questions and uploads documents", "HTTPS/REST")
  Rel(ragAgent, anthropicApi, "LLM inference and tool call results", "HTTPS")
  Rel(ragAgent, redis, "Read/write session memory", "TCP Redis protocol")
  Rel(ragAgent, fileStorage, "Reads source documents for ingestion", "File I/O")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Key Relationships

| From | To | Purpose |
|---|---|---|
| User | RAG QA Agent | Submit questions, upload documents |
| RAG QA Agent | Anthropic API | LLM calls (streaming + sync) |
| RAG QA Agent | Redis | Session memory persistence |
| RAG QA Agent | File Storage | Document ingestion source |

## External Dependencies

- **Anthropic API**: Only external paid API. Requires `ANTHROPIC_API_KEY`. All LLM inference (question answering, summarization) flows through this.
- **Redis**: Can be the bundled Docker container (development) or a managed service like AWS ElastiCache (production).
- **File Storage**: Documents are read from disk at ingest time. The Chroma vector store persists embeddings separately.
