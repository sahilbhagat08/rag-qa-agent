# ADR-001: Architecture Decisions for RAG QA Agent

**Date:** 2026-06-08  
**Status:** Accepted

---

## Context

This document records the key architectural decisions made when designing the `rag-qa-agent` system. Each decision includes the context, the options considered, the choice made, and the rationale.

---

## Decision 1: Native Tool Calling over ReAct

**Context:** LangChain supports two agent paradigms for tool use — ReAct (text-based reasoning/acting loop parsed from LLM output) and native tool calling (structured tool calls via the model's built-in function calling API).

**Options:**
1. ReAct — parse `Thought/Action/Observation` strings from the LLM response
2. Native tool calling — use Claude's structured `tool_use` blocks

**Decision:** Native tool calling via `create_tool_calling_agent` + `bind_tools`.

**Rationale:**
- Claude 3.5 Sonnet has first-class tool calling support; ReAct parsing is fragile and error-prone.
- Structured tool calls are typed and validated — no regex parsing of LLM output strings.
- Lower latency: the model returns a structured tool call object directly, not a string that requires parsing.
- Better observability: tool invocations are discrete events in `astream_events`.

---

## Decision 2: Retrieval as an Agent Tool, not a Fixed Pipeline Step

**Context:** Classic RAG pipelines always retrieve before passing to the LLM. Agent-based RAG lets the LLM decide whether and when to retrieve.

**Options:**
1. Fixed pipeline: always retrieve top-k chunks, prepend to prompt, call LLM once.
2. Tool-based retrieval: expose retrieval as a tool the agent can call zero or more times.

**Decision:** Tool-based retrieval.

**Rationale:**
- Conversational Q&A often has follow-up questions that don't need new retrieval.
- The agent can issue multiple targeted retrieval calls with different queries when needed.
- Avoids polluting the context window with irrelevant chunks on non-document turns.
- Enables multi-hop retrieval (retrieve → reason → retrieve again with refined query).

**Trade-off:** Slightly more LLM calls per turn compared to a fixed pipeline. Acceptable given the accuracy improvement.

---

## Decision 3: SentenceTransformers over OpenAI Embeddings

**Context:** An embedding model is required to convert document chunks and queries into vectors for similarity search.

**Options:**
1. OpenAI `text-embedding-3-small` — high quality, requires additional API key, adds cost
2. `sentence-transformers/all-MiniLM-L6-v2` — open-source, runs locally, no API key, fast

**Decision:** `sentence-transformers/all-MiniLM-L6-v2`.

**Rationale:**
- Eliminates a second external API dependency.
- Model runs entirely on CPU; no GPU required for typical document sizes.
- `all-MiniLM-L6-v2` has strong benchmark performance for its size.
- Model is cached after first download; no per-call cost.
- Easy swap: the `SentenceTransformerEmbeddings` class implements `langchain_core.embeddings.Embeddings` — switching to a different model is a one-line config change.

**Trade-off:** Embedding quality is slightly below state-of-the-art paid models for some domain-specific corpora. Mitigated by `CHUNK_SIZE` and `TOP_K_RESULTS` tuning.

---

## Decision 4: Redis-backed Session Memory over In-Memory Dict

**Context:** The agent needs conversational memory scoped per user session.

**Options:**
1. In-memory Python dict — simplest, but lost on restart, not shareable across replicas
2. Redis-backed `RedisChatMessageHistory` — persistent, TTL-managed, horizontally scalable

**Decision:** Redis-backed memory.

**Rationale:**
- Conversations survive app restarts and deployments.
- Sessions auto-expire via Redis TTL (default 24h) — no manual cleanup required.
- Supports horizontal scaling: multiple app replicas share the same Redis.
- `RedisChatMessageHistory` is a drop-in LangChain integration.

**Trade-off:** Requires a running Redis instance. Mitigated by providing a Redis container in `docker-compose.yml`.

---

## Decision 5: Chroma Embedded over Managed Vector DB

**Context:** A vector store is needed to persist document embeddings.

**Options:**
1. Chroma (embedded, local) — zero infrastructure, persists to disk
2. Pinecone / Qdrant Cloud — managed, scalable, additional cost and config
3. pgvector — good for Postgres shops, but adds DB dependency

**Decision:** Chroma embedded, persisted to a Docker volume.

**Rationale:**
- Zero additional infrastructure for development and small-scale production.
- Chroma's embedded mode persists to disk automatically — no data loss on restart.
- The `ChromaRetriever` wrapper isolates the rest of the codebase from the specific vector store implementation.

**Migration path:** Replacing Chroma with Qdrant or Pinecone requires only changing `rag/retriever.py` and `rag/embeddings.py` — no changes to tools, agent, or API layers.

---

## Decision 6: FastAPI SSE over WebSockets for Streaming

**Context:** Real-time token streaming to the client.

**Options:**
1. WebSockets — bidirectional, more complex client handling
2. Server-Sent Events (SSE) via `StreamingResponse` — unidirectional server→client, simpler

**Decision:** SSE via FastAPI `StreamingResponse`.

**Rationale:**
- Chat streaming is inherently unidirectional (server pushes tokens).
- SSE is simpler: works with `EventSource` in browsers and `curl -N` in terminals.
- No persistent connection management overhead.
- Compatible with all HTTP/1.1 proxies and load balancers.
