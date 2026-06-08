# C4 Level 4 — Code Diagrams

This level documents the key class structures and runtime data flows for the two most complex subsystems.

---

## Class Diagram: Core Classes

```mermaid
classDiagram
  class Settings {
    +str ANTHROPIC_API_KEY
    +str ANTHROPIC_MODEL
    +str REDIS_URL
    +int REDIS_SESSION_TTL_SECONDS
    +str CHROMA_PERSIST_DIR
    +str EMBEDDING_MODEL
    +int CHUNK_SIZE
    +int CHUNK_OVERLAP
    +int TOP_K_RESULTS
    +int MEMORY_WINDOW_K
    +int AGENT_MAX_ITERATIONS
  }

  class SentenceTransformerEmbeddings {
    -SentenceTransformer _model
    +embed_documents(texts) List~List~float~~
    +embed_query(text) List~float~
  }

  class ChromaRetriever {
    -Chroma _store
    -Settings _settings
    +add_documents(docs) None
    +similarity_search(query, k, filter) List~Document~
    +list_sources() List~str~
    +document_count() int
  }

  class DocumentIngestionPipeline {
    -RecursiveCharacterTextSplitter _splitter
    -ChromaRetriever _retriever
    +run(file_path) dict
    -_load(file_path) List~Document~
    -_chunk(docs) List~Document~
  }

  class AgentExecutor {
    +agent
    +tools List~BaseTool~
    +memory ConversationBufferWindowMemory
    +ainvoke(input) dict
    +astream_events(input) AsyncIterator
  }

  SentenceTransformerEmbeddings --> ChromaRetriever : used by
  ChromaRetriever --> DocumentIngestionPipeline : used by
  AgentExecutor --> ChromaRetriever : via retrieve_documents tool
```

---

## Sequence Diagram: Streaming Chat Request

Shows the full request lifecycle for `POST /chat/stream`.

```mermaid
sequenceDiagram
  actor User
  participant ChatRouter as Chat Router<br/>(chat.py)
  participant AgentFactory as Agent Factory<br/>(agent.py)
  participant AgentExec as AgentExecutor<br/>(LangChain)
  participant Redis
  participant Claude as Anthropic API<br/>(Claude)
  participant RetrieverTool as retrieve_documents<br/>(tool)
  participant Chroma as ChromaDB

  User->>ChatRouter: POST /chat/stream {message, session_id}
  ChatRouter->>AgentFactory: create_agent(session_id)
  AgentFactory->>Redis: RedisChatMessageHistory(session_id)
  Redis-->>AgentFactory: chat history (last K turns)
  AgentFactory-->>ChatRouter: AgentExecutor

  ChatRouter->>AgentExec: astream_events({input: message})

  AgentExec->>Claude: chat completion (message + history + tools schema)
  Claude-->>AgentExec: tool_call: retrieve_documents(query="...")

  AgentExec->>RetrieverTool: invoke(query, k)
  RetrieverTool->>Chroma: similarity_search(query, k=5)
  Chroma-->>RetrieverTool: [Document, Document, ...]
  RetrieverTool-->>AgentExec: formatted chunks with citations

  AgentExec->>Claude: tool result + continue generation
  Claude-->>AgentExec: streaming tokens (answer with citations)

  loop each token
    AgentExec-->>ChatRouter: on_chat_model_stream event
    ChatRouter-->>User: data: {"type":"token","content":"..."}
  end

  AgentExec->>Redis: save (human message + AI response)
  ChatRouter-->>User: data: {"type":"done","content":""}
```

---

## Sequence Diagram: Document Ingestion

Shows the lifecycle for `POST /documents/ingest`.

```mermaid
sequenceDiagram
  actor User
  participant DocsRouter as Documents Router<br/>(documents.py)
  participant Pipeline as DocumentIngestionPipeline<br/>(ingestion.py)
  participant Loader as Document Loader<br/>(PyPDF/Text/Markdown)
  participant Splitter as RecursiveCharacterTextSplitter
  participant Retriever as ChromaRetriever
  participant Embedder as SentenceTransformerEmbeddings
  participant Chroma as ChromaDB

  User->>DocsRouter: POST /documents/ingest (multipart file)
  DocsRouter->>DocsRouter: validate extension (.pdf/.txt/.md)
  DocsRouter->>DocsRouter: write to temp file
  DocsRouter->>Pipeline: run(tmp_path)

  Pipeline->>Loader: load()
  Loader-->>Pipeline: List[Document] (raw pages)

  Pipeline->>Splitter: split_documents(docs)
  Splitter-->>Pipeline: List[Document] (chunks)

  Pipeline->>Pipeline: enrich metadata<br/>(source, doc_id, ingested_at, file_type)

  Pipeline->>Retriever: add_documents(chunks)
  Retriever->>Embedder: embed_documents(texts)
  Embedder-->>Retriever: List[List[float]]
  Retriever->>Chroma: upsert(embeddings + metadata)
  Chroma-->>Retriever: OK

  Pipeline-->>DocsRouter: {status, source, doc_id, chunks_stored}
  DocsRouter-->>User: 200 OK IngestResponse
```

---

## Calculator Tool: AST Evaluation Flow

```mermaid
flowchart TD
  A[Input: expression string] --> B[ast.parse in eval mode]
  B --> C{AST node type?}
  C -->|Constant int/float| D[return value]
  C -->|BinOp| E{operator in whitelist?}
  E -->|yes| F[recurse on left + right, apply op]
  E -->|no| G[raise ValueError]
  C -->|UnaryOp| H{operator in whitelist?}
  H -->|yes| I[recurse on operand, apply op]
  H -->|no| G
  C -->|Call| J{func name in safe functions?}
  J -->|yes| K[recurse on args, call func]
  J -->|no| G
  C -->|Name| L{name in safe constants?}
  L -->|yes| M[return constant value]
  L -->|no| G
  C -->|other| G
  F --> N[numeric result]
  I --> N
  K --> N
  D --> N
  M --> N
  N --> O[format: int if .is_integer else 10g float]
  O --> P[return string result]
  G --> Q[return Error message string]
```
