from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import get_chroma_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load heavy resources so the first request is not slow.
    Shutdown: nothing to explicitly clean up (Chroma flushes on exit).
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("startup_begin")

    # Pre-load embedding model (downloads on first run, cached thereafter)
    get_embedding_model()
    # Pre-init Chroma store
    get_chroma_store()

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG QA Agent",
        description=(
            "Document Q&A API powered by Anthropic Claude, LangChain, "
            "ChromaDB (SentenceTransformer embeddings), and Redis session memory."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(documents_router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))

    return app


app = create_app()
