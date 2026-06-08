from __future__ import annotations

from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer model (loaded once at startup)."""
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
        logger.info("embedding_model_loaded", model=settings.EMBEDDING_MODEL)
    return _model


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible Embeddings adapter for SentenceTransformer."""

    def __init__(self) -> None:
        self._model = get_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.debug("embedding_documents", count=len(texts))
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        logger.debug("embedding_query", text_length=len(text))
        embedding = self._model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()
