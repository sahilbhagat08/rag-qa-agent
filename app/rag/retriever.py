from __future__ import annotations

from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import SentenceTransformerEmbeddings

logger = get_logger(__name__)

_chroma_store: Chroma | None = None


def get_chroma_store() -> Chroma:
    """Return a cached Chroma vector store (initialised once at startup)."""
    global _chroma_store
    if _chroma_store is None:
        settings = get_settings()
        logger.info(
            "initialising_chroma",
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection=settings.CHROMA_COLLECTION_NAME,
        )
        _chroma_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        logger.info("chroma_initialised")
    return _chroma_store


class ChromaRetriever:
    """Thin wrapper around Chroma providing add, search, and list operations."""

    def __init__(self) -> None:
        self._store = get_chroma_store()
        self._settings = get_settings()

    def add_documents(self, docs: List[Document]) -> None:
        logger.info("adding_documents_to_chroma", count=len(docs))
        self._store.add_documents(docs)
        logger.info("documents_added", count=len(docs))

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> List[Document]:
        top_k = k or self._settings.TOP_K_RESULTS
        logger.debug("similarity_search", query=query[:80], k=top_k)
        results = self._store.similarity_search(query, k=top_k, filter=filter)
        logger.debug("similarity_search_results", count=len(results))
        return results

    def list_sources(self) -> List[str]:
        """Return deduplicated list of source filenames stored in Chroma."""
        collection = self._store._collection
        results = collection.get(include=["metadatas"])
        sources = {
            meta.get("source", "unknown")
            for meta in results.get("metadatas", [])
            if meta
        }
        return sorted(sources)

    def document_count(self) -> int:
        return self._store._collection.count()
