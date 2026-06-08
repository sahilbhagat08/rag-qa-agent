from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import ChromaRetriever

logger = get_logger(__name__)


def _loader_for(file_path: Path):
    """Select the appropriate LangChain document loader based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    if ext in (".md", ".mdx"):
        return UnstructuredMarkdownLoader(str(file_path))
    # Fallback: plain text (txt, rst, csv, etc.)
    return TextLoader(str(file_path), encoding="utf-8")


class DocumentIngestionPipeline:
    """Load → chunk → enrich metadata → embed → store pipeline."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.CHUNK_SIZE,
            chunk_overlap=self._settings.CHUNK_OVERLAP,
            add_start_index=True,
        )
        self._retriever = ChromaRetriever()

    def run(self, file_path: str | Path) -> dict:
        """
        Ingest a single file. Returns a summary dict with status and counts.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info("ingestion_start", file=str(path))

        # 1. Load
        loader = _loader_for(path)
        raw_docs: List[Document] = loader.load()
        logger.info("documents_loaded", file=str(path), pages=len(raw_docs))

        # 2. Chunk
        chunks: List[Document] = self._splitter.split_documents(raw_docs)
        logger.info("documents_chunked", file=str(path), chunks=len(chunks))

        # 3. Enrich metadata
        doc_id = hashlib.md5(path.name.encode()).hexdigest()[:8]
        ingested_at = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "source": path.name,
                    "doc_id": doc_id,
                    "ingested_at": ingested_at,
                    "file_type": path.suffix.lower().lstrip("."),
                }
            )

        # 4. Embed + store
        self._retriever.add_documents(chunks)
        logger.info("ingestion_complete", file=str(path), chunks=len(chunks))

        return {
            "status": "success",
            "source": path.name,
            "doc_id": doc_id,
            "chunks_stored": len(chunks),
            "pages_loaded": len(raw_docs),
        }
