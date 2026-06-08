from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import DocumentListResponse, IngestResponse
from app.core.logging import get_logger
from app.rag.ingestion import DocumentIngestionPipeline
from app.rag.retriever import ChromaRetriever

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".mdx", ".rst"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    POST /documents/ingest

    Upload a document (PDF, TXT, Markdown) to be chunked, embedded,
    and stored in the Chroma vector database.
    """
    suffix = Path(file.filename or "file").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    logger.info("ingest_request", filename=file.filename, content_type=file.content_type)

    # Write upload to a temp file — loader needs a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        pipeline = DocumentIngestionPipeline()
        # Use the original filename for metadata, not the temp path name
        tmp_path = tmp_path.rename(tmp_path.parent / file.filename)
        result = pipeline.run(tmp_path)
        return IngestResponse(**result)
    except Exception as exc:
        logger.error("ingest_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """
    GET /documents

    List all unique document sources currently stored in the vector database.
    """
    retriever = ChromaRetriever()
    sources = retriever.list_sources()
    return DocumentListResponse(sources=sources, total=len(sources))
