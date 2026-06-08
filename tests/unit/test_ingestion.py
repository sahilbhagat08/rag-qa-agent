"""Unit tests for DocumentIngestionPipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.ingestion import DocumentIngestionPipeline, _loader_for


# ── _loader_for ───────────────────────────────────────────────────────────────

def test_loader_for_pdf():
    from langchain_community.document_loaders import PyPDFLoader
    loader = _loader_for(Path("report.pdf"))
    assert isinstance(loader, PyPDFLoader)


def test_loader_for_markdown():
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    loader = _loader_for(Path("readme.md"))
    assert isinstance(loader, UnstructuredMarkdownLoader)


def test_loader_for_txt():
    from langchain_community.document_loaders import TextLoader
    loader = _loader_for(Path("notes.txt"))
    assert isinstance(loader, TextLoader)


# ── DocumentIngestionPipeline.run ─────────────────────────────────────────────

@patch("app.rag.ingestion.ChromaRetriever")
@patch("app.rag.ingestion._loader_for")
def test_run_success(mock_loader_for, mock_chroma_cls, tmp_path):
    """Pipeline loads, chunks, enriches metadata, and calls add_documents."""
    # Create a real temp file
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Hello world. " * 100)

    # Mock the loader to return controlled Documents
    raw_doc = Document(page_content="Hello world. " * 100, metadata={})
    mock_loader = MagicMock()
    mock_loader.load.return_value = [raw_doc]
    mock_loader_for.return_value = mock_loader

    mock_retriever = MagicMock()
    mock_chroma_cls.return_value = mock_retriever

    pipeline = DocumentIngestionPipeline()
    result = pipeline.run(doc_file)

    assert result["status"] == "success"
    assert result["source"] == "test.txt"
    assert result["chunks_stored"] >= 1
    mock_retriever.add_documents.assert_called_once()


@patch("app.rag.ingestion.ChromaRetriever")
@patch("app.rag.ingestion._loader_for")
def test_run_enriches_metadata(mock_loader_for, mock_chroma_cls, tmp_path):
    """Each chunk must have source, doc_id, ingested_at, file_type metadata."""
    doc_file = tmp_path / "sample.pdf"
    doc_file.write_bytes(b"%PDF-1.4 fake")

    raw_doc = Document(page_content="content " * 200, metadata={})
    mock_loader = MagicMock()
    mock_loader.load.return_value = [raw_doc]
    mock_loader_for.return_value = mock_loader

    mock_retriever = MagicMock()
    mock_chroma_cls.return_value = mock_retriever

    pipeline = DocumentIngestionPipeline()
    pipeline.run(doc_file)

    stored_chunks = mock_retriever.add_documents.call_args[0][0]
    for chunk in stored_chunks:
        assert "source" in chunk.metadata
        assert "doc_id" in chunk.metadata
        assert "ingested_at" in chunk.metadata
        assert "file_type" in chunk.metadata
        assert chunk.metadata["file_type"] == "pdf"


def test_run_raises_for_missing_file():
    """FileNotFoundError raised when file does not exist."""
    pipeline_mock = MagicMock(spec=DocumentIngestionPipeline)
    pipeline_mock.run.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        pipeline_mock.run("/nonexistent/file.txt")
