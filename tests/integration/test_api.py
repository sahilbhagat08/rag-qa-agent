"""Integration tests for FastAPI endpoints (no real LLM/Redis/Chroma calls)."""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── POST /chat ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.api.routes.chat.create_agent")
async def test_chat_returns_answer(mock_create_agent, client):
    mock_executor = AsyncMock()
    mock_executor.ainvoke.return_value = {"output": "Paris is the capital of France."}
    mock_create_agent.return_value = mock_executor

    response = await client.post(
        "/chat",
        json={"message": "What is the capital of France?", "session_id": "sess-1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Paris is the capital of France."
    assert data["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_chat_empty_message_rejected(client):
    response = await client.post(
        "/chat",
        json={"message": "", "session_id": "sess-1"},
    )
    assert response.status_code == 422


# ── POST /documents/ingest ────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.api.routes.documents.DocumentIngestionPipeline")
async def test_ingest_txt_file(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "status": "success",
        "source": "test.txt",
        "doc_id": "abc12345",
        "chunks_stored": 5,
        "pages_loaded": 1,
    }
    mock_pipeline_cls.return_value = mock_pipeline

    file_content = b"This is a test document with some content."
    response = await client.post(
        "/documents/ingest",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_stored"] == 5


@pytest.mark.asyncio
async def test_ingest_unsupported_file_type(client):
    response = await client.post(
        "/documents/ingest",
        files={"file": ("image.jpg", io.BytesIO(b"fake"), "image/jpeg")},
    )
    assert response.status_code == 415


# ── GET /documents ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.api.routes.documents.ChromaRetriever")
async def test_list_documents(mock_chroma_cls, client):
    mock_retriever = MagicMock()
    mock_retriever.list_sources.return_value = ["report.pdf", "readme.md"]
    mock_chroma_cls.return_value = mock_retriever

    response = await client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "report.pdf" in data["sources"]
