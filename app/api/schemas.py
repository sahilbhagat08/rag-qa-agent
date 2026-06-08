from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: str = Field(
        ..., min_length=1, max_length=128, description="Unique session identifier"
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class SSEEvent(BaseModel):
    """Shape of each Server-Sent Event data payload."""
    type: str  # "token" | "done" | "error"
    content: str
    session_id: str


# ── Documents ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    source: str
    doc_id: str
    chunks_stored: int
    pages_loaded: int


class DocumentInfo(BaseModel):
    source: str


class DocumentListResponse(BaseModel):
    sources: List[str]
    total: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
