from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.agent import create_agent
from app.api.schemas import ChatRequest, ChatResponse
from app.core.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


async def _token_stream(message: str, session_id: str) -> AsyncIterator[str]:
    """
    Async generator that streams agent tokens as SSE events.

    Event format:
        data: {"type": "token", "content": "<token>", "session_id": "<id>"}\n\n
        data: {"type": "done",  "content": "",         "session_id": "<id>"}\n\n
    """
    agent = create_agent(session_id)
    try:
        async for event in agent.astream_events(
            {"input": message},
            version="v2",
        ):
            kind = event.get("event", "")
            # Stream only LLM text tokens (not tool call tokens)
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    payload = json.dumps(
                        {"type": "token", "content": chunk.content, "session_id": session_id}
                    )
                    yield f"data: {payload}\n\n"

        # Signal completion
        done_payload = json.dumps({"type": "done", "content": "", "session_id": session_id})
        yield f"data: {done_payload}\n\n"

    except Exception as exc:
        logger.error("stream_error", session_id=session_id, error=str(exc))
        error_payload = json.dumps(
            {"type": "error", "content": str(exc), "session_id": session_id}
        )
        yield f"data: {error_payload}\n\n"


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    POST /chat/stream

    Streams the agent's response token-by-token as Server-Sent Events.
    Connect with an EventSource client or consume with curl:

        curl -N -X POST http://localhost:8000/chat/stream \\
          -H "Content-Type: application/json" \\
          -d '{"message": "What is the main topic?", "session_id": "abc123"}'
    """
    logger.info("chat_stream_request", session_id=request.session_id)
    return StreamingResponse(
        _token_stream(request.message, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    POST /chat

    Non-streaming endpoint — waits for the full answer before returning.
    Use /chat/stream for real-time token output.
    """
    logger.info("chat_request", session_id=request.session_id)
    try:
        agent = create_agent(request.session_id)
        result = await agent.ainvoke({"input": request.message})
        answer = result.get("output", "")
        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as exc:
        logger.error("chat_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
