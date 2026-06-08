from __future__ import annotations

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """
    Return a Redis-backed ConversationBufferWindowMemory for the given session.

    Each session is keyed by session_id in Redis. Messages are automatically
    expired after REDIS_SESSION_TTL_SECONDS.
    """
    settings = get_settings()
    logger.debug("creating_session_memory", session_id=session_id)

    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.REDIS_SESSION_TTL_SECONDS,
        key_prefix="rag_qa:session:",
    )

    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        k=settings.MEMORY_WINDOW_K,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )

    return memory


def clear_session_memory(session_id: str) -> None:
    """Delete all messages for a session from Redis."""
    settings = get_settings()
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        key_prefix="rag_qa:session:",
    )
    history.clear()
    logger.info("session_memory_cleared", session_id=session_id)
