from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@tool
def summarize_document(text: str, max_length: int = 200) -> str:
    """
    Produce a concise summary of the provided text.

    Use this tool when the user explicitly asks for a summary of a document
    or a passage, or when a retrieved chunk is too long and needs condensing
    before including in the final answer.

    Args:
        text: The text content to summarize.
        max_length: Approximate maximum word count for the summary (default 200).

    Returns:
        A concise summary string.
    """
    settings = get_settings()
    logger.info("tool_summarize_document", text_length=len(text), max_length=max_length)

    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=512,
    )

    messages = [
        SystemMessage(
            content=(
                f"You are a precise summarization assistant. "
                f"Summarize the following text in no more than {max_length} words. "
                "Be concise and accurate. Preserve key facts, numbers, and names."
            )
        ),
        HumanMessage(content=text),
    ]

    response = llm.invoke(messages)
    summary = response.content.strip()
    logger.info("tool_summarize_complete", summary_length=len(summary))
    return summary
