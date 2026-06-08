from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import ChromaRetriever

logger = get_logger(__name__)

_retriever: ChromaRetriever | None = None


def _get_retriever() -> ChromaRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever


@tool
def retrieve_documents(query: str, k: int = 5) -> str:
    """
    Search the document knowledge base using semantic similarity.

    Use this tool whenever the user asks a question that requires information
    from ingested documents. Returns the most relevant chunks with citations.

    Args:
        query: The search query in natural language.
        k: Number of results to return (default 5, max 10).

    Returns:
        Formatted string of relevant document chunks with source citations.
    """
    settings = get_settings()
    k = min(k, 10)  # cap to avoid runaway searches
    retriever = _get_retriever()

    logger.info("tool_retrieve_documents", query=query[:80], k=k)
    docs = retriever.similarity_search(query, k=k)

    if not docs:
        return "No relevant documents found for the given query."

    results: List[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        results.append(
            f"[{i}] Source: {source}{page_str}\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(results)
