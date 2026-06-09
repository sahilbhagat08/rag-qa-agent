"""
Pipeline runner — drives real RAG agent calls and collects evaluation traces.

This module runs the actual ``AgentExecutor`` (not mocked) against each
question in the evaluation dataset, intercepting the ``retrieve_documents``
tool's inputs and outputs to capture the retrieved chunks.

The captured data is returned as ``EvalTrace`` objects, which are the
common input format for both the retrieval metrics module and the RAGAS
generation metrics module.

Design choices
--------------
- Uses ``unittest.mock.patch`` to wrap the ``retrieve_documents`` tool's
  underlying ``ChromaRetriever.similarity_search`` call.  This avoids
  modifying any production code while still capturing what was actually
  retrieved for each query.
- Runs one agent per question with a fresh (isolated) session ID to prevent
  conversation history from polluting multi-turn evaluations.
- Synchronous wrapper around the async ``AgentExecutor.ainvoke`` call to keep
  the CLI runner simple.
- On timeout or exception, records the error in the trace and continues with
  the next question.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document

from app.agent.agent import create_agent
from eval.config import AGENT_TIMEOUT_SECONDS, EVAL_SESSION_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvalTrace:
    """
    Captured inputs and outputs for one evaluation question.

    Attributes
    ----------
    question:
        The input question posed to the agent.
    answer:
        The final generated answer from the agent.
    retrieved_chunks:
        Ordered list of chunk texts that ChromaDB returned for this query.
        Empty if the agent did not call retrieve_documents.
    retrieved_metadata:
        Corresponding metadata dicts for each retrieved chunk.
    ground_truth_answer:
        Reference answer from the evaluation dataset.
    ground_truth_contexts:
        Reference chunk texts from the evaluation dataset.
    tool_calls:
        Names of tools the agent actually invoked (in call order).
    latency_seconds:
        Wall-clock time from agent invocation to final answer.
    error:
        Non-empty if the agent raised an exception or timed out.
    """

    question: str
    answer: str = ""
    retrieved_chunks: list[str] = field(default_factory=list)
    retrieved_metadata: list[dict] = field(default_factory=list)
    ground_truth_answer: str = ""
    ground_truth_contexts: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: str = ""

    def has_error(self) -> bool:
        return bool(self.error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_chunks": self.retrieved_chunks,
            "retrieved_metadata": self.retrieved_metadata,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_contexts": self.ground_truth_contexts,
            "tool_calls": self.tool_calls,
            "latency_seconds": round(self.latency_seconds, 3),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_retriever_patch(trace: EvalTrace):  # type: ignore[return]
    """
    Return a context manager that patches ChromaRetriever.similarity_search
    to record retrieved documents into ``trace`` before passing through to
    the real implementation.
    """
    from app.rag.retriever import ChromaRetriever

    original_search = ChromaRetriever.similarity_search

    def _recording_search(self: ChromaRetriever, query: str, k: int | None = None, **kwargs):  # type: ignore[override]
        docs: list[Document] = original_search(self, query, k=k, **kwargs)
        trace.retrieved_chunks.extend(doc.page_content for doc in docs)
        trace.retrieved_metadata.extend(dict(doc.metadata) for doc in docs)
        logger.debug(
            "retriever_patch: captured %d chunks for query=%r",
            len(docs),
            query[:60],
        )
        return docs

    return patch.object(ChromaRetriever, "similarity_search", _recording_search)


def _record_tool_calls(trace: EvalTrace, intermediate_steps: list) -> None:
    """Extract tool names from AgentExecutor intermediate_steps."""
    for action, _ in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        if tool_name:
            trace.tool_calls.append(tool_name)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


async def _run_single_question(
    question: str,
    session_id: str,
    timeout: float = AGENT_TIMEOUT_SECONDS,
) -> tuple[str, list[Document], list[str]]:
    """
    Invoke the agent for a single question.

    Returns (answer, retrieved_docs, tool_names).
    Raises asyncio.TimeoutError on timeout.
    """
    executor = create_agent(session_id=session_id)

    # Return intermediate steps so we can log tool names
    executor.return_intermediate_steps = True

    result = await asyncio.wait_for(
        executor.ainvoke({"input": question}),
        timeout=timeout,
    )

    answer = str(result.get("output", ""))
    intermediate = result.get("intermediate_steps", [])
    tool_names = [
        getattr(action, "tool", "") for action, _ in intermediate if hasattr(action, "tool")
    ]

    return answer, tool_names


async def _run_with_patch(
    question: str,
    trace: EvalTrace,
    timeout: float = AGENT_TIMEOUT_SECONDS,
) -> None:
    """Run one agent call with the retriever patch active."""
    session_id = f"{EVAL_SESSION_ID}-{id(trace)}"

    with _make_retriever_patch(trace):
        t0 = time.monotonic()
        try:
            answer, tool_names = await _run_single_question(
                question=question,
                session_id=session_id,
                timeout=timeout,
            )
            trace.answer = answer
            trace.tool_calls = tool_names
        except TimeoutError:
            trace.error = f"Timeout after {timeout}s"
            logger.warning("Eval timeout for question: %r", question[:80])
        except Exception as exc:  # noqa: BLE001
            trace.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Eval error for question: %r", question[:80])
        finally:
            trace.latency_seconds = time.monotonic() - t0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    dataset: list[dict],
    timeout: float = AGENT_TIMEOUT_SECONDS,
) -> list[EvalTrace]:
    """
    Run the full RAG pipeline against every question in ``dataset``.

    Parameters
    ----------
    dataset:
        List of dicts loaded from ``eval_dataset.json`` or
        ``adversarial_cases.json``.  Each dict must have a ``"question"``
        key; ``"ground_truth_answer"`` and ``"ground_truth_contexts"`` are
        optional but improve metric quality.
    timeout:
        Per-question wall-clock timeout in seconds.

    Returns
    -------
    list[EvalTrace]
        One trace per question, in the same order as ``dataset``.
    """
    traces: list[EvalTrace] = []

    for i, entry in enumerate(dataset):
        question = entry.get("question", "")
        if not question:
            logger.warning("Skipping entry %d: missing 'question' key", i)
            continue

        logger.info(
            "Running question %d/%d: %r",
            i + 1,
            len(dataset),
            question[:80],
        )

        trace = EvalTrace(
            question=question,
            ground_truth_answer=entry.get("ground_truth_answer", ""),
            ground_truth_contexts=entry.get("ground_truth_contexts") or [],
        )

        asyncio.run(_run_with_patch(question, trace, timeout=timeout))

        traces.append(trace)
        status = "ERROR" if trace.has_error() else "OK"
        logger.info(
            "  [%s] latency=%.1fs  tools=%s  chunks_retrieved=%d",
            status,
            trace.latency_seconds,
            trace.tool_calls,
            len(trace.retrieved_chunks),
        )

    return traces
