"""
Generation quality metrics — RAGAS wrappers using Claude as the LLM judge.

Metrics evaluated
-----------------
faithfulness
    Are ALL claims in the generated answer grounded in the retrieved context?
    Score = (claims supported by context) / (total claims in answer).
    Detects hallucination.  Requires: question, answer, retrieved_contexts.

answer_relevancy
    Does the answer actually address the question that was asked?
    Computed by generating hypothetical questions from the answer and comparing
    them to the original question using embedding similarity.
    Requires: question, answer.

context_precision
    Of the retrieved chunks, what fraction are actually relevant to the question?
    Penalises noisy/irrelevant retrieval.
    Requires: question, retrieved_contexts, reference_answer.

context_recall
    Did the retrieved chunks contain all the information needed to answer the
    question correctly?
    Score = (reference sentences attributable to context) / (total reference sentences).
    Requires: question, retrieved_contexts, reference_answer.

answer_correctness
    Factual overlap (F1) between the generated answer and the reference answer.
    Combines semantic similarity and factual statement matching.
    Requires: question, answer, reference_answer.

All metrics use the RAGAS ``evaluate()`` API with ``LangchainLLMWrapper``
wrapping ``ChatAnthropic``, so no additional API keys are needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import get_settings
from app.rag.embeddings import SentenceTransformerEmbeddings
from eval.config import JUDGE_MAX_TOKENS, JUDGE_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class GenerationSample:
    """
    One row for generation metric computation.

    Attributes
    ----------
    question:
        The original user question.
    answer:
        The generated answer from the RAG pipeline.
    retrieved_contexts:
        List of chunk texts that were retrieved and passed to the LLM.
    reference_answer:
        Ground-truth reference answer (required for context_recall,
        context_precision, and answer_correctness).
    """

    question: str
    answer: str
    retrieved_contexts: list[str]
    reference_answer: str = ""


# ---------------------------------------------------------------------------
# RAGAS metric instances (lazily configured with judge LLM)
# ---------------------------------------------------------------------------


def _build_metrics(
    judge_llm: LangchainLLMWrapper,
    judge_embeddings: LangchainEmbeddingsWrapper,
) -> list[Any]:
    """
    Instantiate and configure all five RAGAS metric objects.

    Each metric is configured with the same Claude judge LLM to ensure
    consistent scoring across all dimensions.
    """
    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)
    answer_correctness = AnswerCorrectness(llm=judge_llm, embeddings=judge_embeddings)

    return [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------


@dataclass
class GenerationMetrics:
    """Aggregated RAGAS generation quality scores."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float
    num_samples: int
    # Raw per-sample DataFrame rows (as list of dicts) for debugging
    per_sample: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "num_samples": self.num_samples,
        }

    def __str__(self) -> str:
        lines = [
            "Generation Metrics (RAGAS)",
            "─" * 40,
            f"  Faithfulness       : {self.faithfulness:.3f}",
            f"  Answer Relevancy   : {self.answer_relevancy:.3f}",
            f"  Context Precision  : {self.context_precision:.3f}",
            f"  Context Recall     : {self.context_recall:.3f}",
            f"  Answer Correctness : {self.answer_correctness:.3f}",
            f"  Samples evaluated  : {self.num_samples}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def compute_generation_metrics(
    samples: list[GenerationSample],
) -> GenerationMetrics:
    """
    Run all five RAGAS generation metrics across the provided samples.

    Parameters
    ----------
    samples:
        One ``GenerationSample`` per query, each containing the generated
        answer, retrieved contexts, and (where available) a reference answer.

    Returns
    -------
    GenerationMetrics
        Mean scores across all samples, plus per-sample details.

    Notes
    -----
    - Samples without a ``reference_answer`` will receive NaN for
      context_recall, context_precision, and answer_correctness.
      RAGAS handles NaN by excluding those rows from the mean calculation.
    - Claude API calls are made in parallel by RAGAS (async batch evaluation).
    """
    if not samples:
        return GenerationMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
            answer_correctness=0.0,
            num_samples=0,
        )

    settings = get_settings()

    # Build judge LLM and embeddings
    judge_llm_raw = ChatAnthropic(
        model=JUDGE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    judge_llm = LangchainLLMWrapper(judge_llm_raw)
    judge_embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings())

    metrics = _build_metrics(judge_llm, judge_embeddings)

    # Convert GenerationSample list → HuggingFace Dataset (RAGAS input format)
    dataset_dict: dict[str, list] = {
        "user_input": [s.question for s in samples],
        "response": [s.answer for s in samples],
        "retrieved_contexts": [s.retrieved_contexts for s in samples],
        "reference": [s.reference_answer for s in samples],
    }
    hf_dataset = Dataset.from_dict(dataset_dict)

    logger.info(
        "Running RAGAS evaluation on %d samples with judge=%s",
        len(samples),
        JUDGE_MODEL,
    )

    # Run RAGAS evaluate()
    results = evaluate(dataset=hf_dataset, metrics=metrics)

    # Extract scores from the results object
    scores_df = results.to_pandas()
    per_sample = scores_df.to_dict(orient="records")

    def _mean(col: str) -> float:
        if col not in scores_df.columns:
            return 0.0
        vals = scores_df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else 0.0

    return GenerationMetrics(
        faithfulness=_mean("faithfulness"),
        answer_relevancy=_mean("answer_relevancy"),
        context_precision=_mean("context_precision"),
        context_recall=_mean("context_recall"),
        answer_correctness=_mean("answer_correctness"),
        num_samples=len(samples),
        per_sample=per_sample,
    )
