"""
Retrieval quality metrics — computed without any LLM calls.

These metrics evaluate how well ChromaDB's cosine-similarity search retrieves
relevant chunks before the LLM generates an answer.  They are cheap to run
(no API cost) and run first in the evaluation pipeline.

Metrics
-------
hit_rate_at_k
    Fraction of queries where at least one ground-truth context chunk
    appears in the top-k retrieved chunks.  The canonical retrieval recall
    proxy when you have binary relevance labels.

mean_reciprocal_rank (MRR)
    Average of 1/rank_of_first_relevant_chunk across all queries.
    Penalises systems that rank relevant content low even when it is retrieved.

average_precision_at_k (AP@k)
    Area under the precision-recall curve for the top-k results.
    Aggregated across queries → Mean Average Precision (MAP@k).

precision_at_k
    Fraction of the top-k retrieved chunks that are relevant.

All functions operate on ``EvalTrace`` objects (defined in
``eval/runners/pipeline_runner.py``) and use a configurable similarity
function to decide whether a retrieved chunk "matches" a ground-truth context.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

# ---------------------------------------------------------------------------
# Relevance matching
# ---------------------------------------------------------------------------

_MATCH_MODEL: SentenceTransformer | None = None


def _get_match_model() -> SentenceTransformer:
    """Lazy singleton — reuses the same embedding model as the app."""
    global _MATCH_MODEL
    if _MATCH_MODEL is None:
        _MATCH_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MATCH_MODEL


def semantic_match(
    retrieved: str,
    reference: str,
    threshold: float = 0.85,
) -> bool:
    """
    Return True if ``retrieved`` and ``reference`` are semantically similar.

    Uses cosine similarity between MiniLM-L6-v2 embeddings.  A threshold of
    0.85 is conservative and avoids false positives when chunks partially
    overlap.
    """
    model = _get_match_model()
    emb_ret = model.encode(retrieved, convert_to_tensor=True)
    emb_ref = model.encode(reference, convert_to_tensor=True)
    score = float(st_util.cos_sim(emb_ret, emb_ref)[0][0])
    return score >= threshold


def exact_substring_match(retrieved: str, reference: str) -> bool:
    """
    Lightweight alternative: True if reference text is a substring of retrieved
    or vice versa (after normalising whitespace).
    """
    r = " ".join(retrieved.lower().split())
    g = " ".join(reference.lower().split())
    return g in r or r in g


# Default relevance checker used across all metric functions.
DEFAULT_MATCH_FN: Callable[[str, str], bool] = semantic_match


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class RetrievalSample:
    """
    One row for retrieval metric computation.

    Attributes
    ----------
    question:
        The query that was sent to the retriever.
    retrieved_chunks:
        Ordered list of chunk texts returned by ChromaDB (index 0 = rank 1).
    ground_truth_contexts:
        List of reference chunk texts that contain the correct answer.
    """

    question: str
    retrieved_chunks: list[str]
    ground_truth_contexts: list[str]


# ---------------------------------------------------------------------------
# Individual sample helpers
# ---------------------------------------------------------------------------


def _is_hit(sample: RetrievalSample, match_fn: Callable[[str, str], bool]) -> bool:
    """True if at least one retrieved chunk matches any ground-truth context."""
    for chunk in sample.retrieved_chunks:
        for ref in sample.ground_truth_contexts:
            if ref and match_fn(chunk, ref):
                return True
    return False


def _reciprocal_rank(
    sample: RetrievalSample, match_fn: Callable[[str, str], bool]
) -> float:
    """Return 1/rank of the first relevant chunk, or 0 if none found."""
    for rank, chunk in enumerate(sample.retrieved_chunks, start=1):
        for ref in sample.ground_truth_contexts:
            if ref and match_fn(chunk, ref):
                return 1.0 / rank
    return 0.0


def _average_precision(
    sample: RetrievalSample, match_fn: Callable[[str, str], bool]
) -> float:
    """Compute Average Precision (AP) for a single query."""
    if not sample.ground_truth_contexts:
        return 0.0

    hits = 0
    cumulative_precision = 0.0

    for rank, chunk in enumerate(sample.retrieved_chunks, start=1):
        relevant = any(
            ref and match_fn(chunk, ref) for ref in sample.ground_truth_contexts
        )
        if relevant:
            hits += 1
            cumulative_precision += hits / rank

    if hits == 0:
        return 0.0
    n_ref = len(sample.ground_truth_contexts)
    n_ret = len(sample.retrieved_chunks)
    return cumulative_precision / min(n_ref, n_ret)


def _precision_at_k(
    sample: RetrievalSample, match_fn: Callable[[str, str], bool]
) -> float:
    """Fraction of retrieved chunks that are relevant."""
    if not sample.retrieved_chunks:
        return 0.0
    relevant_count = sum(
        1
        for chunk in sample.retrieved_chunks
        if any(ref and match_fn(chunk, ref) for ref in sample.ground_truth_contexts)
    )
    return relevant_count / len(sample.retrieved_chunks)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval quality metrics across all evaluation samples."""

    hit_rate: float
    mrr: float
    map_at_k: float
    mean_precision_at_k: float
    num_samples: int
    # Per-sample details for debugging
    per_sample: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "map_at_k": round(self.map_at_k, 4),
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "num_samples": self.num_samples,
        }

    def __str__(self) -> str:
        lines = [
            "Retrieval Metrics",
            "─" * 40,
            f"  Hit Rate @k        : {self.hit_rate:.3f}",
            f"  MRR                : {self.mrr:.3f}",
            f"  MAP@k              : {self.map_at_k:.3f}",
            f"  Mean Precision@k   : {self.mean_precision_at_k:.3f}",
            f"  Samples evaluated  : {self.num_samples}",
        ]
        return "\n".join(lines)


def compute_retrieval_metrics(
    samples: list[RetrievalSample],
    match_fn: Callable[[str, str], bool] | None = None,
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics across a list of ``RetrievalSample``s.

    Parameters
    ----------
    samples:
        One sample per query, each containing retrieved chunks and references.
    match_fn:
        Binary relevance function ``(retrieved_chunk, reference_chunk) → bool``.
        Defaults to ``semantic_match`` with threshold=0.85.

    Returns
    -------
    RetrievalMetrics
        Aggregated scores plus per-sample details.
    """
    if match_fn is None:
        match_fn = DEFAULT_MATCH_FN

    if not samples:
        return RetrievalMetrics(
            hit_rate=0.0,
            mrr=0.0,
            map_at_k=0.0,
            mean_precision_at_k=0.0,
            num_samples=0,
        )

    # Queries with no ground-truth contexts are skipped for hit/MRR/MAP
    # but counted in the total (they lower recall-oriented metrics implicitly).
    scorable = [s for s in samples if s.ground_truth_contexts]
    n = len(scorable) if scorable else 1  # avoid div-by-zero

    hit_scores = [1.0 if _is_hit(s, match_fn) else 0.0 for s in scorable]
    rr_scores = [_reciprocal_rank(s, match_fn) for s in scorable]
    ap_scores = [_average_precision(s, match_fn) for s in scorable]
    p_scores = [_precision_at_k(s, match_fn) for s in scorable]

    per_sample = [
        {
            "question": s.question[:120],
            "hit": bool(h),
            "reciprocal_rank": round(rr, 4),
            "average_precision": round(ap, 4),
            "precision_at_k": round(p, 4),
            "num_retrieved": len(s.retrieved_chunks),
            "num_references": len(s.ground_truth_contexts),
        }
        for s, h, rr, ap, p in zip(scorable, hit_scores, rr_scores, ap_scores, p_scores)
    ]

    return RetrievalMetrics(
        hit_rate=sum(hit_scores) / n,
        mrr=sum(rr_scores) / n,
        map_at_k=sum(ap_scores) / n,
        mean_precision_at_k=sum(p_scores) / n,
        num_samples=len(samples),
        per_sample=per_sample,
    )
