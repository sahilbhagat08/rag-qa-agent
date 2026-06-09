"""
CI threshold gate — asserts that the latest evaluation report meets all
configured minimum scores.

These tests are tagged with ``@pytest.mark.eval`` so they can be run
separately from the fast unit/integration suite:

    # Run only eval threshold tests:
    pytest -m eval tests/eval/

    # Skip eval threshold tests in the standard CI run:
    pytest -m "not eval" tests/

The tests rely on a pre-existing report file produced by:

    python -m eval.run_eval

If no report exists yet, the tests are skipped with a clear message rather
than failing noisily.

Test organisation
-----------------
- ``test_report_exists``         — sanity check: confirms report file is present
- ``test_faithfulness``          — hallucination guard
- ``test_answer_relevancy``      — answer addresses the question
- ``test_context_precision``     — retrieved chunks are relevant
- ``test_context_recall``        — all relevant info was retrieved
- ``test_answer_correctness``    — factual match to reference answer
- ``test_hit_rate``              — retrieval hit rate @k
- ``test_mrr``                   — mean reciprocal rank
- ``test_no_excessive_errors``   — fewer than 20 % of traces errored
- ``test_threshold_results_summary`` — holistic all-pass check
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.config import (
    REPORTS_DIR,
    RETRIEVAL_THRESHOLDS,
    THRESHOLDS,
)
from eval.runners.ragas_runner import load_latest_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report() -> dict:
    """
    Load the latest evaluation report.

    Skip the entire module if no report has been generated yet — the eval
    suite should not block the standard unit/integration tests.
    """
    try:
        return load_latest_report(reports_dir=REPORTS_DIR)
    except FileNotFoundError as exc:
        pytest.skip(
            f"No evaluation report found. Run `python -m eval.run_eval` first. ({exc})"
        )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

# Register the 'eval' mark to avoid PytestUnknownMarkWarning.
# (Also declared in pyproject.toml markers section — added by this PR.)
pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def test_report_exists() -> None:
    """The latest report file must exist before threshold checks run."""
    report_path = REPORTS_DIR / "report_latest.json"
    fallback = sorted(REPORTS_DIR.glob("report_2*.json"))
    assert report_path.exists() or fallback, (
        f"No evaluation report found in {REPORTS_DIR}. "
        "Run `python -m eval.run_eval` to generate one."
    )


# ---------------------------------------------------------------------------
# Generation metric thresholds
# ---------------------------------------------------------------------------


def test_faithfulness(report: dict) -> None:
    """
    Faithfulness must meet the configured minimum.

    Faithfulness measures whether every claim in the generated answer is
    supported by the retrieved context chunks.  A score below threshold
    indicates the LLM is hallucinating facts not present in the knowledge base.
    """
    score = report.get("faithfulness", 0.0)
    threshold = THRESHOLDS["faithfulness"]
    assert score >= threshold, (
        f"Faithfulness {score:.3f} is below threshold {threshold}. "
        "The agent may be hallucinating. Check per_sample report for failing questions."
    )


def test_answer_relevancy(report: dict) -> None:
    """
    Answer relevancy must meet the configured minimum.

    Measures whether the generated answer actually addresses the question.
    Low scores indicate the agent is going off-topic or retrieving unrelated content.
    """
    score = report.get("answer_relevancy", 0.0)
    threshold = THRESHOLDS["answer_relevancy"]
    assert score >= threshold, (
        f"Answer relevancy {score:.3f} is below threshold {threshold}. "
        "Check whether the agent prompt needs tuning or retrieval k should increase."
    )


def test_context_precision(report: dict) -> None:
    """
    Context precision must meet the configured minimum.

    Measures what fraction of the retrieved chunks are actually relevant to the
    question.  Low precision = too much noise in the context window, which can
    confuse the LLM.
    """
    score = report.get("context_precision", 0.0)
    threshold = THRESHOLDS["context_precision"]
    assert score >= threshold, (
        f"Context precision {score:.3f} is below threshold {threshold}. "
        "Consider reducing TOP_K_RESULTS or adding a re-ranking step."
    )


def test_context_recall(report: dict) -> None:
    """
    Context recall must meet the configured minimum.

    Measures whether all information needed to answer correctly was retrieved.
    Low recall = the retriever is missing relevant chunks.
    """
    score = report.get("context_recall", 0.0)
    threshold = THRESHOLDS["context_recall"]
    assert score >= threshold, (
        f"Context recall {score:.3f} is below threshold {threshold}. "
        "Consider increasing TOP_K_RESULTS, reducing CHUNK_SIZE, or using hybrid search."
    )


def test_answer_correctness(report: dict) -> None:
    """
    Answer correctness must meet the configured minimum.

    Factual F1 overlap between generated answers and ground-truth reference answers.
    Requires eval_dataset.json to have populated ground_truth_answer fields.
    """
    score = report.get("answer_correctness", 0.0)
    threshold = THRESHOLDS["answer_correctness"]
    assert score >= threshold, (
        f"Answer correctness {score:.3f} is below threshold {threshold}. "
        "Check whether reference answers in the dataset are accurate and complete."
    )


# ---------------------------------------------------------------------------
# Retrieval metric thresholds
# ---------------------------------------------------------------------------


def test_hit_rate(report: dict) -> None:
    """
    Hit rate @k must meet the configured minimum.

    Hit rate = fraction of queries where at least one ground-truth context
    chunk appears in the top-k retrieved results.
    """
    score = report.get("hit_rate", 0.0)
    threshold = RETRIEVAL_THRESHOLDS["hit_rate_at_5"]
    assert score >= threshold, (
        f"Hit rate {score:.3f} is below threshold {threshold}. "
        "ChromaDB may not contain the necessary documents, or the embedding model "
        "is not producing good similarity scores for these queries."
    )


def test_mrr(report: dict) -> None:
    """
    Mean Reciprocal Rank must meet the configured minimum.

    MRR penalises systems where relevant chunks are retrieved but ranked low.
    Low MRR with decent hit rate = retrieval ranking quality issue.
    """
    score = report.get("mrr", 0.0)
    threshold = RETRIEVAL_THRESHOLDS["mrr"]
    assert score >= threshold, (
        f"MRR {score:.3f} is below threshold {threshold}. "
        "Relevant chunks are found but ranked low. Consider adding a cross-encoder re-ranker."
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


def test_no_excessive_errors(report: dict) -> None:
    """
    No more than 20 % of evaluation traces should contain errors.

    Errors indicate agent crashes, timeouts, or other infrastructure issues
    unrelated to answer quality.
    """
    num_samples = report.get("num_samples", 0)
    num_errors = report.get("num_errors", 0)
    if num_samples == 0:
        pytest.skip("No samples in report.")
    error_rate = num_errors / num_samples
    assert error_rate <= 0.20, (
        f"Error rate {error_rate:.1%} exceeds 20 % ({num_errors}/{num_samples} traces failed). "
        "Check agent logs for timeouts or API errors."
    )


def test_threshold_results_summary(report: dict) -> None:
    """
    All individual threshold flags in the report must be True.

    This test consolidates every metric threshold into a single holistic
    assertion.  It will fail with a clear list of which metrics are below
    threshold, complementing the individual tests above.
    """
    threshold_results: dict = report.get("threshold_results", {})
    if not threshold_results:
        pytest.skip("Report does not contain threshold_results — regenerate the report.")

    failures = [metric for metric, passed in threshold_results.items() if not passed]
    assert not failures, (
        f"The following metrics are below threshold: {failures}. "
        "See individual test failures above for details and remediation hints."
    )
