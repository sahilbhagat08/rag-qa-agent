"""
RAGAS runner — converts pipeline traces into RAGAS metrics and saves a report.

This module bridges the gap between the raw ``EvalTrace`` objects produced by
``pipeline_runner.py`` and the RAGAS + custom retrieval metric modules.

Workflow
--------
1.  Convert ``EvalTrace`` list → ``RetrievalSample`` + ``GenerationSample``
2.  Compute retrieval metrics (cheap, no LLM calls)
3.  Compute generation metrics (Claude LLM judge via RAGAS)
4.  Merge all scores into a ``EvalReport``
5.  Save the report to ``eval/reports/report_<timestamp>.json``

The ``EvalReport`` dataclass is also what ``tests/eval/test_eval_thresholds.py``
loads for CI threshold assertions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.config import REPORTS_DIR, RETRIEVAL_THRESHOLDS, THRESHOLDS
from eval.metrics.generation import GenerationMetrics, GenerationSample, compute_generation_metrics
from eval.metrics.retrieval import RetrievalMetrics, RetrievalSample, compute_retrieval_metrics
from eval.runners.pipeline_runner import EvalTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data container
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """
    Full evaluation report combining retrieval and generation metrics.

    This is the canonical artifact persisted to disk and checked in CI.
    """

    # Metadata
    timestamp: str
    num_samples: int
    num_errors: int

    # Generation metrics (RAGAS, LLM-judged)
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float

    # Retrieval metrics (custom, LLM-free)
    hit_rate: float
    mrr: float
    map_at_k: float
    mean_precision_at_k: float

    # Per-sample details for debugging
    generation_per_sample: list[dict] = field(default_factory=list)
    retrieval_per_sample: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)

    # Threshold pass/fail summary
    threshold_results: dict[str, bool] = field(default_factory=dict)

    def passes_all_thresholds(self) -> bool:
        """Return True only if every threshold in config is met."""
        return all(self.threshold_results.values())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def summary_str(self) -> str:
        lines = [
            "=" * 50,
            "RAG Evaluation Report",
            f"Timestamp : {self.timestamp}",
            f"Samples   : {self.num_samples}  (errors: {self.num_errors})",
            "=" * 50,
            "",
            "Generation Metrics (RAGAS / Claude judge)",
            "─" * 40,
            f"  Faithfulness       : {self.faithfulness:.3f}  "
            f"(threshold: {THRESHOLDS['faithfulness']})  "
            + ("PASS" if self.threshold_results.get('faithfulness') else "FAIL"),
            f"  Answer Relevancy   : {self.answer_relevancy:.3f}  "
            f"(threshold: {THRESHOLDS['answer_relevancy']})  "
            + ("PASS" if self.threshold_results.get('answer_relevancy') else "FAIL"),
            f"  Context Precision  : {self.context_precision:.3f}  "
            f"(threshold: {THRESHOLDS['context_precision']})  "
            + ("PASS" if self.threshold_results.get('context_precision') else "FAIL"),
            f"  Context Recall     : {self.context_recall:.3f}  "
            f"(threshold: {THRESHOLDS['context_recall']})  "
            + ("PASS" if self.threshold_results.get('context_recall') else "FAIL"),
            f"  Answer Correctness : {self.answer_correctness:.3f}  "
            f"(threshold: {THRESHOLDS['answer_correctness']})  "
            + ("PASS" if self.threshold_results.get('answer_correctness') else "FAIL"),
            "",
            "Retrieval Metrics (LLM-free)",
            "─" * 40,
            f"  Hit Rate @k        : {self.hit_rate:.3f}  "
            f"(threshold: {RETRIEVAL_THRESHOLDS['hit_rate_at_5']})  "
            + ("PASS" if self.threshold_results.get('hit_rate') else "FAIL"),
            f"  MRR                : {self.mrr:.3f}  "
            f"(threshold: {RETRIEVAL_THRESHOLDS['mrr']})  "
            + ("PASS" if self.threshold_results.get('mrr') else "FAIL"),
            f"  MAP@k              : {self.map_at_k:.3f}",
            f"  Mean Precision@k   : {self.mean_precision_at_k:.3f}",
            "",
            "Overall: "
            + (
                "ALL THRESHOLDS PASSED"
                if self.passes_all_thresholds()
                else "SOME THRESHOLDS FAILED"
            ),
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _traces_to_retrieval_samples(traces: list[EvalTrace]) -> list[RetrievalSample]:
    """Convert EvalTrace list → RetrievalSample list for retrieval metrics."""
    samples = []
    for trace in traces:
        if trace.has_error():
            continue
        samples.append(
            RetrievalSample(
                question=trace.question,
                retrieved_chunks=trace.retrieved_chunks,
                ground_truth_contexts=trace.ground_truth_contexts,
            )
        )
    return samples


def _traces_to_generation_samples(traces: list[EvalTrace]) -> list[GenerationSample]:
    """Convert EvalTrace list → GenerationSample list for RAGAS metrics."""
    samples = []
    for trace in traces:
        if trace.has_error() or not trace.answer:
            continue
        samples.append(
            GenerationSample(
                question=trace.question,
                answer=trace.answer,
                retrieved_contexts=trace.retrieved_chunks,
                reference_answer=trace.ground_truth_answer,
            )
        )
    return samples


def _compute_threshold_results(report: EvalReport) -> dict[str, bool]:
    """Check each metric against its configured threshold."""
    return {
        "faithfulness": report.faithfulness >= THRESHOLDS["faithfulness"],
        "answer_relevancy": report.answer_relevancy >= THRESHOLDS["answer_relevancy"],
        "context_precision": report.context_precision >= THRESHOLDS["context_precision"],
        "context_recall": report.context_recall >= THRESHOLDS["context_recall"],
        "answer_correctness": report.answer_correctness >= THRESHOLDS["answer_correctness"],
        "hit_rate": report.hit_rate >= RETRIEVAL_THRESHOLDS["hit_rate_at_5"],
        "mrr": report.mrr >= RETRIEVAL_THRESHOLDS["mrr"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ragas_evaluation(
    traces: list[EvalTrace],
    save_report: bool = True,
    reports_dir: Path = REPORTS_DIR,
) -> EvalReport:
    """
    Evaluate a list of ``EvalTrace`` objects and produce a full ``EvalReport``.

    Parameters
    ----------
    traces:
        Traces collected by ``pipeline_runner.run_pipeline()``.
    save_report:
        If True, write the report as JSON to ``reports_dir``.
    reports_dir:
        Directory where the report JSON file is saved.

    Returns
    -------
    EvalReport
        Merged retrieval + generation metrics with threshold pass/fail flags.
    """
    num_errors = sum(1 for t in traces if t.has_error())
    logger.info(
        "Starting evaluation: total=%d  errors=%d  valid=%d",
        len(traces),
        num_errors,
        len(traces) - num_errors,
    )

    # 1. Retrieval metrics (cheap, no API calls)
    retrieval_samples = _traces_to_retrieval_samples(traces)
    logger.info("Computing retrieval metrics on %d samples...", len(retrieval_samples))
    retrieval_metrics: RetrievalMetrics = compute_retrieval_metrics(retrieval_samples)
    logger.info("Retrieval metrics done:\n%s", retrieval_metrics)

    # 2. Generation metrics (RAGAS + Claude judge)
    generation_samples = _traces_to_generation_samples(traces)
    logger.info("Running RAGAS evaluation on %d samples...", len(generation_samples))
    generation_metrics: GenerationMetrics = compute_generation_metrics(generation_samples)
    logger.info("Generation metrics done:\n%s", generation_metrics)

    # 3. Assemble report
    report = EvalReport(
        timestamp=datetime.now(UTC).isoformat(),
        num_samples=len(traces),
        num_errors=num_errors,
        # Generation
        faithfulness=generation_metrics.faithfulness,
        answer_relevancy=generation_metrics.answer_relevancy,
        context_precision=generation_metrics.context_precision,
        context_recall=generation_metrics.context_recall,
        answer_correctness=generation_metrics.answer_correctness,
        # Retrieval
        hit_rate=retrieval_metrics.hit_rate,
        mrr=retrieval_metrics.mrr,
        map_at_k=retrieval_metrics.map_at_k,
        mean_precision_at_k=retrieval_metrics.mean_precision_at_k,
        # Per-sample details
        generation_per_sample=generation_metrics.per_sample,
        retrieval_per_sample=retrieval_metrics.per_sample,
        traces=[t.to_dict() for t in traces],
    )

    # 4. Compute threshold pass/fail
    report.threshold_results = _compute_threshold_results(report)

    # 5. Optionally save to disk
    if save_report:
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_path = reports_dir / f"report_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Report saved → %s", report_path)
        # Also write a "latest" symlink / copy for easy CI access
        latest_path = reports_dir / "report_latest.json"
        with open(latest_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Latest report updated → %s", latest_path)

    return report


def load_latest_report(reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """
    Load the most recently saved evaluation report from disk.

    Used by ``tests/eval/test_eval_thresholds.py`` for CI threshold assertions.
    """
    latest = reports_dir / "report_latest.json"
    if not latest.exists():
        # Fallback: find the most recent timestamped file
        candidates = sorted(reports_dir.glob("report_2*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No evaluation report found in {reports_dir}. "
                "Run `python -m eval.run_eval` first to generate one."
            )
        latest = candidates[-1]

    with open(latest, encoding="utf-8") as fh:
        return json.load(fh)
