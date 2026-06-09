"""
RAG Evaluation CLI — main entry point.

Usage
-----
Run the full evaluation pipeline (requires documents already ingested):

    python -m eval.run_eval

Generate a fresh dataset first, then evaluate:

    python -m eval.run_eval --generate-dataset

Evaluate only the adversarial edge cases:

    python -m eval.run_eval --adversarial-only

Use a custom dataset file:

    python -m eval.run_eval --dataset /path/to/my_dataset.json

Skip saving the report to disk (dry run):

    python -m eval.run_eval --no-save-report

Exit codes
----------
0  All metric thresholds passed (or --no-threshold-check flag used).
1  One or more thresholds failed.
2  Fatal error (missing dataset, no ChromaDB documents, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.config import (
    ADVERSARIAL_DATASET_PATH,
    AGENT_TIMEOUT_SECONDS,
    EVAL_DATASET_PATH,
    REPORTS_DIR,
)
from eval.runners.pipeline_runner import run_pipeline
from eval.runners.ragas_runner import EvalReport, run_ragas_evaluation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="Run the RAG evaluation pipeline and produce a scored report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a JSON evaluation dataset (overrides default).",
    )
    dataset_group.add_argument(
        "--adversarial-only",
        action="store_true",
        help="Run only the adversarial edge-case dataset.",
    )

    # Dataset generation
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help=(
            "Generate a fresh QA dataset from ChromaDB before evaluating. "
            "Requires documents to be ingested first."
        ),
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        metavar="N",
        help="Number of QA pairs to generate (only used with --generate-dataset).",
    )

    # Output control
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Do not write the report JSON to disk.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory where report JSON files are saved.",
    )

    # Threshold control
    parser.add_argument(
        "--no-threshold-check",
        action="store_true",
        help="Print the report but always exit 0 (ignore threshold failures).",
    )

    # Execution control
    parser.add_argument(
        "--timeout",
        type=float,
        default=AGENT_TIMEOUT_SECONDS,
        help="Per-question agent timeout in seconds.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )

    return parser


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(2)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        logger.error("Dataset at %s is empty or not a JSON array.", path)
        sys.exit(2)
    return data


def _maybe_generate_dataset(args: argparse.Namespace) -> None:
    """Run the dataset generator if --generate-dataset was passed."""
    if not args.generate_dataset:
        return

    from eval.config import TESTSET_SIZE
    from eval.dataset.generate_dataset import generate_dataset

    size = args.dataset_size or TESTSET_SIZE
    logger.info("Generating evaluation dataset: size=%d ...", size)
    try:
        records = generate_dataset(size=size)
        logger.info("Dataset generation complete: %d records", len(records))
    except RuntimeError as exc:
        logger.error("Dataset generation failed: %s", exc)
        sys.exit(2)


def _resolve_dataset_path(args: argparse.Namespace) -> Path:
    """Determine which dataset file to use based on CLI flags."""
    if args.dataset:
        return args.dataset
    if args.adversarial_only:
        return ADVERSARIAL_DATASET_PATH
    return EVAL_DATASET_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Optionally generate the dataset
    _maybe_generate_dataset(args)

    # Resolve dataset path
    dataset_path = _resolve_dataset_path(args)
    logger.info("Loading dataset from: %s", dataset_path)
    dataset = _load_dataset(dataset_path)
    logger.info("Loaded %d evaluation cases.", len(dataset))

    # Run the pipeline (real agent calls, trace collection)
    logger.info("Running RAG pipeline against evaluation dataset...")
    traces = run_pipeline(dataset=dataset, timeout=args.timeout)

    errors = sum(1 for t in traces if t.has_error())
    if errors:
        logger.warning(
            "%d/%d questions failed during pipeline execution (see traces for details).",
            errors,
            len(traces),
        )

    # Run RAGAS + retrieval evaluation
    logger.info("Computing metrics...")
    report: EvalReport = run_ragas_evaluation(
        traces=traces,
        save_report=not args.no_save_report,
        reports_dir=args.reports_dir,
    )

    # Print summary
    print()
    print(report.summary_str())
    print()

    # Exit code based on thresholds
    if args.no_threshold_check:
        sys.exit(0)

    if report.passes_all_thresholds():
        logger.info("All thresholds passed.")
        sys.exit(0)
    else:
        failed = [k for k, v in report.threshold_results.items() if not v]
        logger.error("Threshold failures: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
