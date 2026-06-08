#!/usr/bin/env python3
"""
CLI script for bulk ingestion of documents into the Chroma vector store.

Usage:
    python scripts/ingest_docs.py --dir ./data
    python scripts/ingest_docs.py --file ./data/report.pdf
    python scripts/ingest_docs.py --dir ./data --pattern "*.pdf"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging, get_logger
from app.rag.ingestion import DocumentIngestionPipeline

setup_logging()
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".mdx", ".rst"}


def ingest_file(pipeline: DocumentIngestionPipeline, file_path: Path) -> dict:
    try:
        result = pipeline.run(file_path)
        print(
            f"  [OK] {result['source']} — {result['chunks_stored']} chunks "
            f"from {result['pages_loaded']} page(s)  [doc_id={result['doc_id']}]"
        )
        return result
    except Exception as exc:
        print(f"  [FAIL] {file_path.name} — {exc}")
        return {"status": "error", "source": file_path.name, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingest documents into Chroma.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Ingest a single file.")
    group.add_argument("--dir", type=Path, help="Ingest all supported files in a directory.")
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern to filter files when using --dir (default: '*')",
    )
    args = parser.parse_args()

    pipeline = DocumentIngestionPipeline()
    results = []

    if args.file:
        if not args.file.exists():
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        print(f"Ingesting file: {args.file}")
        results.append(ingest_file(pipeline, args.file))

    elif args.dir:
        if not args.dir.is_dir():
            print(f"Error: directory not found: {args.dir}")
            sys.exit(1)
        files = [
            f for f in args.dir.glob(args.pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            print(f"No supported files found in {args.dir} with pattern '{args.pattern}'.")
            sys.exit(0)
        print(f"Ingesting {len(files)} file(s) from {args.dir}...")
        for f in sorted(files):
            results.append(ingest_file(pipeline, f))

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - succeeded
    print(f"\nDone: {succeeded} succeeded, {failed} failed.")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
