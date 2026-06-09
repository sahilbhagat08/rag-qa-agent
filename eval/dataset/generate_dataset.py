"""
Dataset generator for RAG evaluation.

Uses RAGAS TestsetGenerator with Claude as both the generator and critic LLM
to synthesise a labelled QA dataset directly from documents already stored in
ChromaDB — no separate document files are needed.

Usage
-----
    # From the repo root:
    python -m eval.dataset.generate_dataset

    # Override the number of test cases:
    python -m eval.dataset.generate_dataset --size 30

    # Save to a custom path:
    python -m eval.dataset.generate_dataset --output /tmp/my_dataset.json

The generated file (eval/dataset/eval_dataset.json by default) is consumed
by eval/runners/pipeline_runner.py and eval/runners/ragas_runner.py.

Dataset schema
--------------
Each entry in the output JSON array contains:
    {
        "question":               str,   # Synthetic question
        "ground_truth_answer":    str,   # Reference answer from source chunks
        "ground_truth_contexts":  [str], # Source chunk texts used to build Q+A
        "metadata": {
            "question_type":  str,       # "simple" | "multi_context" | "reasoning"
            "source_doc_ids": [str],     # doc_id metadata values
            "sources":        [str],     # source filename metadata values
        }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is importable when running as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms

from app.core.config import get_settings
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.retriever import ChromaRetriever
from eval.config import (
    EVAL_DATASET_PATH,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    TESTSET_SIZE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_documents_from_chroma() -> list[Document]:
    """
    Pull all chunks stored in ChromaDB back into LangChain Document objects.

    This avoids re-reading source files — we generate the testset from exactly
    the content that has already been embedded and stored.
    """
    retriever = ChromaRetriever()
    collection = retriever._store._collection
    results = collection.get(include=["documents", "metadatas"])

    documents: list[Document] = []
    raw_docs = results.get("documents") or []
    raw_metas = results.get("metadatas") or []

    for content, meta in zip(raw_docs, raw_metas):
        if content and content.strip():
            documents.append(Document(page_content=content, metadata=meta or {}))

    logger.info("Loaded %d chunks from ChromaDB for dataset generation", len(documents))
    return documents


def _build_knowledge_graph(documents: list[Document]) -> KnowledgeGraph:
    """Convert LangChain Documents into a RAGAS KnowledgeGraph."""
    kg = KnowledgeGraph()
    for doc in documents:
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        kg.nodes.append(node)
    return kg


def _build_llm_and_embeddings(
    settings: Any,
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Instantiate the LLM judge and embeddings wrappers for RAGAS."""
    llm = ChatAnthropic(
        model=JUDGE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    embeddings = SentenceTransformerEmbeddings()
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate_dataset(
    size: int = TESTSET_SIZE,
    output_path: Path = EVAL_DATASET_PATH,
) -> list[dict]:
    """
    Generate a synthetic evaluation dataset and persist it to disk.

    Parameters
    ----------
    size:
        Number of QA pairs to generate.
    output_path:
        Where to write the output JSON file.

    Returns
    -------
    list[dict]
        The generated dataset entries (same content as the saved file).
    """
    settings = get_settings()

    # 1. Load source documents from ChromaDB
    documents = _load_documents_from_chroma()
    if not documents:
        raise RuntimeError(
            "No documents found in ChromaDB. "
            "Please ingest some documents first using POST /documents/ingest "
            "or scripts/ingest_docs.py before generating the evaluation dataset."
        )

    # 2. Build LLM / embeddings
    generator_llm, generator_embeddings = _build_llm_and_embeddings(settings)

    # 3. Build knowledge graph from chunks
    kg = _build_knowledge_graph(documents)

    # 4. Configure and run RAGAS TestsetGenerator
    logger.info("Starting testset generation: size=%d", size)
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )

    transforms = default_transforms(
        documents=documents,
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    testset = generator.generate(
        testset_size=size,
        transforms=transforms,
    )

    # 5. Convert RAGAS Dataset → our schema
    df = testset.to_pandas()
    records: list[dict] = []
    for _, row in df.iterrows():
        # RAGAS columns: user_input, reference, reference_contexts, synthesizer_name
        contexts: list[str] = row.get("reference_contexts") or []
        if isinstance(contexts, str):
            # Some RAGAS versions serialise as JSON string
            try:
                contexts = json.loads(contexts)
            except (json.JSONDecodeError, TypeError):
                contexts = [contexts]

        # Derive source metadata from first context
        sources: list[str] = []
        doc_ids: list[str] = []
        for doc in documents:
            for ctx in contexts:
                if ctx and doc.page_content.strip() == ctx.strip():
                    src = doc.metadata.get("source", "")
                    did = doc.metadata.get("doc_id", "")
                    if src and src not in sources:
                        sources.append(src)
                    if did and did not in doc_ids:
                        doc_ids.append(did)

        records.append(
            {
                "question": str(row.get("user_input", "")),
                "ground_truth_answer": str(row.get("reference", "")),
                "ground_truth_contexts": contexts,
                "metadata": {
                    "question_type": str(row.get("synthesizer_name", "simple")),
                    "source_doc_ids": doc_ids,
                    "sources": sources,
                },
            }
        )

    # 6. Persist to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Dataset saved: %d records → %s",
        len(records),
        output_path,
    )
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAGAS evaluation dataset from ChromaDB documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TESTSET_SIZE,
        help="Number of QA pairs to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DATASET_PATH,
        help="Output path for the generated JSON dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    records = generate_dataset(size=args.size, output_path=args.output)
    print(f"Generated {len(records)} QA pairs → {args.output}")
