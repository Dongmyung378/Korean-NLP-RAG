"""Command-line entry point for the locally runnable Korean RAG portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    DEFAULT_MIN_READER_SCORE,
    DEFAULT_MIN_RERANKER_SCORE,
    DEFAULT_MIN_RRF_SCORE,
    KoreanRAGPipeline,
)
from .retrieval import ArtifactPaths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Korean RAG portfolio in a terminal.")
    parser.add_argument("--question", "-q", help="Korean question to answer")
    parser.add_argument("--outputs-root", type=Path, default=Path(__file__).resolve().parents[1] / "transformer" / "outputs", help="Directory containing week2, week3, and week4 outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-reader", action="store_true", help="Show retrieved evidence without downloading or running the QA reader")
    parser.add_argument("--min-rrf-score", type=float, default=DEFAULT_MIN_RRF_SCORE, help="Minimum hybrid retrieval confidence for an answer")
    parser.add_argument("--min-reranker-score", type=float, default=DEFAULT_MIN_RERANKER_SCORE, help="Minimum Cross-Encoder relevance for an answer")
    parser.add_argument("--min-reader-score", type=float, default=DEFAULT_MIN_READER_SCORE, help="Minimum QA reader confidence for an answer")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def _print_result(result: dict) -> None:
    print(f"\nAnswer: {result['answer'] or 'No reader was used.'}")
    print(f"Source: {result['answer_source']}")
    if "reader_score" in result:
        print(f"Reader confidence: {result['reader_score']:.4f}")
    if "abstention_reasons" in result:
        print("Reason: " + " ".join(result["abstention_reasons"]))
    if "citation" in result:
        citation = result["citation"]
        print(f"Citation: {citation['title']} (chunk {citation['chunk_id']})")
    print("\nTop evidence")
    for item in result["evidence"]:
        preview = item["context"].replace("\n", " ")[:220]
        print(f"{item['rank']}. {item['title']} | reranker={item['reranker_score']:.4f}")
        print(f"   {preview}")


def main() -> None:
    args = _parser().parse_args()
    question = args.question or input("Question: ").strip()
    pipeline = KoreanRAGPipeline(ArtifactPaths(args.outputs_root.resolve()), device=args.device, use_reader=not args.no_reader)
    result = pipeline.answer(
        question,
        min_rrf_score=args.min_rrf_score,
        min_reranker_score=args.min_reranker_score,
        min_reader_score=args.min_reader_score,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "")
    if not args.json:
        _print_result(result)


if __name__ == "__main__":
    main()
