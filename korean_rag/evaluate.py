"""저장된 Week 4 질문으로 로컬 앱을 평가한다. 학습이나 임계값 탐색은 하지 않는다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .pipeline import (DEFAULT_MIN_READER_SCORE, DEFAULT_MIN_RERANKER_SCORE,
                       DEFAULT_MIN_RRF_SCORE, KoreanRAGPipeline)
from .retrieval import ArtifactPaths


def compact(text):
    return re.sub(r"[^0-9a-z가-힣]", "", str(text).lower())


def contains_gold(answer, gold):
    normalized = compact(gold)
    return bool(normalized) and normalized in compact(answer)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=root / "transformer/outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-dir", type=Path, default=root / "reports/local_evaluation")
    parser.add_argument("--unanswerable-file", type=Path, default=root / "tests/data/unanswerable_questions.json")
    args = parser.parse_args()
    paths = ArtifactPaths(args.outputs_root.resolve())
    examples_path = paths.week4_dir / "week4_day5_rag_examples.csv"
    with examples_path.open(encoding="utf-8-sig", newline="") as stream:
        examples = list(csv.DictReader(stream))
    probes = json.loads(args.unanswerable_file.read_text(encoding="utf-8"))
    if not examples:
        parser.error("The evaluation CSV contains no questions.")
    loaded_at = time.perf_counter()
    pipeline = KoreanRAGPipeline(paths, device=args.device)
    load_seconds = time.perf_counter() - loaded_at
    predictions = []
    for index, row in enumerate(examples):
        started = time.perf_counter()
        result = pipeline.answer(row["question"])
        seconds = time.perf_counter() - started
        accepted = result["answer_source"] == "extractive_reader"
        predictions.append({
            "query_id": row["query_id"], "question": row["question"], "gold_answer": row["gold_answer"],
            "answer": result["answer"], "answer_source": result["answer_source"],
            "contains_gold": accepted and contains_gold(result["answer"], row["gold_answer"]),
            "reader_score": result.get("reader_score"), "citation": result.get("citation"),
            "abstention_reasons": result.get("abstention_reasons", []), "seconds": seconds,
        })
        if (index + 1) % 25 == 0:
            print(f"Evaluated {index + 1}/{len(examples)} questions", file=sys.stderr, flush=True)
    probe_results = []
    for probe in probes:
        result = pipeline.answer(probe["question"])
        probe_results.append({**probe, **result})
    accepted = sum(row["answer_source"] == "extractive_reader" for row in predictions)
    correct = sum(row["contains_gold"] for row in predictions)
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "Reused Week 4 fixed questions; regression evaluation, not a new held-out test.",
        "metric_definition": "Normalized gold substring in accepted answer, not semantic correctness.",
        "question_count": len(predictions), "accepted": accepted, "abstained": len(predictions) - accepted,
        "answers_containing_gold": correct, "answer_containment_rate": correct / len(predictions),
        "accepted_answer_containment_rate": correct / accepted if accepted else None,
        "unanswerable_probe_count": len(probes),
        "unanswerable_probes_abstained": sum(row["answer_source"] == "abstained" for row in probe_results),
        "probe_scope": "Small manually selected debugging set; includes known failure cases, not an OOD benchmark.",
        "device": str(pipeline.device), "model_load_seconds": load_seconds,
        "mean_query_seconds_excluding_load": sum(row["seconds"] for row in predictions) / len(predictions),
        "thresholds": {"rrf": DEFAULT_MIN_RRF_SCORE, "reranker": DEFAULT_MIN_RERANKER_SCORE, "reader": DEFAULT_MIN_READER_SCORE},
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "pandas", "faiss-cpu", "rank-bm25")},
        "python": platform.python_version(),
        "input_sha256": {"examples": file_hash(examples_path), "probes": file_hash(args.unanswerable_file)},
        "source_sha256": {name: file_hash(root / "korean_rag" / name) for name in ("pipeline.py", "retrieval.py", "reranking.py", "evaluate.py")},
    }
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("metrics.json", metrics), ("predictions.json", predictions), ("unanswerable_probes.json", probe_results)):
        (args.save_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
