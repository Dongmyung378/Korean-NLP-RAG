"""Select bounded Reader and threshold variants using verified dev caches only.

No model is loaded and no final question is read. Every compared configuration
and per-question prediction is retained, including rejected experiments.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from korean_rag.abstention_benchmark import threshold_promotion_decision, verify_abstention_directory
from korean_rag.benchmark import (canonical_sha256, read_jsonl, score_predictions,
                                  sha256_file, verify_benchmark_directory, write_jsonl)
from korean_rag.runtime import PortableRAG, verify_bundle


BASE_THRESHOLDS = {"min_rrf_score": .025, "min_reranker_score": .5,
                   "min_reader_score": .1, "require_explicit_year": True}
BASE_READER_CONFIG = {"context_k": 3, "context_mode": "chunk", "max_length": 512, "score_mode": "reader"}
NULL_SCORE_MARGIN = 1e-6
READER_VARIANTS = [
    {**BASE_READER_CONFIG, "null_score_margin": NULL_SCORE_MARGIN},
    {"context_k": 3, "context_mode": "parent_window", "max_length": 512, "score_mode": "reader", "null_score_margin": NULL_SCORE_MARGIN},
    {"context_k": 5, "context_mode": "parent_window", "max_length": 512, "score_mode": "reader", "null_score_margin": NULL_SCORE_MARGIN},
    {"context_k": 5, "context_mode": "parent_window", "max_length": 512, "score_mode": "reader_reranker", "null_score_margin": NULL_SCORE_MARGIN},
]
THRESHOLD_VARIANTS = [{"min_rrf_score": rrf, "min_reranker_score": reranker,
                       "min_reader_score": reader, "require_explicit_year": True}
                      for rrf, reranker, reader in itertools.product((.025, .015), (.5, .65), (.05, .1, .2))]


def reader_gain_decision(candidate: dict, incumbent: dict, protocol: dict) -> dict:
    selection = protocol["selection"]
    em_delta = candidate["answerable_exact_match"] - incumbent["answerable_exact_match"]
    f1_delta = candidate["answerable_character_f1"] - incumbent["answerable_character_f1"]
    em_gain = em_delta + 1e-12 >= selection["minimum_answer_em_gain"]
    f1_gain = f1_delta + 1e-12 >= selection["minimum_answer_f1_gain"] and em_delta + 1e-12 >= -selection["maximum_em_regression_for_f1_gain"]
    return {"promote": em_gain or f1_gain, "em_delta": em_delta, "character_f1_delta": f1_delta,
            "scope": "Stage3 development-only Reader comparison against variant0 with unchanged thresholds."}


def exact_question_alignment(rows: list[dict], questions: list[dict], *, require_text: bool) -> None:
    expected = {q["query_id"]: q for q in questions}
    observed = {row.get("query_id"): row for row in rows}
    if len(observed) != len(rows) or set(observed) != set(expected):
        raise ValueError("Result IDs must match the full declared development subset exactly.")
    for query_id, row in observed.items():
        if require_text and row.get("question") != expected[query_id]["question"]:
            raise ValueError(f"Cached question text differs from the frozen manifest: {query_id}")
        if "question" in row and row["question"] != expected[query_id]["question"]:
            raise ValueError(f"Result question text differs from the frozen manifest: {query_id}")


def validate_cached_candidates(rows: list[dict], chunks: list[dict], corpus: dict[str, dict]) -> None:
    by_id = {r["chunk_id"]: r for r in chunks}
    for row in rows:
        evidence = row.get("evidence", [])
        if len(evidence) < 5 or len({r["chunk_id"] for r in evidence}) != len(evidence):
            raise ValueError("Cache evidence must contain unique ranked retrieval candidates.")
        evidence_by_id = {}
        for rank, item in enumerate(evidence, 1):
            expected = by_id.get(item["chunk_id"])
            if item.get("rank") != rank or expected is None or any(item.get(key) != expected[key] for key in ("context_id", "context", "char_start", "char_end")):
                raise ValueError("Cached retrieval evidence differs from the verified bundle.")
            if any(not isinstance(item.get(key), (int, float)) or not math.isfinite(item[key]) for key in ("rrf_score", "reranker_score")):
                raise ValueError("Cached ranking scores must be finite.")
            evidence_by_id[item["chunk_id"]] = item
        pools = row.get("pools", {})
        if set(pools) != {"chunk", "parent_window"}:
            raise ValueError("Reader selection requires both declared inference pools.")
        for mode, candidates in pools.items():
            limit = 3 if mode == "chunk" else 5
            seen_windows = set()
            if not candidates:
                raise ValueError("A Reader cache pool cannot be empty.")
            for item in candidates:
                source = evidence_by_id.get(item.get("chunk_id"))
                if source is None or item.get("rank") != source["rank"] or item["rank"] > limit:
                    raise ValueError("Reader candidate rank is outside its declared pool.")
                if any(item.get(key) != source[key] for key in ("context_id", "rrf_score", "reranker_score")):
                    raise ValueError("Reader candidates must preserve their retrieval identity and scores.")
                start = item.get("reader_window_start")
                context = item.get("context")
                if not isinstance(start, int) or isinstance(start, bool) or start < 0 or not isinstance(context, str):
                    raise ValueError("Reader window offsets must identify a raw source interval.")
                original = corpus[item["context_id"]]["context"]
                if original[start:start + len(context)] != context:
                    raise ValueError("Reader cache context does not match its raw source interval.")
                if mode == "chunk" and (context != source["context"] or start != source["char_start"]):
                    raise ValueError("The chunk Reader pool must use the retrieved chunk unchanged.")
                key = (item["context_id"], start, context)
                if key in seen_windows:
                    raise ValueError("Reader cache windows must be deduplicated.")
                seen_windows.add(key)
                score = item.get("reader_score")
                if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Reader confidence must be finite and in [0,1].")
                null_score = item.get("null_score")
                if not isinstance(null_score, (int, float)) or not math.isfinite(null_score) or not 0 <= null_score <= 1:
                    raise ValueError("Reader caches must retain finite CLS null confidence in [0,1].")
                answer = item.get("answer")
                if not isinstance(answer, str):
                    raise ValueError("Cached answers must be strings.")
                if answer:
                    a, b = item.get("answer_char_start"), item.get("answer_char_end")
                    if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(context) or context[a:b].strip() != answer:
                        raise ValueError("Cached answer offsets do not reconstruct the predicted source span.")


def load_verified_cache(directory: Path, bundle: Path, questions: list[dict], input_metadata: dict) -> tuple[list[dict], dict]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    if report.get("configuration") != config or report.get("bundle_manifest_sha256") != sha256_file(bundle / "manifest.json"):
        raise ValueError("Reader cache configuration or bundle hash differs from the verified bundle.")
    if report.get("inputs") != input_metadata:
        raise ValueError("Reader cache input metadata must match the frozen full development set.")
    if report.get("cache_reader_variants") is not True:
        raise ValueError("The cache was not generated with all declared Reader variants.")
    cache_file = directory / "reader-cache.jsonl"
    if report.get("reader_cache_sha256") != sha256_file(cache_file):
        raise ValueError("Reader cache checksum is missing or mismatched.")
    predictions_file = directory / "predictions.jsonl"
    if report.get("predictions_sha256") != sha256_file(predictions_file):
        raise ValueError("Prediction checksum is missing or mismatched.")
    cache = read_jsonl(cache_file)
    predictions = read_jsonl(predictions_file)
    exact_question_alignment(cache, questions, require_text=True)
    exact_question_alignment(predictions, questions, require_text=False)
    rescored, _ = score_predictions(questions, predictions)
    if rescored != report.get("qa"):
        raise ValueError("Cached QA summary does not match the saved predictions.")
    if config["reader"] != BASE_READER_CONFIG or config["thresholds"] != BASE_THRESHOLDS:
        raise ValueError("Reader comparisons must start from the predeclared fixed baseline configuration.")
    corpus = {r["context_id"]: r for r in read_jsonl(bundle / "corpus.jsonl")}
    validate_cached_candidates(cache, read_jsonl(bundle / "chunks.jsonl"), corpus)
    return cache, report


def load_baseline_predictions(directory: Path, questions: list[dict], *, input_sha256: str, benchmark_sha256: str, attestation: Path | None = None) -> tuple[list[dict], dict]:
    report_file, predictions_file = directory / "metrics.json", directory / "predictions.jsonl"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    metadata = report
    if attestation is not None:
        metadata = json.loads(attestation.read_text(encoding="utf-8"))
        if metadata.get("report_sha256") != sha256_file(report_file):
            raise ValueError("The separate baseline audit does not bind the original report.")
    configuration = metadata.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("reader") != BASE_READER_CONFIG or configuration.get("thresholds") != BASE_THRESHOLDS:
        raise ValueError("Original baseline Reader and threshold settings must be explicitly attested.")
    if metadata.get("predictions_sha256") != sha256_file(predictions_file):
        raise ValueError("Original baseline prediction checksum must be explicitly bound.")
    inputs = metadata.get("inputs", {})
    if metadata.get("question_manifest_sha256", inputs.get("question_file_sha256")) != input_sha256:
        raise ValueError("Original baseline must bind the exact development question file.")
    declared_benchmark = metadata.get("benchmark_sha256", inputs.get("benchmark_sha256", metadata.get("no_answer_benchmark_sha256")))
    if declared_benchmark != benchmark_sha256:
        raise ValueError("Original baseline benchmark digest is missing or mismatched.")
    predictions = read_jsonl(predictions_file)
    exact_question_alignment(predictions, questions, require_text=False)
    rescored, _ = score_predictions(questions, predictions)
    if rescored != report.get("qa"):
        raise ValueError("Original baseline metrics differ from its saved predictions.")
    environment = report.get("environment", {})
    return predictions, {"metrics_sha256": sha256_file(report_file), "predictions_sha256": sha256_file(predictions_file),
                         "attestation_sha256": sha256_file(attestation) if attestation else None, "qa": rescored,
                         "device": metadata.get("device", report.get("device", environment.get("device"))),
                         "gpu_name": metadata.get("gpu_name", report.get("gpu_name", environment.get("gpu_name")))}


def validate_same_device(cache_report: dict, baseline_answerable: dict, baseline_no_answer: dict) -> None:
    environment = cache_report.get("environment", {})
    measured_device = environment.get("device")
    devices = [measured_device, baseline_answerable.get("device"), baseline_no_answer.get("device")]
    if any(not isinstance(device, str) or device.split(":", 1)[0] not in {"cpu", "cuda"} for device in devices):
        raise ValueError("Candidate and both original baselines must declare their measured execution device.")
    if len({device.split(":", 1)[0] for device in devices}) != 1:
        raise ValueError("Promotion requires original answerable and no-answer baselines measured on the candidate device.")
    for baseline in (baseline_answerable, baseline_no_answer):
        if baseline.get("gpu_name") and baseline["gpu_name"] != environment.get("gpu_name"):
            raise ValueError("Candidate and original GPU baseline hardware must match.")


def select_from_cache(questions: list[dict], cache: list[dict], baseline_metrics: dict, protocol: dict) -> tuple[dict, list[dict], list[dict], dict[str, list[dict]]]:
    exact_question_alignment(cache, questions, require_text=True)
    if len(READER_VARIANTS) > protocol["selection"]["max_reader_variants"] or len(THRESHOLD_VARIANTS) > protocol["selection"]["max_abstention_variants"]:
        raise ValueError("The declared comparison exceeds the frozen experiment budget.")
    cache_by_id = {row["query_id"]: row for row in cache}
    all_predictions = {}

    def evaluate_config(reader, thresholds):
        predictions = []
        for question in questions:
            row = cache_by_id[question["query_id"]]
            candidates = [item for item in row["pools"][reader["context_mode"]] if item["rank"] <= reader["context_k"]]
            result = PortableRAG.select_answer(question["question"], row["evidence"], candidates, thresholds, reader)
            compact = {key: value for key, value in result.items() if key != "evidence"}
            if compact.get("citation"):
                compact["citation"] = {key: value for key, value in compact["citation"].items() if key != "context"}
            predictions.append({"query_id": question["query_id"], **compact})
        metrics, scored = score_predictions(questions, predictions)
        for prediction, score in zip(predictions, scored):
            prediction.update(answerable=score["answerable"], exact_match=score["exact_match"], character_f1=score["character_f1"])
        return metrics, predictions

    reader_results = []
    for index, reader in enumerate(READER_VARIANTS):
        metrics, predictions = evaluate_config(reader, BASE_THRESHOLDS)
        reader_results.append({"variant_id": f"reader-{index}", "configuration": reader, "thresholds": BASE_THRESHOLDS, "qa": metrics})
        all_predictions[f"reader-{index}"] = predictions
    incumbent = reader_results[0]["qa"]
    for row in reader_results:
        row["promotion"] = reader_gain_decision(row["qa"], incumbent, protocol)
    eligible_readers = [index for index, row in enumerate(reader_results) if row["promotion"]["promote"]]
    selected_reader = max(eligible_readers, key=lambda index: (reader_results[index]["qa"]["answerable_exact_match"], reader_results[index]["qa"]["answerable_character_f1"], -index)) if eligible_readers else 0

    threshold_results = []
    for index, thresholds in enumerate(THRESHOLD_VARIANTS):
        metrics, predictions = evaluate_config(READER_VARIANTS[selected_reader], thresholds)
        promotion = threshold_promotion_decision(metrics, baseline_metrics, protocol)
        threshold_results.append({"variant_id": f"threshold-{index}", "configuration": thresholds,
                                  "reader_variant_id": f"reader-{selected_reader}", "qa": metrics, "promotion": promotion})
        all_predictions[f"threshold-{index}"] = predictions
    eligible_thresholds = [index for index, row in enumerate(threshold_results) if row["promotion"]["promote"]]
    # Quality first; safety breaks a quality tie. Prefer the originally declared
    # thresholds and then declared order when quality and safety are tied.
    selected_threshold = max(eligible_thresholds, key=lambda index: (
        threshold_results[index]["qa"]["answerable_exact_match"],
        threshold_results[index]["qa"]["answerable_character_f1"],
        -threshold_results[index]["qa"]["no_answer_false_answer_rate"],
        threshold_results[index]["configuration"] == BASE_THRESHOLDS, -index)) if eligible_thresholds else None
    selection = {"reader_variant_id": f"reader-{selected_reader}", "reader_configuration": READER_VARIANTS[selected_reader],
                 "reader_promoted": bool(eligible_readers),
                 "threshold_variant_id": f"threshold-{selected_threshold}" if selected_threshold is not None else None,
                 "thresholds": THRESHOLD_VARIANTS[selected_threshold] if selected_threshold is not None else None,
                 "joint_promotion": selected_threshold is not None,
                 "requires_original_baseline_fallback": selected_threshold is None,
                 "reason": "A declared candidate satisfies every development quality, coverage and safety rule." if selected_threshold is not None else "No bounded threshold candidate satisfies every frozen promotion rule; the new bundle is not approved over the original baseline.",
                 "final_questions_used": 0}
    return selection, reader_results, threshold_results, all_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--no-answer-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-answerable-dir", type=Path, required=True)
    parser.add_argument("--baseline-no-answer-dir", type=Path, required=True)
    parser.add_argument("--baseline-answerable-attestation", type=Path)
    parser.add_argument("--baseline-no-answer-attestation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a configuration selection record.")
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/validation-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    benchmark = verify_benchmark_directory(args.benchmark, protocol_path)
    abstention = verify_abstention_directory(args.no_answer_dir, project / "configs/no-answer-protocol.json")
    if abstention["answerable_benchmark_sha256"] != benchmark["benchmark_sha256"]:
        raise ValueError("Answerable and no-answer benchmark versions differ.")
    answerable = read_jsonl(args.benchmark / "dev.jsonl")
    no_answer = read_jsonl(args.no_answer_dir / "dev.jsonl")
    questions = answerable + no_answer
    if len(answerable) != benchmark["actual_counts"]["dev"] or len(no_answer) != abstention["counts"]["dev"] or any(q.get("benchmark_split") != "dev" for q in questions):
        raise ValueError("Only the full immutable development sets may select configurations.")
    verify_bundle(args.bundle)
    input_metadata = {"split": "dev", "benchmark_sha256": benchmark["benchmark_sha256"],
                      "question_file_sha256": sha256_file(args.benchmark / "dev.jsonl"),
                      "no_answer_benchmark_sha256": abstention["no_answer_benchmark_sha256"],
                      "no_answer_file_sha256": sha256_file(args.no_answer_dir / "dev.jsonl")}
    declaration = {"schema_version": 1, "inputs": input_metadata, "reader_variants": READER_VARIANTS,
                   "reader_fixed_thresholds": BASE_THRESHOLDS, "threshold_variants": THRESHOLD_VARIANTS,
                   "numerical_guard": {"null_score_margin": NULL_SCORE_MARGIN, "selection_tuned": False,
                                       "scope": "One fixed conservative float32 near-tie abstention guard for every candidate. The unchanged original baseline keeps margin0. Device comparisons use the original baseline measured on the same device; this guard does not guarantee universal bitwise equivalence."},
                   "threshold_comparison_baseline": "Original development model on the identical full 304 answerable and 80 synthetic no-answer development questions.",
                   "reader_tie_rule": "Prefer the simpler declared Reader variant when EM and F1 tie; per-variant serving latency is measured separately.",
                   "threshold_tie_rule": "Prefer lower no-answer false-answer rate after EM/F1, then original thresholds, then declared order.",
                   "protocol_sha256": sha256_file(protocol_path), "selector_sha256": sha256_file(Path(__file__)),
                   "final_questions_used": 0}
    cache, cache_report = load_verified_cache(args.cache_dir, args.bundle, questions, input_metadata)
    original_answerable, baseline_answerable = load_baseline_predictions(args.baseline_answerable_dir, answerable,
        input_sha256=input_metadata["question_file_sha256"], benchmark_sha256=benchmark["benchmark_sha256"], attestation=args.baseline_answerable_attestation)
    original_no_answer, baseline_no_answer = load_baseline_predictions(args.baseline_no_answer_dir, no_answer,
        input_sha256=input_metadata["no_answer_file_sha256"], benchmark_sha256=abstention["no_answer_benchmark_sha256"], attestation=args.baseline_no_answer_attestation)
    validate_same_device(cache_report, baseline_answerable, baseline_no_answer)
    baseline_metrics, _ = score_predictions(questions, original_answerable + original_no_answer)
    selection, readers, thresholds, predictions = select_from_cache(questions, cache, baseline_metrics, protocol)
    args.output.mkdir(parents=True)
    for name, value in (("selection-declaration.json", declaration), ("reader-comparison.json", readers), ("threshold-comparison.json", thresholds)):
        with (args.output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for name, rows in predictions.items():
        write_jsonl(args.output / f"{name}-predictions.jsonl", rows)
    selected_config = deepcopy(cache_report["configuration"])
    if selection["joint_promotion"]:
        selected_config.update(reader=selection["reader_configuration"], thresholds=selection["thresholds"])
        with (args.output / "selected-config.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(selected_config, ensure_ascii=False, indent=2) + "\n")
    result = {**selection, "original_baseline_qa": baseline_metrics,
              "cache_sha256": cache_report["reader_cache_sha256"],
              "cache_metrics_sha256": sha256_file(args.cache_dir / "metrics.json"),
              "bundle_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
              "baseline_answerable": baseline_answerable, "baseline_no_answer": baseline_no_answer,
              "declaration_sha256": canonical_sha256(declaration),
              "result_files_sha256": {p.name: sha256_file(p) for p in sorted(args.output.iterdir())}}
    with (args.output / "selection.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
