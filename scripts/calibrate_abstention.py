"""Run nine separately declared development-only threshold refinements.

The frozen original protocol and failed selection records remain unchanged.
This command performs CPU scoring of verified saved Reader candidates, loads no
models, and never opens final question or final prediction records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from korean_rag.abstention_benchmark import threshold_promotion_decision, verify_abstention_directory
from korean_rag.benchmark import canonical_sha256, read_jsonl, score_predictions, sha256_file, verify_benchmark_directory, write_jsonl
from korean_rag.runtime import PortableRAG, verify_bundle
from scripts.evaluate_portable import REQUIRED_FINAL_SOURCES
from scripts.select_configuration import (
    BASE_THRESHOLDS, READER_VARIANTS, THRESHOLD_VARIANTS, exact_question_alignment,
    load_baseline_predictions, load_verified_cache, validate_same_device,
)

CALIBRATION_READER = READER_VARIANTS[3]
CALIBRATION_THRESHOLDS = [
    {"min_rrf_score": rrf, "min_reranker_score": .5, "min_reader_score": reader, "require_explicit_year": True}
    for rrf, reader in itertools.product((.028, .030, .032), (.3, .6, .9))
]


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_plan(plan: dict) -> None:
    expected = {
        "evaluation_split": "dev", "initial_reader_variants": 4,
        "initial_threshold_variants": 12, "additional_threshold_variants": 9,
        "total_threshold_variants": 21, "rrf_thresholds": [.028, .030, .032],
        "reader_thresholds": [.3, .6, .9], "reranker_threshold": .5,
        "require_explicit_year": True, "reader_variant_id": "reader-3",
        "reader_configuration": CALIBRATION_READER, "final_questions_used": 0,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("The follow-up must retain exactly the nine predeclared dev-only configurations.")


def validate_development_questions(questions: list[dict]) -> None:
    if not questions or any(question.get("benchmark_split") != "dev" for question in questions):
        raise ValueError("Final or undeclared questions cannot enter development calibration.")
    exact_question_alignment(questions, questions, require_text=True)


def verify_cache_source(report: dict, project: Path) -> dict[str, str]:
    source = report.get("source_sha256", {})
    if not set(REQUIRED_FINAL_SOURCES).issubset(source):
        raise ValueError("The cache must attest the complete inference and scoring source closure.")
    for name, expected in source.items():
        parts = name.split("/")
        if not name or "\\" in name or ":" in name or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Cache source paths must be canonical project-relative paths.")
        path = project.joinpath(*parts)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Cache inference source differs from the current scoring source: {name}")
    return source


def verify_initial_failure(directory: Path, questions: list[dict], protocol: dict) -> tuple[dict, dict[str, str]]:
    selection_path = directory / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    declaration_path = directory / "selection-declaration.json"
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    if selection.get("joint_promotion") is not False or selection.get("requires_original_baseline_fallback") is not True or selection.get("final_questions_used") != 0:
        raise ValueError("The follow-up requires a recorded failed initial development selection.")
    if selection.get("reader_variant_id") != "reader-3" or selection.get("reader_configuration") != CALIBRATION_READER:
        raise ValueError("Reader configuration cannot be reselected in the follow-up.")
    if declaration.get("reader_variants") != READER_VARIANTS or declaration.get("threshold_variants") != THRESHOLD_VARIANTS or declaration.get("final_questions_used") != 0:
        raise ValueError("Initial four-Reader and twelve-threshold declarations must be preserved.")
    if canonical_sha256(declaration) != selection.get("declaration_sha256"):
        raise ValueError("The original selection declaration was modified.")
    recorded = selection.get("result_files_sha256", {})
    expected_names = {"selection-declaration.json", "reader-comparison.json", "threshold-comparison.json"}
    expected_names.update(f"reader-{i}-predictions.jsonl" for i in range(4))
    expected_names.update(f"threshold-{i}-predictions.jsonl" for i in range(12))
    if set(recorded) != expected_names:
        raise ValueError("All sixteen original variant records must be retained without extra candidates.")
    for name, expected in recorded.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"The failed initial record changed: {name}")
    for family, number in (("reader", 4), ("threshold", 12)):
        rows = json.loads((directory / f"{family}-comparison.json").read_text(encoding="utf-8"))
        if len(rows) != number:
            raise ValueError("The initial record does not contain the declared bounded variants.")
        for i, row in enumerate(rows):
            name = f"{family}-{i}"
            predictions = read_jsonl(directory / f"{name}-predictions.jsonl")
            exact_question_alignment(predictions, questions, require_text=False)
            rescored, _ = score_predictions(questions, predictions)
            if row.get("variant_id") != name or row.get("qa") != rescored:
                raise ValueError("An initial comparison summary differs from its saved predictions.")
            if family == "threshold":
                gate = threshold_promotion_decision(rescored, selection["original_baseline_qa"], protocol)
                if row.get("configuration") != THRESHOLD_VARIANTS[i] or row.get("promotion") != gate or gate["promote"]:
                    raise ValueError("The original threshold failure cannot be silently reclassified.")
            elif row.get("configuration") != READER_VARIANTS[i]:
                raise ValueError("The original Reader variants were changed.")
    return selection, {**recorded, "selection.json": sha256_file(selection_path)}


def calibrate_from_cache(questions: list[dict], cache: list[dict], baseline: dict, protocol: dict) -> tuple[dict, list[dict], dict[str, list[dict]]]:
    validate_development_questions(questions)
    exact_question_alignment(cache, questions, require_text=True)
    cache_by_id = {row["query_id"]: row for row in cache}
    results, all_predictions = [], {}
    for index, thresholds in enumerate(CALIBRATION_THRESHOLDS):
        predictions = []
        for question in questions:
            row = cache_by_id[question["query_id"]]
            candidates = [item for item in row["pools"]["parent_window"] if item["rank"] <= 5]
            result = PortableRAG.select_answer(question["question"], row["evidence"], candidates, thresholds, CALIBRATION_READER)
            compact = {key: value for key, value in result.items() if key != "evidence"}
            if compact.get("citation"):
                compact["citation"] = {key: value for key, value in compact["citation"].items() if key != "context"}
            predictions.append({"query_id": question["query_id"], **compact})
        metrics, scored = score_predictions(questions, predictions)
        for prediction, score in zip(predictions, scored):
            prediction.update(answerable=score["answerable"], exact_match=score["exact_match"], character_f1=score["character_f1"])
        variant_id = f"calibration-v2-{index}"
        results.append({"variant_id": variant_id, "reader_configuration": CALIBRATION_READER,
                        "thresholds": thresholds, "qa": metrics,
                        "promotion": threshold_promotion_decision(metrics, baseline, protocol)})
        all_predictions[variant_id] = predictions
    eligible = [index for index, row in enumerate(results) if row["promotion"]["promote"]]
    chosen = max(eligible, key=lambda index: (
        results[index]["qa"]["answerable_exact_match"], results[index]["qa"]["answerable_character_f1"],
        -results[index]["qa"]["no_answer_false_answer_rate"], results[index]["thresholds"] == BASE_THRESHOLDS, -index,
    )) if eligible else None
    selection = {
        "phase": "calibration-v2", "reader_variant_id": "reader-3", "reader_configuration": CALIBRATION_READER,
        "threshold_variant_id": f"calibration-v2-{chosen}" if chosen is not None else None,
        "thresholds": CALIBRATION_THRESHOLDS[chosen] if chosen is not None else None,
        "joint_promotion": chosen is not None, "requires_original_baseline_fallback": chosen is None,
        "initial_protocol_succeeded": False, "initial_threshold_variants": 12, "follow_up_threshold_variants": 9,
        "total_threshold_variants": 21, "final_questions_used": 0,
        "reason": "The separately declared development refinement satisfies all unchanged quality, coverage and same-device safety gates." if chosen is not None else "Neither the original twelve-threshold search nor the nine-candidate follow-up met every unchanged gate; preserve the original conservative default.",
    }
    return selection, results, all_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "benchmark", "no-answer-dir", "cache-dir", "initial-selection-dir", "baseline-answerable-dir", "baseline-no-answer-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-answerable-attestation", type=Path)
    parser.add_argument("--baseline-no-answer-attestation", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a calibration declaration or result.")
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/validation-protocol.json"
    plan_path = project / "configs/calibration-v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    benchmark = verify_benchmark_directory(args.benchmark, protocol_path)
    no_answer = verify_abstention_directory(args.no_answer_dir, project / "configs/no-answer-protocol.json")
    if no_answer["answerable_benchmark_sha256"] != benchmark["benchmark_sha256"]:
        raise ValueError("The no-answer benchmark must bind the frozen answerable benchmark.")
    answerable_questions = read_jsonl(args.benchmark / "dev.jsonl")
    no_answer_questions = read_jsonl(args.no_answer_dir / "dev.jsonl")
    questions = answerable_questions + no_answer_questions
    validate_development_questions(questions)
    if len(answerable_questions) != benchmark["actual_counts"]["dev"] or len(no_answer_questions) != no_answer["counts"]["dev"]:
        raise ValueError("The entire frozen development subsets are required.")
    inputs = {"split": "dev", "benchmark_sha256": benchmark["benchmark_sha256"],
              "question_file_sha256": sha256_file(args.benchmark / "dev.jsonl"),
              "no_answer_benchmark_sha256": no_answer["no_answer_benchmark_sha256"],
              "no_answer_file_sha256": sha256_file(args.no_answer_dir / "dev.jsonl")}
    verify_bundle(args.bundle)
    cache, cache_report = load_verified_cache(args.cache_dir, args.bundle, questions, inputs)
    source_hashes = verify_cache_source(cache_report, project)
    initial, initial_hashes = verify_initial_failure(args.initial_selection_dir, questions, protocol)
    if initial.get("cache_sha256") != cache_report["reader_cache_sha256"] or initial.get("cache_metrics_sha256") != sha256_file(args.cache_dir / "metrics.json") or initial.get("bundle_manifest_sha256") != sha256_file(args.bundle / "manifest.json"):
        raise ValueError("The follow-up must reuse the exact original Reader development cache and bundle.")
    declaration = json.loads((args.initial_selection_dir / "selection-declaration.json").read_text(encoding="utf-8"))
    if declaration.get("inputs") != inputs or declaration.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Initial input and unchanged protocol hashes must match.")
    original_answerable, baseline_answerable = load_baseline_predictions(args.baseline_answerable_dir, answerable_questions,
        input_sha256=inputs["question_file_sha256"], benchmark_sha256=benchmark["benchmark_sha256"], attestation=args.baseline_answerable_attestation)
    original_no_answer, baseline_no_answer = load_baseline_predictions(args.baseline_no_answer_dir, no_answer_questions,
        input_sha256=inputs["no_answer_file_sha256"], benchmark_sha256=no_answer["no_answer_benchmark_sha256"], attestation=args.baseline_no_answer_attestation)
    validate_same_device(cache_report, baseline_answerable, baseline_no_answer)
    baseline, _ = score_predictions(questions, original_answerable + original_no_answer)
    if baseline != initial["original_baseline_qa"] or baseline_answerable != initial["baseline_answerable"] or baseline_no_answer != initial["baseline_no_answer"]:
        raise ValueError("The original same-device baseline cannot change in the follow-up.")
    binding = {"schema_version": 1, "declared_at_utc": datetime.now(timezone.utc).isoformat(),
               "plan": plan, "inputs": inputs, "reader_configuration": CALIBRATION_READER,
               "threshold_variants": CALIBRATION_THRESHOLDS, "protocol_sha256": sha256_file(protocol_path),
               "calibration_plan_sha256": sha256_file(plan_path), "calibrator_sha256": sha256_file(Path(__file__)),
               "source_sha256": source_hashes, "cache_sha256": cache_report["reader_cache_sha256"],
               "cache_metrics_sha256": sha256_file(args.cache_dir / "metrics.json"),
               "bundle_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
               "initial_result_files_sha256": initial_hashes, "original_baseline_qa": baseline,
               "baseline_answerable": baseline_answerable, "baseline_no_answer": baseline_no_answer,
               "declaration_timing": "Written before any of the nine follow-up candidates is scored.",
               "final_questions_used": 0}
    args.output.mkdir(parents=True)
    write_json(args.output / "calibration-declaration.json", binding)
    # The declaration is durable before the new nine-candidate evaluation.
    selection, results, predictions = calibrate_from_cache(questions, cache, baseline, protocol)
    write_json(args.output / "calibration-comparison.json", results)
    for name, rows in predictions.items():
        write_jsonl(args.output / f"{name}-predictions.jsonl", rows)
    if selection["joint_promotion"]:
        selected_config = deepcopy(cache_report["configuration"])
        selected_config.update(reader=CALIBRATION_READER, thresholds=selection["thresholds"])
        write_json(args.output / "selected-config.json", selected_config)
    # Guard against shared-workspace mutation while this record is produced.
    verify_cache_source(cache_report, project)
    _, preserved_hashes = verify_initial_failure(args.initial_selection_dir, questions, protocol)
    if preserved_hashes != initial_hashes or sha256_file(protocol_path) != binding["protocol_sha256"]:
        raise ValueError("An immutable original input changed during calibration.")
    summary = {**selection, "original_baseline_qa": baseline, "declaration_sha256": canonical_sha256(binding),
               "result_files_sha256": {path.name: sha256_file(path) for path in sorted(args.output.iterdir())}}
    write_json(args.output / "selection.json", summary)
    print(json.dumps({**selection, "variants": [{"id": row["variant_id"], "thresholds": row["thresholds"],
                     "em": row["qa"]["answerable_exact_match"], "f1": row["qa"]["answerable_character_f1"],
                     "answered_answerable": row["qa"]["answered_answerable_count"],
                     "false_answers": row["qa"]["no_answer_false_answer_count"], "promote": row["promotion"]["promote"]}
                    for row in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
