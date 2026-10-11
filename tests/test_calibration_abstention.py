from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from korean_rag.abstention_benchmark import threshold_promotion_decision
from korean_rag.benchmark import canonical_sha256, score_predictions, sha256_file, write_jsonl
from scripts.calibrate_abstention import (
    CALIBRATION_READER, CALIBRATION_THRESHOLDS, calibrate_from_cache,
    validate_development_questions, validate_plan, verify_cache_source, verify_initial_failure, write_json,
)
from scripts.select_configuration import READER_VARIANTS, THRESHOLD_VARIANTS


class CalibrationAbstentionTests(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[1]
        self.protocol = json.loads((self.project / "configs/validation-protocol.json").read_text(encoding="utf-8"))
        self.plan = json.loads((self.project / "configs/calibration-v2.json").read_text(encoding="utf-8"))
        self.questions = [
            {"query_id": "a", "question": "질문?", "benchmark_split": "dev", "answerable": True, "answers": [{"text": "정답"}]},
            {"query_id": "b", "question": "없음?", "benchmark_split": "dev", "answerable": False, "answers": []},
        ]

    def test_plan_is_separate_and_cannot_expand_or_weaken_fixed_variants(self):
        validate_plan(self.plan)
        self.assertEqual(self.protocol["selection"]["max_abstention_variants"], 12)
        self.assertEqual(len(CALIBRATION_THRESHOLDS), 9)
        for key, value in (("reader_thresholds", [.2, .3, .9]), ("additional_threshold_variants", 10),
                           ("evaluation_split", "final"), ("reader_configuration", READER_VARIANTS[0])):
            changed = {**self.plan, key: value}
            with self.assertRaisesRegex(ValueError, "exactly"):
                validate_plan(changed)

    def test_final_or_duplicate_questions_are_rejected_before_scoring(self):
        final = deepcopy(self.questions)
        final[0]["benchmark_split"] = "final"
        with self.assertRaisesRegex(ValueError, "Final"):
            validate_development_questions(final)
        with self.assertRaisesRegex(ValueError, "IDs"):
            validate_development_questions([self.questions[0], self.questions[0]])

    def make_cache(self):
        cache = []
        for question, answer, score in ((self.questions[0], "정답", .95), (self.questions[1], "오답", .4)):
            candidate = {"answer": answer, "context": answer, "rrf_score": .03, "reranker_score": .8,
                         "reader_score": score, "null_score": .01, "rank": 1, "title": "t", "chunk_id": 0,
                         "context_id": "d", "reader_window_start": 0, "answer_char_start": 0, "answer_char_end": len(answer)}
            cache.append({"query_id": question["query_id"], "question": question["question"], "evidence": [],
                          "pools": {"parent_window": [candidate]}})
        return cache

    def test_selection_uses_unchanged_safety_and_declared_order_after_quality_ties(self):
        baseline = {"answerable_exact_match": 0., "answerable_character_f1": 0., "answerable_coverage": 1., "no_answer_false_answer_rate": 0.}
        selection, results, predictions = calibrate_from_cache(self.questions, self.make_cache(), baseline, self.protocol)
        self.assertEqual(selection["threshold_variant_id"], "calibration-v2-1")
        self.assertFalse(results[0]["promotion"]["promote"])
        self.assertTrue(results[1]["promotion"]["promote"])
        self.assertEqual(selection["reader_configuration"], CALIBRATION_READER)
        self.assertEqual(len(results), 9)
        self.assertEqual(len(predictions), 9)
        self.assertFalse(selection["initial_protocol_succeeded"])
        self.assertEqual(selection["total_threshold_variants"], 21)
        self.assertEqual(selection["final_questions_used"], 0)

    def test_coverage_loss_cannot_be_hidden_by_better_no_answer_abstention(self):
        baseline = {"answerable_exact_match": 1., "answerable_character_f1": 1., "answerable_coverage": 1., "no_answer_false_answer_rate": 0.}
        selection, results, _ = calibrate_from_cache(self.questions, self.make_cache(), baseline, self.protocol)
        self.assertFalse(selection["joint_promotion"])
        self.assertTrue(selection["requires_original_baseline_fallback"])
        self.assertTrue(all(not row["promotion"]["promote"] for row in results))

    def test_cache_source_mismatch_or_incomplete_attestation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "source.py").write_text("source\n", encoding="utf-8")
            report = {"source_sha256": {"source.py": sha256_file(project / "source.py")}}
            with patch("scripts.calibrate_abstention.REQUIRED_FINAL_SOURCES", ("source.py",)):
                verify_cache_source(report, project)
                with self.assertRaisesRegex(ValueError, "complete"):
                    verify_cache_source({"source_sha256": {}}, project)
                (project / "source.py").write_text("modified\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_cache_source(report, project)

    def test_failed_initial_records_must_all_remain_exact_and_failed(self):
        predictions = [{"query_id": "a", "answer": "정답", "answered": True},
                       {"query_id": "b", "answer": "오답", "answered": True}]
        metrics, _ = score_predictions(self.questions, predictions)
        baseline = {"answerable_exact_match": 0., "answerable_character_f1": 0., "answerable_coverage": 1., "no_answer_false_answer_rate": 0.}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            declaration = {"reader_variants": READER_VARIANTS, "threshold_variants": THRESHOLD_VARIANTS, "final_questions_used": 0}
            write_json(directory / "selection-declaration.json", declaration)
            for family, variants in (("reader", READER_VARIANTS), ("threshold", THRESHOLD_VARIANTS)):
                rows = []
                for index, configuration in enumerate(variants):
                    name = f"{family}-{index}"
                    write_jsonl(directory / f"{name}-predictions.jsonl", predictions)
                    rows.append({"variant_id": name, "configuration": configuration, "qa": metrics,
                                 "promotion": threshold_promotion_decision(metrics, baseline, self.protocol)})
                write_json(directory / f"{family}-comparison.json", rows)
            selection = {"joint_promotion": False, "requires_original_baseline_fallback": True, "final_questions_used": 0,
                         "reader_variant_id": "reader-3", "reader_configuration": CALIBRATION_READER,
                         "declaration_sha256": canonical_sha256(declaration), "original_baseline_qa": baseline,
                         "result_files_sha256": {path.name: sha256_file(path) for path in directory.iterdir()}}
            write_json(directory / "selection.json", selection)
            verify_initial_failure(directory, self.questions, self.protocol)
            with (directory / "threshold-0-predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "record changed"):
                verify_initial_failure(directory, self.questions, self.protocol)

    def test_existing_declaration_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration-declaration.json"
            write_json(path, {"declared": True})
            with self.assertRaises(FileExistsError):
                write_json(path, {"declared": False})
