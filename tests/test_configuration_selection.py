import json
from pathlib import Path
import unittest

from scripts.select_configuration import (BASE_READER_CONFIG, BASE_THRESHOLDS, NULL_SCORE_MARGIN, READER_VARIANTS, THRESHOLD_VARIANTS,
                                          exact_question_alignment, reader_gain_decision,
                                          select_from_cache, validate_cached_candidates, validate_same_device)


class ConfigurationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/validation-protocol.json").read_text(encoding="utf-8"))

    def test_variants_are_bounded_and_baseline_threshold_is_included(self):
        self.assertEqual(len(READER_VARIANTS), 4)
        self.assertEqual(len(THRESHOLD_VARIANTS), 12)
        self.assertIn(BASE_THRESHOLDS, THRESHOLD_VARIANTS)
        self.assertTrue(all(row["require_explicit_year"] for row in THRESHOLD_VARIANTS))
        self.assertTrue(all(row["null_score_margin"] == 1e-6 for row in READER_VARIANTS))
        self.assertNotIn("null_score_margin", BASE_READER_CONFIG)

    def test_fixed_near_null_margin_abstains_without_changing_original_baseline(self):
        from korean_rag.runtime import PortableRAG
        candidate = {"answer": "정답", "context": "정답", "rrf_score": .03, "reranker_score": .8,
                     "reader_score": .125 + NULL_SCORE_MARGIN / 2, "null_score": .125,
                     "rank": 1, "title": "t", "chunk_id": 0, "context_id": "d",
                     "reader_window_start": 0, "answer_char_start": 0, "answer_char_end": 2}
        for delta in (0., NULL_SCORE_MARGIN / 2, NULL_SCORE_MARGIN):
            near_tie = {**candidate, "reader_score": candidate["null_score"] + delta}
            result = PortableRAG.select_answer("질문?", [], [near_tie], BASE_THRESHOLDS, READER_VARIANTS[0])
            self.assertFalse(result["answered"])
        original = PortableRAG.select_answer("질문?", [], [candidate], BASE_THRESHOLDS, BASE_READER_CONFIG)
        self.assertTrue(original["answered"])
        clear_answer = {**candidate, "reader_score": .9}
        promoted = PortableRAG.select_answer("질문?", [], [clear_answer], BASE_THRESHOLDS, READER_VARIANTS[0])
        self.assertTrue(promoted["answered"])

    def test_incomplete_mismatched_or_duplicate_question_cache_rejected(self):
        questions = [{"query_id": "a", "question": "질문?"}, {"query_id": "b", "question": "다른 질문?"}]
        for rows in ([{"query_id": "a", "question": "질문?"}],
                     [{"query_id": "a", "question": "잘못된 질문?"}, {"query_id": "b", "question": "다른 질문?"}],
                     [{"query_id": "a", "question": "질문?"}, {"query_id": "a", "question": "질문?"}]):
            with self.assertRaises(ValueError):
                exact_question_alignment(rows, questions, require_text=True)

    def test_f1_gain_cannot_hide_em_regression(self):
        baseline = {"answerable_exact_match": .7, "answerable_character_f1": .8}
        self.assertTrue(reader_gain_decision({"answerable_exact_match": .71, "answerable_character_f1": .8}, baseline, self.protocol)["promote"])
        self.assertFalse(reader_gain_decision({"answerable_exact_match": .69, "answerable_character_f1": .9}, baseline, self.protocol)["promote"])

    def test_same_device_control_cannot_be_replaced_by_cpu_false_answer_rate(self):
        candidate = {"environment": {"device": "cuda:0", "gpu_name": "GPU A"}}
        gpu = {"device": "cuda", "gpu_name": "GPU A"}
        validate_same_device(candidate, gpu, gpu)
        with self.assertRaisesRegex(ValueError, "candidate device"):
            validate_same_device(candidate, gpu, {"device": "cpu"})
        with self.assertRaisesRegex(ValueError, "declare"):
            validate_same_device(candidate, {}, gpu)
        with self.assertRaisesRegex(ValueError, "hardware"):
            validate_same_device(candidate, gpu, {"device": "cuda", "gpu_name": "GPU B"})

    def test_all_unpromotable_candidates_require_explicit_baseline_fallback(self):
        questions = [{"query_id": "a", "question": "질문?", "answerable": True, "answers": [{"text": "정답"}]},
                     {"query_id": "b", "question": "없음?", "answerable": False, "answers": []}]
        def candidate(answer):
            return {"answer": answer, "context": answer, "rrf_score": .03, "reranker_score": .8,
                    "reader_score": .9, "null_score": .01, "rank": 1, "title": "t", "chunk_id": 0, "context_id": "d",
                    "reader_window_start": 0, "answer_char_start": 0, "answer_char_end": len(answer)}
        cache = [{"query_id": q["query_id"], "question": q["question"], "evidence": [],
                  "pools": {"chunk": [candidate("오답")], "parent_window": [candidate("오답")]}} for q in questions]
        baseline = {"answerable_exact_match": 1., "answerable_character_f1": 1., "answerable_coverage": 1., "no_answer_false_answer_rate": 0.}
        selection, readers, thresholds, predictions = select_from_cache(questions, cache, baseline, self.protocol)
        self.assertTrue(selection["requires_original_baseline_fallback"])
        self.assertFalse(selection["joint_promotion"])
        self.assertEqual(len(readers), 4)
        self.assertEqual(len(thresholds), 12)
        self.assertEqual(len(predictions), 16)
        self.assertEqual(selection["final_questions_used"], 0)

    def test_source_mismatched_cache_is_rejected(self):
        chunks = [{"chunk_id": i, "context_id": "doc", "context": "原文", "char_start": 0, "char_end": 2} for i in range(5)]
        evidence = [{**chunk, "rank": i + 1, "rrf_score": .03, "reranker_score": .8} for i, chunk in enumerate(chunks)]
        candidates = [{**evidence[0], "reader_window_start": 0, "context": "改変", "answer": "", "reader_score": .1}]
        rows = [{"evidence": evidence, "pools": {"chunk": candidates, "parent_window": candidates}}]
        with self.assertRaisesRegex(ValueError, "raw source"):
            validate_cached_candidates(rows, chunks, {"doc": {"context": "原文"}})
