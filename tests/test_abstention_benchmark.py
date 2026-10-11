import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from korean_rag.abstention_benchmark import (construct_no_answer_sets, historical_regression_probes,
                                             marker_matches_corpus, summarize_no_answer, threshold_promotion_decision)
from korean_rag.benchmark import text_id


class AbstentionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/no-answer-protocol.json").read_text(encoding="utf-8"))
        self.protocol.update(development_question_count=4, final_question_count=4)
        self.corpus = [{"id": text_id(str(i)), "title": f"주제{i}", "text": f"원문{i}"} for i in range(4)]
        self.bases = {split: [{"query_id": f"q{i}", "context_id": self.corpus[i]["id"], "article_id": f"a{i}", "title": self.corpus[i]["title"]} for i in indexes]
                      for split, indexes in [("dev", [0, 1]), ("final", [2, 3])]}

    def test_reproducible_disjoint_construction_and_complete_corpus_checks(self):
        first, audit = construct_no_answer_sets(self.corpus, self.bases, self.protocol)
        second, _ = construct_no_answer_sets(self.corpus, self.bases, self.protocol)
        self.assertEqual(first, second)
        self.assertEqual(audit["counts"], {"dev": 4, "final": 4})
        self.assertEqual(audit["base_paragraph_overlap"], 0)
        self.assertEqual(audit["surface_template_overlap"], 0)
        entities = [row["fictional_entity"] for rows in first.values() for row in rows]
        self.assertEqual(len(entities), len(set(entities)))
        self.assertTrue(all(row["label_provenance"]["corpus_paragraphs_checked"] == 4 for rows in first.values() for row in rows))

    def test_normalized_collision_is_not_ignored(self):
        self.assertEqual(marker_matches_corpus("A 서울연구회", [{"id": "doc", "text": "Ａ서울 연구회", "title": ""}]), ["doc"])
        rows, _ = construct_no_answer_sets(self.corpus, self.bases, self.protocol)
        contaminated = [*self.corpus, {"id": "unexpected", "text": rows["dev"][0]["fictional_entity"], "title": ""}]
        with self.assertRaisesRegex(ValueError, "accidentally"):
            construct_no_answer_sets(contaminated, self.bases, self.protocol)

    def test_base_paragraph_overlap_and_template_reuse_rejected(self):
        overlapping = {"dev": self.bases["dev"], "final": self.bases["dev"]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            construct_no_answer_sets(self.corpus, overlapping, self.protocol)
        self.protocol["templates"]["final"]["absent_entity_location"] = self.protocol["templates"]["dev"]["absent_entity_location"]
        with self.assertRaisesRegex(ValueError, "templates"):
            construct_no_answer_sets(self.corpus, self.bases, self.protocol)

    def test_subtype_metrics_count_false_acceptance(self):
        questions = [{"query_id": "a", "answerable": False, "answers": [], "category": "absent_entity"},
                     {"query_id": "b", "answerable": False, "answers": [], "category": "false_relation"}]
        predictions = [{"query_id": "a", "answer_source": "abstained", "answer": "Unavailable"},
                       {"query_id": "b", "answer_source": "extractive_reader", "answer": "서울"}]
        result = summarize_no_answer(questions, predictions)
        self.assertEqual(result["no_answer_false_answer_rate"], 0.5)
        self.assertEqual(result["by_subtype"]["false_relation"]["no_answer_false_answer_count"], 1)

    def test_regression_probes_are_separate_and_exposed(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "probes.json"
            path.write_text(json.dumps([{"id": "old", "question": "기존 질문?"}]), encoding="utf-8")
            rows = historical_regression_probes(path)
        self.assertEqual(rows[0]["benchmark_split"], "regression")
        self.assertTrue(rows[0]["label_provenance"]["previously_exposed"])

    def test_threshold_promotion_preserves_coverage_and_safety(self):
        validation_protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/validation-protocol.json").read_text(encoding="utf-8"))
        baseline = {"answerable_exact_match": 0.6, "answerable_character_f1": 0.7, "answerable_coverage": 0.8, "no_answer_false_answer_rate": 0.3}
        better = {**baseline, "answerable_exact_match": 0.61}
        self.assertTrue(threshold_promotion_decision(better, baseline, validation_protocol)["promote"])
        for worse in ({**better, "answerable_coverage": 0.77}, {**better, "no_answer_false_answer_rate": 0.31}):
            self.assertFalse(threshold_promotion_decision(worse, baseline, validation_protocol)["promote"])
        aspiration_only = {**baseline, "no_answer_false_answer_rate": 0.1, "answerable_coverage": 0.1}
        self.assertFalse(threshold_promotion_decision(aspiration_only, baseline, validation_protocol)["promote"])
