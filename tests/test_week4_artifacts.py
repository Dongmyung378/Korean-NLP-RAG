import json
import re
import unittest
from pathlib import Path

import pandas as pd


ARTIFACTS = Path(__file__).resolve().parents[1] / "transformer/outputs/week4/week4_day1_3_artifacts"


def compact(value):
    return re.sub(r"[^0-9a-z가-힣]", "", str(value).lower())


@unittest.skipUnless((ARTIFACTS / "week4_day5_rag_examples.csv").exists(), "Download the Week 4 artifacts to validate saved results.")
class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = pd.read_csv(ARTIFACTS / "week4_day5_rag_examples.csv").fillna("")
        cls.candidates = pd.read_csv(ARTIFACTS / "week4_day5_reader_candidates.csv").fillna("")
        cls.chunks = pd.read_csv(ARTIFACTS / "week4_knowledge_base_chunks.csv")

    def test_metrics_match_individual_answers(self):
        metrics = json.loads((ARTIFACTS / "week4_day5_generation_metrics.json").read_text())
        covered = [bool(compact(row.gold_answer)) and compact(row.gold_answer) in compact(row.generated_answer)
                   for row in self.examples.itertuples()]
        self.assertEqual(covered, self.examples.gold_answer_in_generated_answer.tolist())
        self.assertEqual(metrics["generation_query_count"], len(covered))
        self.assertAlmostEqual(metrics["generated_answer_coverage"], sum(covered) / len(covered))
        self.assertAlmostEqual(metrics["citation_coverage"], (self.examples.citation_count > 0).mean())
        self.assertEqual(self.examples.query_id.nunique(), len(covered))

    def test_reader_cites_its_selected_chunk(self):
        selected = self.candidates.sort_values("candidate_rank").groupby("query_id").first()
        self.assertFalse(self.candidates.answer.eq("").any())
        self.assertFalse(self.candidates.answer.eq("근거 문맥에서 답을 찾을 수 없습니다.").any())
        for row in self.examples.itertuples():
            if row.answer_source != "extractive_reader":
                continue
            chosen = selected.loc[row.query_id]
            self.assertEqual(str(chosen.answer), row.generated_answer)
            citations = json.loads(row.citations)
            self.assertEqual(row.citation_count, len(citations))
            self.assertEqual([item["chunk_id"] for item in citations], [int(chosen.chunk_id)])
            self.assertIn(compact(row.generated_answer), compact(self.chunks.iloc[int(chosen.chunk_id)].chunk_text))

    def test_strategy_and_error_summaries(self):
        strategies = pd.read_csv(ARTIFACTS / "week4_day5_generation_strategy_examples.csv")
        summary = pd.read_csv(ARTIFACTS / "week4_day6_generation_strategy_summary.csv")
        for row in summary.itertuples():
            selected = strategies.loc[strategies.strategy == row.strategy]
            self.assertEqual(len(selected), row.query_count)
            self.assertAlmostEqual(selected.gold_answer_in_generated_answer.mean(), row.generated_answer_coverage)
            self.assertAlmostEqual(selected.answer_is_grounded.mean(), row.grounded_answer_rate)
        errors = pd.read_csv(ARTIFACTS / "week4_day6_rag_error_summary.csv")
        self.assertEqual(errors.query_count.sum(), len(self.examples))


if __name__ == "__main__":
    unittest.main()
