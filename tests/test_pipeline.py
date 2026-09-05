import unittest
from types import SimpleNamespace

import numpy as np

from korean_rag.pipeline import KoreanRAGPipeline, decode_answer_span


def evidence(rank, context, rrf=0.03, reranker=0.95):
    return {"rank": rank, "chunk_id": rank, "title": "test", "context": context,
            "rrf_score": rrf, "reranker_score": reranker}


class AnswerTests(unittest.TestCase):
    def pipeline(self, items, spans):
        pipeline = object.__new__(KoreanRAGPipeline)
        pipeline.reader = True
        pipeline.search = lambda question: items
        pipeline._read_span = lambda question, context: spans[context]
        return pipeline

    def test_low_scoring_second_chunk_cannot_bypass_gate(self):
        pipeline = self.pipeline(
            [evidence(1, "정답"), evidence(2, "오답", rrf=0.01)],
            {"정답": ("정답", 0.3), "오답": ("오답", 0.99)},
        )
        self.assertEqual(pipeline.answer("질문")["answer"], "정답")

    def test_low_reranker_candidate_cannot_bypass_gate(self):
        pipeline = self.pipeline([evidence(1, "오답", reranker=0.01)], {})
        self.assertEqual(pipeline.answer("질문")["answer_source"], "abstained")

    def test_valid_second_chunk_is_not_rejected_with_first(self):
        pipeline = self.pipeline(
            [evidence(1, "오답", rrf=0.01), evidence(2, "정답")],
            {"정답": ("정답", 0.8)},
        )
        self.assertEqual(pipeline.answer("질문")["citation"]["chunk_id"], 2)

    def test_empty_answer_cannot_beat_real_answer(self):
        pipeline = self.pipeline(
            [evidence(1, "문맥"), evidence(2, "정답")],
            {"문맥": ("", 0.99), "정답": ("정답", 0.8)},
        )
        self.assertEqual(pipeline.answer("질문")["answer"], "정답")

    def test_single_character_answer_is_valid(self):
        pipeline = self.pipeline([evidence(1, "총 3개")], {"총 3개": ("3", 0.8)})
        result = pipeline.answer("몇 개?")
        self.assertEqual(result["answer"], "3")
        self.assertTrue(result["answer_in_evidence"])
        self.assertNotIn("answer_supported", result)

    def test_unsubstantiated_span_does_not_hide_valid_candidate(self):
        pipeline = self.pipeline(
            [evidence(1, "문맥"), evidence(2, "정답")],
            {"문맥": ("없는 답", 0.99), "정답": ("정답", 0.8)},
        )
        self.assertEqual(pipeline.answer("질문")["answer"], "정답")

    def test_explicit_year_missing_from_context(self):
        pipeline = self.pipeline([evidence(1, "2014년 3월 29일")], {})
        self.assertEqual(pipeline.answer("2030년 몇 월 며칠?")["answer_source"], "abstained")
        self.assertTrue(pipeline._explicit_years_match("2014년 날짜?", "2014년 3월 29일"))

    def test_reader_below_threshold_and_missing_evidence(self):
        pipeline = self.pipeline([evidence(1, "정답")], {"정답": ("정답", 0.01)})
        self.assertEqual(pipeline.answer("질문")["answer_source"], "abstained")
        pipeline.search = lambda question: []
        self.assertEqual(pipeline.answer("질문")["answer_source"], "abstained")

    def test_retrieval_only(self):
        pipeline = self.pipeline([evidence(1, "문맥")], {})
        pipeline.reader = None
        result = pipeline.answer("질문")
        self.assertEqual(result["answer_source"], "retrieval_only")
        self.assertIsNone(result["answer"])

    def test_invalid_options(self):
        for options in [(0, .025, .5, .1), (21, .025, .5, .1),
                        (3, -.1, .5, .1), (3, .04, .5, .1),
                        (3, .025, float("nan"), .1), (3, .025, .5, 1.1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                KoreanRAGPipeline.validate_answer_options(*options)


class SpanTests(unittest.TestCase):
    def decode(self, starts, ends):
        return decode_answer_span("가 나", [(0, 0), (0, 2), (0, 1), (2, 3), (0, 0)],
                                  [None, 0, 1, 1, None], starts, ends, 0)

    def test_null_wins(self):
        self.assertEqual(self.decode([10, 0, 1, 1, 0], [10, 0, 1, 1, 0])[0], "")

    def test_question_and_special_tokens_are_masked(self):
        answer, score = self.decode([0, 100, 10, 0, 100], [0, 100, 10, 0, 100])
        self.assertEqual(answer, "가")
        self.assertGreater(score, .99)

    def test_no_context(self):
        self.assertEqual(decode_answer_span("", [(0, 0)], [None], [1], [1], 0), ("", 0.0))


class ArtifactTests(unittest.TestCase):
    def test_index_alignment(self):
        import pandas as pd

        pipeline = object.__new__(KoreanRAGPipeline)
        pipeline.chunks = pd.DataFrame({"chunk_id": [0, 1], "title": ["a", "b"], "chunk_text": ["a", "b"]})
        pipeline.dense_index = SimpleNamespace(ntotal=2, d=768)
        pipeline.bm25 = SimpleNamespace(corpus_size=2)
        pipeline.retriever = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hidden_size=768)))
        pipeline._validate_index_alignment()
        pipeline.dense_index.ntotal = 1
        with self.assertRaises(ValueError):
            pipeline._validate_index_alignment()
        pipeline.dense_index.ntotal = 2
        pipeline.chunks["chunk_id"] = [1, 0]
        with self.assertRaises(ValueError):
            pipeline._validate_index_alignment()
        pipeline.chunks["chunk_id"] = [0, 1]
        pipeline.dense_index.d = 128
        with self.assertRaises(ValueError):
            pipeline._validate_index_alignment()


if __name__ == "__main__":
    unittest.main()
