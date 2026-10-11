import unittest

from korean_rag.qa_metrics import character_f1_score, exact_match_score, normalize_answer, score_answer, wilson_interval


class QAMetricTests(unittest.TestCase):
    def test_official_normalization_preserves_token_boundaries(self):
        self.assertEqual(normalize_answer(" 《서울》 (SEOUL),  ‘한국’! "), "서울 seoul 한국")
        self.assertEqual(exact_match_score("서울 특별시", "서울특별시"), 0.0)
        self.assertEqual(character_f1_score("서울 특별시", "서울특별시"), 1.0)

    def test_character_counts_and_multiple_annotations(self):
        self.assertAlmostEqual(character_f1_score("가가나", "가나나"), 2 / 3)
        self.assertEqual(score_answer("대한민국", ["한국", "대한민국"]), {"exact_match": 1.0, "character_f1": 1.0})
        self.assertEqual(character_f1_score("", ""), 0.0)
        self.assertEqual(character_f1_score("서울", "부산"), 0.0)

    def test_no_empty_answerable_gold(self):
        for answers in ([], [""]):
            with self.assertRaises(ValueError):
                score_answer("", answers)

    def test_wilson_interval(self):
        low, high = wilson_interval(48, 48)
        self.assertAlmostEqual(low, 0.9258998703, places=8)
        self.assertAlmostEqual(high, 1.0)
        self.assertIsNone(wilson_interval(0, 0))
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
