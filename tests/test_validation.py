import contextlib
import io
import unittest
from unittest.mock import patch

from korean_rag.__main__ import main
from korean_rag.metrics import ranking_summary
from korean_rag.validation import (
    MAX_QUESTION_CHARS, MAX_QUESTION_TOKENS, QuestionValidationError,
    validate_question_text, validate_question_tokens,
)


class QuestionTests(unittest.TestCase):
    def test_character_limit_and_empty_question(self):
        for question in ("", "   ", "가" * (MAX_QUESTION_CHARS + 1)):
            with self.subTest(question_length=len(question)), self.assertRaises(QuestionValidationError):
                validate_question_text(question)
        self.assertEqual(validate_question_text(" 질문 "), "질문")
        self.assertEqual(len(validate_question_text("가" * MAX_QUESTION_CHARS)), MAX_QUESTION_CHARS)

    def test_token_limit(self):
        def tokenizer(question, **kwargs):
            self.assertFalse(kwargs["truncation"])
            self.assertFalse(kwargs["add_special_tokens"])
            return {"input_ids": question.split()}
        self.assertTrue(validate_question_tokens("가 " * MAX_QUESTION_TOKENS, tokenizer))
        with self.assertRaises(QuestionValidationError):
            validate_question_tokens("가 " * (MAX_QUESTION_TOKENS + 1), tokenizer)

    def test_cli_rejects_long_text_before_loading_models(self):
        with patch("sys.argv", ["korean_rag", "-q", "가" * (MAX_QUESTION_CHARS + 1)]), \
             patch("korean_rag.__main__.KoreanRAGPipeline") as pipeline, \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            pipeline.assert_not_called()
        self.assertIn("질문이 너무 깁니다", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_reports_token_error_without_traceback(self):
        with patch("sys.argv", ["korean_rag", "-q", "질문"]), \
             patch("korean_rag.__main__.KoreanRAGPipeline") as pipeline, \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            pipeline.return_value.answer.side_effect = QuestionValidationError("질문이 너무 깁니다.")
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
        self.assertNotIn("Traceback", stderr.getvalue())


class RankingSummaryTests(unittest.TestCase):
    def test_generator_matches_tuple(self):
        self.assertEqual(ranking_summary(iter([1, 3, None]), iter([1, 5, 10])),
                         ranking_summary([1, 3, None], (1, 5, 10)))

    def test_empty_cutoffs(self):
        self.assertEqual(ranking_summary([1], iter([])), {})

    def test_invalid_cutoffs(self):
        for cutoffs in ([0], [-1], [1.5], [True]):
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                ranking_summary([1], cutoffs)
