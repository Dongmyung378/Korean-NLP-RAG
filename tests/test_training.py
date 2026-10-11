import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from korean_rag.training import (
    Passage, PlateauStopper, TrainingConfig, TrainingTriplet,
    accumulation_weight, prepare_matched_negatives, train_bounded, validate_split_isolation,
)


def triplet(identifier, document, question, answer="서울", negative="부산"):
    return TrainingTriplet(identifier, document, question, "p:" + identifier,
                           "정답은 " + answer, "n:" + identifier, negative,
                           (answer,), "hard")


class TrainingPreparationTests(unittest.TestCase):
    def test_both_arms_share_questions_and_positives_after_filtering(self):
        passages = [Passage("p1", "d1", "대한민국 수도는 서울", "수도"),
                    Passage("same-doc", "d1", "이 문단은 답이 없다", "수도"),
                    Passage("false-negative", "d2", "서 울은 수도다", "다른 제목"),
                    Passage("same-title", "d3", "제목이 같다", "수도"),
                    Passage("valid", "d4", "부산은 항구 도시", "도시")]
        questions = [{"id": "q1", "document_id": "d1", "question": "수도는?", "answers": ["서울"]},
                     {"id": "q2", "document_id": "d1", "question": "서울의 위치는?", "answers": ["서울"]}]
        rankings = {"q1": ["p1", "same-doc", "false-negative", "same-title", "valid"],
                    "q2": ["p1", "same-doc", "false-negative"]}
        arms, report = prepare_matched_negatives(questions, passages, rankings)
        self.assertEqual(report["eligible_questions"], 1)
        self.assertTrue(report["same_questions_and_positives"])
        self.assertEqual(arms["hard"][0].negative_id, "valid")
        self.assertEqual(arms["random"][0].negative_id, "valid")
        self.assertEqual(report["dropped"], {"missing_hard_negative": 1})

    def test_blocked_document_cannot_be_used_as_negative(self):
        questions = [{"id": "q", "document_id": "d", "question": "답?", "answers": ["정답"]}]
        passages = [Passage("p", "d", "정답"), Passage("heldout", "test", "없는 답"),
                    Passage("n", "other", "다른 내용")]
        arms, _ = prepare_matched_negatives(questions, passages,
                                            {"q": ["heldout", "n"]}, blocked_document_ids=["test"])
        self.assertEqual(arms["hard"][0].negative_id, "n")

    def test_spacing_equivalent_duplicate_questions_are_rejected(self):
        rows = [{"id": "q1", "document_id": "d", "question": "수도는?", "answers": ["서울"]},
                {"id": "q2", "document_id": "d", "question": "수 도는", "answers": ["서울"]}]
        with self.assertRaisesRegex(ValueError, "equivalent questions"):
            prepare_matched_negatives(rows, [Passage("p", "d", "서울"),
                                             Passage("n", "other", "부산")], {"q1": ["n"]})

    def test_deserialized_false_negative_is_rejected(self):
        from dataclasses import asdict
        with self.assertRaisesRegex(ValueError, "Negative passage"):
            TrainingTriplet.from_row(asdict(triplet("q", "d", "수도는?", negative="서 울")))

    def test_split_checks_document_and_question_leakage(self):
        train = [triplet("train", "d", "수도는?")]
        with self.assertRaisesRegex(ValueError, "positive documents"):
            validate_split_isolation(train, [triplet("dev", "d", "서울은?")])
        with self.assertRaisesRegex(ValueError, "normalized questions"):
            validate_split_isolation(train, [triplet("dev", "other", "수 도는")])

    def test_original_incumbent_is_retained_on_plateau(self):
        stopper = PlateauStopper(0.8, patience=1, min_delta=0.001)
        self.assertEqual(stopper.update(0.799), (False, True))
        self.assertEqual(stopper.best, 0.8)

    def test_improvement_resets_patience(self):
        stopper = PlateauStopper(0.8, patience=2, min_delta=0.001)
        self.assertEqual(stopper.update(0.8), (False, False))
        self.assertEqual(stopper.update(0.82), (True, False))
        self.assertEqual(stopper.update(0.82), (False, False))
        self.assertEqual(stopper.update(0.82), (False, True))

    def test_config_rejects_fractional_budget_and_nonfinite_rate(self):
        with self.assertRaises(ValueError):
            TrainingConfig(micro_batch_size=1.5)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=float("nan"))

    def test_partial_accumulation_uses_actual_sample_counts(self):
        weights = [accumulation_weight(i, sample_count=5, micro_batch_size=2,
                                       accumulation_steps=2) for i in range(3)]
        self.assertEqual(weights, [0.5, 0.5, 1.0])
        weights = [accumulation_weight(i, sample_count=3, micro_batch_size=2,
                                       accumulation_steps=2) for i in range(2)]
        self.assertAlmostEqual(weights[0], 2 / 3)
        self.assertAlmostEqual(weights[1], 1 / 3)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "PyTorch and Transformers are optional for data preparation")
class CPUTrainingSmokeTests(unittest.TestCase):
    def test_tiny_cpu_training_respects_budget_and_saves_portable_checkpoint(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

        vocabulary = {token: index for index, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "정답은", "서울", "부산", "수도는", "항구는", "?"])}
        tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer_backend.pre_tokenizer = Whitespace()
        tokenizer_backend.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend,
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            cls_token="[CLS]", sep_token="[SEP]")
        model = RobertaModel(RobertaConfig(vocab_size=len(vocabulary), hidden_size=16,
                                           num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=32, max_position_embeddings=64,
                                           type_vocab_size=2, pad_token_id=0))
        config = TrainingConfig(epochs=1, micro_batch_size=1, gradient_accumulation_steps=2,
                                query_max_length=16, passage_max_length=16, evaluation_batch_size=2,
                                gradient_checkpointing=False, max_optimizer_steps=1)
        train = [triplet("train1", "td1", "수도는 ?"), triplet("train2", "td2", "항구는 ?", "부산", "서울")]
        dev = [triplet("dev", "dd", "수도는 서울 ?")]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory(prefix=".training-test-", dir=Path(__file__).resolve().parents[1]) as directory:
                target = Path(directory) / "run"
                report = train_bounded(model, tokenizer, train, dev, target, config, device_name="cpu")
                self.assertEqual(report["optimizer_steps"], 1)
                self.assertFalse(report["mixed_precision_used"])
                self.assertEqual(report["device"], "cpu")
                self.assertTrue((target / "best_model" / "model.safetensors").exists())
                saved_config = json.loads((target / "best_model" / "config.json").read_text())
                self.assertEqual(saved_config["num_attention_heads"], 2)
                with self.assertRaises(FileExistsError):
                    train_bounded(model, tokenizer, train, dev, target, config, device_name="cpu")
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
