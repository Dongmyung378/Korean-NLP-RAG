import re
import unittest

from scripts.prepare_training_data import choose_groups, natural_chunks


class WordTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        result = {"input_ids": list(range(len(offsets) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class TrainingDataTests(unittest.TestCase):
    def test_natural_windows_preserve_offsets_and_token_budget(self):
        text = "첫 문장은 짧다. 두 번째 문장은 조금 더 길다. 세 번째 문장도 길게 적는다. 마지막 문장이다."
        tokenizer = WordTokenizer()
        chunks = natural_chunks(text, tokenizer, max_tokens=12, overlap_tokens=2)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(text[chunk["start_char"]:chunk["end_char"]], chunk["text"])
            self.assertLessEqual(len(tokenizer(chunk["text"])["input_ids"]), 12)
        self.assertEqual(chunks[-1]["end_char"], len(text))

    def test_whole_paragraph_groups_do_not_cross_internal_split(self):
        rows = [{"id": f"q{i}", "document_id": f"p{i // 3}", "question": f"질문 {i}"} for i in range(24)]
        selected, duplicates = choose_groups(rows, seed=378, train_target=10, dev_target=4)
        train_docs = {row["document_id"] for row in selected["train"]}
        dev_docs = {row["document_id"] for row in selected["dev"]}
        self.assertFalse(train_docs & dev_docs)
        self.assertGreaterEqual(len(selected["train"]), 10)
        self.assertGreaterEqual(len(selected["dev"]), 4)
        self.assertEqual(duplicates, 0)
        self.assertEqual(selected, choose_groups(rows[::-1], seed=378, train_target=10, dev_target=4)[0])

    def test_question_duplicates_are_removed_before_group_selection(self):
        rows = [{"id": "a", "document_id": "p1", "question": "수도는?"},
                {"id": "b", "document_id": "p2", "question": "수 도는"},
                {"id": "c", "document_id": "p3", "question": "항구는?"},
                {"id": "d", "document_id": "p4", "question": "인구는?"}]
        _, duplicates = choose_groups(rows, seed=1, train_target=1, dev_target=1)
        self.assertEqual(duplicates, 1)


if __name__ == "__main__":
    unittest.main()
