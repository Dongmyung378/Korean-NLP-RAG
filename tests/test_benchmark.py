import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from korean_rag.benchmark import (build_splits, canonical_sha256, corpus_records, freeze_configuration, grouped_records,
                                  read_exposure_files, read_korquad, relevant_chunk_ids,
                                  retrieval_metrics, score_predictions, sha256_file, text_id,
                                  verify_benchmark_directory, verify_final_freeze, write_jsonl)


def example(index, context=None, question=None, article=None):
    context = context or f"정답{index}을 설명한 원문입니다."
    answer = context[:3]
    return {"query_id": str(index), "question": question or f"질문 {index}?", "context": context,
            "context_id": text_id(context), "context_key": __import__("hashlib").sha1(context.encode()).hexdigest(),
            "article_id": article or str(index), "title": str(index), "source_split": "validation",
            "answers": [{"text": answer, "answer_start": 0}], "answerable": True}


class BenchmarkTests(unittest.TestCase):
    def test_duplicates_connect_paragraph_groups(self):
        rows = [example(1, "서울의 원문", "같은 질문"), example(2, "부산의 원문", " 같은   질문 "), example(3, "부산의 원문")]
        self.assertEqual(len(grouped_records(rows)), 1)

    def test_final_excludes_whole_exposed_paragraph_and_is_deterministic(self):
        rows = [example(i) for i in range(20)] + [example(100, context=example(2)["context"])]
        first, audit = build_splits([], rows, {"2"}, set(), dev_target=4, final_target=5)
        second, _ = build_splits([], [dict(row) for row in rows], {"2"}, set(), dev_target=4, final_target=5)
        self.assertEqual(first, second)
        self.assertFalse({"2", "100"} & {row["query_id"] for row in first["final"]})
        self.assertEqual(audit["final_known_exposure_count"], 0)
        self.assertEqual(len(corpus_records(rows)), 20)

    def test_train_overlap_removed_and_insufficient_article_holdout_fails(self):
        validation = [example(i, article="one") for i in range(10)]
        train = [example(100, question="質問"), example(101, context=validation[0]["context"])]
        splits, audit = build_splits(train, validation, set(), set(), dev_target=2, final_target=2)
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(audit["train_removed_for_validation_overlap"], 1)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_splits([], validation, {"1"}, set(), dev_target=2, final_target=2, grouping="article")

    def test_gold_requires_full_answer_span_and_same_paragraph(self):
        question = example(1, "앞 서울특별시 뒤")
        question["answers"] = [{"text": "서울특별시", "answer_start": 2}]
        chunks = [{"chunk_id": i, "context_id": context_id, "char_start": start, "char_end": end}
                  for i, context_id, start, end in [(0, question["context_id"], 0, 4), (1, question["context_id"], 2, 7), (2, "other", 0, 20)]]
        self.assertEqual(relevant_chunk_ids(question, chunks), [1])
        self.assertEqual(relevant_chunk_ids(question, chunks[:1]), [])
        metrics = retrieval_metrics([[0], [1]], [[], [1]], (1,))
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["complete_gold_chunk_coverage"], 0.5)

    def test_ndcg_accounts_for_multiple_relevant_chunks(self):
        metrics = retrieval_metrics([[5, 3, 7]], [[3, 7]], (3,))
        self.assertLess(metrics["ndcg_at_3"], 1.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr_at_3"], 0.5)

    def test_selective_denominators_and_no_answer(self):
        questions = [example(1), example(2), {"query_id": "none", "answerable": False, "answers": []}]
        predictions = [{"query_id": "1", "answer_source": "extractive_reader", "answer": questions[0]["answers"][0]["text"]},
                       {"query_id": "2", "answer_source": "abstained", "answer": "Unavailable"},
                       {"query_id": "none", "answer_source": "abstained", "answer": "Unavailable"}]
        metrics, _ = score_predictions(questions, predictions)
        self.assertEqual(metrics["answerable_exact_match"], 0.5)
        self.assertEqual(metrics["answered_answerable_exact_match"], 1.0)
        self.assertEqual(metrics["answerable_coverage"], 0.5)
        self.assertEqual(metrics["no_answer_abstention_rate"], 1.0)
        with self.assertRaises(ValueError):
            score_predictions(questions, predictions[:-1])

    def test_source_offset_validation_and_exposure_import(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.json"
            payload = {"data": [{"title": "서울", "paragraphs": [{"context": "서울은 도시", "qas": [{"id": "q", "question": "어디?", "answers": [{"text": "서울", "answer_start": 0}]}]}]}]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = read_korquad(path, "validation")
            self.assertEqual(records[0]["answers"][0]["answer_start"], 0)
            payload["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "offset mismatch"):
                read_korquad(path, "validation")
            exposure = Path(folder) / "legacy.csv"
            exposure.write_text("query_id,question\nq,어디?\n", encoding="utf-8")
            ids, questions, provenance = read_exposure_files([exposure])
            self.assertEqual(ids, {"q"})
            self.assertEqual(questions, {"어디?"})
            self.assertEqual(len(provenance[0]["sha256"]), 64)

    def test_final_freeze_rejects_model_or_configuration_changes(self):
        arguments = {"benchmark_sha256": "a" * 64, "protocol_sha256": "b" * 64,
                     "configuration": {"context_k": 3}, "source_sha256": {"pipeline.py": "c" * 64},
                     "artifact_sha256": {"reader": "d" * 64}}
        frozen = freeze_configuration(**arguments)
        verify_final_freeze(frozen, **arguments)
        for field, replacement in (("configuration", {"context_k": 5}), ("artifact_sha256", {"reader": "e" * 64})):
            with self.assertRaisesRegex(ValueError, "Final inference"):
                verify_final_freeze(frozen, **{**arguments, field: replacement})

    def test_modified_benchmark_rows_are_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = root / "dev.jsonl"
            write_jsonl(rows, [example(1)])
            manifest = {"manifest_sha256": {rows.name: sha256_file(rows)}}
            manifest["benchmark_sha256"] = canonical_sha256(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            verify_benchmark_directory(root)
            with rows.open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "modified"):
                verify_benchmark_directory(root)
