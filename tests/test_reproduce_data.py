import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.reproduce_data import fetch_source, reconstruct_splits


def record(identifier, document):
    return {"query_id": identifier, "id": identifier, "context_id": document,
            "document_id": document, "article_id": "article:" + document,
            "question": "질문 " + identifier}


class DataReproductionTests(unittest.TestCase):
    def recipe(self):
        return {"grouping": "paragraph", "train_removed_ids": ["removed"],
                "ordered_ids": {"dev": ["dev2", "dev1"], "final": ["final"], "unused": ["unused"]}}

    def test_published_order_and_training_removals_are_preserved(self):
        train = [record("train", "t"), record("removed", "overlap")]
        validation = [record("dev1", "d1"), record("dev2", "d2"),
                      record("final", "f"), record("unused", "u")]
        splits = reconstruct_splits(train, validation, self.recipe())
        self.assertEqual([row["query_id"] for row in splits["train"]], ["train"])
        self.assertEqual([row["query_id"] for row in splits["dev"]], ["dev2", "dev1"])
        self.assertEqual(splits["final"][0]["benchmark_split"], "final")
        self.assertNotIn("benchmark_split", validation[0])

    def test_missing_validation_id_fails_before_evaluation(self):
        recipe = self.recipe()
        recipe["ordered_ids"]["unused"] = []
        validation = [record("dev1", "d1"), record("dev2", "d2"),
                      record("final", "f"), record("unused", "u")]
        with self.assertRaisesRegex(ValueError, "complete official source"):
            reconstruct_splits([record("train", "t"), record("removed", "r")], validation, recipe)

    def test_duplicate_validation_ids_are_rejected(self):
        recipe = self.recipe()
        recipe["ordered_ids"]["final"] = ["dev1"]
        validation = [record("dev1", "d1"), record("dev2", "d2"), record("unused", "u")]
        with self.assertRaisesRegex(ValueError, "unique and disjoint"):
            reconstruct_splits([record("train", "t"), record("removed", "r")], validation, recipe)

    def test_pinned_source_reuses_matching_bytes_and_never_overwrites_modified_file(self):
        content = b'{"data":[]}'
        source = {"file": "KorQuAD_v1.0_train.json", "url": "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json",
                  "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory(prefix=".data-test-", dir=Path(__file__).resolve().parents[1]) as directory:
            target = Path(directory) / source["file"]
            target.write_bytes(content)
            self.assertEqual(fetch_source(source, Path(directory), offline=True), target)
            target.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                fetch_source(source, Path(directory), offline=True)
            self.assertEqual(target.read_bytes(), b"modified")

    def test_nonofficial_url_is_rejected_without_network_access(self):
        source = {"file": "source.json", "url": "https://example.com/source.json", "bytes": 0,
                  "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "official HTTPS"):
            fetch_source(source, Path("."), offline=True)


if __name__ == "__main__":
    unittest.main()
