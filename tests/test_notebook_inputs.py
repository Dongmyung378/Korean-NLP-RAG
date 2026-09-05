import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class NotebookInputTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "transformer/notebooks/Week - 4/hybrid-rag.ipynb").read_text(encoding="utf-8"))
        self.source = "".join(notebook["cells"][3]["source"])
        function = next(node for node in ast.parse(self.source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "select_input_artifact")
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "input_selector", "exec"), scope)
        self.select = scope["select_input_artifact"]

    def test_single_candidate_and_duplicate_paths(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            path.touch()
            self.assertEqual(self.select([path, path], None, "model"), path.resolve())

    def test_multiple_candidates_require_selection(self):
        with TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.pt", Path(folder) / "b.pt"
            first.touch()
            second.touch()
            with self.assertRaisesRegex(ValueError, "Multiple"):
                self.select([first, second], None, "model")
            self.assertEqual(self.select([first, second], second, "model"), second.resolve())

    def test_missing_or_wrong_path_type(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                self.select([], None, "model")
            with self.assertRaises(FileNotFoundError):
                self.select([], Path(folder) / "missing", "model")
            with self.assertRaises(FileNotFoundError):
                self.select([], folder, "model")
            self.assertEqual(self.select([], folder, "model", directory=True), Path(folder).resolve())

    def test_config_cannot_fall_back_to_another_checkpoint(self):
        self.assertIn('week2_checkpoint_path.parent / "retrieval_config.json"', self.source)
        self.assertIn('week2_checkpoint_path.parent / "tokenizer"', self.source)
        self.assertNotIn("week2_config_candidates[0]", self.source)
