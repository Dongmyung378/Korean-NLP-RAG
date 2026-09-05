import ast
import inspect
import json
import re
import unittest
from pathlib import Path

from korean_rag.pipeline import decode_answer_span


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_reader_citations_allow_single_character_without_changing_qwen(self):
        import pandas as pd

        path = ROOT / "transformer/notebooks/Week - 4/hybrid-rag.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        scope = {"re": re, "chunk_df": pd.DataFrame({"chunk_text": ["욕계에는 진이 존재한다."], "title": ["불교"]})}
        names = {"normalize_text", "compact_match_text", "answer_is_grounded", "build_answer_citations"}
        for index in (8, 36):
            tree = ast.parse("".join(notebook["cells"][index]["source"]))
            selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            exec(compile(ast.Module(body=selected, type_ignores=[]), "citation_helpers", "exec"), scope)
        self.assertFalse(scope["answer_is_grounded"]("진", [0]))
        self.assertTrue(scope["answer_is_grounded"]("진", [0], allow_short=True))
        self.assertEqual(len(scope["build_answer_citations"]("진", [0], allow_short=True)), 1)
        for value in ("", "근거 문맥에서 답을 찾을 수 없습니다.", "없는 답"):
            self.assertFalse(scope["answer_is_grounded"](value, [0], allow_short=True))
            self.assertEqual(scope["build_answer_citations"](value, [0], allow_short=True), [])
        self.assertNotIn("citation_chunk_ids", "".join(notebook["cells"][37]["source"]))

    def test_python_cells_compile(self):
        notebooks = list((ROOT / "transformer/notebooks").rglob("*.ipynb"))
        self.assertEqual(len(notebooks), 4)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                # 셸 설치 명령은 Python 문법 검사 대상에서 제외한다.
                source = "".join(line for line in cell["source"] if not line.lstrip().startswith(("!", "%")))
                with self.subTest(notebook=path.name, cell=index):
                    compile(source, f"{path.name}:cell{index}", "exec")

    def test_reader_helper_preserves_empty_answer(self):
        path = ROOT / "transformer/notebooks/Week - 4/hybrid-rag.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "def clean_reader_answer(" in "".join(cell["source"]))
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "clean_reader_answer")
        scope = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "reader_helper", "exec"), scope)
        clean = scope["clean_reader_answer"]
        for value in (None, "", "  ", "근거 문맥에서 답을 찾을 수 없습니다."):
            self.assertEqual(clean(value), "")
        self.assertEqual(clean("3"), "3")
        self.assertEqual(clean("  정답  "), "정답")
        decoder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_answer_span")
        self.assertEqual(ast.dump(decoder), ast.dump(ast.parse(inspect.getsource(decode_answer_span)).body[0]))


if __name__ == "__main__":
    unittest.main()
