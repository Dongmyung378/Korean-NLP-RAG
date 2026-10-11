"""Artifact installation validates contents before publishing a usable folder."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.download_artifacts import download_archive, extract_archive, install_bundle, materialize_source_data, safe_relative, verify_bundle


def fixture_bundle(root: Path):
    files = {}
    for name in ["config.json", "corpus.jsonl", "chunks.jsonl", "dense.index",
                 "models/retriever/model.safetensors", "models/reranker/model.safetensors", "models/reader/model.safetensors"]:
        data = ("fixture " + name).encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"files": files}))
    return files


class ArtifactDownloadTests(unittest.TestCase):
    def direct_file_fixture(self, root: Path, *, include_direct_in_archive=False):
        source = root / "source"
        source.mkdir()
        files = fixture_bundle(source)
        direct_name = "models/reader/model.safetensors"
        archive = root / "metadata.zip"
        with zipfile.ZipFile(archive, "w") as target:
            for path in source.rglob("*"):
                if path.is_file() and (include_direct_in_archive or path.relative_to(source).as_posix() != direct_name):
                    target.write(path, path.relative_to(source).as_posix())
        descriptor = {"schema_version": 1, "archives": [{"url": "https://example.test/metadata.zip",
                      "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}],
                      "files": [{"path": direct_name, "url": "https://example.test/pinned-model.safetensors",
                                 **files[direct_name]}]}
        path = root / "download.json"
        path.write_text(json.dumps(descriptor))
        return descriptor, path, archive, source / direct_name

    def test_direct_artifact_is_streamed_verified_and_installed_atomically(self):
        class Response(io.BytesIO):
            def __init__(self, content, url):
                super().__init__(content)
                self.url = url
            def geturl(self):
                return self.url
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, descriptor, archive, model = self.direct_file_fixture(root)
            content = {"https://example.test/metadata.zip": archive.read_bytes(),
                       "https://example.test/pinned-model.safetensors": model.read_bytes()}
            destination = root / "artifacts/runtime"
            with patch("scripts.download_artifacts.urlopen", side_effect=lambda request, **kw: Response(content[request.full_url], request.full_url)):
                self.assertTrue(install_bundle(descriptor, destination)["valid"])
            self.assertEqual((destination / "models/reader/model.safetensors").read_bytes(), model.read_bytes())

    def test_individual_artifact_paths_are_bounded_safe_and_unique_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, descriptor, _, _ = self.direct_file_fixture(root)
            entries = original["files"]
            invalid_lists = [entries * 65, [entries[0], {**entries[0], "path": entries[0]["path"].upper()}],
                             [{**entries[0], "path": "../outside.safetensors"}],
                             [{**entries[0], "path": "model.pkl"}],
                             [{**entries[0], "url": "https://user:password@example.test/model"}]]
            for files in invalid_lists:
                descriptor.write_text(json.dumps({**original, "files": files}))
                with self.subTest(files=files[:2]), patch("scripts.download_artifacts.urlopen") as download, self.assertRaises(ValueError):
                    install_bundle(descriptor, root / "artifacts/runtime")
                download.assert_not_called()

    def test_individual_artifact_cannot_replace_archive_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, descriptor, archive, _ = self.direct_file_fixture(root, include_direct_in_archive=True)
            destination = root / "artifacts/runtime"
            with patch("scripts.download_artifacts.download_archive", side_effect=lambda entry, path: shutil.copyfile(archive, path)) as download, self.assertRaises(ValueError):
                install_bundle(descriptor, destination)
            self.assertEqual(download.call_count, 1)
            self.assertFalse(destination.exists())

    def test_modified_direct_artifact_does_not_publish_partial_bundle(self):
        class Response(io.BytesIO):
            def geturl(self):
                return "https://example.test/pinned-model.safetensors"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, descriptor, archive, model = self.direct_file_fixture(root)
            def response(request, **kw):
                return Response(archive.read_bytes() if request.full_url.endswith(".zip") else b"X" * model.stat().st_size)
            destination = root / "artifacts/runtime"
            with patch("scripts.download_artifacts.urlopen", side_effect=response), self.assertRaises(ValueError):
                install_bundle(descriptor, destination)
            self.assertFalse(destination.exists())

    def local_retriever_fixture(self, root: Path, *, include_model_in_archive=False):
        source = root / "source"
        source.mkdir()
        fixture_bundle(source)
        name = "models/retriever/model.safetensors"
        archive = root / "metadata.zip"
        with zipfile.ZipFile(archive, "w") as target:
            for path in source.rglob("*"):
                if path.is_file() and (include_model_in_archive or path.relative_to(source).as_posix() != name):
                    target.write(path, path.relative_to(source).as_posix())
        descriptor = root / "download.json"
        descriptor.write_text(json.dumps({"schema_version": 1, "archives": [{"url": "https://example.test/metadata.zip",
                              "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
        return descriptor, archive, source / name

    def test_exact_local_original_retriever_is_added_to_its_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, archive, model = self.local_retriever_fixture(root)
            destination = root / "artifacts/original"
            with patch("scripts.download_artifacts.download_archive", side_effect=lambda entry, path: shutil.copyfile(archive, path)):
                self.assertTrue(install_bundle(descriptor, destination, local_retriever=model)["valid"])
            self.assertEqual((destination / "models/retriever/model.safetensors").read_bytes(), model.read_bytes())

    def test_changed_local_original_retriever_never_publishes_partial_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, archive, model = self.local_retriever_fixture(root)
            model.write_bytes(b"X" * model.stat().st_size)
            destination = root / "artifacts/original"
            with patch("scripts.download_artifacts.download_archive", side_effect=lambda entry, path: shutil.copyfile(archive, path)), self.assertRaisesRegex(ValueError, "SHA256"):
                install_bundle(descriptor, destination, local_retriever=model)
            self.assertFalse(destination.exists())

    def test_local_original_retriever_cannot_replace_archive_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, archive, model = self.local_retriever_fixture(root, include_model_in_archive=True)
            destination = root / "artifacts/original"
            with patch("scripts.download_artifacts.download_archive", side_effect=lambda entry, path: shutil.copyfile(archive, path)), self.assertRaisesRegex(ValueError, "collides"):
                install_bundle(descriptor, destination, local_retriever=model)
            self.assertFalse(destination.exists())

    def test_traversal_windows_special_names_and_pickles_rejected(self):
        for name in ["../outside", "/absolute", "C:/drive", "a\\b", "a/../b", "a//b", "a/./b",
                     "NUL.json", "a./b", "model.pt", "index.pkl", "a?.json", "a:b.json"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)
        self.assertEqual(safe_relative("models/reader/model.safetensors"), Path("models/reader/model.safetensors"))

    def test_valid_bundle_then_tampered_file_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_bundle(root)
            self.assertTrue(verify_bundle(root)["valid"])
            (root / "chunks.jsonl").write_text("tampered")
            with self.assertRaises(ValueError):
                verify_bundle(root)

    def test_undeclared_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_bundle(root)
            (root / "extra.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_bundle(root)

    def test_zip_links_and_case_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("link", "case"):
                archive = root / (kind + ".zip")
                with zipfile.ZipFile(archive, "w") as target:
                    if kind == "link":
                        info = zipfile.ZipInfo("models/link.json")
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        target.writestr(info, "../../outside")
                    else:
                        target.writestr("Model.json", "a")
                        target.writestr("model.json", "b")
                stage = root / kind
                stage.mkdir()
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    extract_archive(archive, stage, set())

    def test_archive_sha_mismatch_rejected(self):
        class Response(io.BytesIO):
            def geturl(self):
                return "https://example.test/frozen.zip"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "archive.zip"
            with patch("scripts.download_artifacts.urlopen", return_value=Response(b"changed")), self.assertRaises(ValueError):
                download_archive({"url": "https://example.test/frozen.zip", "bytes": 7, "sha256": "0" * 64}, destination)

    def test_complete_install_then_reuse_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            fixture_bundle(source)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as target:
                for file in source.rglob("*"):
                    if file.is_file():
                        target.write(file, file.relative_to(source).as_posix())
            descriptor = root / "download.json"
            descriptor.write_text(json.dumps({"schema_version": 1, "archives": [{"url": "https://example.test/frozen.zip",
                                  "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
            destination = root / "artifacts" / "runtime"
            with patch("scripts.download_artifacts.download_archive", side_effect=lambda entry, path: shutil.copyfile(archive, path)):
                self.assertFalse(install_bundle(descriptor, destination)["reused"])
            with patch("scripts.download_artifacts.download_archive") as download:
                self.assertTrue(install_bundle(descriptor, destination)["reused"])
                download.assert_not_called()

    def test_incomplete_download_never_publishes_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = root / "download.json"
            descriptor.write_text(json.dumps({"schema_version": 1, "archives": [{"url": "https://example.test/frozen.zip"}]}))
            destination = root / "artifacts" / "runtime"
            with patch("scripts.download_artifacts.download_archive", side_effect=ValueError("transfer failed")), self.assertRaises(ValueError):
                install_bundle(descriptor, destination)
            self.assertFalse(destination.exists())

    def test_source_recipe_reconstructs_only_declared_offsets_and_newlines(self):
        context = "서울은 대한민국의 수도입니다."
        context_id = hashlib.sha256(context.encode()).hexdigest()
        source_data = {"version": "KorQuAD_v1.0_dev", "data": [{"title": "서울", "paragraphs": [
            {"context": context, "qas": [{"id": "fixture", "question": "대한민국의 수도는?",
             "answers": [{"text": "서울", "answer_start": 0}]}]}]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official.json"
            official.write_text(json.dumps(source_data, ensure_ascii=False), encoding="utf-8")
            recipe = {"source": {"url": "https://example.test/official.json", "bytes": official.stat().st_size,
                      "sha256": hashlib.sha256(official.read_bytes()).hexdigest()}, "source_split": "validation",
                      "chunks": [{"chunk_id": 0, "context_id": context_id, "char_start": 0, "char_end": len(context)}]}
            for newline in ("\n", "\r\n"):
                bundle = root / ("lf" if newline == "\n" else "crlf")
                bundle.mkdir()
                with patch("scripts.download_artifacts.download_archive", side_effect=lambda entry, path: shutil.copyfile(official, path)):
                    materialize_source_data({**recipe, "chunks_newline": newline}, bundle, bundle.parent / (bundle.name + ".json"))
                expected = {"chunk_id": 0, "context_id": context_id, "title": "서울", "char_start": 0,
                            "char_end": len(context), "context": context}
                self.assertEqual((bundle / "chunks.jsonl").read_bytes(), (json.dumps(expected, ensure_ascii=False) + newline).encode())
                corpus = json.loads((bundle / "corpus.jsonl").read_text(encoding="utf-8"))
                self.assertNotIn("answers", corpus)
                self.assertNotIn("question", corpus)

    def test_recipe_does_not_accept_preexisting_redistributed_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "corpus.jsonl").write_text("existing")
            with self.assertRaises(ValueError):
                materialize_source_data({"source": {}, "chunks": [{}]}, root, root / "source.json")


if __name__ == "__main__":
    unittest.main()
