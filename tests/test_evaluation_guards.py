from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from korean_rag.benchmark import freeze_configuration, sha256_file
from scripts.evaluate_portable import (REQUIRED_FINAL_SOURCES, environment_metadata,
                                       validate_relative_hash_paths, verify_frozen_final)


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        for name in REQUIRED_FINAL_SOURCES:
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'fixture for {name}\n', encoding='utf-8')
        (self.bundle / 'manifest.json').write_text(json.dumps({'files': {}}), encoding='utf-8')
        self.inputs = {'split': 'final', 'benchmark_sha256': 'a' * 64,
                       'question_file_sha256': 'b' * 64, 'no_answer_benchmark_sha256': 'c' * 64,
                       'no_answer_file_sha256': 'd' * 64}
        self.configuration = {'evaluation_binding': self.inputs, 'reader': {'null_score_margin': 1e-6}}
        self.benchmark = {'benchmark_sha256': 'a' * 64, 'protocol_sha256': 'e' * 64}
        self.frozen = freeze_configuration(
            **self.benchmark, configuration=self.configuration,
            source_sha256={name: sha256_file(self.project / name) for name in REQUIRED_FINAL_SOURCES},
            artifact_sha256={'manifest.json': sha256_file(self.bundle / 'manifest.json')})

    def verify(self, frozen=None, inputs=None):
        verify_frozen_final(self.frozen if frozen is None else frozen, configuration=self.configuration,
                            input_metadata=self.inputs if inputs is None else inputs,
                            benchmark_manifest=self.benchmark, project_root=self.project,
                            bundle_directory=self.bundle)

    def test_complete_binding_verifies_without_models_or_question_inference(self):
        self.verify()

    def test_every_inference_and_metric_dependency_is_required(self):
        for name in REQUIRED_FINAL_SOURCES:
            with self.subTest(name=name):
                partial = deepcopy(self.frozen)
                del partial['source_sha256'][name]
                with self.assertRaisesRegex(ValueError, 'omits'):
                    self.verify(partial)

    def test_artifact_manifest_is_required(self):
        partial = deepcopy(self.frozen)
        partial['artifact_sha256'] = {'config.json': 'f' * 64}
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.verify(partial)

    def test_portable_relative_paths_reject_traversal_and_windows_separators(self):
        for name in ('../runtime.py', '/runtime.py', 'C:/runtime.py', 'korean_rag\\runtime.py',
                     'korean_rag//runtime.py', 'korean_rag/./runtime.py'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'forward-slash'):
                validate_relative_hash_paths({name: 'a' * 64}, 'Source')

    def test_changed_source_or_manifest_is_rejected(self):
        source = self.project / 'korean_rag/runtime.py'
        original = source.read_bytes()
        source.write_bytes(original + b'changed\n')
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            self.verify()
        source.write_bytes(original)
        (self.bundle / 'manifest.json').write_text(json.dumps({'files': {'extra': {}}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            self.verify()

    def test_changed_no_answer_input_cannot_reuse_the_same_freeze(self):
        wrong_inputs = {**self.inputs, 'no_answer_file_sha256': 'f' * 64}
        with self.assertRaisesRegex(ValueError, 'data hashes'):
            self.verify(inputs=wrong_inputs)

    def test_execution_metadata_records_actual_cpu_and_numeric_settings(self):
        metadata = environment_metadata(SimpleNamespace(device='cpu'))
        self.assertEqual(metadata['device'], 'cpu')
        self.assertIsNone(metadata['gpu_name'])
        self.assertTrue(metadata['python_version'])
        self.assertIn('transformers', metadata['dependency_versions'])
        self.assertIn('matmul_allow_tf32', metadata)
        self.assertIn('cudnn_allow_tf32', metadata)
        self.assertIn('torch_num_threads', metadata)
        self.assertIn('float32_matmul_precision', metadata)


if __name__ == '__main__':
    unittest.main()
