"""Runtime setup must stay project-local and resist environment redirection."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.setup_runtime import child_environment, environment_python, require_inside, setup


class RuntimeSetupTests(unittest.TestCase):
    def test_paths_must_stay_under_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(require_inside(root, root / ".venv"), (root / ".venv").resolve())
            for target in (root, root.parent / "global-python", root / ".." / "elsewhere"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    require_inside(root, target)

    def test_python_and_pip_redirections_removed_but_proxy_kept(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "PYTHONPATH": "untrusted", "PYTHONHOME": "elsewhere", "PIP_TARGET": "global",
            "PIP_USER": "1", "PIP_INDEX_URL": "https://untrusted.invalid",
            "PIP_NO_INDEX": "1", "PIP_LOG": "outside.log", "PIP_BUILD_TRACKER": "outside",
            "HTTPS_PROXY": "http://proxy.example:8080",
        }):
            env = child_environment(Path(directory))
            for name in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_USER", "PIP_INDEX_URL", "PIP_NO_INDEX", "PIP_LOG", "PIP_BUILD_TRACKER"):
                self.assertNotIn(name, env)
            self.assertEqual(env["HTTPS_PROXY"], "http://proxy.example:8080")
            self.assertTrue(Path(env["TEMP"]).is_relative_to(Path(directory)))
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_existing_incompatible_environment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / ".venv"
            venv.mkdir()
            saved = venv / "saved.txt"
            saved.write_text("preserve me")
            with patch("scripts.setup_runtime.healthy", return_value=False), self.assertRaises(ValueError):
                setup(root, venv, "cpu")
            self.assertEqual(saved.read_text(), "preserve me")

    def test_healthy_environment_is_reused_without_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / ".venv"
            with patch("scripts.setup_runtime.healthy", return_value=True), patch("scripts.setup_runtime.subprocess.run") as run:
                self.assertEqual(setup(root, venv, "cpu"), environment_python(venv))
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
