"""Legacy conversion rejects altered input before any weights-only loader call."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.convert_original_retriever import convert


class OriginalRetrieverConversionTests(unittest.TestCase):
    def test_wrong_checkpoint_digest_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "original.pt"
            checkpoint.write_bytes(b"altered checkpoint")
            with patch("scripts.convert_original_retriever.SOURCE_BYTES", checkpoint.stat().st_size), \
                 patch("scripts.convert_original_retriever.load_initial_model") as model_loader, \
                 self.assertRaisesRegex(ValueError, "SHA256"):
                convert(checkpoint=checkpoint, metadata_directory=root / "metadata",
                        output_directory=root / "portable")
            model_loader.assert_not_called()
            self.assertFalse((root / "portable").exists())

    def test_unexpected_metadata_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "original.pt"
            checkpoint.write_bytes(b"fixture")
            metadata = root / "metadata"
            metadata.mkdir()
            (metadata / "model.pt").write_bytes(b"unexpected")
            with patch("scripts.convert_original_retriever.verify_file"), \
                 patch("scripts.convert_original_retriever.load_initial_model") as model_loader, \
                 self.assertRaisesRegex(ValueError, "exactly"):
                convert(checkpoint=checkpoint, metadata_directory=metadata,
                        output_directory=root / "portable")
            model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
