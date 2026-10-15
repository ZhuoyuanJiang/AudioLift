"""Check model downloads, integrity checks, offline use and partial assets."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import download_models as models


class ModelDownloads(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="audiolift models ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.content = b"verified model fixture"
        patch = mock.patch.dict(models.MODELS, {
            "2": ("AudioLift-gaussianlifter.pt", hashlib.sha256(self.content).hexdigest()),
        })
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"})
        patch.start()
        self.addCleanup(patch.stop)
        self.path = self.root / models.MODELS["2"][0]

    def test_first_download_uses_release_name_and_pinned_revision(self):
        def download(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.write_bytes(self.content)
            return str(target)

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch:
            self.assertEqual(models.ensure_checkpoint("2", self.root), self.path)
        self.assertEqual(fetch.call_args.kwargs["repo_id"], "DanJZY/AudioLift")
        self.assertEqual(fetch.call_args.kwargs["revision"], models.MODEL_REVISION)
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_valid_local_checkpoint_needs_no_network(self):
        self.path.write_bytes(self.content)
        with mock.patch("huggingface_hub.hf_hub_download") as fetch:
            models.ensure_checkpoint("2", self.root, allow_download=False)
        fetch.assert_not_called()

    def test_wrong_local_checkpoint_is_rejected_without_overwrite(self):
        self.path.write_bytes(b"wrong checkpoint")
        with mock.patch("huggingface_hub.hf_hub_download") as fetch:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                models.ensure_checkpoint("2", self.root)
        fetch.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"wrong checkpoint")

    def test_downloaded_bytes_must_match_release_checksum(self):
        def download(**kwargs):
            self.path.write_bytes(b"incomplete download")

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=download):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                models.ensure_checkpoint("2", self.root)

    def test_offline_and_check_only_do_not_download_missing_weights(self):
        with mock.patch("huggingface_hub.hf_hub_download") as fetch:
            with self.assertRaisesRegex(FileNotFoundError, "downloads disabled"):
                models.ensure_checkpoint("2", self.root, allow_download=False)
            with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                with self.assertRaisesRegex(FileNotFoundError, "downloads disabled"):
                    models.ensure_checkpoint("2", self.root)
        fetch.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_partial_upstream_directory_downloads_only_missing_files(self):
        (self.root / "wan").mkdir()
        (self.root / "wan/config.json").write_bytes(b"{}")

        def download(**kwargs):
            self.assertEqual(kwargs["allow_patterns"], ["wan/model.bin"])
            self.assertEqual(kwargs["revision"], models.UPSTREAM_REVISION)
            (self.root / "wan/model.bin").write_bytes(b"model")

        with mock.patch.dict(models.UPSTREAM_FILES, {"wan/config.json": 2, "wan/model.bin": 5}, clear=True):
            with mock.patch("huggingface_hub.snapshot_download", side_effect=download) as fetch:
                models.ensure_upstream(self.root)
                models.ensure_upstream(self.root, allow_download=False)
            fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
