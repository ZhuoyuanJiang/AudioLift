"""Exercise the portable launchers with temporary assets and no GPU work."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Stage1Entrypoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audiolift entrypoints ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clip = self.root / "prepared clip"
        self.clip.mkdir()
        for name in ("reference_image.jpg", "audio.wav"):
            (self.clip / name).write_bytes(b"test fixture")
        self.weights = self.root / "input.pt"
        self.weights.write_bytes(b"explicit new-lineage test weights")
        self.capture = self.root / "arguments.json"
        self.stub = self.root / "capture-python"
        self.stub.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\n"
            "with open(os.environ['ARGUMENT_CAPTURE'], 'w') as f:\n"
            "    json.dump(sys.argv[1:], f)\n"
            "sys.exit(int(os.environ.get('CAPTURE_EXIT', '0')))\n"
        )
        self.stub.chmod(0o755)
        self.env = dict(os.environ, STAGE1_PYTHON=str(self.stub),
                        ARGUMENT_CAPTURE=str(self.capture))
        self.env.pop("CKPT", None)
        self.env.pop("STAGE1_ASSET_ROOT", None)





    def infer(self, *extra):
        args = [sys.executable, str(ROOT / "inference_stage1.py"),
                "--reference", "prepared clip/reference_image.jpg", "--audio", "prepared clip/audio.wav",
                "--checkpoint", "input.pt", "--out", "inference output", "--dry_run", *extra]
        return subprocess.run(args, cwd=self.root, env=self.env, capture_output=True, text=True)

    def test_inference_passes_explicit_checkpoint_to_the_original_loader(self):
        result = self.infer()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.capture.read_text())
        self.assertEqual(args[args.index("--transformer_path") + 1], str(self.weights))

    def test_inference_resolves_relative_paths_before_changing_directory(self):
        result = self.infer()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.capture.read_text())
        self.assertEqual(args[0], str(ROOT / "stage1/pipeline.py"))
        self.assertIn(str(self.clip / "reference_image.jpg"), args)
        self.assertIn(str(self.root / "inference output"), args)

    def test_reference_refusal_exit_status_is_preserved(self):
        self.env["CAPTURE_EXIT"] = "3"
        self.assertEqual(self.infer().returncode, 3)

    def test_strict_review_reaches_stage1_pipeline(self):
        result = self.infer("--strict-reference")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--strict-reference", json.loads(self.capture.read_text()))


if __name__ == "__main__":
    unittest.main()
