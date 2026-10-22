"""Check two-stage orchestration using subprocess stand-ins, without model execution."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

import inference

ROOT = Path(__file__).resolve().parents[1]


class UnifiedInference(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audiolift unified ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("portrait.png", "speech.wav", "stage1.pt", "stage2.pt"):
            (self.root / name).write_bytes(b"fixture")
        self.log = self.root / "calls.jsonl"
        self.stub = self.root / "stage stand-in.py"
        self.stub.write_text('''import json, os, sys
from pathlib import Path
stage = sys.argv[1]
args = sys.argv[2:]
def value(flag):
    return args[args.index(flag) + 1]
with open(os.environ["INFERENCE_TEST_LOG"], "a") as log:
    log.write(json.dumps({"stage": stage, "args": args,
                         "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "python": os.environ.get("STAGE" + stage + "_PYTHON")}) + "\\n")
code = int(os.environ.get("INFERENCE_TEST_EXIT_" + stage, "0"))
if code:
    sys.exit(code)
out = Path(value("--out"))
if stage == "2":
    previous = Path(value("--stage1-dir"))
    assert all((previous / name).is_file() for name in ("grid.mp4", "front.mp4"))
out.mkdir(parents=True, exist_ok=True)
names = ("grid.mp4", "front.mp4") if stage == "1" else ("frontal.mp4", "coverage.mp4", "orbit360.mp4")
for name in names:
    if name != os.environ.get("INFERENCE_TEST_OMIT"):
        (out / name).write_bytes(b"video stand-in")
''')
        self.pythons = []
        for stage in (1, 2):
            executable = self.root / f"stage {stage} python"
            # The executable emulates a stage-specific Python and runs an isolated
            # process. The real stage entrypoint path must still be selected correctly.
            executable.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                f"assert sys.argv[1] == {str(ROOT / f'inference_stage{stage}.py')!r}\n"
                f"os.execv({sys.executable!r}, [{sys.executable!r}, {str(self.stub)!r}, {str(stage)!r}, *sys.argv[2:]])\n"
            )
            executable.chmod(0o755)
            self.pythons.append(executable)
        self.env = dict(os.environ, STAGE1_PYTHON=str(self.pythons[0]),
                        STAGE2_PYTHON=str(self.pythons[1]), INFERENCE_TEST_LOG=str(self.log))
        self.env.pop("CUDA_VISIBLE_DEVICES", None)

    def run_inference(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "inference.py"), "--reference", "portrait.png",
             "--audio", "speech.wav", "--out", "result with spaces", *extra],
            cwd=self.root, env=self.env, capture_output=True, text=True,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_two_environments_handoff_paths_options_and_visible_gpu(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "5,2"
        result = self.run_inference("--gpu", "1", "--steps", "24", "--seed", "123",
                                    "--turns", "1.5", "--stage1-checkpoint", "stage1.pt",
                                    "--stage2-checkpoint", "stage2.pt", "--no-download",
                                    "--skip_reference_check")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["stage"] for call in calls], ["1", "2"])
        for index, call in enumerate(calls):
            args = call["args"]
            self.assertEqual(call["python"], str(self.pythons[index]))
            self.assertEqual(call["gpu"], "2")
            self.assertEqual(args[args.index("--gpu") + 1], "2" if index == 0 else "0")
            self.assertEqual(args[args.index("--audio") + 1], str(self.root / "speech.wav"))
            self.assertEqual(args[args.index("--out") + 1], str(self.root / "result with spaces" / f"stage{index + 1}"))
            self.assertEqual(args[args.index("--checkpoint") + 1], str(self.root / f"stage{index + 1}.pt"))
            self.assertIn("--no-download", args)
        for flag, expected in (("--steps", "24"), ("--seed", "123")):
            self.assertEqual(calls[0]["args"][calls[0]["args"].index(flag) + 1], expected)
        self.assertEqual(calls[1]["args"][calls[1]["args"].index("--turns") + 1], "1.5")
        self.assertIn("--skip_reference_check", calls[0]["args"])
        self.assertNotIn("--skip_reference_check", calls[1]["args"])
        self.assertEqual(calls[1]["args"][calls[1]["args"].index("--stage1-dir") + 1],
                         str(self.root / "result with spaces/stage1"))
        self.assertTrue((self.root / "result with spaces/stage2/orbit360.mp4").is_file())

    def test_stage1_failure_and_reference_refusal_never_start_stage2(self):
        for code in (1, 3):
            with self.subTest(code=code):
                self.log.unlink(missing_ok=True)
                self.env["INFERENCE_TEST_EXIT_1"] = str(code)
                result = self.run_inference()
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual([call["stage"] for call in self.calls()], ["1"])
                if code == 3:
                    self.assertIn("_gate_candidate", result.stderr)

    def test_strict_reference_option_only_reaches_stage1(self):
        result = self.run_inference("--strict-reference")
        self.assertEqual(result.returncode, 0, result.stderr)
        first, second = self.calls()
        self.assertIn("--strict-reference", first["args"])
        self.assertNotIn("--strict-reference", second["args"])

    def test_missing_stage1_video_prevents_stage2(self):
        self.env["INFERENCE_TEST_OMIT"] = "front.mp4"
        result = self.run_inference()
        self.assertEqual(result.returncode, 1)
        self.assertIn("front.mp4", result.stderr)
        self.assertEqual([call["stage"] for call in self.calls()], ["1"])

    def test_stage2_failure_preserves_stage1_for_retry(self):
        self.env["INFERENCE_TEST_EXIT_2"] = "7"
        result = self.run_inference()
        self.assertEqual(result.returncode, 7)
        self.assertIn("inference_stage2.py", result.stderr)
        self.assertTrue((self.root / "result with spaces/stage1/grid.mp4").is_file())

    def test_missing_stage2_environment_fails_before_stage1(self):
        self.env["STAGE2_PYTHON"] = str(self.root / "missing python")
        result = self.run_inference()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Stage 2 Python not found", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_explicit_python_overrides_environment(self):
        self.env["STAGE2_PYTHON"] = str(self.root / "missing python")
        result = self.run_inference("--stage2-python", self.pythons[1].name)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_directory_is_accepted_and_unrelated_files_are_preserved(self):
        out = self.root / "result with spaces"
        out.mkdir()
        (out / "keep.txt").write_text("keep")
        result = self.run_inference()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call["stage"] for call in self.calls()], ["1", "2"])
        self.assertEqual((out / "keep.txt").read_text(), "keep")

    def test_complete_results_do_not_regenerate_only_stage1(self):
        self.assertEqual(self.run_inference().returncode, 0)
        self.assertEqual(self.run_inference().returncode, 0)
        self.assertEqual([call["stage"] for call in self.calls()], ["1", "2"])

    def test_invalid_input_checkpoint_or_gpu_fails_before_launch(self):
        for extra in (("--reference", "missing.png"), ("--stage2-checkpoint", "missing.pt"),
                      ("--gpu", "-1")):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_inference(*extra).returncode, 1)
                self.assertEqual(self.calls(), [])
        self.env["CUDA_VISIBLE_DEVICES"] = ""
        self.assertEqual(self.run_inference().returncode, 1)
        self.assertEqual(self.calls(), [])

    def test_default_venv_python_keeps_its_environment(self):
        destination = self.root / ".venv-stage1"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(destination)
        with patch.object(inference, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            executable = inference.python_for_stage(1, None)
        result = subprocess.run([executable, "-c", "import sys; print(sys.prefix)"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(Path(result.stdout.strip()), destination)


if __name__ == "__main__":
    unittest.main()
