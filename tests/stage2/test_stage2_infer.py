"""CPU regressions for paper-compatible frame handling and complete audio."""

import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage2 import pipeline as stage2


class InferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stage2-infer-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.inputs = cls.root / "inputs"
        cls.inputs.mkdir()
        cls.video(cls.inputs / "grid.mp4", "832x1216", 81)
        cls.video(cls.inputs / "front.mp4", "416x608", 81)
        cls.audio = cls.root / "audio.wav"
        stage2.ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=3.24",
                      "-c:a", "pcm_s16le", cls.audio)

    @staticmethod
    def video(path, size, frames, fps=25):
        stage2.ffmpeg("-f", "lavfi", "-i", f"color=white:s={size}:r={fps}",
                      "-frames:v", frames, "-c:v", "libx264", "-preset", "ultrafast", path)

    def test_stage1_frame_plan_handoff_for_short_and_long_audio(self):
        # Execute the real Stage-1 planning functions without loading its models.
        source = stage2.ROOT / "stage1/pipeline.py"
        tree = ast.parse(source.read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name in ("build_windows", "plan_windows")]
        namespace = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
        for seconds in (3.24, 4, 13, 14, 15, 16, 25, 40):
            with self.subTest(seconds=seconds):
                frames, _, _, _ = namespace["plan_windows"](int(seconds * 25), 4, 81, 5)
                inputs = self.root / f"plan-{seconds}"
                inputs.mkdir()
                self.video(inputs / "grid.mp4", "64x64", frames)
                self.video(inputs / "front.mp4", "32x32", frames)
                audio = inputs / "audio.wav"
                stage2.ffmpeg("-f", "lavfi", "-i", f"sine=sample_rate=16000:duration={seconds}",
                              "-c:a", "pcm_s16le", audio)
                result = stage2.inspect_inputs(inputs, audio)
                self.assertEqual(result["frames"], frames)
                self.assertAlmostEqual(result["audio"]["duration"], seconds, places=3)

    def test_extra_frontal_frames_do_not_reject_the_grid(self):
        inputs = self.root / "long-front"
        inputs.mkdir()
        (inputs / "grid.mp4").symlink_to(self.inputs / "grid.mp4")
        self.video(inputs / "front.mp4", "416x608", 82)
        self.assertEqual(stage2.inspect_inputs(inputs, self.audio)["frames"], 81)

    def test_media_metadata_does_not_impose_new_fps_geometry_or_start_time_rules(self):
        inputs = self.root / "other-media"
        inputs.mkdir()
        self.video(inputs / "grid.mp4", "64x64", 30, fps=30)
        self.video(inputs / "front.mp4", "32x32", 30, fps=30)
        audio = inputs / "offset.m4a"
        stage2.ffmpeg("-i", self.audio, "-af", "asetpts=PTS+1/TB", "-c:a", "aac", audio)
        self.assertEqual(stage2.inspect_inputs(inputs, audio)["frames"], 30)

    def test_existing_directory_and_local_checkpoint_are_accepted(self):
        # A dry run does not load this stand-in; loading belongs to the model.
        with contextlib.redirect_stdout(io.StringIO()):
            result = stage2.main(["--stage1-dir", str(self.inputs), "--audio", str(self.audio),
                                  "--checkpoint", str(self.audio), "--out", str(self.inputs), "--dry-run"])
        self.assertEqual(result, 0)
        self.assertTrue((self.inputs / "grid.mp4").is_file())

    def test_completed_outputs_are_kept_as_in_the_paper_renderer(self):
        out = self.root / "completed"
        out.mkdir()
        for name in ("frontal", "coverage", "orbit360"):
            (out / f"{name}.mp4").write_bytes(b"existing paper output")
        with mock.patch.object(stage2, "render_frames") as render, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2.main(["--stage1-dir", str(self.inputs), "--audio", str(self.audio),
                                          "--out", str(out)]), 0)
        render.assert_not_called()
        self.assertEqual((out / "frontal.mp4").read_bytes(), b"existing paper output")

    def test_camera_paths_keep_historical_endpoints_and_world_up(self):
        self.assertEqual([stage2.coverage_azimuth(i, 81) for i in (0, 20, 40, 60, 80)],
                         [0, 51, 0, -40, 0])
        self.assertEqual(stage2.coverage_azimuth(0, 1), 0)
        self.assertEqual(stage2.orbit_azimuth(80, 81, 1), 360)
        self.assertAlmostEqual(stage2.orbit_azimuth(80, 81, 2), 720 * 80 / 81)
        np.testing.assert_allclose(stage2.rotation_z(90) @ [1, 0, 2, 1], [0, 1, 2, 1], atol=1e-7)

    def test_default_model_download_precedes_rendering_and_respects_offline(self):
        repo = self.root / "download-repository"
        repo.mkdir()
        for name in ("configs", "stage2"):
            (repo / name).symlink_to(stage2.ROOT / name, target_is_directory=True)
        content = b"downloaded Gaussian model fixture"
        checkpoint = repo / "checkpoints" / "AudioLift-gaussianlifter.pt"

        def download(stage, directory, *, allow_download):
            self.assertEqual(stage, "2")
            self.assertEqual(directory, repo / "checkpoints")
            if not allow_download:
                raise FileNotFoundError("downloads disabled")
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(content)
            return checkpoint

        base = ["--stage1-dir", str(self.inputs), "--audio", str(self.audio)]
        with mock.patch.object(stage2, "ROOT", repo), \
                mock.patch.object(stage2, "ensure_checkpoint", side_effect=download) as fetch, \
                mock.patch.object(stage2, "render_frames", side_effect=RuntimeError("stop before GPU")) as render:
            for option in ("--dry-run", "--no-download"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    stage2.main([*base, "--out", str(self.root / "offline-output"), option])
                self.assertFalse(checkpoint.exists())
                render.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stage2.main([*base, "--out", str(self.root / "download-output")])
            self.assertEqual(checkpoint.read_bytes(), content)
            render.assert_called_once()
            self.assertTrue(fetch.call_args.kwargs["allow_download"])

    def test_natural_audio_tail_reaches_all_outputs_without_padding_or_trimming(self):
        inputs = self.root / "natural-tail"
        inputs.mkdir()
        self.video(inputs / "grid.mp4", "64x64", 97)
        self.video(inputs / "front.mp4", "32x32", 97)
        audio = inputs / "audio.wav"
        stage2.ffmpeg("-f", "lavfi", "-i", "sine=sample_rate=16000:duration=4",
                      "-c:a", "pcm_s16le", audio)
        out = self.root / "partial-output"
        out.mkdir()
        (out / "frontal.mp4").write_bytes(b"old partial result")
        (out / "keep.txt").write_text("unrelated file")

        def render(_inputs, _checkpoint, work, frames, _turns, _gpu):
            self.assertEqual(frames, 97)
            for frame in range(frames):
                for name in ("frontal", "coverage", "orbit360"):
                    Image.new("RGB", (64, 64), (frame * 2, 0, 0)).save(work / f"{name}_{frame:06d}.png")
            return {"test": "GPU reconstruction replaced; real decoding and mux"}

        with mock.patch.object(stage2, "render_frames", side_effect=render), \
                contextlib.redirect_stdout(io.StringIO()):
            result = stage2.main(["--stage1-dir", str(inputs), "--audio", str(audio),
                                  "--checkpoint", str(self.audio), "--out", str(out)])
        self.assertEqual(result, 0)
        receipt = json.loads((out / "run.json").read_text())
        self.assertEqual(receipt["frames"], 97)
        for name in ("frontal", "coverage", "orbit360"):
            self.assertEqual(stage2.probe_video(out / f"{name}.mp4")["frames"], 97)
            self.assertAlmostEqual(stage2.probe_audio(out / f"{name}.mp4")["duration"], 4, places=2)
        self.assertEqual((out / "keep.txt").read_text(), "unrelated file")

    def test_all_81_frames_and_complete_aligned_audio_survive_mux(self):
        work = self.root / "encode"
        work.mkdir()
        for frame in range(81):
            Image.new("RGB", (64, 64), (frame * 3, 0, 0)).save(work / f"frontal_{frame:06d}.png")
        encoded = stage2.encode_audio(self.audio, work)
        output = self.root / "frontal.mp4"
        result = stage2.encode_video(work, "frontal", encoded, output)
        self.assertEqual(result["video"]["frames"], 81)
        self.assertAlmostEqual(result["audio"]["duration"], 3.24, places=2)

        def samples(path):
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                      "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                     check=True, capture_output=True)
            return np.frombuffer(decoded.stdout, dtype="<f4")

        reference, delivered = samples(self.audio), samples(output)
        self.assertEqual(len(reference), 51840)
        self.assertGreaterEqual(len(delivered), len(reference))
        self.assertGreater(np.corrcoef(reference, delivered[:len(reference)])[0, 1], .999)


if __name__ == "__main__":
    unittest.main()
