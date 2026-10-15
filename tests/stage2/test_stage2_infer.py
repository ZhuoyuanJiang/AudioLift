"""CPU regressions for alignment and the audio-tail bug in 81-frame clips."""

import contextlib
import hashlib
import io
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

    def test_aligned_inputs_and_explicit_prefix(self):
        full = stage2.validate_inputs(self.inputs, self.audio)
        self.assertEqual(full["frames"], 81)
        self.assertEqual(stage2.validate_inputs(self.inputs, self.audio, 3)["frames"], 3)
        for invalid in (0, -1, 82):
            with self.assertRaisesRegex(ValueError, "max-frames"):
                stage2.validate_inputs(self.inputs, self.audio, invalid)

    def test_reject_mismatched_frame_counts(self):
        inputs = self.root / "short-front"
        inputs.mkdir()
        (inputs / "grid.mp4").symlink_to(self.inputs / "grid.mp4")
        self.video(inputs / "front.mp4", "416x608", 80)
        with self.assertRaisesRegex(ValueError, "same frame count"):
            stage2.validate_inputs(inputs, self.audio)

    def test_reject_unaligned_audio_and_trim_smoke_audio(self):
        work = self.root / "smoke-audio"
        work.mkdir()
        audio = stage2.encode_audio(self.audio, work, trim_seconds=3 / 25)
        self.assertAlmostEqual(stage2.probe_audio(audio)["duration"], .12, places=2)
        with self.assertRaisesRegex(ValueError, "Audio duration must match"):
            stage2.validate_inputs(self.inputs, audio)

    def test_reject_wrong_fps_and_grid_geometry(self):
        wrong_fps = self.root / "wrong-fps.mp4"
        self.video(wrong_fps, "416x608", 5, fps=30)
        with self.assertRaisesRegex(ValueError, "25 fps"):
            stage2.probe_video(wrong_fps)
        inputs = self.root / "wrong-grid"
        inputs.mkdir()
        (inputs / "grid.mp4").symlink_to(self.inputs / "front.mp4")
        (inputs / "front.mp4").symlink_to(self.inputs / "front.mp4")
        with self.assertRaisesRegex(ValueError, "832x1216"):
            stage2.validate_inputs(inputs, self.audio)

    def test_reject_wrong_checkpoint_and_existing_outputs(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            stage2.verify_hash(self.audio, stage2.CHECKPOINT_SHA256)
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as exit_status:
            stage2.main(["--stage1-dir", str(self.inputs), "--audio", str(self.audio),
                         "--checkpoint", str(self.audio), "--out", str(self.inputs), "--dry-run"])
        self.assertEqual(exit_status.exception.code, 1)
        self.assertIn("Output already exists", error.getvalue())
        self.assertTrue((self.inputs / "grid.mp4").is_file())

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
                mock.patch.object(stage2, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest()), \
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

    def test_all_81_frames_and_complete_aligned_audio_survive_mux(self):
        work = self.root / "encode"
        work.mkdir()
        for frame in range(81):
            Image.new("RGB", (64, 64), (frame * 3, 0, 0)).save(work / f"frontal_{frame:06d}.png")
        encoded = stage2.encode_audio(self.audio, work)
        output = self.root / "frontal.mp4"
        result = stage2.encode_video(work, "frontal", 81, encoded, output)
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
