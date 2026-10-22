"""Check automatic preparation, original-pixel fallbacks and strict review."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.preprocess import preprocess_reference as prep


class ReferencePreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audiolift reference ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.cache = self.output / "_gate_candidate"
        self.source = self.root / "input.png"
        self.image = Image.new("RGB", (100, 100), (40, 70, 100))
        self.image.paste((190, 80, 50), (20, 20, 80, 80))
        self.image.save(self.source)
        self.original_bytes = self.source.read_bytes()

    def run_prep(self, **kwargs):
        return Path(prep.ensure_reference(self.source, width=104, height=152,
                                         cache_dir=str(self.cache), cpu=True, **kwargs))

    def record(self):
        return json.loads((self.output / "reference_processing.json").read_text())

    def face_candidate(self, *args, **kwargs):
        self.cache.mkdir(parents=True, exist_ok=True)
        candidate = self.cache / "input_face.png"
        prepared = Image.new("RGB", (104, 152), "white")
        prepared.paste((180, 90, 60), (20, 20, 84, 132))
        prepared.save(candidate)
        return str(candidate)

    def test_prepared_jpeg_keeps_decoded_pixels_and_never_runs_matting(self):
        self.source = self.root / "prepared.jpg"
        original = Image.new("RGB", (208, 304), "white")
        original.paste((120, 45, 30), (50, 30, 160, 250))
        original.save(self.source)
        with patch.object(prep, "preprocess_one") as face, patch.object(prep, "matte_rembg") as matte:
            result = self.run_prep()
        face.assert_not_called()
        matte.assert_not_called()
        np.testing.assert_array_equal(np.asarray(Image.open(result)),
                                      np.asarray(Image.open(self.source).convert("RGB")))
        self.assertEqual(self.record()["method"], "original")
        self.assertEqual(self.record()["reference_used"], str(result))

    def test_raw_image_automatically_uses_the_prepared_candidate(self):
        with patch.object(prep, "preprocess_one", side_effect=self.face_candidate):
            result = self.run_prep()
        np.testing.assert_array_equal(np.asarray(Image.open(result)),
                                      np.asarray(Image.open(self.cache / "input_face.png")))
        self.assertEqual(self.record()["method"], "rvm_face")
        self.assertEqual(self.record()["status"], "ready")
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_no_face_uses_rembg_on_original_pixels_and_preserves_aspect(self):
        fg = np.asarray(self.image, dtype=np.float32) / 255
        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[20:80, 20:80] = 1

        def matte(image):
            np.testing.assert_array_equal(np.asarray(image), np.asarray(self.image))
            return fg, alpha

        with patch.object(prep, "preprocess_one", return_value=None), \
                patch.object(prep, "matte_rembg", side_effect=matte):
            result = self.run_prep()
        expected = prep.frame_fitpad(prep.composite_white(fg, alpha), 104, 152)
        np.testing.assert_array_equal(np.asarray(Image.open(result)), np.asarray(expected))
        self.assertEqual(self.record()["method"], "rembg_fitpad")
        self.assertEqual(self.record()["attempts"][0]["status"], "no_face")

    def test_nearly_empty_matte_keeps_original_subject_and_background(self):
        fg = np.zeros((100, 100, 3), dtype=np.float32)
        alpha = np.full((100, 100), 0.01, dtype=np.float32)
        with patch.object(prep, "preprocess_one", return_value=None), \
                patch.object(prep, "matte_rembg", return_value=(fg, alpha)):
            result = self.run_prep()
        np.testing.assert_array_equal(np.asarray(Image.open(result)),
                                      np.asarray(prep.frame_fitpad(self.image, 104, 152)))
        self.assertEqual(self.record()["method"], "original_fitpad")

    def test_preprocessing_failures_are_recorded_and_keep_original_pixels(self):
        with patch.object(prep, "preprocess_one", side_effect=RuntimeError("face asset unavailable")), \
                patch.object(prep, "matte_rembg", side_effect=RuntimeError("matting asset unavailable")):
            result = self.run_prep()
        np.testing.assert_array_equal(np.asarray(Image.open(result)),
                                      np.asarray(prep.frame_fitpad(self.image, 104, 152)))
        record = self.record()
        self.assertEqual(record["method"], "original_fitpad")
        self.assertIn("face asset unavailable", record["attempts"][0]["error"])
        self.assertIn("matting asset unavailable", record["attempts"][1]["reason"])

    def test_strict_mode_prepares_candidate_but_never_uses_it(self):
        with patch.object(prep, "preprocess_one", side_effect=self.face_candidate), \
                self.assertRaises(prep.ReferenceGateError):
            self.run_prep(strict=True)
        record = self.record()
        self.assertEqual(record["status"], "review_required")
        self.assertTrue(Path(record["candidate"]).is_file())
        self.assertFalse((self.output / "reference_used.png").exists())

    def test_explicit_overrides_still_record_the_original_image(self):
        for options in ({"skip": True}, {"allow_noncompliant": True}):
            with self.subTest(options=options), patch.object(prep, "preprocess_one") as face:
                result = self.run_prep(**options)
            face.assert_not_called()
            np.testing.assert_array_equal(np.asarray(Image.open(result)), np.asarray(self.image))
            self.assertEqual(self.record()["method"], "original")

    def test_unreadable_image_records_error_instead_of_faking_success(self):
        self.source.write_text("not an image")
        with self.assertRaises(prep.ReferenceGateError):
            self.run_prep()
        self.assertEqual(self.record()["status"], "error")
        self.assertFalse((self.output / "reference_used.png").exists())

    def test_no_face_does_not_reuse_candidate_from_a_previous_attempt(self):
        self.face_candidate()
        def no_face(args):
            (self.cache / "input_meta.json").write_text(json.dumps({"face_mode": "skipped_no_face"}))
        with patch.object(prep, "process_images", side_effect=no_face):
            result = prep.preprocess_one(str(self.source), str(self.cache))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
