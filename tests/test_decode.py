import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

try:
    import cv2  # noqa: F401
except ImportError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_NEAREST = 0
    cv2_stub.resize = lambda array, size, interpolation: np.zeros(
        (size[1], size[0]), dtype=array.dtype
    )
    sys.modules["cv2"] = cv2_stub

spec = importlib.util.spec_from_file_location("ghost_decode_cli", ROOT / "decode.py")
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class OutputPreparationTests(unittest.TestCase):
    def test_cropping_mask_preserves_full_heatmap_context(self):
        mask = np.zeros((1200, 1200), dtype=np.uint8)
        heat = np.zeros_like(mask)
        mask[500:600, 500:600] = 255
        heat[500:600, 500:600] = 200
        heat[500:600, 100:200] = 80

        display_mask, display_heat = decoder.frame_to_text(mask, heat)

        self.assertLess(display_mask.shape[0], mask.shape[0])
        self.assertLess(display_mask.shape[1], mask.shape[1])
        self.assertIs(display_heat, heat)
        self.assertEqual(display_heat.shape, (1200, 1200))
        self.assertGreater(int(display_heat[:, 100:200].sum()), 0)

    def test_empty_mask_keeps_both_outputs_unchanged(self):
        mask = np.zeros((1200, 1200), dtype=np.uint8)
        heat = np.ones_like(mask)

        display_mask, display_heat = decoder.frame_to_text(mask, heat)

        self.assertIs(display_mask, mask)
        self.assertIs(display_heat, heat)


class CliTests(unittest.TestCase):
    def test_retry_options_are_supported(self):
        args = decoder.build_parser().parse_args(
            [
                "video.mp4",
                "-o",
                "out",
                "--method",
                "farneback",
                "--stride",
                "2",
                "--max-frames",
                "200",
            ]
        )
        self.assertEqual(args.method, "farneback")
        self.assertEqual(args.stride, 2)
        self.assertEqual(args.max_frames, 200)

    def test_stride_must_be_positive(self):
        with self.assertRaises(decoder.argparse.ArgumentTypeError):
            decoder.positive_int("0")


class OcrDiscoveryTests(unittest.TestCase):
    def test_windows_tesseract_location_is_discovered_when_not_on_path(self):
        expected = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        with mock.patch.object(decoder.shutil, "which", return_value=None):
            with mock.patch.object(
                decoder.os.path,
                "isfile",
                side_effect=lambda candidate: candidate == expected,
            ):
                self.assertEqual(decoder.find_tesseract(), expected)


if __name__ == "__main__":
    unittest.main()
