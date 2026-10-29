"""Grid rendering and preservation of existing outputs, without API calls."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diag"))
import build_grid_video as grid


class TestGridVideo(unittest.TestCase):
    def test_title_uses_available_japanese_font(self):
        trials = [
            {"trial": index, "simulation": {"goal": True, "end_time": 12.0}}
            for index in range(1, 6)
        ]
        frame = grid.title_cell("Reference", trials)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(float(frame.std()), 10.0)

    def test_result_banner_preserves_source_frame(self):
        source = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = grid.frozen_frame(source, {"goal": True, "end_time": 12.0})
        self.assertEqual(frame.shape, source.shape)
        self.assertTrue(np.all(source == 0))
        self.assertGreater(int(frame.sum()), 0)

    def test_existing_output_is_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.mp4"
            output.write_bytes(b"keep")
            with patch.object(sys, "argv", [
                "build_grid_video.py", "--batch-dir", directory,
                "--title", "Reference", "--output", str(output),
            ]):
                with self.assertRaises(FileExistsError):
                    grid.main()
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
