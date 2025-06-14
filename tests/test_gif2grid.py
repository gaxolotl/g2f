# tests/test_gif2grid.py
import os
import unittest
from main import gif2grid_zip

class TestGif2Grid(unittest.TestCase):
    def test_basic_functionality(self):
        gif_path = "tests/test.gif"  # Add a small test GIF here
        output_dir = "tests/output"
        os.makedirs(output_dir, exist_ok=True)

        def dummy_status(msg):
            print(msg)

        def dummy_progress(val):
            print(f"Progress: {val}")

        gif2grid_zip(
            gif_path,
            frame_start=0,
            frame_end=3,
            output_dir=output_dir,
            apply_filter="None",
            update_status=dummy_status,
            update_progress=dummy_progress,
            zip_only=True
        )

        # Check if ZIP file was created
        self.assertTrue(os.path.exists(os.path.join(output_dir, "gif_tiles_export.zip")))

if __name__ == "__main__":
    unittest.main()
