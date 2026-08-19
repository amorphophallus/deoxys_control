import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deoxys.utils.video_utils import H264VideoWriter


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg missing",
)
class H264VideoWriterTest(unittest.TestCase):
    def test_writes_browser_compatible_h264_yuv420p(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "test.mp4"
            writer = H264VideoWriter(output_path, fps=10, frame_shape=(48, 64, 3))
            for value in (0, 64, 128, 255):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()

            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt",
                        "-of",
                        "json",
                        str(output_path),
                    ],
                    text=True,
                )
            )
            stream = probe["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(stream["pix_fmt"], "yuv420p")


if __name__ == "__main__":
    unittest.main()
