"""Video encoding helpers with browser-compatible H.264 output."""

import subprocess
from pathlib import Path

import numpy as np


class H264VideoWriter:
    """Stream BGR uint8 frames to ffmpeg as H.264/yuv420p MP4."""

    def __init__(self, path, fps, frame_shape, crf=20, preset="medium"):
        self.path = Path(path)
        height, width = frame_shape[:2]
        if height % 2 or width % 2:
            raise ValueError("H.264 yuv420p requires even frame dimensions")
        self.height = int(height)
        self.width = int(width)
        self._released = False
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(float(fps)),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                str(preset),
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def isOpened(self):
        return self._process.poll() is None and not self._released

    def write(self, frame):
        if self._released:
            raise RuntimeError("video writer is already released")
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"expected {(self.height, self.width, 3)}, got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 video frame, got {frame.dtype}")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            error = self._process.stderr.read().decode("utf-8", errors="replace")
            self._process.stderr.close()
            raise RuntimeError(f"ffmpeg stopped while encoding: {error}") from exc

    def release(self):
        if self._released:
            return
        self._released = True
        self._process.stdin.close()
        error = self._process.stderr.read().decode("utf-8", errors="replace")
        self._process.stderr.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with status {return_code}: {error.strip()}"
            )
