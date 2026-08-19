"""Prompt Depth Anything adapter for RealSense RGB-D observations.

The model is loaded lazily so normal Deoxys imports do not require PyTorch or
the PromptDA submodule. RealSense invalid and saturated samples are removed
before the metric-depth prompt is passed to PromptDA.
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


MODEL_IDS = {
    "vits": "depth-anything/prompt-depth-anything-vits",
    "vitl": "depth-anything/prompt-depth-anything-vitl",
    "vits-transparent": "depth-anything/prompt-depth-anything-vits-transparent",
}


def _promptda_root():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "third_party" / "PromptDA"
    if not path.is_dir():
        raise RuntimeError(
            f"PromptDA submodule is missing at {path}. Run: "
            "git submodule update --init --recursive"
        )
    return path


def _inference_shape(height, width, max_size, patch_size=14):
    max_size = max(patch_size, int(max_size) // patch_size * patch_size)
    scale = max_size / float(max(height, width))
    output_height = max(
        patch_size,
        int(round(height * scale / patch_size)) * patch_size,
    )
    output_width = max(patch_size, int(round(width * scale / patch_size)) * patch_size)
    return output_height, output_width


def prepare_prompt_depth(
    depth_m,
    output_width=256,
    output_height=192,
    min_depth_m=0.05,
    max_depth_m=5.0,
):
    """Create a dense, bounded metric prompt from a noisy RealSense depth map."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")

    valid = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    if np.count_nonzero(valid) < 16:
        raise ValueError("fewer than 16 valid RealSense depth pixels")

    valid_values = depth[valid]
    low, high = np.percentile(valid_values, [1.0, 99.0])
    low = max(float(low), float(min_depth_m))
    high = min(float(high), float(max_depth_m))
    if high - low < 1e-4:
        low = float(np.min(valid_values))
        high = float(np.max(valid_values))
    if high - low < 1e-4:
        raise ValueError("RealSense depth has no usable metric range")

    clipped = np.clip(depth, low, high)
    weighted_depth = cv2.resize(
        np.where(valid, clipped, 0.0),
        (int(output_width), int(output_height)),
        interpolation=cv2.INTER_AREA,
    )
    weights = cv2.resize(
        valid.astype(np.float32),
        (int(output_width), int(output_height)),
        interpolation=cv2.INTER_AREA,
    )
    prompt = np.zeros_like(weighted_depth, dtype=np.float32)
    resized_valid = weights >= 0.05
    prompt[resized_valid] = weighted_depth[resized_valid] / weights[resized_valid]

    missing = (~resized_valid).astype(np.uint8)
    if np.any(missing):
        prompt = cv2.inpaint(prompt, missing, 3.0, cv2.INPAINT_NS)
    prompt = np.nan_to_num(prompt, nan=low, posinf=high, neginf=low)
    prompt = np.clip(prompt, low, high).astype(np.float32, copy=False)
    return prompt, {
        "raw_valid_fraction": float(np.mean(valid)),
        "prompt_min_m": float(np.min(prompt)),
        "prompt_max_m": float(np.max(prompt)),
        "raw_percentile_01_m": low,
        "raw_percentile_99_m": high,
    }


def has_usable_depth(depth_m, min_depth_m=0.05, max_depth_m=5.0):
    depth = np.asarray(depth_m)
    valid = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    if np.count_nonzero(valid) < 16:
        return False
    values = depth[valid].astype(np.float32, copy=False)
    low, high = np.percentile(values, [1.0, 99.0])
    if high - low >= 1e-4:
        return True
    return bool(float(np.max(values)) - float(np.min(values)) >= 1e-4)


DEPTH_COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def depth_display_bounds(
    depth_m,
    min_depth_m=0.05,
    max_depth_m=3.0,
    low_percentile=2.0,
    high_percentile=98.0,
):
    """Return robust visualization bounds without treating invalid depth as zero."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    if np.count_nonzero(valid) < 16:
        return float(min_depth_m), float(max_depth_m)
    low, high = np.percentile(
        depth[valid],
        [float(low_percentile), float(high_percentile)],
    )
    low = max(float(low), float(min_depth_m))
    high = min(float(high), float(max_depth_m))
    if high - low < 0.01:
        center = (low + high) / 2.0
        low = max(float(min_depth_m), center - 0.005)
        high = min(float(max_depth_m), center + 0.005)
    return low, high


def colorize_depth(
    depth_m,
    min_depth_m=0.05,
    max_depth_m=3.0,
    colormap="viridis",
):
    """Render metric depth with a fixed scale; invalid pixels stay black."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be greater than min_depth_m")
    normalized[valid] = np.clip(
        255.0
        * (depth[valid] - float(min_depth_m))
        / (float(max_depth_m) - float(min_depth_m)),
        0.0,
        255.0,
    ).astype(np.uint8)
    if colormap not in DEPTH_COLORMAPS:
        raise ValueError(f"unknown depth colormap: {colormap}")
    rendered = cv2.applyColorMap(255 - normalized, DEPTH_COLORMAPS[colormap])
    rendered[~valid] = 0
    return rendered


class PromptDepthAnythingEstimator:
    """Lazy PromptDA inference wrapper that accepts RGB uint8 and depth meters."""

    def __init__(
        self,
        model="vits",
        device="cuda",
        max_size=448,
        prompt_width=256,
        prompt_height=192,
        min_depth_m=0.05,
        max_depth_m=5.0,
        use_amp=True,
    ):
        if model not in MODEL_IDS:
            raise ValueError(f"unknown PromptDA model: {model}")
        self.model_name = model
        self.model_id = MODEL_IDS[model]
        self.device_name = device
        self.max_size = int(max_size)
        self.prompt_width = int(prompt_width)
        self.prompt_height = int(prompt_height)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.use_amp = bool(use_amp)
        self._torch = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        promptda_root = str(_promptda_root())
        if promptda_root not in sys.path:
            sys.path.insert(0, promptda_root)

        import torch
        from promptda.promptda import PromptDA

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("PromptDA requested CUDA, but CUDA is unavailable")
        self._torch = torch
        model_kwargs = {"encoder": self.model_name.split("-")[0]}
        try:
            self._model = PromptDA.from_pretrained(
                self.model_id,
                model_kwargs=model_kwargs,
                local_files_only=True,
            )
        except Exception:
            self._model = PromptDA.from_pretrained(
                self.model_id,
                model_kwargs=model_kwargs,
            )
        self._model.eval().to(self.device_name)

    def enhance(self, rgb, depth_m, prompt_depth_m=None):
        rgb = np.asarray(rgb)
        depth = np.asarray(depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB must be HxWx3, got {rgb.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth shapes do not match: {rgb.shape[:2]} vs {depth.shape}"
            )

        prompt_source = depth if prompt_depth_m is None else prompt_depth_m
        prompt, stats = prepare_prompt_depth(
            prompt_source,
            output_width=self.prompt_width,
            output_height=self.prompt_height,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )
        input_height, input_width = _inference_shape(
            rgb.shape[0],
            rgb.shape[1],
            self.max_size,
        )
        resized_rgb = cv2.resize(
            rgb,
            (input_width, input_height),
            interpolation=(
                cv2.INTER_AREA if input_width < rgb.shape[1] else cv2.INTER_CUBIC
            ),
        )

        with self._lock:
            self._load()
            torch = self._torch
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(resized_rgb.transpose(2, 0, 1))
            ).unsqueeze(0).to(self.device_name, dtype=torch.float32) / 255.0
            prompt_tensor = torch.from_numpy(prompt).unsqueeze(0).unsqueeze(0).to(
                self.device_name,
                dtype=torch.float32,
            )
            amp_enabled = self.use_amp and self.device_name.startswith("cuda")
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = self._model.predict(image_tensor, prompt_tensor)
            if self.device_name.startswith("cuda"):
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - started) * 1000.0
            enhanced = prediction.squeeze().float().cpu().numpy()

        enhanced = cv2.resize(
            enhanced,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced = np.nan_to_num(
            enhanced,
            nan=stats["prompt_min_m"],
            posinf=stats["prompt_max_m"],
            neginf=stats["prompt_min_m"],
        )
        enhanced = np.clip(
            enhanced,
            stats["prompt_min_m"],
            stats["prompt_max_m"],
        ).astype(np.float32, copy=False)
        stats.update(
            {
                "inference_ms": float(inference_ms),
                "input_height": input_height,
                "input_width": input_width,
                "model": self.model_name,
                "used_external_prompt": prompt_depth_m is not None,
            }
        )
        return enhanced, stats


class PromptDepthWorker:
    """Latest-frame worker for non-blocking dual-camera live preview."""

    def __init__(self, estimator, cameras=("wrist", "front")):
        self.estimator = estimator
        self.cameras = tuple(cameras)
        self._condition = threading.Condition()
        self._pending = None
        self._latest = None
        self._running = False
        self._thread = None
        self._last_token = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="prompt_depth_anything",
            daemon=True,
        )
        self._thread.start()

    def submit(self, camera_sample):
        if camera_sample is None:
            return
        token = camera_sample.get("camera_capture_wall_time_ns", id(camera_sample))
        if token == self._last_token:
            return
        self._last_token = token
        copied = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in camera_sample.items()
        }
        with self._condition:
            self._pending = copied
            self._condition.notify()

    def latest(self):
        with self._condition:
            return self._latest

    def stop(self):
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self):
        camera_fields = {
            "wrist": ("color_image1", "depth_image1"),
            "front": ("color_image2", "depth_image2"),
        }
        last_usable_depth = {}
        while True:
            with self._condition:
                while self._running and self._pending is None:
                    self._condition.wait()
                if not self._running:
                    return
                sample = self._pending
                self._pending = None

            result = {
                "camera_sample": sample,
                "depths": {},
                "stats": {},
                "fallbacks": {},
            }
            errors = []
            for camera_name in self.cameras:
                color_key, depth_key = camera_fields[camera_name]
                current_depth = sample[depth_key]
                prompt_depth = None
                if has_usable_depth(
                    current_depth,
                    self.estimator.min_depth_m,
                    self.estimator.max_depth_m,
                ):
                    last_usable_depth[camera_name] = current_depth
                else:
                    prompt_depth = last_usable_depth.get(camera_name)
                    result["fallbacks"][camera_name] = prompt_depth is not None
                try:
                    enhanced, stats = self.estimator.enhance(
                        sample[color_key],
                        current_depth,
                        prompt_depth_m=prompt_depth,
                    )
                    result["depths"][depth_key] = enhanced
                    result["stats"][camera_name] = stats
                except Exception as exc:
                    errors.append(f"{camera_name}: {type(exc).__name__}: {exc}")
            if errors:
                result["error"] = "; ".join(errors)
            with self._condition:
                self._latest = result
