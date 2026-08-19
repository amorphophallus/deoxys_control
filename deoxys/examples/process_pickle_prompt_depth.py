"""Enhance RealSense depth in FurnitureBench-compatible pickle episodes."""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from deoxys.utils.prompt_depth_anything import (
    MODEL_IDS,
    PromptDepthAnythingEstimator,
    colorize_depth,
    depth_display_bounds,
    has_usable_depth,
)
from deoxys.utils.video_utils import H264VideoWriter


CAMERAS = {
    "wrist": ("color_image1", "depth_image1"),
    "front": ("color_image2", "depth_image2"),
}


class MetricAccumulator:
    def __init__(self):
        self.frames = []

    def add(self, raw_depth, enhanced_depth, min_depth_m, max_depth_m):
        raw = np.asarray(raw_depth, dtype=np.float32)
        enhanced = np.asarray(enhanced_depth, dtype=np.float32)
        raw_valid = (
            np.isfinite(raw)
            & (raw >= float(min_depth_m))
            & (raw <= float(max_depth_m))
        )
        enhanced_valid = (
            np.isfinite(enhanced)
            & (enhanced >= float(min_depth_m))
            & (enhanced <= float(max_depth_m))
        )
        common = raw_valid & enhanced_valid
        values = {
            "raw_valid_fraction": float(np.mean(raw_valid)),
            "enhanced_valid_fraction": float(np.mean(enhanced_valid)),
            "hole_fill_fraction": float(np.mean((~raw_valid) & enhanced_valid)),
        }
        if np.any(common):
            error = np.abs(enhanced[common] - raw[common])
            values.update(
                {
                    "raw_anchor_mae_m": float(np.mean(error)),
                    "raw_anchor_rmse_m": float(np.sqrt(np.mean(error**2))),
                    "raw_anchor_abs_rel": float(
                        np.mean(error / np.maximum(raw[common], 1e-6))
                    ),
                }
            )
        self.frames.append(values)

    def summary(self):
        if not self.frames:
            return {"frame_count": 0}
        keys = sorted({key for frame in self.frames for key in frame})
        result = {"frame_count": len(self.frames)}
        for key in keys:
            values = [frame[key] for frame in self.frames if key in frame]
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"p05_{key}"] = float(np.percentile(values, 5.0))
            result[f"p95_{key}"] = float(np.percentile(values, 95.0))
        return result


def _label(panel, text):
    cv2.putText(
        panel,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _comparison_frame(
    observation,
    enhanced_depths,
    display_bounds,
    colormap,
):
    rows = []
    for camera_name in ("wrist", "front"):
        color_key, depth_key = CAMERAS[camera_name]
        display_min_m, display_max_m = display_bounds[camera_name]
        rgb = cv2.cvtColor(observation[color_key], cv2.COLOR_RGB2BGR)
        raw = colorize_depth(
            observation[depth_key],
            display_min_m,
            display_max_m,
            colormap,
        )
        enhanced = colorize_depth(
            enhanced_depths.get(depth_key, observation[depth_key]),
            display_min_m,
            display_max_m,
            colormap,
        )
        rows.append(
            cv2.hconcat(
                [
                    _label(rgb, f"{camera_name.upper()} RGB"),
                    _label(
                        raw,
                        f"RealSense {display_min_m:.2f}-{display_max_m:.2f}m",
                    ),
                    _label(
                        enhanced,
                        f"PromptDA {display_min_m:.2f}-{display_max_m:.2f}m",
                    ),
                ]
            )
        )
    return cv2.vconcat(rows)


def _open_video(path, frame, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = H264VideoWriter(path, fps, frame.shape)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer


def _output_path(input_path, output_dir, model, max_size):
    directory = Path(output_dir).expanduser() if output_dir else input_path.parent
    resolution_suffix = "" if int(max_size) == 448 else f"_{int(max_size)}"
    return directory / (
        f"{input_path.stem}_promptda_{model}{resolution_suffix}.pkl"
    )


def _selected_indices(observation_count, preview_only, preview_count):
    if not preview_only:
        return list(range(observation_count))
    count = min(int(preview_count), observation_count)
    return np.unique(
        np.linspace(0, observation_count - 1, count, dtype=np.int64)
    ).tolist()


def _usable_depth_indices(observations, depth_key, min_depth_m, max_depth_m):
    indices = [
        index
        for index, observation in enumerate(observations)
        if has_usable_depth(
            observation[depth_key],
            min_depth_m,
            max_depth_m,
        )
    ]
    if not indices:
        raise ValueError(f"no frame has usable depth for {depth_key}")
    return np.asarray(indices, dtype=np.int64)


def _nearest_index(sorted_indices, target):
    position = int(np.searchsorted(sorted_indices, target))
    candidates = []
    if position < len(sorted_indices):
        candidates.append(int(sorted_indices[position]))
    if position > 0:
        candidates.append(int(sorted_indices[position - 1]))
    return min(candidates, key=lambda index: abs(index - target))


def _episode_display_bounds(
    observations,
    selected_indices,
    camera_names,
    min_depth_m,
    max_depth_m,
):
    bounds = {}
    for camera_name in camera_names:
        _, depth_key = CAMERAS[camera_name]
        samples = []
        for observation_index in selected_indices:
            depth = np.asarray(
                observations[observation_index][depth_key],
                dtype=np.float32,
            )[::4, ::4]
            valid = (
                np.isfinite(depth)
                & (depth >= float(min_depth_m))
                & (depth <= float(max_depth_m))
            )
            if np.any(valid):
                samples.append(depth[valid])
        if samples:
            bounds[camera_name] = depth_display_bounds(
                np.concatenate(samples),
                min_depth_m,
                max_depth_m,
            )
        else:
            bounds[camera_name] = (float(min_depth_m), float(max_depth_m))
    return bounds


def process_episode(input_path, args, estimator):
    input_path = Path(input_path).expanduser().resolve()
    output_path = _output_path(
        input_path,
        args.output_dir,
        args.model,
        args.max_size,
    ).resolve()
    if args.preview_only:
        preview_path = output_path.with_name(output_path.stem + "_preview.mp4")
        metrics_path = output_path.with_name(
            output_path.stem + "_preview.metrics.json"
        )
    else:
        preview_path = output_path.with_name(output_path.stem + "_comparison.mp4")
        metrics_path = output_path.with_suffix(".metrics.json")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not args.preview_only and output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output_path}")

    print(f"Loading {input_path}", flush=True)
    with input_path.open("rb") as input_file:
        payload = pickle.load(input_file)
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("pickle has no non-empty observations list")

    camera_names = (
        ("wrist", "front") if args.cameras == "both" else (args.cameras,)
    )
    selected = _selected_indices(
        len(observations),
        args.preview_only,
        args.preview_count,
    )
    accumulators = {name: MetricAccumulator() for name in camera_names}
    usable_indices = {
        name: _usable_depth_indices(
            observations,
            CAMERAS[name][1],
            args.min_depth_m,
            args.max_depth_m,
        )
        for name in camera_names
    }
    fallback_offsets = {name: [] for name in camera_names}
    display_bounds = _episode_display_bounds(
        observations,
        selected,
        ("wrist", "front"),
        args.min_depth_m,
        args.display_max_m,
    )
    video_writer = None
    started = time.perf_counter()
    try:
        for completed, observation_index in enumerate(selected, start=1):
            observation = observations[observation_index]
            enhanced_depths = {}
            for camera_name in camera_names:
                color_key, depth_key = CAMERAS[camera_name]
                prompt_depth = None
                if not has_usable_depth(
                    observation[depth_key],
                    args.min_depth_m,
                    args.max_depth_m,
                ):
                    prompt_index = _nearest_index(
                        usable_indices[camera_name],
                        observation_index,
                    )
                    prompt_depth = observations[prompt_index][depth_key]
                    fallback_offsets[camera_name].append(
                        abs(prompt_index - observation_index)
                    )
                enhanced, _ = estimator.enhance(
                    observation[color_key],
                    observation[depth_key],
                    prompt_depth_m=prompt_depth,
                )
                enhanced_depths[depth_key] = enhanced
                accumulators[camera_name].add(
                    observation[depth_key],
                    enhanced,
                    args.min_depth_m,
                    args.max_depth_m,
                )

            if args.comparison_video or args.preview_only:
                frame = _comparison_frame(
                    observation,
                    enhanced_depths,
                    display_bounds,
                    args.depth_colormap,
                )
                if video_writer is None:
                    video_writer = _open_video(
                        preview_path,
                        frame,
                        args.preview_fps if args.preview_only else args.video_fps,
                    )
                video_writer.write(frame)

            if not args.preview_only:
                for camera_name in camera_names:
                    _, depth_key = CAMERAS[camera_name]
                    raw_depth = observation[depth_key]
                    if not args.drop_original_depth:
                        observation[f"{depth_key}_realsense"] = raw_depth
                    observation[depth_key] = enhanced_depths[depth_key].astype(
                        np.float16
                    )

            if completed == 1 or completed % 25 == 0 or completed == len(selected):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-6)
                remaining = (len(selected) - completed) / max(rate, 1e-6)
                print(
                    f"{input_path.name}: {completed}/{len(selected)} frames, "
                    f"{rate:.2f} frame/s, ETA {remaining:.0f}s",
                    flush=True,
                )
    finally:
        if video_writer is not None:
            video_writer.release()

    report = {
        "input": str(input_path),
        "output": None if args.preview_only else str(output_path),
        "comparison_video": (
            str(preview_path)
            if args.comparison_video or args.preview_only
            else None
        ),
        "model": args.model,
        "model_id": MODEL_IDS[args.model],
        "max_size": args.max_size,
        "prompt_size": [args.prompt_width, args.prompt_height],
        "depth_range_m": [args.min_depth_m, args.max_depth_m],
        "visualization": {
            "colormap": args.depth_colormap,
            "camera_ranges_m": display_bounds,
        },
        "processed_observations": len(selected),
        "total_observations": len(observations),
        "metrics_are_no_ground_truth_proxies": True,
        "camera_metrics": {
            name: {
                **accumulator.summary(),
                "temporal_prompt_fallback_frames": len(fallback_offsets[name]),
                "mean_temporal_prompt_offset_frames": (
                    float(np.mean(fallback_offsets[name]))
                    if fallback_offsets[name]
                    else 0.0
                ),
                "max_temporal_prompt_offset_frames": (
                    int(np.max(fallback_offsets[name]))
                    if fallback_offsets[name]
                    else 0
                ),
            }
            for name, accumulator in accumulators.items()
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(report, metrics_file, indent=2, ensure_ascii=False)

    if not args.preview_only:
        metadata = payload.setdefault("metadata", {})
        metadata["prompt_depth_anything"] = {
            "model": args.model,
            "model_id": MODEL_IDS[args.model],
            "max_size": args.max_size,
            "prompt_width": args.prompt_width,
            "prompt_height": args.prompt_height,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "canonical_depth_fields": "prompt_depth_anything",
            "original_depth_fields": (
                None if args.drop_original_depth else "depth_image{1,2}_realsense"
            ),
            "source_pickle": str(input_path),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("wb") as output_file:
            pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_paths", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--model",
        choices=("vits", "vitl", "vits-transparent"),
        default="vits",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-size", type=int, default=448)
    parser.add_argument("--prompt-width", type=int, default=256)
    parser.add_argument("--prompt-height", type=int, default=192)
    parser.add_argument("--cameras", choices=("both", "front", "wrist"), default="both")
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--display-max-m", type=float, default=3.0)
    parser.add_argument(
        "--depth-colormap",
        choices=("viridis", "turbo", "inferno", "jet"),
        default="viridis",
    )
    parser.add_argument("--comparison-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--preview-fps", type=float, default=3.0)
    parser.add_argument("--drop-original-depth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    estimator = PromptDepthAnythingEstimator(
        model=args.model,
        device=args.device,
        max_size=args.max_size,
        prompt_width=args.prompt_width,
        prompt_height=args.prompt_height,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    for pickle_path in args.pickle_paths:
        process_episode(pickle_path, args, estimator)


if __name__ == "__main__":
    main()
