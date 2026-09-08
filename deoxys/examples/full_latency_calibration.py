#!/usr/bin/env python3
"""Calibrate every timing quantity used by Deoxys collection and evaluation.

The default ``--component all`` run measures both RealSense observation paths,
robot and gripper proprioception paths, arm and gripper execution latency, and
clock offset/drift diagnostics. Action latency is the command-to-observed motion
interval minus the matching observation latency, following UMI's definition.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


SHARED_CAMERA_CLOCK_DOMAINS = ("global_time", "system_time")
PING_RTT_RE = re.compile(r"time[=<]([0-9]+(?:\.[0-9]+)?)\s*ms")


def describe_ms(values_ms: Sequence[float], *, include_samples: bool = True) -> dict:
    if not values_ms:
        raise ValueError("at least one sample is required")
    values = np.asarray(values_ms, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    result = {
        "num_samples": int(values.size),
        "min_ms": round(float(np.min(values)), 6),
        "median_ms": round(float(np.median(values)), 6),
        "mean_ms": round(float(np.mean(values)), 6),
        "p95_ms": round(float(np.percentile(values, 95)), 6),
        "max_ms": round(float(np.max(values)), 6),
    }
    if include_samples:
        result["samples_ms"] = [round(float(value), 6) for value in values]
    return result


def summarize_latency_ms(values_ms: Sequence[float], recommendation: str) -> dict:
    result = describe_ms(values_ms)
    if result["min_ms"] < 0:
        raise ValueError("latency samples must be non-negative")
    if recommendation not in {"median", "p95"}:
        raise ValueError("recommendation must be 'median' or 'p95'")
    result["recommended_statistic"] = recommendation
    result["recommended_action_latency_ms"] = round(
        float(result[f"{recommendation}_ms"]), 3
    )
    return result


def summarize_observation_latency_ms(
    values_ms: Sequence[float], recommendation: str
) -> dict:
    result = describe_ms(values_ms)
    if result["min_ms"] < 0:
        raise ValueError(
            "observation latency is negative; the source clock is not synchronized "
            "to the host wall clock"
        )
    if recommendation not in {"median", "p95"}:
        raise ValueError("recommendation must be 'median' or 'p95'")
    result["recommended_statistic"] = recommendation
    result["recommended_observation_latency_ms"] = round(
        float(result[f"{recommendation}_ms"]), 3
    )
    return result


def fit_clock_mapping(source_ms: Sequence[float], receive_ms: Sequence[float]) -> dict:
    """Fit host_ms = slope * source_ms + offset_ms with centered arithmetic."""
    source = np.asarray(source_ms, dtype=np.float64)
    receive = np.asarray(receive_ms, dtype=np.float64)
    if source.shape != receive.shape or source.ndim != 1 or source.size < 2:
        raise ValueError("clock fit requires equal one-dimensional arrays with >=2 samples")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(receive)):
        raise ValueError("clock samples must be finite")
    slope, centered_offset = np.polyfit(
        source - source[0], receive - receive[0], 1
    )
    host_epoch_offset_ms = receive[0] + centered_offset - slope * source[0]
    residual = receive - (slope * source + host_epoch_offset_ms)
    return {
        "mapping": "host_wall_ms = slope * source_ms + host_epoch_offset_ms",
        "slope": float(slope),
        "drift_ppm": round(float((slope - 1.0) * 1e6), 6),
        "host_epoch_offset_ms": round(float(host_epoch_offset_ms), 6),
        "source_span_s": round(float((source[-1] - source[0]) / 1000.0), 6),
        "residual": describe_ms(residual.tolist()),
        "absolute_residual": describe_ms(np.abs(residual).tolist()),
    }


def parse_ping_rtts_ms(output: str) -> list[float]:
    return [float(match.group(1)) for match in PING_RTT_RE.finditer(output)]


def measure_ping_rtt_ms(
    host: str, *, samples: int, interval_s: float, timeout_s: float
) -> tuple[list[float], dict]:
    command = [
        "ping", "-n", "-c", str(samples), "-i", str(interval_s),
        "-W", str(max(1, int(math.ceil(timeout_s)))), host,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    rtts = parse_ping_rtts_ms(result.stdout + "\n" + result.stderr)
    if not rtts:
        raise RuntimeError(
            f"no ICMP replies from NUC {host}; command exited {result.returncode}"
        )
    if len(rtts) < max(2, samples // 2):
        raise RuntimeError(
            f"only {len(rtts)}/{samples} ICMP replies from NUC {host}; "
            "observation latency estimate is unreliable"
        )
    return rtts, {
        "host": host,
        "requested_samples": samples,
        "received_samples": len(rtts),
        "packet_loss_fraction": round(1.0 - len(rtts) / samples, 6),
        "rtt": describe_ms(rtts),
    }


def detect_motion_onset_ns(
    samples: Iterable[tuple[int, object]], *, command_wall_time_ns: int,
    baseline: object, threshold: float, confirm_samples: int,
) -> int | None:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if confirm_samples < 1:
        raise ValueError("confirm_samples must be at least 1")
    baseline_array = np.asarray(baseline, dtype=np.float64).reshape(-1)
    run_start_ns = None
    run_length = 0
    for timestamp_ns, value in samples:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns < int(command_wall_time_ns):
            continue
        value_array = np.asarray(value, dtype=np.float64).reshape(-1)
        if value_array.shape != baseline_array.shape:
            raise ValueError(
                f"sample shape {value_array.shape} != baseline shape {baseline_array.shape}"
            )
        if float(np.linalg.norm(value_array - baseline_array)) >= threshold:
            if run_length == 0:
                run_start_ns = timestamp_ns
            run_length += 1
            if run_length >= confirm_samples:
                return int(run_start_ns)
        else:
            run_start_ns = None
            run_length = 0
    return None


def corrected_execution_latency_ms(
    end_to_end_ms: float, observation_latency_ms: float
) -> float:
    value = float(end_to_end_ms) - float(observation_latency_ms)
    if value < 0:
        raise ValueError(
            f"end-to-end latency {end_to_end_ms:.3f} ms is smaller than observation "
            f"latency {observation_latency_ms:.3f} ms"
        )
    return value


def build_updated_latency_profile(
    base_profile: Mapping[str, object], calibration: Mapping[str, object], *,
    measured_at: str,
) -> dict:
    """Replace every successfully calibrated field and preserve the remainder."""
    profile = dict(base_profile)
    field_sources = (
        ("front_observation", "front_observation_ms", "recommended_observation_latency_ms"),
        ("wrist_observation", "wrist_observation_ms", "recommended_observation_latency_ms"),
        ("robot_observation", "robot_observation_ms", "recommended_observation_latency_ms"),
        ("gripper_observation", "gripper_observation_ms", "recommended_observation_latency_ms"),
        ("arm", "robot_action_ms", "recommended_action_latency_ms"),
        ("gripper", "gripper_action_ms", "recommended_action_latency_ms"),
    )
    calibrated_fields = []
    for result_key, profile_key, value_key in field_sources:
        result = calibration.get(result_key)
        if result is not None:
            profile[profile_key] = float(result[value_key])
            calibrated_fields.append(profile_key)
    if not calibrated_fields:
        raise ValueError("calibration contains no profile results")
    all_fields = [entry[1] for entry in field_sources] + ["action_stale_guard_ms"]
    profile["schema_version"] = max(int(profile.get("schema_version", 1)), 2)
    profile["measured_at"] = measured_at
    profile["latency_source"] = "measured"
    profile["basis"] = (
        "RealSense sensor-to-local-receive latency on a shared wall clock; "
        "robot/gripper observation latency estimated as half NUC ICMP RTT; "
        "execution latency is publish-to-observed-onset minus observation latency."
    )
    profile["calibrated_fields"] = calibrated_fields
    profile["inherited_fields"] = [
        key for key in all_fields
        if key in base_profile and key not in calibrated_fields
    ]
    profile["calibration_clock"] = "deoxys_host_time.time_ns"
    return profile


def _camera_clock_result(records: list[dict], camera_name: str) -> tuple[dict, dict]:
    source_ms = [float(record["sensor_timestamp_ms"]) for record in records]
    receive_ms = [float(record["receive_wall_time_ns"]) / 1e6 for record in records]
    domains = sorted({str(record["timestamp_domain"]).lower() for record in records})
    if not domains or any(
        not any(shared in domain for shared in SHARED_CAMERA_CLOCK_DOMAINS)
        for domain in domains
    ):
        raise RuntimeError(
            f"{camera_name} timestamp domain {domains!r} is not a shared wall clock"
        )
    latencies = [receive - source for source, receive in zip(source_ms, receive_ms)]
    observation = {
        **summarize_observation_latency_ms(latencies, records[0]["recommendation"]),
        "method": "sensor_timestamp_to_local_receive",
        "timestamp_domains": domains,
        "receive_definition": (
            "time.time_ns after frame retrieval, depth alignment, and array copy"
        ),
    }
    clock = fit_clock_mapping(source_ms, receive_ms)
    clock.update({
        "source": f"{camera_name}_realsense_sensor_timestamp",
        "timestamp_domains": domains,
        "apparent_offset_including_pipeline": describe_ms(latencies),
    })
    return observation, clock


def calibrate_cameras(args) -> tuple[dict, dict, dict]:
    from deoxys.utils.furniture_bench_utils import RealSenseCamera

    front = RealSenseCamera(
        args.front_camera_serial, args.front_color_width, args.front_color_height,
        args.front_color_fps, enable_depth=not args.disable_camera_depth,
        depth_width=args.front_depth_width, depth_height=args.front_depth_height,
        depth_fps=args.front_depth_fps,
    )
    wrist = RealSenseCamera(
        args.wrist_camera_serial, args.wrist_color_width, args.wrist_color_height,
        args.wrist_color_fps, enable_depth=not args.disable_camera_depth,
        depth_width=args.wrist_depth_width, depth_height=args.wrist_depth_height,
        depth_fps=args.wrist_depth_fps,
    )
    records = {"front": [], "wrist": []}
    try:
        front.start()
        wrist.start()
        if not front.global_time_enabled or not wrist.global_time_enabled:
            raise RuntimeError(
                "both RealSense devices must support and enable global_time_enabled"
            )
        total = args.camera_warmup_frames + args.camera_samples
        for index in range(total):
            front_frame = front.read(timeout_ms=args.camera_timeout_ms)
            wrist_frame = wrist.read(timeout_ms=args.camera_timeout_ms)
            if front_frame is None or wrist_frame is None:
                raise TimeoutError("timed out waiting for a RealSense frame")
            if index < args.camera_warmup_frames:
                continue
            for name, frame in (("front", front_frame), ("wrist", wrist_frame)):
                records[name].append({
                    "frame_number": int(frame["frame_number"]),
                    "sensor_timestamp_ms": float(frame["sensor_timestamp_ms"]),
                    "receive_wall_time_ns": int(frame["wall_time_ns"]),
                    "timestamp_domain": str(frame["timestamp_domain"]),
                    "recommendation": args.recommendation,
                })
            completed = index - args.camera_warmup_frames + 1
            if completed == 1 or completed % 50 == 0 or completed == args.camera_samples:
                print(f"camera samples: {completed}/{args.camera_samples}")
    finally:
        wrist.stop()
        front.stop()

    front_observation, front_clock = _camera_clock_result(records["front"], "front")
    wrist_observation, wrist_clock = _camera_clock_result(records["wrist"], "wrist")
    signed_skew = [
        wrist_record["sensor_timestamp_ms"] - front_record["sensor_timestamp_ms"]
        for front_record, wrist_record in zip(records["front"], records["wrist"])
    ]
    return front_observation, wrist_observation, {
        "clock": {
            "front": front_clock,
            "wrist": wrist_clock,
            "front_wrist_signed_skew": describe_ms(signed_skew),
            "front_wrist_absolute_skew": describe_ms(np.abs(signed_skew).tolist()),
            "pairing_method": "one front read followed by one wrist read",
        },
        "samples": records,
    }


def _duration_ms(message) -> float:
    seconds = float(getattr(message.time, "toSec", 0.0))
    if seconds > 0:
        return seconds * 1000.0
    milliseconds = float(getattr(message.time, "toMSec", 0.0))
    if milliseconds > 0:
        return milliseconds
    raise ValueError("state message has no positive device timestamp")


def _state_clock_diagnostics(records: list[dict], stream_name: str) -> dict:
    if len(records) < 2:
        raise RuntimeError(f"{stream_name} state clock needs at least two samples")
    source_ms = [_duration_ms(record["message"]) for record in records]
    receive_ms = [record["receive_wall_time_ns"] / 1e6 for record in records]
    source_delta = np.diff(source_ms)
    receive_delta = np.diff(receive_ms)
    if np.any(source_delta <= 0):
        raise RuntimeError(f"{stream_name} device timestamps are not strictly increasing")
    result = fit_clock_mapping(source_ms, receive_ms)
    result.update({
        "source": f"{stream_name}_protobuf_uptime",
        "warning": (
            "host_epoch_offset includes the unknown device boot epoch and cannot "
            "by itself identify one-way observation latency"
        ),
        "source_sample_period": describe_ms(source_delta.tolist()),
        "receive_sample_period": describe_ms(receive_delta.tolist()),
        "samples": [
            {
                "device_timestamp_ms": round(source, 6),
                "receive_wall_time_ns": int(record["receive_wall_time_ns"]),
            }
            for source, record in zip(source_ms, records)
        ],
    })
    return result


def calibrate_proprioception(robot, nuc_ip: str, args) -> tuple[dict, dict, dict]:
    start_ns = time.time_ns()
    rtts, ping = measure_ping_rtt_ms(
        nuc_ip, samples=args.ping_samples, interval_s=args.ping_interval_s,
        timeout_s=args.ping_timeout_s,
    )
    remaining_s = args.state_sample_s - (time.time_ns() - start_ns) / 1e9
    if remaining_s > 0:
        time.sleep(remaining_s)
    robot_records = [
        record for record in robot.timestamped_robot_state_buffer()
        if int(record["receive_wall_time_ns"]) >= start_ns
    ]
    gripper_records = [
        record for record in robot.timestamped_gripper_state_buffer()
        if int(record["receive_wall_time_ns"]) >= start_ns
    ]
    half_rtts = [value / 2.0 for value in rtts]
    base = summarize_observation_latency_ms(half_rtts, args.recommendation)
    limitation = "does not separately identify NUC publisher processing delay"
    return (
        {**base, "method": "half_icmp_rtt_to_nuc", "stream": "robot", "limitation": limitation},
        {**base, "method": "half_icmp_rtt_to_nuc", "stream": "gripper", "limitation": limitation},
        {
            "ping": ping,
            "robot_clock": _state_clock_diagnostics(robot_records, "robot"),
            "gripper_clock": _state_clock_diagnostics(gripper_records, "gripper"),
        },
    )


def _message_arm_position(message) -> np.ndarray:
    return _message_arm_pose(message)[:3, 3].copy()


def _message_arm_pose(message) -> np.ndarray:
    return np.asarray(message.O_T_EE, dtype=np.float64).reshape(4, 4).T.copy()


def _message_gripper_width(message) -> np.ndarray:
    return np.asarray(message.width, dtype=np.float64).reshape(-1)


def _latest_value(records: list[dict], extractor: Callable) -> np.ndarray:
    if not records:
        raise RuntimeError("state stream is empty")
    return np.asarray(extractor(records[-1]["message"]), dtype=np.float64)


def _wait_for_streams(robot, *, arm: bool, gripper: bool, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        arm_ready = not arm or bool(robot.timestamped_robot_state_buffer(max_records=1))
        gripper_ready = not gripper or bool(robot.timestamped_gripper_state_buffer(max_records=1))
        if arm_ready and gripper_ready:
            return
        time.sleep(0.05)
    missing = []
    if arm and not robot.timestamped_robot_state_buffer(max_records=1):
        missing.append("robot")
    if gripper and not robot.timestamped_gripper_state_buffer(max_records=1):
        missing.append("gripper")
    raise TimeoutError(f"timed out waiting for state stream(s): {', '.join(missing)}")


def _wait_for_motion(
    get_records: Callable[[], list[dict]], extractor: Callable, *,
    command_wall_time_ns: int, baseline: np.ndarray, threshold: float,
    confirm_samples: int, timeout_s: float, diagnostic_name: Optional[str] = None,
) -> int:
    started_monotonic = time.monotonic()
    deadline = started_monotonic + timeout_s
    next_diagnostic = started_monotonic
    latest_delta = float("nan")
    max_delta = 0.0
    latest_receive_ns = None
    sample_count = 0
    while time.monotonic() < deadline:
        samples = [
            (int(record["receive_wall_time_ns"]), extractor(record["message"]))
            for record in get_records()
            if int(record["receive_wall_time_ns"]) >= command_wall_time_ns
        ]
        sample_count = len(samples)
        if samples:
            latest_receive_ns, latest_value = samples[-1]
            latest_delta = float(np.linalg.norm(latest_value - baseline))
            max_delta = max(
                max_delta,
                max(float(np.linalg.norm(value - baseline)) for _, value in samples),
            )
        onset_ns = detect_motion_onset_ns(
            samples, command_wall_time_ns=command_wall_time_ns,
            baseline=baseline, threshold=threshold, confirm_samples=confirm_samples,
        )
        if onset_ns is not None:
            if diagnostic_name is not None:
                print(
                    f"[{diagnostic_name}] onset receive_wall_time_ns={onset_ns} "
                    f"delta={latest_delta:.6f} max_delta={max_delta:.6f}",
                    flush=True,
                )
            return onset_ns
        now = time.monotonic()
        if diagnostic_name is not None and now >= next_diagnostic:
            state_age_ms = (
                float("nan")
                if latest_receive_ns is None
                else (time.time_ns() - latest_receive_ns) / 1e6
            )
            print(
                f"[{diagnostic_name}] waiting elapsed_s={now - started_monotonic:.3f} "
                f"post_command_states={sample_count} "
                f"latest_state_age_ms={state_age_ms:.3f} "
                f"delta={latest_delta:.6f} max_delta={max_delta:.6f} "
                f"threshold={threshold:.6f}",
                flush=True,
            )
            next_diagnostic = now + 0.25
        time.sleep(0.005)
    state_age_ms = (
        float("nan")
        if latest_receive_ns is None
        else (time.time_ns() - latest_receive_ns) / 1e6
    )
    raise TimeoutError(
        f"no persistent state change >= {threshold:g} within {timeout_s:g}s; "
        f"observed {sample_count} post-command states, "
        f"latest delta={latest_delta:.6f}, max delta={max_delta:.6f}, "
        f"latest state age={state_age_ms:.3f}ms"
    )


def _wait_for_arm_motion_with_resend(
    robot, controller_cfg, target_action: np.ndarray, *,
    first_command_wall_time_ns: int, baseline: np.ndarray, threshold: float,
    confirm_samples: int, timeout_s: float, command_frequency_hz: float,
) -> tuple[int, int, list[int]]:
    """Wait for arm onset while mirroring policy eval's 5 Hz target hold."""
    period_s = 1.0 / float(command_frequency_hz)
    started_monotonic = time.monotonic()
    deadline = started_monotonic + timeout_s
    next_resend = started_monotonic + period_s
    next_diagnostic = started_monotonic
    send_times_ns = [int(first_command_wall_time_ns)]
    latest_delta_m = float("nan")
    max_delta_m = 0.0
    latest_receive_ns = None
    post_command_records = 0

    while time.monotonic() < deadline:
        records = robot.timestamped_robot_state_buffer()
        samples = [
            (int(record["receive_wall_time_ns"]), _message_arm_position(record["message"]))
            for record in records
            if int(record["receive_wall_time_ns"]) >= first_command_wall_time_ns
        ]
        post_command_records = len(samples)
        if samples:
            latest_receive_ns, latest_position = samples[-1]
            latest_delta_m = float(np.linalg.norm(latest_position - baseline))
            max_delta_m = max(
                max_delta_m,
                max(float(np.linalg.norm(position - baseline)) for _, position in samples),
            )
        onset_ns = detect_motion_onset_ns(
            samples, command_wall_time_ns=first_command_wall_time_ns,
            baseline=baseline, threshold=threshold,
            confirm_samples=confirm_samples,
        )
        if onset_ns is not None:
            causal_command_ns = max(
                sent_ns for sent_ns in send_times_ns if sent_ns <= onset_ns
            )
            print(
                "[arm] onset "
                f"receive_wall_time_ns={onset_ns} "
                f"causal_send_wall_time_ns={causal_command_ns} "
                f"send_attempt={send_times_ns.index(causal_command_ns) + 1} "
                f"delta_m={latest_delta_m:.6f}",
                flush=True,
            )
            return onset_ns, causal_command_ns, send_times_ns

        now = time.monotonic()
        if now >= next_resend:
            command_result = robot.control(
                controller_type="OSC_POSE", action=target_action,
                controller_cfg=controller_cfg, control_gripper=False,
                enforce_control_frequency=False,
            )
            sent_ns = int(command_result["robot_command_wall_time_ns"])
            send_times_ns.append(sent_ns)
            print(
                f"[arm] target resend attempt={len(send_times_ns)} "
                f"send_wall_time_ns={sent_ns}",
                flush=True,
            )
            next_resend = now + period_s

        if now >= next_diagnostic:
            state_age_ms = (
                float("nan")
                if latest_receive_ns is None
                else (time.time_ns() - latest_receive_ns) / 1e6
            )
            print(
                "[arm] waiting "
                f"elapsed_s={now - started_monotonic:.3f} "
                f"post_command_states={post_command_records} "
                f"latest_state_age_ms={state_age_ms:.3f} "
                f"delta_m={latest_delta_m:.6f} threshold_m={threshold:.6f}",
                flush=True,
            )
            next_diagnostic = now + 0.25
        time.sleep(0.005)

    state_age_ms = (
        float("nan")
        if latest_receive_ns is None
        else (time.time_ns() - latest_receive_ns) / 1e6
    )
    raise TimeoutError(
        f"no persistent arm state change >= {threshold:g} within {timeout_s:g}s; "
        f"sent target {len(send_times_ns)} times, observed {post_command_records} "
        f"post-command states, latest delta={latest_delta_m:.6f}m, "
        f"max delta={max_delta_m:.6f}m, "
        f"latest state age={state_age_ms:.3f}ms"
    )


def _inside_workspace(position: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.all(position >= lower) and np.all(position <= upper))


def _absolute_pose_action(pose: np.ndarray, gripper_sign: float = -1.0) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"expected a 4x4 EE pose, got {pose.shape}")
    return np.r_[
        pose[:3, 3],
        Rotation.from_matrix(pose[:3, :3]).as_rotvec(),
        float(gripper_sign),
    ]


def _configure_eval_absolute_controller(controller_cfg, time_fraction: float):
    """Apply the OSC_POSE settings used by ``src.real.evaluate_policy``."""
    if time_fraction <= 0:
        raise ValueError("controller time fraction must be positive")
    controller_cfg.is_delta = False
    controller_cfg.action_scale.translation = 1.0
    controller_cfg.action_scale.rotation = 1.0
    controller_cfg.traj_interpolator_cfg.traj_interpolator_type = "LINEAR_POSE"
    controller_cfg.traj_interpolator_cfg.time_fraction = float(time_fraction)
    return controller_cfg


def _keep_arm_controller_alive(
    robot, controller_cfg, hold_action: np.ndarray, duration_s: float,
    command_frequency_hz: float,
) -> None:
    """Resend an absolute pose using the same hold path as policy eval."""
    if duration_s <= 0:
        return
    if command_frequency_hz <= 0:
        raise ValueError("arm command frequency must be positive")
    hold_action = np.asarray(hold_action, dtype=np.float64)
    if hold_action.shape != (7,):
        raise ValueError(f"expected a 7D absolute action, got {hold_action.shape}")
    period_s = 1.0 / float(command_frequency_hz)
    deadline = time.monotonic() + float(duration_s)
    while True:
        robot.control(
            controller_type="OSC_POSE",
            action=hold_action,
            controller_cfg=controller_cfg,
            control_gripper=False,
            enforce_control_frequency=False,
        )
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(period_s, remaining_s))


def _terminate_arm_controller(robot, controller_cfg) -> None:
    stop_pose = _latest_value(
        robot.timestamped_robot_state_buffer(max_records=1), _message_arm_pose
    )
    stop_action = _absolute_pose_action(stop_pose)
    result = robot.control(
        controller_type="OSC_POSE", action=stop_action,
        controller_cfg=controller_cfg, termination=True,
        control_gripper=False, enforce_control_frequency=False,
    )
    print(
        "[arm] controller termination sent "
        f"send_wall_time_ns={result['robot_command_wall_time_ns']}",
        flush=True,
    )


def calibrate_arm(
    robot, controller_cfg, args, observation_latency_ms: float
) -> tuple[dict, list[dict]]:
    if bool(controller_cfg.is_delta):
        raise ValueError("arm calibration requires the policy-eval absolute controller")
    current_pose = _latest_value(
        robot.timestamped_robot_state_buffer(max_records=1), _message_arm_pose
    )
    hold_action = _absolute_pose_action(current_pose)
    print(
        "[arm] warmup absolute pose "
        f"xyz={current_pose[:3, 3].tolist()} action={hold_action.tolist()}",
        flush=True,
    )
    warmup_result = robot.control(
        controller_type="OSC_POSE", action=hold_action, controller_cfg=controller_cfg,
        control_gripper=False, enforce_control_frequency=False,
    )
    print(
        "[arm] warmup sent "
        f"send_wall_time_ns={warmup_result['robot_command_wall_time_ns']}",
        flush=True,
    )
    _keep_arm_controller_alive(
        robot, controller_cfg, hold_action, args.settle_s,
        args.arm_command_frequency_hz,
    )
    axis_index = {"x": 0, "y": 1, "z": 2}[args.arm_axis]
    lower = np.asarray(args.workspace_min, dtype=np.float64)
    upper = np.asarray(args.workspace_max, dtype=np.float64)
    trials = []
    for trial_index in range(args.trials):
        signed_step = args.arm_step_m * (1.0 if trial_index % 2 == 0 else -1.0)
        baseline_pose = _latest_value(
            robot.timestamped_robot_state_buffer(max_records=1), _message_arm_pose
        )
        baseline = baseline_pose[:3, 3].copy()
        target_pose = baseline_pose.copy()
        target_pose[axis_index, 3] += signed_step
        target = target_pose[:3, 3]
        if not _inside_workspace(baseline, lower, upper):
            raise RuntimeError(f"current EE position {baseline.tolist()} is outside workspace")
        if not _inside_workspace(target, lower, upper):
            raise RuntimeError(f"arm pulse target {target.tolist()} is outside workspace")
        target_action = _absolute_pose_action(target_pose)
        print(
            f"[arm] trial={trial_index + 1} baseline_xyz={baseline.tolist()} "
            f"target_xyz={target.tolist()} action={target_action.tolist()}",
            flush=True,
        )
        command_result = robot.control(
            controller_type="OSC_POSE",
            action=target_action,
            controller_cfg=controller_cfg, control_gripper=False,
            enforce_control_frequency=False,
        )
        first_command_ns = int(command_result["robot_command_wall_time_ns"])
        print(
            f"[arm] target sent attempt=1 send_wall_time_ns={first_command_ns}",
            flush=True,
        )
        onset_ns, command_ns, send_times_ns = _wait_for_arm_motion_with_resend(
            robot, controller_cfg, target_action,
            first_command_wall_time_ns=first_command_ns, baseline=baseline,
            threshold=args.arm_threshold_m,
            confirm_samples=args.confirm_samples,
            timeout_s=args.motion_timeout_s,
            command_frequency_hz=args.arm_command_frequency_hz,
        )
        end_to_end_ms = (onset_ns - command_ns) / 1e6
        latency_ms = corrected_execution_latency_ms(end_to_end_ms, observation_latency_ms)
        trials.append({
            "trial": trial_index + 1, "axis": args.arm_axis,
            "signed_step_m": signed_step, "baseline_position_m": baseline.tolist(),
            "first_command_wall_time_ns": first_command_ns,
            "command_wall_time_ns": command_ns,
            "target_send_wall_times_ns": send_times_ns,
            "target_send_attempts": len(send_times_ns),
            "onset_receive_wall_time_ns": onset_ns,
            "end_to_end_ms": round(end_to_end_ms, 6),
            "observation_latency_subtracted_ms": round(observation_latency_ms, 6),
            "action_execution_latency_ms": round(latency_ms, 6),
            "latency_ms": round(latency_ms, 6),
        })
        print(
            f"arm trial {trial_index + 1}/{args.trials}: e2e={end_to_end_ms:.3f} ms, "
            f"execution={latency_ms:.3f} ms"
        )
        _keep_arm_controller_alive(
            robot, controller_cfg, target_action, args.settle_s,
            args.arm_command_frequency_hz,
        )
    if args.trials % 2:
        restore_pose = _latest_value(
            robot.timestamped_robot_state_buffer(max_records=1), _message_arm_pose
        )
        restore_pose[axis_index, 3] -= args.arm_step_m
        restore_action = _absolute_pose_action(restore_pose)
        robot.control(
            controller_type="OSC_POSE",
            action=restore_action,
            controller_cfg=controller_cfg, control_gripper=False,
            enforce_control_frequency=False,
        )
        _keep_arm_controller_alive(
            robot, controller_cfg, restore_action, args.settle_s,
            args.arm_command_frequency_hz,
        )
    summary = summarize_latency_ms(
        [trial["action_execution_latency_ms"] for trial in trials], args.recommendation
    )
    summary.update({
        "method": (
            "policy_eval_absolute_pose_publish_to_observed_onset_minus_"
            "robot_observation_latency"
        ),
        "state_signal": "O_T_EE translation", "motion_threshold_m": args.arm_threshold_m,
        "pulse_axis": args.arm_axis, "pulse_step_m": args.arm_step_m,
        "command_frequency_hz": args.arm_command_frequency_hz,
        "controller_time_fraction": args.controller_time_fraction,
        "observation_latency_subtracted_ms": observation_latency_ms,
        "end_to_end": describe_ms([trial["end_to_end_ms"] for trial in trials]),
    })
    return summary, trials


def calibrate_gripper(
    robot, args, observation_latency_ms: float
) -> tuple[dict, list[dict]]:
    initial_width = _latest_value(
        robot.timestamped_gripper_state_buffer(max_records=1), _message_gripper_width
    )
    initial_open = float(initial_width[0]) >= args.gripper_switch_width_m
    trials = []
    for trial_index in range(args.trials):
        baseline = _latest_value(
            robot.timestamped_gripper_state_buffer(max_records=1), _message_gripper_width
        )
        command_close = float(baseline[0]) >= args.gripper_switch_width_m
        command = "close" if command_close else "open"
        send_times_ns = []
        onset_ns = None
        command_ns = None
        last_timeout = None
        for attempt in range(1, args.gripper_command_attempts + 1):
            robot.gripper_control(1.0 if command_close else -1.0)
            command_ns = int(robot.last_gripper_command_wall_time_ns)
            send_times_ns.append(command_ns)
            print(
                f"[gripper trial={trial_index + 1} command={command}] "
                f"send attempt={attempt}/{args.gripper_command_attempts} "
                f"baseline_width_m={float(baseline[0]):.6f} "
                f"send_wall_time_ns={command_ns}",
                flush=True,
            )
            try:
                onset_ns = _wait_for_motion(
                    robot.timestamped_gripper_state_buffer, _message_gripper_width,
                    command_wall_time_ns=command_ns, baseline=baseline,
                    threshold=args.gripper_threshold_m,
                    confirm_samples=args.confirm_samples,
                    timeout_s=args.motion_timeout_s,
                    diagnostic_name=(
                        f"gripper trial={trial_index + 1} command={command} "
                        f"attempt={attempt}"
                    ),
                )
                break
            except TimeoutError as exc:
                last_timeout = exc
                print(
                    f"[gripper trial={trial_index + 1} command={command}] "
                    f"attempt={attempt} timed out: {exc}",
                    flush=True,
                )
        if onset_ns is None or command_ns is None:
            raise TimeoutError(
                f"gripper trial {trial_index + 1} command={command} failed after "
                f"{args.gripper_command_attempts} send attempts"
            ) from last_timeout
        end_to_end_ms = (onset_ns - command_ns) / 1e6
        latency_ms = corrected_execution_latency_ms(end_to_end_ms, observation_latency_ms)
        trials.append({
            "trial": trial_index + 1, "command": command,
            "baseline_width_m": float(baseline[0]), "command_wall_time_ns": command_ns,
            "command_send_wall_times_ns": send_times_ns,
            "command_send_attempts": len(send_times_ns),
            "onset_receive_wall_time_ns": onset_ns,
            "end_to_end_ms": round(end_to_end_ms, 6),
            "observation_latency_subtracted_ms": round(observation_latency_ms, 6),
            "action_execution_latency_ms": round(latency_ms, 6),
            "latency_ms": round(latency_ms, 6),
        })
        print(
            f"gripper trial {trial_index + 1}/{args.trials}: "
            f"e2e={end_to_end_ms:.3f} ms, execution={latency_ms:.3f} ms"
        )
        time.sleep(args.settle_s)
    final_width = _latest_value(
        robot.timestamped_gripper_state_buffer(max_records=1), _message_gripper_width
    )
    if (float(final_width[0]) >= args.gripper_switch_width_m) != initial_open:
        robot.gripper_control(-1.0 if initial_open else 1.0)
        time.sleep(args.settle_s)
    summary = summarize_latency_ms(
        [trial["action_execution_latency_ms"] for trial in trials], args.recommendation
    )
    by_command = {}
    for command in ("open", "close"):
        values = [
            trial["action_execution_latency_ms"] for trial in trials
            if trial["command"] == command
        ]
        if values:
            by_command[command] = summarize_latency_ms(values, args.recommendation)
    summary.update({
        "method": "publish_to_observed_onset_minus_gripper_observation_latency",
        "state_signal": "FrankaGripperStateMessage.width",
        "motion_threshold_m": args.gripper_threshold_m,
        "switch_width_m": args.gripper_switch_width_m,
        "observation_latency_subtracted_ms": observation_latency_ms,
        "end_to_end": describe_ms([trial["end_to_end_ms"] for trial in trials]),
        "by_command": by_command,
    })
    return summary, trials


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure camera/robot/gripper observation latency, clock diagnostics, "
            "and arm/gripper execution latency."
        )
    )
    parser.add_argument("--interface-cfg", default="charmander.yml")
    parser.add_argument("--controller-cfg", default="osc-pose-controller.yml")
    parser.add_argument(
        "--component",
        choices=("all", "observation", "cameras", "state", "arm", "gripper", "both"),
        default="all",
        help="'both' is the backward-compatible arm+gripper action-only mode.",
    )
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--arm-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--arm-step-m", type=float, default=0.010)
    parser.add_argument("--arm-threshold-m", type=float, default=0.0005)
    parser.add_argument("--gripper-threshold-m", type=float, default=0.002)
    parser.add_argument("--gripper-switch-width-m", type=float, default=0.04)
    parser.add_argument(
        "--gripper-command-attempts", type=int, default=3,
        help="Maximum state-confirmed sends per open/close calibration trial.",
    )
    parser.add_argument("--confirm-samples", type=int, default=2)
    parser.add_argument("--state-timeout-s", type=float, default=10.0)
    parser.add_argument("--state-sample-s", type=float, default=10.0)
    parser.add_argument("--motion-timeout-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument(
        "--arm-command-frequency-hz", type=float, default=5.0,
        help="Absolute-pose resend frequency; defaults to policy eval's 5 Hz.",
    )
    parser.add_argument(
        "--controller-time-fraction", type=float, default=2.0,
        help="LINEAR_POSE time_fraction; defaults to policy eval's value.",
    )
    parser.add_argument("--ping-samples", type=int, default=20)
    parser.add_argument("--ping-interval-s", type=float, default=0.05)
    parser.add_argument("--ping-timeout-s", type=float, default=1.0)
    parser.add_argument("--front-camera-serial", default="327122071654")
    parser.add_argument("--wrist-camera-serial", default="001622071252")
    parser.add_argument("--front-color-width", type=int, default=1280)
    parser.add_argument("--front-color-height", type=int, default=720)
    parser.add_argument("--front-color-fps", type=int, default=30)
    parser.add_argument("--front-depth-width", type=int, default=1280)
    parser.add_argument("--front-depth-height", type=int, default=720)
    parser.add_argument("--front-depth-fps", type=int, default=30)
    parser.add_argument("--wrist-color-width", type=int, default=424)
    parser.add_argument("--wrist-color-height", type=int, default=240)
    parser.add_argument("--wrist-color-fps", type=int, default=30)
    parser.add_argument("--wrist-depth-width", type=int, default=480)
    parser.add_argument("--wrist-depth-height", type=int, default=270)
    parser.add_argument("--wrist-depth-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=60)
    parser.add_argument("--camera-samples", type=int, default=300)
    parser.add_argument("--camera-timeout-ms", type=int, default=2000)
    parser.add_argument("--disable-camera-depth", action="store_true")
    parser.add_argument(
        "--workspace-min", type=float, nargs=3, default=(0.30, -0.35, 0.00)
    )
    parser.add_argument(
        "--workspace-max", type=float, nargs=3, default=(0.75, 0.35, 0.60)
    )
    parser.add_argument(
        "--recommendation", choices=("median", "p95"), default="median",
        help="Statistic copied into the generated profile.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-latency-profile", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument(
        "--execute", action="store_true",
        help="Required when arm or gripper calibration will move hardware.",
    )
    args = parser.parse_args(argv)
    moves_hardware = args.component in {"all", "arm", "gripper", "both"}
    if moves_hardware and not args.execute:
        parser.error("--execute is required because action calibration moves hardware")
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    if args.arm_step_m <= 0 or args.arm_threshold_m <= 0:
        parser.error("arm step and threshold must be positive")
    if args.arm_threshold_m >= args.arm_step_m:
        parser.error("--arm-threshold-m must be smaller than --arm-step-m")
    if args.gripper_threshold_m <= 0:
        parser.error("--gripper-threshold-m must be positive")
    if args.gripper_command_attempts < 1:
        parser.error("--gripper-command-attempts must be at least 1")
    if args.confirm_samples < 1:
        parser.error("--confirm-samples must be at least 1")
    if args.camera_samples < 2 or args.camera_warmup_frames < 0:
        parser.error("camera samples must be >=2 and warmup frames must be >=0")
    if args.state_sample_s <= 0 or args.ping_samples < 2:
        parser.error("state sample duration must be positive and ping samples >=2")
    if args.arm_command_frequency_hz <= 0 or args.controller_time_fraction <= 0:
        parser.error("arm command frequency and controller time fraction must be positive")
    if args.profile_output is not None and args.base_latency_profile is None:
        parser.error("--profile-output requires --base-latency-profile")
    if np.any(np.asarray(args.workspace_min) >= np.asarray(args.workspace_max)):
        parser.error("each --workspace-min value must be below --workspace-max")
    return args


def _resolve_cfg(path_value: str, config_root: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(config_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


def choose_non_overwriting_output_paths(
    output: Path, profile_output: Path | None
) -> tuple[Path, Path | None, int]:
    """Keep a full-calibration/profile pair together without replacing old runs."""

    output = output.expanduser().resolve()
    profile_output = (
        None if profile_output is None else profile_output.expanduser().resolve()
    )
    for run_number in range(1, 10_000):
        suffix = "" if run_number == 1 else f"-run{run_number:02d}"

        def candidate(path: Path | None) -> Path | None:
            if path is None or not suffix:
                return path
            return path.with_name(f"{path.stem}{suffix}{path.suffix}")

        output_candidate = candidate(output)
        profile_candidate = candidate(profile_output)
        assert output_candidate is not None
        if not output_candidate.exists() and (
            profile_candidate is None or not profile_candidate.exists()
        ):
            return output_candidate, profile_candidate, run_number
    raise RuntimeError("could not allocate a free calibration run suffix")


def _write_json(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")


def _profile_observation_latency(
    calibration: Mapping[str, object], base_profile: Mapping[str, object],
    result_key: str, profile_key: str,
) -> float:
    result = calibration.get(result_key)
    if result is not None:
        # Execution latency is a central physical time offset.  Always remove
        # the median observation path even when the user asks the profile to
        # carry p95 values; subtracting a tail percentile would bias execution
        # latency downward and can even make valid trials negative.
        return float(result["median_ms"])
    if profile_key in base_profile:
        print(f"using {profile_key}={base_profile[profile_key]} ms from base profile")
        return float(base_profile[profile_key])
    raise ValueError(
        f"action-only calibration needs {profile_key} in --base-latency-profile"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output, args.profile_output, output_run_number = (
        choose_non_overwriting_output_paths(args.output, args.profile_output)
    )
    if output_run_number > 1:
        print(
            "requested calibration output already exists; preserving it and writing "
            f"run {output_run_number} to {args.output}",
            flush=True,
        )
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils.config_utils import verify_controller_config
    from deoxys.utils.yaml_config import YamlConfig

    interface_cfg = _resolve_cfg(args.interface_cfg, config_root)
    controller_cfg_path = _resolve_cfg(args.controller_cfg, config_root)
    general_cfg = YamlConfig(interface_cfg).as_easydict()
    controller_cfg = verify_controller_config(
        YamlConfig(controller_cfg_path).as_easydict()
    )
    controller_cfg = _configure_eval_absolute_controller(
        controller_cfg, args.controller_time_fraction
    )
    base_profile = {}
    if args.base_latency_profile is not None:
        base_profile = json.loads(args.base_latency_profile.expanduser().read_text())

    need_cameras = args.component in {"all", "observation", "cameras"}
    need_state = args.component in {"all", "observation", "state"}
    need_arm = args.component in {"all", "arm", "both"}
    need_gripper = args.component in {"all", "gripper", "both"}
    if need_arm:
        print(
            "[arm] runtime "
            f"calibration_module={Path(__file__).resolve()} "
            f"franka_interface_module={Path(inspect.getfile(FrankaInterface)).resolve()} "
            f"interface_cfg={interface_cfg} controller_cfg={controller_cfg_path}",
            flush=True,
        )
        print(
            "[arm] controller "
            f"type={controller_cfg.controller_type} "
            f"is_delta={controller_cfg.is_delta} "
            f"translation_scale={controller_cfg.action_scale.translation} "
            f"rotation_scale={controller_cfg.action_scale.rotation} "
            "interpolator="
            f"{controller_cfg.traj_interpolator_cfg.traj_interpolator_type} "
            f"time_fraction={controller_cfg.traj_interpolator_cfg.time_fraction} "
            f"command_frequency_hz={args.arm_command_frequency_hz} "
            f"Kp_translation={list(controller_cfg.Kp.translation)} "
            f"Kp_rotation={list(controller_cfg.Kp.rotation)}",
            flush=True,
        )
    measured_at = datetime.now(timezone.utc).astimezone().isoformat()
    calibration = {
        "front_observation": None, "wrist_observation": None,
        "robot_observation": None, "gripper_observation": None,
        "arm": None, "gripper": None,
    }
    diagnostics = {
        "host_reference": {
            "clock": "time.time_ns (CLOCK_REALTIME)",
            "offset_ms": 0.0,
            "drift_ppm": 0.0,
            "role": "master clock",
        },
        "camera": None,
        "proprioception": None,
    }
    trial_details = {"arm": [], "gripper": []}

    if need_cameras:
        print("Measuring front/wrist RealSense latency and clock drift...")
        (
            calibration["front_observation"],
            calibration["wrist_observation"],
            diagnostics["camera"],
        ) = calibrate_cameras(args)

    robot = None
    arm_controller_started = False
    if need_state or need_arm or need_gripper:
        if need_arm or need_gripper:
            print("WARNING: physical calibration will move the arm and/or gripper.")
            print("Clear the workspace and gripper, and keep E-stop ready.")
        robot = FrankaInterface(
            general_cfg_file=interface_cfg,
            control_freq=args.arm_command_frequency_hz,
            state_freq=100.0,
            has_gripper=True, use_visualizer=False, automatic_gripper_reset=False,
        )
        try:
            _wait_for_streams(
                robot, arm=need_state or need_arm, gripper=need_state or need_gripper,
                timeout_s=args.state_timeout_s,
            )
            if need_state:
                print("Measuring NUC RTT and robot/gripper state clocks...")
                (
                    calibration["robot_observation"],
                    calibration["gripper_observation"],
                    diagnostics["proprioception"],
                ) = calibrate_proprioception(robot, str(general_cfg.NUC.IP), args)
            if need_arm:
                robot_obs_ms = _profile_observation_latency(
                    calibration, base_profile, "robot_observation", "robot_observation_ms"
                )
                arm_controller_started = True
                calibration["arm"], trial_details["arm"] = calibrate_arm(
                    robot, controller_cfg, args, robot_obs_ms
                )
                _terminate_arm_controller(robot, controller_cfg)
                arm_controller_started = False
            if need_gripper:
                gripper_obs_ms = _profile_observation_latency(
                    calibration, base_profile, "gripper_observation",
                    "gripper_observation_ms",
                )
                calibration["gripper"], trial_details["gripper"] = calibrate_gripper(
                    robot, args, gripper_obs_ms
                )
        finally:
            if arm_controller_started:
                try:
                    _terminate_arm_controller(robot, controller_cfg)
                except Exception as exc:
                    print(f"WARNING: failed to send controller termination: {exc}")
            robot.close()

    payload = {
        "schema": "deoxys_full_latency_calibration_v2",
        "measured_at": measured_at, "host": socket.gethostname(),
        "clock": "deoxys_host_time.time_ns (CLOCK_REALTIME)",
        "interface_cfg": interface_cfg, "controller_cfg": controller_cfg_path,
        "component": args.component, "recommendation": args.recommendation,
        "settings": {
            "front_camera": {
                "serial": args.front_camera_serial,
                "color": [args.front_color_width, args.front_color_height, args.front_color_fps],
                "depth": [args.front_depth_width, args.front_depth_height, args.front_depth_fps],
            },
            "wrist_camera": {
                "serial": args.wrist_camera_serial,
                "color": [args.wrist_color_width, args.wrist_color_height, args.wrist_color_fps],
                "depth": [args.wrist_depth_width, args.wrist_depth_height, args.wrist_depth_fps],
            },
            "camera_depth_enabled": not args.disable_camera_depth,
            "camera_warmup_frames": args.camera_warmup_frames,
            "camera_samples": args.camera_samples,
            "state_sample_s": args.state_sample_s,
            "ping_samples": args.ping_samples,
            "ping_interval_s": args.ping_interval_s,
            "trials": args.trials,
            "confirm_samples": args.confirm_samples,
            "arm_axis": args.arm_axis,
            "arm_step_m": args.arm_step_m,
            "arm_threshold_m": args.arm_threshold_m,
            "arm_command_frequency_hz": args.arm_command_frequency_hz,
            "arm_controller_mode": "absolute_pose",
            "controller_time_fraction": args.controller_time_fraction,
            "gripper_threshold_m": args.gripper_threshold_m,
            "gripper_switch_width_m": args.gripper_switch_width_m,
            "gripper_command_attempts": args.gripper_command_attempts,
            "workspace_min": list(args.workspace_min),
            "workspace_max": list(args.workspace_max),
        },
        "calibration": calibration, "clock_diagnostics": diagnostics,
        "trials": trial_details,
    }
    _write_json(args.output, payload)
    print(f"wrote full calibration: {args.output.expanduser().resolve()}")
    if args.profile_output is not None:
        updated_profile = build_updated_latency_profile(
            base_profile, calibration, measured_at=measured_at
        )
        updated_profile["calibration_file"] = str(args.output.expanduser().resolve())
        _write_json(args.profile_output, updated_profile)
        print(f"wrote eval latency profile: {args.profile_output.expanduser().resolve()}")

    for key, name in (
        ("front_observation", "front_observation_ms"),
        ("wrist_observation", "wrist_observation_ms"),
        ("robot_observation", "robot_observation_ms"),
        ("gripper_observation", "gripper_observation_ms"),
    ):
        if calibration[key] is not None:
            print(f"recommended {name}={calibration[key]['recommended_observation_latency_ms']}")
    if calibration["arm"] is not None:
        print(f"recommended robot_action_ms={calibration['arm']['recommended_action_latency_ms']}")
    if calibration["gripper"] is not None:
        print(
            "recommended gripper_action_ms="
            f"{calibration['gripper']['recommended_action_latency_ms']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
