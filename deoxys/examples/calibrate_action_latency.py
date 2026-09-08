#!/usr/bin/env python3
"""Calibrate observation clocks and Franka arm/gripper execution latency.

The measured interval starts when the local control process finishes publishing a
command and ends when that same process receives the first persistent robot or
gripper state change.  Both endpoints use ``time.time_ns()`` on the Deoxys host;
the NUC protobuf timestamp is deliberately not used.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


def detect_motion_onset_ns(
    samples: Iterable[tuple[int, object]],
    *,
    command_wall_time_ns: int,
    baseline: object,
    threshold: float,
    confirm_samples: int,
) -> int | None:
    """Return the first timestamp of a persistent threshold crossing.

    ``samples`` contains local receive wall timestamps and scalar/vector state
    values.  A crossing is accepted only after ``confirm_samples`` consecutive
    values exceed the Euclidean distance from ``baseline``.
    """
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
        changed = float(np.linalg.norm(value_array - baseline_array)) >= threshold
        if changed:
            if run_length == 0:
                run_start_ns = timestamp_ns
            run_length += 1
            if run_length >= confirm_samples:
                return int(run_start_ns)
        else:
            run_start_ns = None
            run_length = 0
    return None


def summarize_latency_ms(values_ms: Sequence[float], recommendation: str) -> dict:
    if not values_ms:
        raise ValueError("at least one latency sample is required")
    values = np.asarray(values_ms, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("latency samples must be finite and non-negative")
    median_ms = float(np.median(values))
    p95_ms = float(np.percentile(values, 95))
    recommended_ms = median_ms if recommendation == "median" else p95_ms
    if recommendation not in {"median", "p95"}:
        raise ValueError("recommendation must be 'median' or 'p95'")
    return {
        "num_samples": int(values.size),
        "samples_ms": [round(float(value), 6) for value in values],
        "min_ms": round(float(np.min(values)), 6),
        "median_ms": round(median_ms, 6),
        "mean_ms": round(float(np.mean(values)), 6),
        "p95_ms": round(p95_ms, 6),
        "max_ms": round(float(np.max(values)), 6),
        "recommended_statistic": recommendation,
        "recommended_action_latency_ms": round(recommended_ms, 3),
    }


def build_updated_latency_profile(
    base_profile: dict,
    calibration: dict,
    *,
    measured_at: str,
) -> dict:
    """Replace calibrated action fields while preserving observation fields."""
    profile = dict(base_profile)
    calibrated_fields = []
    if calibration.get("arm") is not None:
        profile["robot_action_ms"] = calibration["arm"][
            "recommended_action_latency_ms"
        ]
        calibrated_fields.append("robot_action_ms")
    if calibration.get("gripper") is not None:
        profile["gripper_action_ms"] = calibration["gripper"][
            "recommended_action_latency_ms"
        ]
        calibrated_fields.append("gripper_action_ms")
    if not calibrated_fields:
        raise ValueError("calibration contains neither arm nor gripper results")

    inherited_fields = [
        key
        for key in (
            "front_observation_ms",
            "wrist_observation_ms",
            "robot_observation_ms",
            "gripper_observation_ms",
            "action_stale_guard_ms",
        )
        if key in base_profile
    ]
    profile["schema_version"] = max(int(profile.get("schema_version", 1)), 2)
    profile["measured_at"] = measured_at
    profile["latency_source"] = "measured"
    profile["basis"] = (
        "Action fields calibrated by Deoxys host send-to-persistent-state-onset "
        "using host CLOCK_REALTIME/time.time_ns; non-action fields inherited "
        "unchanged from the base profile."
    )
    profile["calibrated_fields"] = calibrated_fields
    profile["inherited_fields"] = inherited_fields
    profile["action_calibration_clock"] = "deoxys_host_time.time_ns"
    return profile


def _message_arm_position(message) -> np.ndarray:
    pose = np.asarray(message.O_T_EE, dtype=np.float64).reshape(4, 4).T
    return pose[:3, 3].copy()


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
        gripper_ready = not gripper or bool(
            robot.timestamped_gripper_state_buffer(max_records=1)
        )
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
    get_records: Callable[[], list[dict]],
    extractor: Callable,
    *,
    command_wall_time_ns: int,
    baseline: np.ndarray,
    threshold: float,
    confirm_samples: int,
    timeout_s: float,
) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        samples = [
            (record["receive_wall_time_ns"], extractor(record["message"]))
            for record in get_records()
            if int(record["receive_wall_time_ns"]) >= command_wall_time_ns
        ]
        onset_ns = detect_motion_onset_ns(
            samples,
            command_wall_time_ns=command_wall_time_ns,
            baseline=baseline,
            threshold=threshold,
            confirm_samples=confirm_samples,
        )
        if onset_ns is not None:
            return onset_ns
        time.sleep(0.005)
    raise TimeoutError(
        f"no persistent state change >= {threshold:g} within {timeout_s:g}s"
    )


def _inside_workspace(position: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.all(position >= lower) and np.all(position <= upper))


def _arm_delta_action(controller_cfg, axis_index: int, displacement_m: float) -> np.ndarray:
    scale = np.asarray(controller_cfg.action_scale.translation, dtype=np.float64).reshape(-1)
    axis_scale = float(scale[0] if scale.size == 1 else scale[axis_index])
    if axis_scale <= 0:
        raise ValueError("controller translation action scale must be positive")
    normalized = displacement_m / axis_scale
    if abs(normalized) > 1.0:
        raise ValueError(
            f"requested arm step needs normalized action {normalized:.3f}, outside [-1, 1]"
        )
    action = np.zeros(7, dtype=np.float64)
    action[axis_index] = normalized
    action[-1] = -1.0
    return action


def calibrate_arm(robot, controller_cfg, args) -> tuple[dict, list[dict]]:
    if not bool(controller_cfg.is_delta):
        raise ValueError("arm latency calibration requires an OSC_POSE delta controller")

    # Warm up the controller transition so preprocess() is outside measured trials.
    zero_action = np.zeros(7, dtype=np.float64)
    zero_action[-1] = -1.0
    robot.control(
        controller_type="OSC_POSE",
        action=zero_action,
        controller_cfg=controller_cfg,
        control_gripper=False,
        enforce_control_frequency=False,
    )
    time.sleep(args.settle_s)

    axis_index = {"x": 0, "y": 1, "z": 2}[args.arm_axis]
    lower = np.asarray(args.workspace_min, dtype=np.float64)
    upper = np.asarray(args.workspace_max, dtype=np.float64)
    trials = []
    for trial_index in range(args.trials):
        signed_step = args.arm_step_m * (1.0 if trial_index % 2 == 0 else -1.0)
        records = robot.timestamped_robot_state_buffer(max_records=1)
        baseline = _latest_value(records, _message_arm_position)
        target = baseline.copy()
        target[axis_index] += signed_step
        if not _inside_workspace(baseline, lower, upper):
            raise RuntimeError(f"current EE position {baseline.tolist()} is outside workspace")
        if not _inside_workspace(target, lower, upper):
            raise RuntimeError(f"arm pulse target {target.tolist()} is outside workspace")

        command_result = robot.control(
            controller_type="OSC_POSE",
            action=_arm_delta_action(controller_cfg, axis_index, signed_step),
            controller_cfg=controller_cfg,
            control_gripper=False,
            enforce_control_frequency=False,
        )
        command_ns = int(command_result["robot_command_wall_time_ns"])
        onset_ns = _wait_for_motion(
            robot.timestamped_robot_state_buffer,
            _message_arm_position,
            command_wall_time_ns=command_ns,
            baseline=baseline,
            threshold=args.arm_threshold_m,
            confirm_samples=args.confirm_samples,
            timeout_s=args.motion_timeout_s,
        )
        latency_ms = (onset_ns - command_ns) / 1e6
        trials.append(
            {
                "trial": trial_index + 1,
                "axis": args.arm_axis,
                "signed_step_m": signed_step,
                "baseline_position_m": baseline.tolist(),
                "command_wall_time_ns": command_ns,
                "onset_receive_wall_time_ns": onset_ns,
                "latency_ms": round(latency_ms, 6),
            }
        )
        print(f"arm trial {trial_index + 1}/{args.trials}: {latency_ms:.3f} ms")
        time.sleep(args.settle_s)

    if args.trials % 2:
        robot.control(
            controller_type="OSC_POSE",
            action=_arm_delta_action(controller_cfg, axis_index, -args.arm_step_m),
            controller_cfg=controller_cfg,
            control_gripper=False,
            enforce_control_frequency=False,
        )
        time.sleep(args.settle_s)

    summary = summarize_latency_ms(
        [trial["latency_ms"] for trial in trials], args.recommendation
    )
    summary.update(
        {
            "state_signal": "O_T_EE translation",
            "motion_threshold_m": args.arm_threshold_m,
            "pulse_axis": args.arm_axis,
            "pulse_step_m": args.arm_step_m,
        }
    )
    return summary, trials


def calibrate_gripper(robot, args) -> tuple[dict, list[dict]]:
    initial_width = _latest_value(
        robot.timestamped_gripper_state_buffer(max_records=1),
        _message_gripper_width,
    )
    initial_open = float(initial_width[0]) >= args.gripper_switch_width_m
    trials = []
    for trial_index in range(args.trials):
        baseline = _latest_value(
            robot.timestamped_gripper_state_buffer(max_records=1),
            _message_gripper_width,
        )
        command_close = float(baseline[0]) >= args.gripper_switch_width_m
        action = 1.0 if command_close else -1.0
        robot.gripper_control(action)
        command_ns = int(robot.last_gripper_command_wall_time_ns)
        onset_ns = _wait_for_motion(
            robot.timestamped_gripper_state_buffer,
            _message_gripper_width,
            command_wall_time_ns=command_ns,
            baseline=baseline,
            threshold=args.gripper_threshold_m,
            confirm_samples=args.confirm_samples,
            timeout_s=args.motion_timeout_s,
        )
        latency_ms = (onset_ns - command_ns) / 1e6
        trials.append(
            {
                "trial": trial_index + 1,
                "command": "close" if command_close else "open",
                "baseline_width_m": float(baseline[0]),
                "command_wall_time_ns": command_ns,
                "onset_receive_wall_time_ns": onset_ns,
                "latency_ms": round(latency_ms, 6),
            }
        )
        print(f"gripper trial {trial_index + 1}/{args.trials}: {latency_ms:.3f} ms")
        time.sleep(args.settle_s)

    final_width = _latest_value(
        robot.timestamped_gripper_state_buffer(max_records=1),
        _message_gripper_width,
    )
    final_open = float(final_width[0]) >= args.gripper_switch_width_m
    if final_open != initial_open:
        robot.gripper_control(-1.0 if initial_open else 1.0)
        time.sleep(args.settle_s)

    summary = summarize_latency_ms(
        [trial["latency_ms"] for trial in trials], args.recommendation
    )
    summary.update(
        {
            "state_signal": "FrankaGripperStateMessage.width",
            "motion_threshold_m": args.gripper_threshold_m,
            "switch_width_m": args.gripper_switch_width_m,
        }
    )
    return summary, trials


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Measure Deoxys arm/gripper command-to-observed-motion latency."
    )
    parser.add_argument("--interface-cfg", default="charmander.yml")
    parser.add_argument("--controller-cfg", default="osc-pose-controller.yml")
    parser.add_argument("--component", choices=("arm", "gripper", "both"), default="both")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--arm-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--arm-step-m", type=float, default=0.003)
    parser.add_argument("--arm-threshold-m", type=float, default=0.0008)
    parser.add_argument("--gripper-threshold-m", type=float, default=0.002)
    parser.add_argument("--gripper-switch-width-m", type=float, default=0.04)
    parser.add_argument("--confirm-samples", type=int, default=2)
    parser.add_argument("--state-timeout-s", type=float, default=10.0)
    parser.add_argument("--motion-timeout-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument(
        "--workspace-min", type=float, nargs=3, default=(0.30, -0.35, 0.00)
    )
    parser.add_argument(
        "--workspace-max", type=float, nargs=3, default=(0.75, 0.35, 0.60)
    )
    parser.add_argument(
        "--recommendation",
        choices=("median", "p95"),
        default="median",
        help="Statistic copied into generated profile; median is best for time alignment.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-latency-profile", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that the script will move the robot and gripper.",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required because calibration commands physical motion")
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    if args.arm_step_m <= 0 or args.arm_threshold_m <= 0:
        parser.error("arm step and threshold must be positive")
    if args.arm_threshold_m >= args.arm_step_m:
        parser.error("--arm-threshold-m must be smaller than --arm-step-m")
    if args.gripper_threshold_m <= 0:
        parser.error("--gripper-threshold-m must be positive")
    if args.confirm_samples < 1:
        parser.error("--confirm-samples must be at least 1")
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


def _write_json(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils.yaml_config import YamlConfig

    interface_cfg = _resolve_cfg(args.interface_cfg, config_root)
    controller_cfg_path = _resolve_cfg(args.controller_cfg, config_root)
    controller_cfg = YamlConfig(controller_cfg_path).as_easydict()

    print("WARNING: physical calibration will move the arm and/or gripper.")
    print("Clear the workspace, remove objects from the gripper, and keep E-stop ready.")
    print("Do not touch the robot during measurement.")

    need_arm = args.component in {"arm", "both"}
    need_gripper = args.component in {"gripper", "both"}
    measured_at = datetime.now(timezone.utc).astimezone().isoformat()
    calibration = {"arm": None, "gripper": None}
    trial_details = {"arm": [], "gripper": []}
    robot = FrankaInterface(
        general_cfg_file=interface_cfg,
        control_freq=100.0,
        state_freq=100.0,
        has_gripper=True,
        use_visualizer=False,
        automatic_gripper_reset=False,
    )
    arm_controller_started = False
    try:
        _wait_for_streams(
            robot,
            arm=need_arm,
            gripper=need_gripper,
            timeout_s=args.state_timeout_s,
        )
        if need_arm:
            arm_controller_started = True
            calibration["arm"], trial_details["arm"] = calibrate_arm(
                robot, controller_cfg, args
            )
        if need_gripper:
            calibration["gripper"], trial_details["gripper"] = calibrate_gripper(
                robot, args
            )
    finally:
        if arm_controller_started:
            try:
                stop_action = np.zeros(7, dtype=np.float64)
                stop_action[-1] = -1.0
                robot.control(
                    controller_type="OSC_POSE",
                    action=stop_action,
                    controller_cfg=controller_cfg,
                    termination=True,
                    control_gripper=False,
                    enforce_control_frequency=False,
                )
            except Exception as exc:  # cleanup must not hide calibration failure
                print(f"WARNING: failed to send controller termination: {exc}")
        robot.close()

    payload = {
        "schema": "deoxys_action_latency_calibration_v1",
        "measured_at": measured_at,
        "host": socket.gethostname(),
        "clock": "deoxys_host_time.time_ns (CLOCK_REALTIME)",
        "interval_definition": (
            "local publish completion to first locally received persistent state change"
        ),
        "interface_cfg": interface_cfg,
        "controller_cfg": controller_cfg_path,
        "component": args.component,
        "recommendation": args.recommendation,
        "confirm_samples": args.confirm_samples,
        "calibration": calibration,
        "trials": trial_details,
    }
    _write_json(args.output, payload)
    print(f"wrote calibration: {args.output.expanduser().resolve()}")

    if args.profile_output is not None:
        base_profile = json.loads(args.base_latency_profile.expanduser().read_text())
        updated_profile = build_updated_latency_profile(
            base_profile, calibration, measured_at=measured_at
        )
        updated_profile["action_calibration_file"] = str(args.output.expanduser().resolve())
        _write_json(args.profile_output, updated_profile)
        print(f"wrote eval latency profile: {args.profile_output.expanduser().resolve()}")

    arm_value = calibration["arm"]
    gripper_value = calibration["gripper"]
    if arm_value is not None:
        print(f"recommended --robot-action-latency-ms {arm_value['recommended_action_latency_ms']}")
    if gripper_value is not None:
        print(
            "recommended --gripper-action-latency-ms "
            f"{gripper_value['recommended_action_latency_ms']}"
        )
    return 0


# Keep the historical module/CLI path while the implementation now covers the
# complete timing profile. Importing here intentionally replaces the legacy
# helpers above as well as ``main`` for existing callers and tests.
from examples.full_latency_calibration import (  # noqa: E402,F401
    _absolute_pose_action,
    _configure_eval_absolute_controller,
    _wait_for_arm_motion_with_resend,
    build_updated_latency_profile,
    calibrate_gripper,
    choose_non_overwriting_output_paths,
    corrected_execution_latency_ms,
    describe_ms,
    detect_motion_onset_ns,
    fit_clock_mapping,
    _keep_arm_controller_alive,
    main,
    parse_args,
    parse_ping_rtts_ms,
    summarize_latency_ms,
    summarize_observation_latency_ms,
    _write_json,
)


if __name__ == "__main__":
    raise SystemExit(main())
