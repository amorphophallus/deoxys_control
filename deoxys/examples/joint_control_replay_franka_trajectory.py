#!/usr/bin/env python3
"""Replay recorded Franka trajectories with recorded joint positions."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils import YamlConfig

from osc_control_replay_robot_eval import (
    DEFAULT_LEFT_SERIAL,
    DEFAULT_RIGHT_SERIAL,
    DualRealSenseVideoRecorder,
    _gripper_action_from_state,
)
from osc_control_replay_robot_eval_ee_pose import (
    _get_gripper_width,
    _jsonable,
    _optional_float,
    get_gripper_width_event_plan,
    maybe_send_gripper_width_event,
    resolve_gripper_mode,
    should_send_initial_gripper_command,
    sleep_until_timestamp,
)

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay trajectories recorded by record_current_franka_ee_trajectory.py "
            "using recorded joint_positions instead of EE pose OSC targets."
        )
    )
    parser.add_argument(
        "--traj-npz",
        type=str,
        default=None,
        help="Path to franka_ee_trajectory.npz containing joint_positions.",
    )
    parser.add_argument(
        "--traj-json",
        type=str,
        default=None,
        help="Optional robot_eval.json containing per-frame joint_positions.",
    )
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--controller-type",
        choices=("JOINT_POSITION", "JOINT_IMPEDANCE"),
        default="JOINT_IMPEDANCE",
        help="Joint-space controller used for replay.",
    )
    parser.add_argument(
        "--controller-cfg",
        type=str,
        default="joint-impedance-min-jerk-controller.yml",
        help=(
            "Controller config YAML. Defaults to joint-impedance-min-jerk-controller.yml."
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="Replay every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip leading frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to replay")
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=0,
        help="Additional repeated control steps at each recorded joint target.",
    )
    parser.add_argument(
        "--advance-mode",
        choices=("reached", "time", "stream"),
        default="time",
        help=(
            "reached repeats each joint target until the robot reaches it, matching "
            "joint_position_control_replay.py. time sends each frame once according "
            "to timestamps. stream continuously tracks the recorded time axis with "
            "lookahead and adaptive slowdown."
        ),
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1e-3,
        help="Max absolute joint error required before advancing in reached mode.",
    )
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=3.0,
        help="Maximum seconds to wait for one target in reached mode.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Do not move to the first recorded joint configuration before replay.",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=10.0,
        help="Maximum seconds for the initial move to the first recorded joint target.",
    )
    parser.add_argument(
        "--reset-tolerance",
        type=float,
        default=0.01,
        help="Joint max-absolute-error tolerance for initial reset completion.",
    )
    parser.add_argument("--open-gripper-action", type=float, default=-1.0)
    parser.add_argument("--closed-gripper-action", type=float, default=1.0)
    parser.add_argument(
        "--gripper-mode",
        choices=("auto", "action", "width-events", "none"),
        default="action",
        help=(
            "Gripper replay mode. auto uses width-events when gripper widths are available; "
            "action sends the eighth joint-control action value."
        ),
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.06,
        help="Width in meters below which width-events/action mode treats gripper as closed.",
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.07,
        help="Width in meters above which width-events mode treats gripper as open.",
    )
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    parser.add_argument("--gripper-move-speed", type=float, default=0.1)
    parser.add_argument(
        "--gripper-grasp-width",
        type=float,
        default=None,
        help=(
            "Target width in meters for grasp commands. Defaults to the recorded "
            "width at the close event; pass 0.0 to force a fully closed target."
        ),
    )
    parser.add_argument("--gripper-grasp-force", type=float, default=15.0)
    parser.add_argument("--gripper-grasp-speed", type=float, default=0.5)
    parser.add_argument(
        "--gripper-command-delay",
        type=float,
        default=0.05,
        help="Delay in seconds after gripper stop before sending open/grasp commands.",
    )
    parser.add_argument(
        "--no-initial-gripper-command",
        action="store_true",
        help="Do not send the first width-derived gripper command at frame 0.",
    )
    parser.add_argument(
        "--initial-gripper-command",
        action="store_true",
        help="Send the first width-derived gripper command at frame 0.",
    )
    parser.add_argument(
        "--gripper-stop-before-command",
        action="store_true",
        help="Send a gripper stop command immediately before each open/grasp command.",
    )
    parser.add_argument(
        "--hold-closed-after-first-grasp",
        action="store_true",
        help="In width-events mode, ignore later open events after the first grasp.",
    )
    parser.add_argument(
        "--respect-timestamps",
        action="store_true",
        default=True,
        help="Sleep between target points according to timestamp_sec from the trajectory.",
    )
    parser.add_argument(
        "--ignore-timestamps",
        action="store_false",
        dest="respect_timestamps",
        help="Disable sleeping according to trajectory timestamps.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for recorded timing. Used by stream mode and by time mode "
            "when --respect-timestamps is enabled."
        ),
    )
    parser.add_argument(
        "--lookahead-time",
        type=float,
        default=0.15,
        help=(
            "Seconds ahead of the current replay time to command in stream mode. "
            "Larger values reduce tracking lag but may cut corners."
        ),
    )
    parser.add_argument(
        "--stream-slowdown-error",
        type=float,
        default=0.025,
        help=(
            "Max absolute joint tracking error, in radians, where stream mode starts "
            "slowing the recorded time axis."
        ),
    )
    parser.add_argument(
        "--stream-pause-error",
        type=float,
        default=0.06,
        help=(
            "Max absolute joint tracking error, in radians, where stream mode pauses "
            "recorded time advancement until the robot catches up."
        ),
    )
    parser.add_argument(
        "--stream-final-tolerance",
        type=float,
        default=0.01,
        help="Max absolute joint error required for the final settle in stream mode.",
    )
    parser.add_argument(
        "--stream-final-timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to settle on the final joint target in stream mode.",
    )
    parser.add_argument(
        "--replay-log-root",
        type=str,
        default="data/replay_logs",
        help="Root directory for timestamped replay logs.",
    )
    parser.add_argument(
        "--replay-log-dir",
        type=str,
        default=None,
        help="Exact directory for replay logs. Overrides --replay-log-root.",
    )
    parser.add_argument(
        "--disable-replay-log",
        action="store_true",
        help="Disable structured replay log files.",
    )
    parser.add_argument(
        "--log-frame-stride",
        type=int,
        default=1,
        help="Log every Nth replayed frame to replay_events.jsonl. Use 0 to disable frame logs.",
    )
    parser.add_argument(
        "--camera-output-root",
        type=str,
        default="data/dual_realsense_recordings",
        help="Directory where dual RealSense videos will be saved.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Enable dual RealSense recording during joint trajectory replay.",
    )
    parser.add_argument("--left-camera-serial", type=str, default=DEFAULT_LEFT_SERIAL)
    parser.add_argument("--right-camera-serial", type=str, default=DEFAULT_RIGHT_SERIAL)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-align",
        choices=("color", "none"),
        default="color",
        help="Align depth to color for each RealSense pipeline.",
    )
    parser.add_argument(
        "--show-camera-preview",
        action="store_true",
        default=False,
        help="Show a live preview window while recording.",
    )
    parser.add_argument(
        "--no-camera-preview",
        action="store_false",
        dest="show_camera_preview",
        help="Disable the live preview window.",
    )
    return parser.parse_args()


def load_controller_cfg(controller_type, controller_cfg_arg):
    if controller_cfg_arg is None:
        if controller_type == "JOINT_POSITION":
            controller_cfg_arg = "joint-position-controller.yml"
        elif controller_type == "JOINT_IMPEDANCE":
            controller_cfg_arg = "joint-impedance-controller.yml"
        else:
            raise ValueError(f"Unsupported joint controller type: {controller_type}")

    cfg_path = Path(controller_cfg_arg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = Path(config_root) / cfg_path
    return YamlConfig(str(cfg_path)).as_easydict()


class JointReplayRunLogger:
    def __init__(self, args, traj_source, payload, effective_gripper_mode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.replay_log_dir is not None:
            self.output_dir = Path(args.replay_log_dir).expanduser().resolve()
        else:
            self.output_dir = (
                Path(args.replay_log_root).expanduser().resolve()
                / f"joint_replay_{timestamp}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.output_dir / "replay_events.jsonl"
        self.summary_path = self.output_dir / "replay_summary.json"
        self.text_log_path = self.output_dir / "replay.log"
        self._events_file = self.events_path.open("w", encoding="utf-8")
        self._start_monotonic = time.monotonic()
        self._counts = {}
        self._closed = False

        self._file_handler = logging.FileHandler(self.text_log_path, encoding="utf-8")
        self._file_handler.setLevel(logging.INFO)
        self._file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s %(levelname)s] %(message)s (%(filename)s:%(lineno)d)"
            )
        )
        logger.addHandler(self._file_handler)

        self.summary = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "traj_source": str(traj_source),
            "trajectory_payload": _jsonable(payload),
            "effective_gripper_mode": effective_gripper_mode,
            "args": _jsonable(vars(args)),
            "log_dir": str(self.output_dir),
            "events_path": str(self.events_path),
            "text_log_path": str(self.text_log_path),
        }
        logger.info("Joint replay logs will be written to %s", self.output_dir)
        self.event("run_start", **self.summary)

    def event(self, event_type, **payload):
        if self._closed:
            return
        self._counts[event_type] = self._counts.get(event_type, 0) + 1
        record = {
            "event_type": event_type,
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_sec": time.monotonic() - self._start_monotonic,
            **payload,
        }
        self._events_file.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._events_file.flush()

    def close(self, status="completed", error=None):
        if self._closed:
            return
        self.summary.update(
            {
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "error": error,
                "event_counts": dict(self._counts),
                "duration_sec": time.monotonic() - self._start_monotonic,
            }
        )
        self.summary_path.write_text(
            json.dumps(_jsonable(self.summary), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._events_file.close()
        logger.removeHandler(self._file_handler)
        self._file_handler.close()
        self._closed = True


def _valid_joint_vector(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr.shape == (7,) and np.all(np.isfinite(arr))


def _gripper_action_from_width(width, args):
    width = _optional_float(width)
    if width is None:
        return float(args.open_gripper_action)
    if width <= args.gripper_close_threshold:
        return float(args.closed_gripper_action)
    return float(args.open_gripper_action)


def _make_joint_point(
    frame_index,
    joint_positions,
    gripper_width=None,
    gripper_action=-1.0,
    timestamp_sec=None,
):
    return {
        "frame_index": int(frame_index),
        "joint_positions": np.asarray(joint_positions, dtype=np.float64).reshape(7),
        "gripper_width": _optional_float(gripper_width),
        "gripper_action": float(gripper_action),
        "timestamp_sec": _optional_float(timestamp_sec),
    }


def load_npz_joint_traj(
    traj_npz,
    args,
    stride=1,
    start_frame=0,
    max_frames=None,
):
    data = np.load(traj_npz)
    if "joint_positions" not in data:
        raise RuntimeError("NPZ must contain joint_positions for joint replay")

    joints = np.asarray(data["joint_positions"], dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 7:
        raise RuntimeError(f"joint_positions must have shape (N, 7), got {joints.shape}")
    num_frames = joints.shape[0]

    timestamps = (
        np.asarray(data["timestamps_sec"], dtype=np.float64)
        if "timestamps_sec" in data
        else np.full(num_frames, np.nan, dtype=np.float64)
    )
    if timestamps.shape != (num_frames,):
        raise RuntimeError(
            f"timestamps_sec must have shape ({num_frames},), got {timestamps.shape}"
        )
    gripper_widths = (
        np.asarray(data["gripper_widths_m"], dtype=np.float64)
        if "gripper_widths_m" in data
        else np.full(num_frames, np.nan, dtype=np.float64)
    )
    if gripper_widths.shape != (num_frames,):
        raise RuntimeError(
            f"gripper_widths_m must have shape ({num_frames},), got {gripper_widths.shape}"
        )

    indices = np.arange(num_frames)
    if start_frame > 0:
        indices = indices[start_frame:]
    if stride > 1:
        indices = indices[::stride]
    if max_frames is not None:
        indices = indices[:max_frames]

    traj = []
    skipped_invalid = 0
    for index in indices:
        q = joints[index]
        if not _valid_joint_vector(q):
            skipped_invalid += 1
            continue
        width = gripper_widths[index]
        traj.append(
            _make_joint_point(
                frame_index=int(index),
                joint_positions=q,
                gripper_width=width,
                gripper_action=_gripper_action_from_width(width, args),
                timestamp_sec=timestamps[index],
            )
        )

    payload = {
        "coordinate_frame": "franka_joint_space",
        "trajectory_type": "joint_positions_npz",
        "num_frames": int(num_frames),
        "npz_keys": sorted(data.files),
        "source_path": str(traj_npz),
    }
    logger.info(
        "Loaded %d joint targets from %s, skipped %d invalid rows",
        len(traj),
        traj_npz,
        skipped_invalid,
    )
    return payload, traj


def load_json_joint_traj(
    traj_json,
    args,
    stride=1,
    start_frame=0,
    max_frames=None,
):
    payload = json.loads(Path(traj_json).read_text())
    frames = payload.get("frames", [])
    if start_frame > 0:
        frames = frames[start_frame:]
    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    traj = []
    skipped_invalid = 0
    for fallback_index, frame in enumerate(frames):
        q = frame.get("joint_positions")
        if not _valid_joint_vector(q):
            skipped_invalid += 1
            continue
        width = frame.get("gripper_width_m")
        traj.append(
            _make_joint_point(
                frame_index=int(frame.get("frame_index", fallback_index)),
                joint_positions=q,
                gripper_width=width,
                gripper_action=_gripper_action_from_state(
                    frame,
                    open_value=args.open_gripper_action,
                    closed_value=args.closed_gripper_action,
                )
                if "gripper_state" in frame
                else _gripper_action_from_width(width, args),
                timestamp_sec=frame.get("timestamp_sec"),
            )
        )

    logger.info(
        "Loaded %d joint targets from %s, skipped %d invalid frames",
        len(traj),
        traj_json,
        skipped_invalid,
    )
    return payload, traj


def wait_for_robot_state(robot_interface):
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)


def move_to_initial_joint_target(
    robot_interface,
    target_q,
    gripper_action,
    timeout,
    tolerance,
    run_logger=None,
):
    controller_type = "JOINT_POSITION"
    controller_cfg = load_controller_cfg(controller_type, "joint-position-controller.yml")
    target_q = np.asarray(target_q, dtype=np.float64).reshape(7)
    action = target_q.tolist() + [float(gripper_action)]
    start_time = time.monotonic()
    last_error = None

    logger.info("Resetting to first recorded joint target: %s", target_q.tolist())
    while True:
        wait_for_robot_state(robot_interface)
        current_q = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
        last_error = float(np.max(np.abs(current_q - target_q)))
        if last_error <= tolerance:
            logger.info("Initial joint reset reached tolerance %.4f", last_error)
            if run_logger is not None:
                run_logger.event(
                    "reset_completed",
                    target_joint_positions=target_q.tolist(),
                    max_abs_joint_error=last_error,
                )
            return
        if time.monotonic() - start_time > timeout:
            logger.warning(
                "Initial joint reset timed out with max_abs_joint_error=%.4f",
                last_error,
            )
            if run_logger is not None:
                run_logger.event(
                    "reset_timeout",
                    target_joint_positions=target_q.tolist(),
                    max_abs_joint_error=last_error,
                    timeout=timeout,
                )
            return
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


def _make_stream_time_axis(traj):
    timestamps = np.array(
        [
            np.nan
            if point.get("timestamp_sec") is None
            else float(point["timestamp_sec"])
            for point in traj
        ],
        dtype=np.float64,
    )
    if (
        len(timestamps) >= 2
        and np.all(np.isfinite(timestamps))
        and np.all(np.diff(timestamps) > 0)
    ):
        return timestamps - timestamps[0], "timestamps_sec"

    return np.arange(len(traj), dtype=np.float64) * 0.05, "synthetic_20hz"


def _interp_joint_positions(times, joint_positions, query_time):
    query_time = float(np.clip(query_time, times[0], times[-1]))
    return np.array(
        [np.interp(query_time, times, joint_positions[:, joint_idx]) for joint_idx in range(7)],
        dtype=np.float64,
    )


def _nearest_traj_index(times, query_time):
    query_time = float(np.clip(query_time, times[0], times[-1]))
    right = int(np.searchsorted(times, query_time, side="left"))
    if right <= 0:
        return 0
    if right >= len(times):
        return len(times) - 1
    left = right - 1
    if query_time - times[left] <= times[right] - query_time:
        return left
    return right


def _stream_time_advance_scale(tracking_error, slowdown_error, pause_error):
    if tracking_error <= slowdown_error:
        return 1.0
    if tracking_error >= pause_error:
        return 0.0
    return float((pause_error - tracking_error) / (pause_error - slowdown_error))


def follow_joint_traj_stream(
    robot_interface,
    controller_type,
    controller_cfg,
    traj,
    gripper_mode,
    should_stop=None,
    args=None,
    run_logger=None,
):
    times, time_source = _make_stream_time_axis(traj)
    joints = np.array([point["joint_positions"] for point in traj], dtype=np.float64)
    duration = float(times[-1])
    replay_time = 0.0
    loop_idx = 0
    previous_gripper_state = None
    last_wall_time = time.monotonic()

    logger.info(
        "Stream replay time_source=%s duration=%.3fs lookahead=%.3fs time_scale=%.3f",
        time_source,
        duration,
        args.lookahead_time,
        args.time_scale,
    )
    logger.info(
        "Stream adaptive slowdown: slowdown_error=%.4f pause_error=%.4f",
        args.stream_slowdown_error,
        args.stream_pause_error,
    )
    if run_logger is not None:
        run_logger.event(
            "stream_started",
            time_source=time_source,
            duration=duration,
            lookahead_time=args.lookahead_time,
            time_scale=args.time_scale,
            stream_slowdown_error=args.stream_slowdown_error,
            stream_pause_error=args.stream_pause_error,
        )

    while replay_time < duration:
        if should_stop is not None and should_stop():
            logger.warning("Stop requested before stream replay completed")
            break

        wait_for_robot_state(robot_interface)
        now = time.monotonic()
        wall_dt = max(0.0, now - last_wall_time)
        last_wall_time = now

        current_q = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
        nominal_q = _interp_joint_positions(times, joints, replay_time)
        tracking_error = float(np.max(np.abs(nominal_q - current_q)))
        advance_scale = _stream_time_advance_scale(
            tracking_error,
            args.stream_slowdown_error,
            args.stream_pause_error,
        )
        replay_time = min(duration, replay_time + wall_dt * advance_scale / args.time_scale)

        command_time = min(duration, replay_time + args.lookahead_time)
        target_q = _interp_joint_positions(times, joints, command_time)
        point_idx = _nearest_traj_index(times, replay_time)
        command_point_idx = _nearest_traj_index(times, command_time)
        point = traj[point_idx]

        if gripper_mode == "width-events":
            previous_gripper_state = maybe_send_gripper_width_event(
                robot_interface,
                point,
                previous_gripper_state,
                args,
                force_initial=(loop_idx == 0 and should_send_initial_gripper_command(args)),
                run_logger=run_logger,
            )

        gripper_action = 0.0 if gripper_mode in ("width-events", "none") else point["gripper_action"]
        action = target_q.tolist() + [float(gripper_action)]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

        if (
            run_logger is not None
            and args.log_frame_stride > 0
            and loop_idx % args.log_frame_stride == 0
        ):
            current_q_after = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
            command_error_after = float(np.max(np.abs(target_q - current_q_after)))
            run_logger.event(
                "stream_step",
                loop_idx=loop_idx,
                replay_time=replay_time,
                command_time=command_time,
                frame_index=point["frame_index"],
                command_frame_index=traj[command_point_idx]["frame_index"],
                target_joint_positions=target_q.tolist(),
                nominal_joint_positions=nominal_q.tolist(),
                current_joint_positions=current_q_after.tolist(),
                max_abs_tracking_error=tracking_error,
                max_abs_command_error_after=command_error_after,
                advance_scale=advance_scale,
                gripper_action=gripper_action,
                recorded_gripper_width=point.get("gripper_width"),
                measured_gripper_width_after=_get_gripper_width(robot_interface),
                gripper_state_memory=previous_gripper_state,
            )
        loop_idx += 1

    final_q = joints[-1]
    final_gripper_action = 0.0 if gripper_mode in ("width-events", "none") else traj[-1]["gripper_action"]
    final_action = final_q.tolist() + [float(final_gripper_action)]
    settle_start = time.monotonic()
    settle_steps = 0
    while True:
        if should_stop is not None and should_stop():
            logger.warning("Stop requested during stream final settle")
            break
        wait_for_robot_state(robot_interface)
        current_q = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
        final_error = float(np.max(np.abs(final_q - current_q)))
        if final_error <= args.stream_final_tolerance:
            logger.info("Stream final target reached tolerance %.4f", final_error)
            break
        if time.monotonic() - settle_start > args.stream_final_timeout:
            logger.warning(
                "Stream final settle timed out with max_abs_joint_error=%.4f",
                final_error,
            )
            break
        robot_interface.control(
            controller_type=controller_type,
            action=final_action,
            controller_cfg=controller_cfg,
        )
        settle_steps += 1

    if run_logger is not None:
        run_logger.event(
            "stream_finished",
            loop_count=loop_idx,
            replay_time=replay_time,
            duration=duration,
            settle_steps=settle_steps,
            final_error=final_error if "final_error" in locals() else None,
        )


def follow_joint_traj(
    robot_interface,
    controller_type,
    controller_cfg,
    traj,
    hold_steps=0,
    should_stop=None,
    args=None,
    run_logger=None,
):
    wait_for_robot_state(robot_interface)

    gripper_mode = resolve_gripper_mode(args.gripper_mode, traj) if args is not None else "action"
    logger.info("Joint replay controller: %s", controller_type)
    logger.info("Gripper replay mode: %s", gripper_mode)
    logger.info("Advance mode: %s", args.advance_mode)
    if gripper_mode == "width-events":
        event_plan = get_gripper_width_event_plan(traj, args)
        logger.info("Planned gripper width events: %d", len(event_plan))
        for event in event_plan:
            logger.info(
                "Planned gripper event point=%s frame=%s timestamp=%s width=%s -> %s",
                event["point_idx"],
                event["frame_index"],
                event["timestamp_sec"],
                event["width"],
                event["state"],
            )
            if run_logger is not None:
                run_logger.event("planned_gripper_event", **event)

    first_timestamp = None
    if args is not None and args.respect_timestamps and args.advance_mode == "time":
        for point in traj:
            if point.get("timestamp_sec") is not None:
                first_timestamp = point["timestamp_sec"]
                break
        logger.info("Respecting trajectory timestamps with time_scale=%.3f", args.time_scale)
    elif args is not None and args.respect_timestamps and args.advance_mode == "reached":
        logger.info(
            "Ignoring --respect-timestamps because --advance-mode reached matches "
            "joint_position_control_replay.py by waiting for each target."
        )
    elif args is not None and args.respect_timestamps and args.advance_mode == "stream":
        logger.info(
            "Stream mode always uses trajectory timestamps when available; "
            "--respect-timestamps is not required."
        )
    replay_start_time = time.monotonic()

    original_has_gripper = robot_interface.has_gripper
    original_automatic_gripper_reset = robot_interface.automatic_gripper_reset
    if gripper_mode in ("width-events", "none"):
        robot_interface.has_gripper = False
        robot_interface.automatic_gripper_reset = False

    previous_gripper_state = None
    try:
        if args is not None and args.advance_mode == "stream":
            follow_joint_traj_stream(
                robot_interface,
                controller_type,
                controller_cfg,
                traj=traj,
                gripper_mode=gripper_mode,
                should_stop=should_stop,
                args=args,
                run_logger=run_logger,
            )
            return

        for point_idx, point in enumerate(traj):
            if should_stop is not None and should_stop():
                logger.warning("Stop requested before joint trajectory replay completed")
                break

            if (
                args is not None
                and args.respect_timestamps
                and args.advance_mode == "time"
            ):
                sleep_until_timestamp(
                    point,
                    first_timestamp=first_timestamp,
                    replay_start_time=replay_start_time,
                    time_scale=args.time_scale,
                )

            if gripper_mode == "width-events":
                previous_gripper_state = maybe_send_gripper_width_event(
                    robot_interface,
                    point,
                    previous_gripper_state,
                    args,
                    force_initial=(
                        point_idx == 0 and should_send_initial_gripper_command(args)
                    ),
                    run_logger=run_logger,
                )

            target_q = point["joint_positions"]
            gripper_action = 0.0 if gripper_mode in ("width-events", "none") else point["gripper_action"]
            action = target_q.tolist() + [float(gripper_action)]
            current_q_before = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
            joint_error_before = float(np.max(np.abs(target_q - current_q_before)))
            measured_gripper_before = _get_gripper_width(robot_interface)

            control_steps = 0
            target_reached = False
            target_timed_out = False
            target_start_time = time.monotonic()

            if args is not None and args.advance_mode == "reached":
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                control_steps += 1
                while True:
                    if should_stop is not None and should_stop():
                        logger.warning("Stop requested while moving to current joint target")
                        break
                    wait_for_robot_state(robot_interface)
                    current_q = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
                    joint_error = float(np.max(np.abs(target_q - current_q)))
                    if joint_error <= args.target_tolerance:
                        target_reached = True
                        break
                    if time.monotonic() - target_start_time > args.target_timeout:
                        target_timed_out = True
                        logger.warning(
                            "Timed out moving to frame=%s with max_abs_joint_error=%.4f",
                            point["frame_index"],
                            joint_error,
                        )
                        break
                    robot_interface.control(
                        controller_type=controller_type,
                        action=action,
                        controller_cfg=controller_cfg,
                    )
                    control_steps += 1
            else:
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                control_steps += 1

            for _ in range(hold_steps):
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                control_steps += 1

            if (
                run_logger is not None
                and args is not None
                and args.log_frame_stride > 0
                and point_idx % args.log_frame_stride == 0
            ):
                current_q_after = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
                joint_error_after = float(np.max(np.abs(target_q - current_q_after)))
                run_logger.event(
                    "frame_executed",
                    point_idx=point_idx,
                    frame_index=point["frame_index"],
                    timestamp_sec=point.get("timestamp_sec"),
                    target_joint_positions=target_q.tolist(),
                    current_joint_positions_before=current_q_before.tolist(),
                    current_joint_positions_after=current_q_after.tolist(),
                    max_abs_joint_error_before=joint_error_before,
                    max_abs_joint_error_after=joint_error_after,
                    control_steps=control_steps,
                    target_reached=target_reached,
                    target_timed_out=target_timed_out,
                    gripper_action=gripper_action,
                    recorded_gripper_width=point.get("gripper_width"),
                    measured_gripper_width_before=measured_gripper_before,
                    measured_gripper_width_after=_get_gripper_width(robot_interface),
                    gripper_state_memory=previous_gripper_state,
                )
    finally:
        robot_interface.has_gripper = original_has_gripper
        robot_interface.automatic_gripper_reset = original_automatic_gripper_reset


def main():
    args = parse_args()
    if (args.traj_npz is None) == (args.traj_json is None):
        raise ValueError("Specify exactly one of --traj-npz or --traj-json")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.hold_steps < 0:
        raise ValueError("--hold-steps cannot be negative")
    if args.target_tolerance <= 0:
        raise ValueError("--target-tolerance must be positive")
    if args.target_timeout <= 0:
        raise ValueError("--target-timeout must be positive")
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    if args.lookahead_time < 0:
        raise ValueError("--lookahead-time cannot be negative")
    if args.stream_slowdown_error <= 0:
        raise ValueError("--stream-slowdown-error must be positive")
    if args.stream_pause_error <= args.stream_slowdown_error:
        raise ValueError("--stream-pause-error must be larger than --stream-slowdown-error")
    if args.stream_final_tolerance <= 0:
        raise ValueError("--stream-final-tolerance must be positive")
    if args.stream_final_timeout <= 0:
        raise ValueError("--stream-final-timeout must be positive")
    if args.reset_timeout <= 0:
        raise ValueError("--reset-timeout must be positive")
    if args.reset_tolerance <= 0:
        raise ValueError("--reset-tolerance must be positive")
    if args.gripper_command_delay < 0:
        raise ValueError("--gripper-command-delay cannot be negative")
    if (
        args.gripper_grasp_width is not None
        and (args.gripper_grasp_width < 0 or args.gripper_grasp_width > args.gripper_open_width)
    ):
        raise ValueError("--gripper-grasp-width must be between 0 and --gripper-open-width")
    if args.log_frame_stride < 0:
        raise ValueError("--log-frame-stride cannot be negative")
    if args.gripper_close_threshold >= args.gripper_open_threshold:
        raise ValueError("--gripper-close-threshold must be smaller than --gripper-open-threshold")

    if args.traj_npz is not None:
        payload, traj = load_npz_joint_traj(
            traj_npz=args.traj_npz,
            args=args,
            stride=args.stride,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
        )
        traj_source = args.traj_npz
    else:
        payload, traj = load_json_joint_traj(
            traj_json=args.traj_json,
            args=args,
            stride=args.stride,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
        )
        traj_source = args.traj_json

    if not traj:
        raise RuntimeError("No valid joint trajectory points found")

    effective_gripper_mode = resolve_gripper_mode(args.gripper_mode, traj)
    logger.info(
        "Loaded %d joint trajectory points from %s",
        len(traj),
        traj_source,
    )

    run_logger = None
    if not args.disable_replay_log:
        run_logger = JointReplayRunLogger(
            args=args,
            traj_source=traj_source,
            payload=payload,
            effective_gripper_mode=effective_gripper_mode,
        )
        run_logger.event(
            "trajectory_loaded",
            num_points=len(traj),
            traj_source=traj_source,
            effective_gripper_mode=effective_gripper_mode,
        )

    use_action_gripper = effective_gripper_mode not in ("width-events", "none")
    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}",
        use_visualizer=False,
        has_gripper=use_action_gripper,
        automatic_gripper_reset=use_action_gripper,
    )
    controller_cfg = load_controller_cfg(args.controller_type, args.controller_cfg)

    camera_recorder = None
    if args.record_video:
        camera_recorder = DualRealSenseVideoRecorder(
            output_root=args.camera_output_root,
            left_serial=args.left_camera_serial,
            right_serial=args.right_camera_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            align_mode=args.camera_align,
            show_preview=args.show_camera_preview,
        )

    recorder_stop_error = None
    run_status = "completed"
    run_error = None
    try:
        if camera_recorder is not None:
            camera_recorder.start()
            if run_logger is not None:
                run_logger.event(
                    "camera_recording_started",
                    output_dir=str(camera_recorder.output_dir),
                    left_video_path=str(camera_recorder.left_video_path),
                    right_video_path=str(camera_recorder.right_video_path),
                    metadata_path=str(camera_recorder.metadata_path),
                    left_camera_serial=camera_recorder.left_serial,
                    right_camera_serial=camera_recorder.right_serial,
                    width=camera_recorder.width,
                    height=camera_recorder.height,
                    fps=camera_recorder.fps,
                    align_mode=camera_recorder.align_mode,
                    show_preview=camera_recorder.show_preview,
                )

        if not args.skip_reset:
            move_to_initial_joint_target(
                robot_interface,
                target_q=traj[0]["joint_positions"],
                gripper_action=traj[0]["gripper_action"],
                timeout=args.reset_timeout,
                tolerance=args.reset_tolerance,
                run_logger=run_logger,
            )
        elif run_logger is not None:
            run_logger.event("reset_skipped")

        follow_joint_traj(
            robot_interface,
            args.controller_type,
            controller_cfg,
            traj=traj,
            hold_steps=args.hold_steps,
            should_stop=None if camera_recorder is None else camera_recorder.should_stop,
            args=args,
            run_logger=run_logger,
        )
        if camera_recorder is not None:
            camera_recorder.raise_if_failed()
    except Exception as exc:
        run_status = "error"
        run_error = repr(exc)
        if run_logger is not None:
            run_logger.event("exception", error=run_error)
        raise
    finally:
        if camera_recorder is not None:
            try:
                camera_recorder.stop()
                if run_logger is not None:
                    run_logger.event(
                        "camera_recording_stopped",
                        output_dir=str(camera_recorder.output_dir),
                        left_video_path=str(camera_recorder.left_video_path),
                        right_video_path=str(camera_recorder.right_video_path),
                        metadata_path=str(camera_recorder.metadata_path),
                        left_frame_count=camera_recorder.left_frame_count,
                        right_frame_count=camera_recorder.right_frame_count,
                    )
            except Exception as exc:
                recorder_stop_error = exc
                run_status = "error"
                run_error = repr(exc)
                if run_logger is not None:
                    run_logger.event("camera_recording_stop_error", error=run_error)
        robot_interface.close()
        if run_logger is not None:
            run_logger.event("robot_interface_closed")
            run_logger.close(status=run_status, error=run_error)

    if recorder_stop_error is not None:
        raise recorder_stop_error
    if camera_recorder is not None:
        camera_recorder.raise_if_failed()


if __name__ == "__main__":
    main()
