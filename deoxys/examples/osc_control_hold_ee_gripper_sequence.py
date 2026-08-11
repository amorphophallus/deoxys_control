#!/usr/bin/env python3
"""Record video while replaying a gripper-width sequence without arm motion."""

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger

from osc_control_replay_robot_eval import (
    DEFAULT_LEFT_SERIAL,
    DEFAULT_RIGHT_SERIAL,
    DualRealSenseVideoRecorder,
    osc_move,
)
from osc_control_replay_robot_eval_ee_pose import send_gripper_move, send_gripper_stop

logger = get_deoxys_example_logger()


TARGET_T_BASE_EE = np.array(
    [
        [0.243956, 0.146129, 0.958714, 0.572814],
        [-0.066034, -0.983785, 0.166754, 0.110633],
        [0.967536, -0.103989, -0.23035, 0.102349],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

RESET_JOINT_POSITIONS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Start dual RealSense recording and command gripper widths: "
            "open, half, closed, open. No Franka arm control commands are sent."
        )
    )
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE", help=argparse.SUPPRESS)
    parser.add_argument("--skip-reset", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approach-steps", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument(
        "--approach-interpolation",
        choices=("min-jerk", "linear"),
        default="min-jerk",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--approach-mode",
        choices=("staged", "pose-interp"),
        default="staged",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--approach-log-stride", type=int, default=25, help=argparse.SUPPRESS)
    parser.add_argument("--post-approach-hold-sec", type=float, default=0.75, help=argparse.SUPPRESS)
    parser.add_argument("--position-tolerance", type=float, default=0.006, help=argparse.SUPPRESS)
    parser.add_argument("--orientation-tolerance", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--no-require-pose-tolerance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--open-width",
        type=float,
        default=0.08,
        help="Maximum gripper opening width in meters.",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=0.06,
        help="Half-close gripper width in meters. Defaults to 0.06 m.",
    )
    parser.add_argument(
        "--closed-width",
        type=float,
        default=0.05,
        help="Closed-stage gripper width in meters. Defaults to 0.05 m.",
    )
    parser.add_argument(
        "--gripper-move-speed",
        type=float,
        default=0.05,
        help="Gripper move speed in m/s.",
    )
    parser.add_argument(
        "--gripper-step-timeout-sec",
        type=float,
        default=2.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gripper-min-hold-sec",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gripper-stage-hold-sec",
        type=float,
        default=5.0,
        help="Seconds to stay in each gripper stage after sending the width command.",
    )
    parser.add_argument(
        "--gripper-width-tolerance",
        type=float,
        default=0.004,
        help="Width tolerance in meters used to decide whether a gripper command reached its target.",
    )
    parser.add_argument(
        "--gripper-stop-before-command",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send gripper stop commands around each width stage.",
    )
    parser.add_argument(
        "--gripper-command-delay",
        type=float,
        default=0.05,
        help="Delay after optional gripper stop before sending the next width command.",
    )
    parser.add_argument(
        "--pre-gripper-record-delay",
        type=float,
        default=0.5,
        help="Seconds to record before sending the first gripper command.",
    )
    parser.add_argument(
        "--camera-output-root",
        type=str,
        default="data/dual_realsense_recordings",
        help=(
            "Root directory where timestamped dual RealSense recording folders "
            "will be saved. Ignored when --camera-output-dir is set."
        ),
    )
    parser.add_argument(
        "--camera-output-dir",
        type=str,
        default=None,
        help=(
            "Exact directory where this run's videos and metadata will be saved. "
            "If omitted, a timestamped directory is created under --camera-output-root."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=True,
        help="Enable dual RealSense recording. This is enabled by default.",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_false",
        dest="record_video",
        help="Disable video recording.",
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
    parser.add_argument(
        "--log-root",
        type=str,
        default="data/gripper_sequence_logs",
        help="Root directory for timestamped JSON run logs when --run-log is enabled.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Exact directory for JSON run logs when --run-log is enabled. Overrides --log-root.",
    )
    parser.add_argument(
        "--run-log",
        action="store_true",
        help="Write JSON run logs under --log-root or --log-dir.",
    )
    parser.add_argument("--disable-run-log", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


class RunLogger:
    def __init__(self, args):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.log_dir is not None:
            self.output_dir = Path(args.log_dir).expanduser().resolve()
        else:
            self.output_dir = (
                Path(args.log_root).expanduser().resolve()
                / f"hold_ee_gripper_sequence_{timestamp}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._events_file = self.events_path.open("w", encoding="utf-8")
        self._start_monotonic = time.monotonic()
        self._counts = {}
        self._closed = False
        self.summary = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "arm_control": "disabled",
            "gripper_sequence": "open_max_initial -> half_closed -> fully_closed -> open_max_final",
            "args": _jsonable(vars(args)),
            "log_dir": str(self.output_dir),
            "events_path": str(self.events_path),
        }
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
        self.event("run_end", status=status, error=error)
        self.summary.update(
            {
                "status": status,
                "error": error,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": time.monotonic() - self._start_monotonic,
                "event_counts": self._counts,
            }
        )
        self.summary_path.write_text(
            json.dumps(_jsonable(self.summary), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._events_file.close()
        self._closed = True


def make_camera_recorder(args):
    return DualRealSenseVideoRecorder(
        output_root=args.camera_output_root,
        left_serial=args.left_camera_serial,
        right_serial=args.right_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        align_mode=args.camera_align,
        show_preview=args.show_camera_preview,
        output_dir=args.camera_output_dir,
    )


def wait_for_robot_state(robot_interface):
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)


def gripper_width_or_none(robot_interface):
    width = robot_interface.last_gripper_q
    if width is None:
        return None
    return float(np.asarray(width).reshape(-1)[0])


def target_pose_from_matrix(T_base_ee):
    target_pos = np.asarray(T_base_ee[:3, 3], dtype=np.float64).reshape(3, 1)
    target_quat = transform_utils.mat2quat(T_base_ee[:3, :3]).astype(np.float64)
    quat_norm = float(np.linalg.norm(target_quat))
    if quat_norm <= 1e-8:
        raise ValueError("Target rotation produced an invalid quaternion")
    return target_pos, target_quat / quat_norm


def get_pose_error(robot_interface, target_pos, target_quat):
    current_pose = robot_interface.last_eef_pose
    current_pos = current_pose[:3, 3:]
    current_quat = transform_utils.mat2quat(current_pose[:3, :3])
    if np.dot(target_quat, current_quat) < 0.0:
        current_quat = -current_quat
    quat_diff = transform_utils.quat_distance(target_quat, current_quat)
    axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
    return (
        float(np.linalg.norm(target_pos - current_pos)),
        float(np.linalg.norm(axis_angle_diff)),
    )


def get_pose_error_detail(robot_interface, target_pos, target_quat):
    current_pose = robot_interface.last_eef_pose
    current_pos = current_pose[:3, 3:]
    current_quat = transform_utils.mat2quat(current_pose[:3, :3])
    if np.dot(target_quat, current_quat) < 0.0:
        current_quat = -current_quat
    quat_diff = transform_utils.quat_distance(target_quat, current_quat)
    axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
    pos_error_vec = (target_pos - current_pos).flatten()
    return {
        "position_error_m": float(np.linalg.norm(pos_error_vec)),
        "position_error_xyz_m": pos_error_vec.tolist(),
        "orientation_error_rad": float(np.linalg.norm(axis_angle_diff)),
        "current_position_xyz": current_pos.flatten().tolist(),
        "current_quat_xyzw": current_quat.tolist(),
    }


def interpolation_alpha(fraction, mode):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if mode == "linear":
        return fraction
    return 10.0 * fraction**3 - 15.0 * fraction**4 + 6.0 * fraction**5


def hold_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pos,
    target_quat,
    duration_sec,
    should_stop=None,
):
    if duration_sec <= 0.0:
        return
    end_time = time.monotonic() + duration_sec
    while time.monotonic() < end_time:
        if should_stop is not None and should_stop():
            raise RuntimeError("Stop requested while holding target pose")
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_steps=1,
            gripper_action=0.0,
        )


def osc_position_step(
    robot_interface,
    controller_cfg,
    target_pos,
    position_gain=10.0,
    max_action=1.0,
):
    current_pose = robot_interface.last_eef_pose
    current_pos = current_pose[:3, 3:]
    action_pos = np.clip(
        (target_pos - current_pos).flatten() * position_gain,
        -max_action,
        max_action,
    )
    action = action_pos.tolist() + [0.0, 0.0, 0.0, 0.0]
    robot_interface.control(
        controller_type="OSC_POSITION",
        action=action,
        controller_cfg=controller_cfg,
    )


class PoseHoldWorker:
    def __init__(
        self,
        robot_interface,
        controller_type,
        controller_cfg,
        target_pos,
        target_quat,
    ):
        self.robot_interface = robot_interface
        self.controller_type = controller_type
        self.controller_cfg = controller_cfg
        self.target_pos = target_pos
        self.target_quat = target_quat
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                osc_move(
                    self.robot_interface,
                    self.controller_type,
                    self.controller_cfg,
                    (self.target_pos, self.target_quat),
                    num_steps=1,
                    gripper_action=0.0,
                )
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.error is not None:
            raise RuntimeError("Pose hold worker failed") from self.error


def move_to_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pos,
    target_quat,
    args,
    run_logger=None,
):
    start_pose = robot_interface.last_eef_pose.copy()
    start_pos = start_pose[:3, 3:].copy()
    start_quat = transform_utils.mat2quat(start_pose[:3, :3])
    if np.dot(target_quat, start_quat) < 0.0:
        target_quat = -target_quat

    logger.info(
        "Approach starts from xyz=%s quat_xyzw=%s",
        start_pos.flatten().tolist(),
        start_quat.tolist(),
    )
    if run_logger is not None:
        run_logger.event(
            "approach_start",
            start_position_xyz=start_pos.flatten().tolist(),
            start_quat_xyzw=start_quat.tolist(),
            target_position_xyz=target_pos.flatten().tolist(),
            target_quat_xyzw=target_quat.tolist(),
            approach_steps=args.approach_steps,
            interpolation=args.approach_interpolation,
            approach_mode=args.approach_mode,
        )

    if args.approach_mode == "staged":
        position_steps = max(1, int(round(args.approach_steps * 0.55)))
        orientation_steps = max(1, int(round(args.approach_steps * 0.35)))
        refine_steps = max(1, args.approach_steps - position_steps - orientation_steps)
        position_controller_cfg = get_default_controller_config("OSC_POSITION")

        logger.info(
            "Staged approach: position_steps=%d orientation_steps=%d refine_steps=%d",
            position_steps,
            orientation_steps,
            refine_steps,
        )
        move_position_only_to_target(
            robot_interface,
            position_controller_cfg,
            start_pos,
            target_pos,
            args,
            position_steps,
            run_logger=run_logger,
        )
        current_quat = transform_utils.mat2quat(robot_interface.last_eef_pose[:3, :3])
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        move_pose_orientation_to_target(
            robot_interface,
            controller_type,
            controller_cfg,
            target_pos,
            current_quat,
            target_quat,
            args,
            orientation_steps,
            stage_name="orientation",
            run_logger=run_logger,
        )
        refine_pose_to_target(
            robot_interface,
            controller_type,
            controller_cfg,
            target_pos,
            target_quat,
            args,
            refine_steps,
            run_logger=run_logger,
        )
    else:
        move_pose_interp_to_target(
            robot_interface,
            controller_type,
            controller_cfg,
            start_pos,
            start_quat,
            target_pos,
            target_quat,
            args,
            args.approach_steps,
            stage_name="pose_interp",
            run_logger=run_logger,
        )

    hold_target_pose(
        robot_interface,
        controller_type,
        controller_cfg,
        target_pos,
        target_quat,
        args.post_approach_hold_sec,
    )
    error_detail = get_pose_error_detail(robot_interface, target_pos, target_quat)
    pos_error = error_detail["position_error_m"]
    ori_error = error_detail["orientation_error_rad"]
    logger.info(
        "Final target error before recording: pos_error=%.5f m "
        "pos_error_xyz=%s ori_error=%.5f rad",
        pos_error,
        np.round(error_detail["position_error_xyz_m"], 5).tolist(),
        ori_error,
    )
    if run_logger is not None:
        run_logger.event(
            "target_pose_reached",
            **error_detail,
            measured_gripper_width=gripper_width_or_none(robot_interface),
        )
    if (
        not args.no_require_pose_tolerance
        and (
            pos_error > args.position_tolerance
            or ori_error > args.orientation_tolerance
        )
    ):
        raise RuntimeError(
            "Target pose tolerance was not met before recording: "
            f"pos_error={pos_error:.5f} m, ori_error={ori_error:.5f} rad"
        )


def should_log_approach_step(step_idx, total_steps, log_stride):
    return step_idx == total_steps - 1 or (
        log_stride > 0 and step_idx % log_stride == 0
    )


def move_position_only_to_target(
    robot_interface,
    position_controller_cfg,
    start_pos,
    target_pos,
    args,
    num_steps,
    run_logger=None,
):
    interp_steps = max(1, int(round(num_steps * 0.75)))
    for step_idx in range(num_steps):
        if step_idx < interp_steps:
            fraction = float(step_idx + 1) / float(interp_steps)
        else:
            fraction = 1.0
        alpha = interpolation_alpha(fraction, args.approach_interpolation)
        waypoint_pos = start_pos + alpha * (target_pos - start_pos)
        osc_position_step(
            robot_interface,
            position_controller_cfg,
            waypoint_pos,
        )
        if should_log_approach_step(step_idx, num_steps, args.approach_log_stride):
            current_pose = robot_interface.last_eef_pose
            pos_error_vec = (target_pos - current_pose[:3, 3:]).flatten()
            logger.info(
                "Approach[position] step=%d/%d alpha=%.3f pos_error=%.5f m pos_error_xyz=%s",
                step_idx + 1,
                num_steps,
                alpha,
                float(np.linalg.norm(pos_error_vec)),
                np.round(pos_error_vec, 5).tolist(),
            )
            if run_logger is not None:
                run_logger.event(
                    "approach_position_progress",
                    step=step_idx + 1,
                    approach_steps=num_steps,
                    alpha=alpha,
                    position_error_m=float(np.linalg.norm(pos_error_vec)),
                    position_error_xyz_m=pos_error_vec.tolist(),
                    current_position_xyz=current_pose[:3, 3].tolist(),
                )


def move_pose_orientation_to_target(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pos,
    start_quat,
    target_quat,
    args,
    num_steps,
    stage_name,
    run_logger=None,
):
    for step_idx in range(num_steps):
        fraction = float(step_idx + 1) / float(num_steps)
        alpha = interpolation_alpha(fraction, args.approach_interpolation)
        waypoint_quat = transform_utils.quat_slerp(start_quat, target_quat, alpha)
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, waypoint_quat),
            num_steps=1,
            gripper_action=0.0,
        )
        if should_log_approach_step(step_idx, num_steps, args.approach_log_stride):
            error_detail = get_pose_error_detail(robot_interface, target_pos, target_quat)
            logger.info(
                "Approach[%s] step=%d/%d alpha=%.3f pos_error=%.5f m "
                "pos_error_xyz=%s ori_error=%.5f rad",
                stage_name,
                step_idx + 1,
                num_steps,
                alpha,
                error_detail["position_error_m"],
                np.round(error_detail["position_error_xyz_m"], 5).tolist(),
                error_detail["orientation_error_rad"],
            )
            if run_logger is not None:
                run_logger.event(
                    "approach_progress",
                    stage=stage_name,
                    step=step_idx + 1,
                    approach_steps=num_steps,
                    alpha=alpha,
                    **error_detail,
                )


def move_pose_interp_to_target(
    robot_interface,
    controller_type,
    controller_cfg,
    start_pos,
    start_quat,
    target_pos,
    target_quat,
    args,
    num_steps,
    stage_name,
    run_logger=None,
):
    for step_idx in range(num_steps):
        fraction = float(step_idx + 1) / float(num_steps)
        alpha = interpolation_alpha(fraction, args.approach_interpolation)
        waypoint_pos = start_pos + alpha * (target_pos - start_pos)
        waypoint_quat = transform_utils.quat_slerp(start_quat, target_quat, alpha)
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (waypoint_pos, waypoint_quat),
            num_steps=1,
            gripper_action=0.0,
        )
        if should_log_approach_step(step_idx, num_steps, args.approach_log_stride):
            error_detail = get_pose_error_detail(robot_interface, target_pos, target_quat)
            logger.info(
                "Approach[%s] step=%d/%d alpha=%.3f pos_error=%.5f m "
                "pos_error_xyz=%s ori_error=%.5f rad",
                stage_name,
                step_idx + 1,
                num_steps,
                alpha,
                error_detail["position_error_m"],
                np.round(error_detail["position_error_xyz_m"], 5).tolist(),
                error_detail["orientation_error_rad"],
            )
            if run_logger is not None:
                run_logger.event(
                    "approach_progress",
                    stage=stage_name,
                    step=step_idx + 1,
                    approach_steps=num_steps,
                    alpha=alpha,
                    **error_detail,
                )


def refine_pose_to_target(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pos,
    target_quat,
    args,
    num_steps,
    run_logger=None,
):
    for step_idx in range(num_steps):
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_steps=1,
            gripper_action=0.0,
        )
        if should_log_approach_step(step_idx, num_steps, args.approach_log_stride):
            error_detail = get_pose_error_detail(robot_interface, target_pos, target_quat)
            logger.info(
                "Approach[refine] step=%d/%d pos_error=%.5f m pos_error_xyz=%s "
                "ori_error=%.5f rad",
                step_idx + 1,
                num_steps,
                error_detail["position_error_m"],
                np.round(error_detail["position_error_xyz_m"], 5).tolist(),
                error_detail["orientation_error_rad"],
            )
            if run_logger is not None:
                run_logger.event(
                    "approach_progress",
                    stage="refine",
                    step=step_idx + 1,
                    approach_steps=num_steps,
                    **error_detail,
                )


def hold_until_gripper_width_or_timeout(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pos,
    target_quat,
    target_width,
    args,
    should_stop=None,
):
    start_time = time.monotonic()
    last_width = None
    while True:
        now = time.monotonic()
        elapsed = now - start_time
        if should_stop is not None and should_stop():
            raise RuntimeError("Stop requested during gripper sequence")
        if elapsed >= args.gripper_step_timeout_sec:
            return False, last_width, elapsed

        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_steps=1,
            gripper_action=0.0,
        )
        last_width = gripper_width_or_none(robot_interface)
        if (
            last_width is not None
            and abs(last_width - target_width) <= args.gripper_width_tolerance
        ):
            if elapsed >= args.gripper_min_hold_sec:
                return True, last_width, elapsed


def wait_until_gripper_width_or_timeout(
    robot_interface,
    target_width,
    args,
    should_stop=None,
):
    start_time = time.monotonic()
    last_width = None
    while True:
        elapsed = time.monotonic() - start_time
        if should_stop is not None and should_stop():
            raise RuntimeError("Stop requested during gripper sequence")
        if elapsed >= args.gripper_step_timeout_sec:
            return False, last_width, elapsed

        last_width = gripper_width_or_none(robot_interface)
        if (
            last_width is not None
            and abs(last_width - target_width) <= args.gripper_width_tolerance
            and elapsed >= args.gripper_min_hold_sec
        ):
            return True, last_width, elapsed
        time.sleep(0.02)


def hold_gripper_stage(
    robot_interface,
    name,
    target_width,
    args,
    should_stop=None,
    run_logger=None,
):
    start_time = time.monotonic()
    samples = []
    while True:
        elapsed = time.monotonic() - start_time
        if should_stop is not None and should_stop():
            raise RuntimeError("Stop requested during gripper sequence")
        if elapsed >= args.gripper_stage_hold_sec:
            break

        width = gripper_width_or_none(robot_interface)
        if width is not None:
            samples.append(width)
        time.sleep(0.02)

    measured_after = gripper_width_or_none(robot_interface)
    reached = (
        measured_after is not None
        and abs(measured_after - target_width) <= args.gripper_width_tolerance
    )
    elapsed = time.monotonic() - start_time
    logger.info(
        "Gripper stage %s held for %.3f sec: target_width=%.4f reached=%s measured_after=%s",
        name,
        elapsed,
        target_width,
        reached,
        measured_after,
    )
    if run_logger is not None:
        run_logger.event(
            "gripper_stage_held",
            name=name,
            target_width=target_width,
            reached=reached,
            measured_width_after=measured_after,
            elapsed_sec=elapsed,
            measured_width_min=None if not samples else float(np.min(samples)),
            measured_width_max=None if not samples else float(np.max(samples)),
        )
    return reached, measured_after, elapsed


def command_gripper_width(
    robot_interface,
    width,
    name,
    args,
    run_logger=None,
):
    width = float(np.clip(width, 0.0, args.open_width))
    if args.gripper_stop_before_command:
        send_gripper_stop(robot_interface)
        if run_logger is not None:
            run_logger.event("gripper_command", name=name, command="stop")
        if args.gripper_command_delay > 0.0:
            time.sleep(args.gripper_command_delay)

    measured_before = gripper_width_or_none(robot_interface)
    logger.info(
        "Gripper command %s: move width=%.4f speed=%.4f measured_before=%s",
        name,
        width,
        args.gripper_move_speed,
        measured_before,
    )
    if run_logger is not None:
        run_logger.event(
            "gripper_command",
            name=name,
            command="move",
            target_width=width,
            speed=args.gripper_move_speed,
            measured_width_before=measured_before,
        )
    send_gripper_move(robot_interface, width=width, speed=args.gripper_move_speed)
    return width


def run_gripper_sequence(
    robot_interface,
    args,
    should_stop=None,
    run_logger=None,
):
    sequence = [
        ("open_max_initial", args.open_width),
        ("half_closed", args.half_width),
        ("fully_closed", args.closed_width),
        ("open_max_final", args.open_width),
    ]

    for name, width in sequence:
        target_width = command_gripper_width(
            robot_interface,
            width=width,
            name=name,
            args=args,
            run_logger=run_logger,
        )
        reached, measured_after, elapsed = hold_gripper_stage(
            robot_interface,
            name,
            target_width,
            args,
            should_stop=should_stop,
            run_logger=run_logger,
        )
        logger.info(
            "Gripper step %s finished: reached=%s measured_after=%s elapsed=%.3f",
            name,
            reached,
            measured_after,
            elapsed,
        )
        if run_logger is not None:
            run_logger.event(
                "gripper_step_finished",
                name=name,
                target_width=target_width,
                reached=reached,
                measured_width_after=measured_after,
                elapsed_sec=elapsed,
            )
        if args.gripper_stop_before_command:
            send_gripper_stop(robot_interface)
            if run_logger is not None:
                run_logger.event("gripper_command", name=name, command="stop_after_stage")


def validate_args(args):
    if args.open_width <= 0.0 or args.open_width > 0.08:
        raise ValueError("--open-width must be in (0.0, 0.08]")
    if args.half_width is not None and (args.half_width < 0.0 or args.half_width > args.open_width):
        raise ValueError("--half-width must be between 0.0 and --open-width")
    if args.closed_width < 0.0 or args.closed_width > args.open_width:
        raise ValueError("--closed-width must be between 0.0 and --open-width")
    if args.gripper_move_speed <= 0.0:
        raise ValueError("--gripper-move-speed must be positive")
    if args.gripper_stage_hold_sec <= 0.0:
        raise ValueError("--gripper-stage-hold-sec must be positive")
    if args.gripper_step_timeout_sec <= 0.0:
        raise ValueError("--gripper-step-timeout-sec must be positive")
    if args.gripper_min_hold_sec < 0.0:
        raise ValueError("--gripper-min-hold-sec cannot be negative")
    if args.gripper_min_hold_sec > args.gripper_step_timeout_sec:
        raise ValueError("--gripper-min-hold-sec cannot exceed --gripper-step-timeout-sec")
    if args.gripper_width_tolerance <= 0.0:
        raise ValueError("--gripper-width-tolerance must be positive")
    if args.gripper_command_delay < 0.0:
        raise ValueError("--gripper-command-delay cannot be negative")
    if args.pre_gripper_record_delay < 0.0:
        raise ValueError("--pre-gripper-record-delay cannot be negative")


def main():
    args = parse_args()
    validate_args(args)

    run_logger = RunLogger(args) if args.run_log and not args.disable_run_log else None
    robot_interface = None
    camera_recorder = None
    recorder_stop_error = None
    run_status = "completed"
    run_error = None

    try:
        robot_interface = FrankaInterface(
            config_root + f"/{args.interface_cfg}",
            use_visualizer=False,
            has_gripper=False,
            automatic_gripper_reset=False,
        )
        if run_logger is not None:
            run_logger.event(
                "robot_interface_ready",
                has_gripper=robot_interface.has_gripper,
                automatic_gripper_reset=robot_interface.automatic_gripper_reset,
            )

        if args.record_video:
            camera_recorder = make_camera_recorder(args)
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
        else:
            logger.warning("Video recording disabled by --no-record-video")
            if run_logger is not None:
                run_logger.event("camera_recording_disabled")

        logger.info("Arm control is disabled. Executing gripper sequence only.")
        if args.pre_gripper_record_delay > 0.0:
            logger.info(
                "Recording %.3f seconds before first gripper command",
                args.pre_gripper_record_delay,
            )
            time.sleep(args.pre_gripper_record_delay)

        run_gripper_sequence(
            robot_interface,
            args,
            should_stop=None if camera_recorder is None else camera_recorder.should_stop,
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
        if robot_interface is not None:
            robot_interface.close()
            if run_logger is not None:
                run_logger.event("robot_interface_closed")
        if run_logger is not None:
            run_logger.close(status=run_status, error=run_error)

    if recorder_stop_error is not None:
        raise recorder_stop_error
    if camera_recorder is not None:
        camera_recorder.raise_if_failed()


if __name__ == "__main__":
    main()
