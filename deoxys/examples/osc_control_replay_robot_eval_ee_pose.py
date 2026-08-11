"""Replay robot_eval.json trajectories with per-frame EE position and orientation."""

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.proto.franka_interface import franka_controller_pb2
from deoxys.proto.franka_interface import franka_robot_state_pb2
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger

from osc_control_replay_robot_eval import (
    DEFAULT_CAMERA_HIGH_SERIAL,
    DEFAULT_CAMERA_WRIST_SERIAL,
    DualRealSenseVideoRecorder,
    _gripper_action_from_state,
    osc_move,
)

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay EE pose trajectories under OSC_POSE from JSON or NPZ"
    )
    parser.add_argument("--traj-json", type=str, default=None, help="Path to robot_eval.json")
    parser.add_argument(
        "--traj-npz",
        type=str,
        default=None,
        help="Path to franka_ee_trajectory.npz recorded by record_current_franka_ee_trajectory.py",
    )
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--replay-mode",
        choices=("ee-pose", "joint", "observe", "auto"),
        default="ee-pose",
        help=(
            "Replay mode. ee-pose preserves the original OSC_POSE behavior; "
            "joint replays recorded joint_positions; observe records the live "
            "robot state and videos without sending robot commands; auto selects "
            "joint only when the source has no valid EE pose trajectory."
        ),
    )
    parser.add_argument(
        "--observe-fps",
        type=float,
        default=20.0,
        help="State recording frequency in Hz for --replay-mode observe.",
    )
    parser.add_argument(
        "--observe-duration",
        type=float,
        default=None,
        help="Optional maximum recording duration in seconds for observe mode.",
    )
    parser.add_argument(
        "--observe-max-frames",
        type=int,
        default=None,
        help="Optional maximum number of state samples to record in observe mode.",
    )
    parser.add_argument(
        "--observe-print-every",
        type=int,
        default=30,
        help="Log one observe progress line every N samples. Use 0 to disable.",
    )
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument(
        "--joint-controller-type",
        choices=("JOINT_POSITION", "JOINT_IMPEDANCE"),
        default="JOINT_IMPEDANCE",
        help="Joint-space controller used when --replay-mode joint.",
    )
    parser.add_argument(
        "--joint-controller-cfg",
        type=str,
        default="joint-impedance-min-jerk-controller.yml",
        help="Controller YAML used when --replay-mode joint.",
    )
    parser.add_argument(
        "--joint-advance-mode",
        choices=("time", "reached"),
        default="time",
        help=(
            "Joint replay advancement. time sends one command per recorded frame; "
            "reached waits for each target until tolerance or timeout."
        ),
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1e-3,
        help="Max absolute joint error required before advancing in reached joint mode.",
    )
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=3.0,
        help="Maximum seconds to wait for one target in reached joint mode.",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=10.0,
        help="Maximum seconds for the initial move to the first joint target.",
    )
    parser.add_argument(
        "--reset-tolerance",
        type=float,
        default=0.01,
        help="Joint max-absolute-error tolerance for initial joint reset completion.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Replay every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip leading trajectory points")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of points to replay")
    parser.add_argument("--num-steps", type=int, default=5, help="OSC control steps per target point")
    parser.add_argument("--hold-steps", type=int, default=0, help="Additional hold steps at each target")
    parser.add_argument("--open-gripper-action", type=float, default=-1.0)
    parser.add_argument("--closed-gripper-action", type=float, default=1.0)
    parser.add_argument(
        "--ee-to-center",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Optional offset from EE origin to target point in EE frame (meters)",
    )
    parser.add_argument("--skip-reset", action="store_true", help="Do not reset joints before replay")
    parser.add_argument(
        "--gripper-mode",
        choices=("auto", "action", "width-events", "none"),
        default="auto",
        help=(
            "Gripper replay mode. auto uses width-events when gripper_width_m is available, "
            "otherwise action."
        ),
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.06,
        help="Width in meters below which width-events mode sends a grasp command.",
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.07,
        help="Width in meters above which width-events mode sends an open command.",
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
        help="Do not send an initial open/grasp command in width-events mode. This is the default.",
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
        default=None,
        help="Sleep between target points according to timestamp_sec from the trajectory.",
    )
    parser.add_argument(
        "--ignore-timestamps",
        action="store_false",
        dest="respect_timestamps",
        help="Do not sleep according to trajectory timestamps.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiplier for recorded timing when --respect-timestamps is enabled.",
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
        help=(
            "Parent directory for auto-created demoN video folders. If the path "
            "itself is named demoN, that exact directory is used."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Enable dual RealSense recording during trajectory replay",
    )
    parser.add_argument("--camera-high-serial", type=str, default=DEFAULT_CAMERA_HIGH_SERIAL)
    parser.add_argument("--camera-wrist-serial", type=str, default=DEFAULT_CAMERA_WRIST_SERIAL)
    parser.add_argument("--left-camera-serial", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--right-camera-serial", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-align",
        choices=("color", "none"),
        default="color",
        help="Align depth to color for each RealSense pipeline",
    )
    parser.add_argument(
        "--show-camera-preview",
        action="store_true",
        default=False,
        help="Show a live preview window while recording",
    )
    parser.add_argument(
        "--no-camera-preview",
        action="store_false",
        dest="show_camera_preview",
        help="Disable the live preview window",
    )
    return parser.parse_args()


def _is_demo_dir_name(name):
    return name.startswith("demo") and name[4:].isdigit()


def resolve_camera_output_dir(camera_output_root):
    root = Path(camera_output_root).expanduser().resolve()
    if _is_demo_dir_name(root.name):
        root.mkdir(parents=True, exist_ok=True)
        return root

    root.mkdir(parents=True, exist_ok=True)
    next_demo_index = 1
    for child in root.iterdir():
        if child.is_dir() and _is_demo_dir_name(child.name):
            next_demo_index = max(next_demo_index, int(child.name[4:]) + 1)

    while True:
        candidate = root / f"demo{next_demo_index}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            next_demo_index += 1


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


class ReplayRunLogger:
    def __init__(self, args, traj_source, payload, effective_gripper_mode, log_prefix="ee_pose_replay"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.replay_log_dir is not None:
            self.output_dir = Path(args.replay_log_dir).expanduser().resolve()
        else:
            self.output_dir = (
                Path(args.replay_log_root).expanduser().resolve()
                / f"{log_prefix}_{timestamp}"
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
        logger.info("Replay logs will be written to %s", self.output_dir)
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
        logger.removeHandler(self._file_handler)
        self._file_handler.close()
        self._closed = True


class PassiveFrankaStateReader:
    """Subscribe to robot/gripper state without binding control publishers."""

    def __init__(self, general_cfg_file, max_buffer_size=2000):
        general_cfg = YamlConfig(general_cfg_file).as_easydict()
        self._ip = general_cfg.NUC.IP
        self._sub_port = general_cfg.NUC.PUB_PORT
        self._gripper_sub_port = general_cfg.NUC.GRIPPER_PUB_PORT
        self._max_buffer_size = int(max_buffer_size)

        self.has_gripper = False
        self.automatic_gripper_reset = False

        self._context = zmq.Context()
        self._subscriber = self._context.socket(zmq.SUB)
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._subscriber.connect(f"tcp://{self._ip}:{self._sub_port}")

        self._gripper_subscriber = self._context.socket(zmq.SUB)
        self._gripper_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._gripper_subscriber.connect(f"tcp://{self._ip}:{self._gripper_sub_port}")

        self._state_buffer = []
        self._gripper_state_buffer = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._state_sub_thread = threading.Thread(
            target=self._state_loop,
            name="passive_franka_state_sub",
            daemon=True,
        )
        self._gripper_sub_thread = threading.Thread(
            target=self._gripper_state_loop,
            name="passive_franka_gripper_state_sub",
            daemon=True,
        )
        self._state_sub_thread.start()
        self._gripper_sub_thread.start()

    def _append_state(self, buffer, message):
        with self._lock:
            buffer.append(message)
            if len(buffer) > self._max_buffer_size:
                del buffer[: len(buffer) - self._max_buffer_size]

    def _state_loop(self):
        while not self._stop_event.is_set():
            try:
                message = self._subscriber.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            except Exception:
                if not self._stop_event.is_set():
                    logger.debug("Passive state receive failed", exc_info=True)
                    time.sleep(0.01)
                continue

            franka_robot_state = franka_robot_state_pb2.FrankaRobotStateMessage()
            try:
                franka_robot_state.ParseFromString(message)
            except Exception:
                logger.debug("Failed to parse passive robot state", exc_info=True)
                continue
            self._append_state(self._state_buffer, franka_robot_state)

    def _gripper_state_loop(self):
        while not self._stop_event.is_set():
            try:
                message = self._gripper_subscriber.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            except Exception:
                if not self._stop_event.is_set():
                    logger.debug("Passive gripper state receive failed", exc_info=True)
                    time.sleep(0.01)
                continue

            franka_gripper_state = franka_robot_state_pb2.FrankaGripperStateMessage()
            try:
                franka_gripper_state.ParseFromString(message)
            except Exception:
                logger.debug("Failed to parse passive gripper state", exc_info=True)
                continue
            self._append_state(self._gripper_state_buffer, franka_gripper_state)

    def close(self):
        self._stop_event.set()
        self._state_sub_thread.join(1.0)
        self._gripper_sub_thread.join(1.0)
        self._subscriber.close(linger=0)
        self._gripper_subscriber.close(linger=0)
        self._context.term()

    @property
    def state_buffer_size(self):
        with self._lock:
            return len(self._state_buffer)

    @property
    def gripper_state_buffer_size(self):
        with self._lock:
            return len(self._gripper_state_buffer)

    @property
    def last_q(self):
        with self._lock:
            if not self._state_buffer:
                return None
            return np.array(self._state_buffer[-1].q)

    @property
    def last_dq(self):
        with self._lock:
            if not self._state_buffer:
                return None
            return np.array(self._state_buffer[-1].dq)

    @property
    def last_gripper_q(self):
        with self._lock:
            if not self._gripper_state_buffer:
                return None
            return np.array(self._gripper_state_buffer[-1].width)

    @property
    def last_eef_pose(self):
        with self._lock:
            if not self._state_buffer:
                return None
            return np.array(self._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()

    def last_robot_state_snapshot(self):
        with self._lock:
            if not self._state_buffer:
                return None
            state = self._state_buffer[-1]
            q = np.array(state.q)
            dq = np.array(state.dq)
            raw_eef_pose = np.array(state.O_T_EE)
            eef_pose = (
                None
                if raw_eef_pose.size != 16
                else raw_eef_pose.reshape(4, 4).transpose()
            )
        return q, dq, eef_pose


class ReplayStateRecorder:
    def __init__(self, output_dir, args, source_traj_path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "replayed_joint_trajectory.json"
        self.npz_path = self.output_dir / "replayed_joint_trajectory.npz"
        self.args = args
        self.source_traj_path = str(source_traj_path)
        self.records = []

    def _qpos8(self, joints, gripper_width):
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        width = _optional_float(gripper_width)
        if width is None:
            gripper_norm = np.nan
        else:
            gripper_norm = float(np.clip(width / self.args.gripper_open_width, 0.0, 1.0))
        return np.concatenate([joints, np.array([gripper_norm], dtype=np.float64)])

    def _ee_pose_fields(self, eef_pose):
        if eef_pose is None:
            pose = np.full((4, 4), np.nan, dtype=np.float64)
            position = np.full(3, np.nan, dtype=np.float64)
            quat_xyzw = np.full(4, np.nan, dtype=np.float64)
        else:
            pose = np.asarray(eef_pose, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                pose = np.full((4, 4), np.nan, dtype=np.float64)
                position = np.full(3, np.nan, dtype=np.float64)
                quat_xyzw = np.full(4, np.nan, dtype=np.float64)
            else:
                position = pose[:3, 3].astype(np.float64)
                quat_xyzw = transform_utils.mat2quat(pose[:3, :3]).astype(np.float64)
                quat_norm = float(np.linalg.norm(quat_xyzw))
                if quat_norm > 1e-8:
                    quat_xyzw = quat_xyzw / quat_norm
                else:
                    quat_xyzw = np.full(4, np.nan, dtype=np.float64)

        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        return {
            "observed_ee_pose_matrix_after": pose.tolist(),
            "observed_ee_position_after": position.tolist(),
            "observed_ee_quat_xyzw_after": quat_xyzw.tolist(),
            "observed_ee_quat_wxyz_after": quat_wxyz.tolist(),
            "observed_endpose_xyzw_after": np.concatenate([position, quat_xyzw]).tolist(),
            "observed_endpose_wxyz_after": np.concatenate([position, quat_wxyz]).tolist(),
        }

    def record(
        self,
        point_idx,
        point,
        target_q,
        current_q_before,
        current_q_after,
        current_dq_after,
        measured_gripper_before,
        measured_gripper_after,
        gripper_state_memory,
        gripper_action,
        control_steps,
        max_abs_joint_error_before,
        max_abs_joint_error_after,
        current_eef_pose_after=None,
    ):
        timestamp_monotonic_sec = time.monotonic()
        wall_time_ns = time.time_ns()
        target_q = np.asarray(target_q, dtype=np.float64).reshape(7)
        current_q_before = np.asarray(current_q_before, dtype=np.float64).reshape(7)
        current_q_after = np.asarray(current_q_after, dtype=np.float64).reshape(7)
        if current_dq_after is None:
            current_dq_after = np.full(7, np.nan, dtype=np.float64)
        else:
            current_dq_after = np.asarray(current_dq_after, dtype=np.float64).reshape(7)

        recorded_gripper_width = point.get("gripper_width")
        observed_gripper_width_for_qpos = (
            measured_gripper_after
            if measured_gripper_after is not None
            else recorded_gripper_width
        )
        record = {
            "point_idx": int(point_idx),
            "frame_index": int(point["frame_index"]),
            "recorded_timestamp_sec": point.get("timestamp_sec"),
            "replay_monotonic_sec": float(timestamp_monotonic_sec),
            "replay_wall_time_ns": int(wall_time_ns),
            "target_joint_positions": target_q.tolist(),
            "observed_joint_positions_before": current_q_before.tolist(),
            "observed_joint_positions_after": current_q_after.tolist(),
            "observed_joint_velocities_after": current_dq_after.tolist(),
            "target_qpos8": self._qpos8(target_q, recorded_gripper_width).tolist(),
            "observed_qpos8_after": self._qpos8(current_q_after, observed_gripper_width_for_qpos).tolist(),
            "recorded_gripper_width_m": recorded_gripper_width,
            "measured_gripper_width_before_m": measured_gripper_before,
            "measured_gripper_width_after_m": measured_gripper_after,
            "gripper_action": float(gripper_action),
            "gripper_state_memory": gripper_state_memory,
            "control_steps": int(control_steps),
            "max_abs_joint_error_before": float(max_abs_joint_error_before),
            "max_abs_joint_error_after": float(max_abs_joint_error_after),
            **self._ee_pose_fields(current_eef_pose_after),
        }
        self.records.append(record)

    def save(self):
        if not self.records:
            return None

        metadata = {
            "schema": "deoxys_replayed_joint_trajectory_v1",
            "source_traj_path": self.source_traj_path,
            "num_frames": len(self.records),
            "gripper_open_width": self.args.gripper_open_width,
            "ee_pose_source": "FrankaRobotStateMessage.O_T_EE",
            "ee_pose_matrix_convention": "row-major 4x4 homogeneous O_T_EE",
            "ee_pose_endpose_wxyz_convention": "x,y,z,qw,qx,qy,qz",
            "json_path": str(self.json_path),
            "npz_path": str(self.npz_path),
        }
        payload = {**metadata, "frames": self.records}
        self.json_path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        def arr(key, dtype=np.float64):
            return np.array([record[key] for record in self.records], dtype=dtype)

        def optional_float_arr(key):
            values = []
            for record in self.records:
                value = record.get(key)
                values.append(np.nan if value is None else float(value))
            return np.array(values, dtype=np.float64)

        np.savez(
            self.npz_path,
            replay_monotonic_sec=arr("replay_monotonic_sec"),
            replay_wall_time_ns=arr("replay_wall_time_ns", dtype=np.int64),
            point_indices=arr("point_idx", dtype=np.int64),
            frame_indices=arr("frame_index", dtype=np.int64),
            recorded_timestamps_sec=optional_float_arr("recorded_timestamp_sec"),
            target_joint_positions=arr("target_joint_positions"),
            observed_joint_positions_before=arr("observed_joint_positions_before"),
            observed_joint_positions_after=arr("observed_joint_positions_after"),
            observed_joint_velocities_after=arr("observed_joint_velocities_after"),
            observed_ee_pose_matrix_after=arr("observed_ee_pose_matrix_after"),
            observed_ee_position_after=arr("observed_ee_position_after"),
            observed_ee_quat_xyzw_after=arr("observed_ee_quat_xyzw_after"),
            observed_ee_quat_wxyz_after=arr("observed_ee_quat_wxyz_after"),
            observed_endpose_xyzw_after=arr("observed_endpose_xyzw_after"),
            observed_endpose_wxyz_after=arr("observed_endpose_wxyz_after"),
            target_qpos8=arr("target_qpos8"),
            observed_qpos8_after=arr("observed_qpos8_after"),
            recorded_gripper_widths_m=optional_float_arr("recorded_gripper_width_m"),
            measured_gripper_widths_before_m=optional_float_arr("measured_gripper_width_before_m"),
            measured_gripper_widths_after_m=optional_float_arr("measured_gripper_width_after_m"),
            gripper_actions=arr("gripper_action"),
            control_steps=arr("control_steps", dtype=np.int64),
            max_abs_joint_error_before=arr("max_abs_joint_error_before"),
            max_abs_joint_error_after=arr("max_abs_joint_error_after"),
        )
        return self.json_path, self.npz_path


def _valid_vector(values, expected_len):
    if not isinstance(values, list) or len(values) != expected_len:
        return False
    return all(np.isfinite(float(v)) for v in values)


def _optional_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _get_gripper_width(robot_interface):
    width = robot_interface.last_gripper_q
    if width is None:
        return None
    return float(np.asarray(width).reshape(-1)[0])


def _load_frame_orientation(frame_entry):
    quat = frame_entry.get("orientation_quat_xyzw")
    if _valid_vector(quat, 4):
        quat_np = np.array(quat, dtype=np.float64).reshape(4)
        quat_norm = float(np.linalg.norm(quat_np))
        if quat_norm > 1e-8:
            quat_np = quat_np / quat_norm
            rot_np = transform_utils.quat2mat(quat_np)
            return quat_np, rot_np, "orientation_quat_xyzw"

    rotation = frame_entry.get("rotation_matrix")
    rotation_np = np.array(rotation, dtype=np.float64) if rotation is not None else None
    if rotation_np is not None and rotation_np.shape == (3, 3) and np.all(np.isfinite(rotation_np)):
        quat_np = transform_utils.mat2quat(rotation_np).astype(np.float64)
        quat_norm = float(np.linalg.norm(quat_np))
        if quat_norm > 1e-8:
            quat_np = quat_np / quat_norm
            return quat_np, rotation_np, "rotation_matrix"

    return None


def _make_traj_point(
    frame_index,
    position,
    quaternion,
    rotation,
    orientation_source,
    gripper_action=-1.0,
    gripper_width=None,
    timestamp_sec=None,
):
    return {
        "frame_index": int(frame_index),
        "position": np.array(position, dtype=np.float64).reshape(3, 1),
        "quaternion": np.array(quaternion, dtype=np.float64).reshape(4),
        "rotation": np.array(rotation, dtype=np.float64).reshape(3, 3),
        "orientation_source": orientation_source,
        "gripper_action": float(gripper_action),
        "gripper_width": _optional_float(gripper_width),
        "timestamp_sec": _optional_float(timestamp_sec),
    }


def load_robot_eval_ee_pose_traj(
    traj_json,
    stride=1,
    start_frame=0,
    max_frames=None,
    open_gripper_action=-1.0,
    closed_gripper_action=1.0,
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
    skipped_invalid_position = 0
    skipped_invalid_orientation = 0
    for frame in frames:
        pos = frame.get("position_abs_m")
        if not _valid_vector(pos, 3):
            skipped_invalid_position += 1
            continue

        orientation = _load_frame_orientation(frame)
        if orientation is None:
            skipped_invalid_orientation += 1
            continue

        target_quat, target_rot, orientation_source = orientation
        traj.append(
            _make_traj_point(
                frame_index=frame["frame_index"],
                position=pos,
                quaternion=target_quat,
                rotation=target_rot,
                orientation_source=orientation_source,
                gripper_action=_gripper_action_from_state(
                    frame,
                    open_value=open_gripper_action,
                    closed_value=closed_gripper_action,
                ),
                gripper_width=frame.get("gripper_width_m"),
                timestamp_sec=frame.get("timestamp_sec"),
            )
        )

    logger.info(
        "Loaded %d EE pose targets from %s, skipped %d invalid positions and %d invalid orientations",
        len(traj),
        traj_json,
        skipped_invalid_position,
        skipped_invalid_orientation,
    )
    return payload, traj


def load_npz_ee_pose_traj(
    traj_npz,
    stride=1,
    start_frame=0,
    max_frames=None,
):
    data = np.load(traj_npz)
    if "positions_abs_m" in data:
        positions = np.asarray(data["positions_abs_m"], dtype=np.float64)
    elif "T_base_ee" in data:
        positions = np.asarray(data["T_base_ee"], dtype=np.float64)[:, :3, 3]
    else:
        raise RuntimeError("NPZ must contain positions_abs_m or T_base_ee")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise RuntimeError(f"positions_abs_m must have shape (N, 3), got {positions.shape}")
    num_frames = positions.shape[0]

    if "quaternions_xyzw" in data:
        quaternions = np.asarray(data["quaternions_xyzw"], dtype=np.float64)
    elif "rotations" in data:
        quaternions = np.array(
            [transform_utils.mat2quat(rot) for rot in data["rotations"]],
            dtype=np.float64,
        )
    elif "T_base_ee" in data:
        quaternions = np.array(
            [transform_utils.mat2quat(pose[:3, :3]) for pose in data["T_base_ee"]],
            dtype=np.float64,
        )
    else:
        raise RuntimeError("NPZ must contain quaternions_xyzw, rotations, or T_base_ee")

    if quaternions.shape != (num_frames, 4):
        raise RuntimeError(
            f"quaternions_xyzw must have shape ({num_frames}, 4), got {quaternions.shape}"
        )

    if "rotations" in data:
        rotations = np.asarray(data["rotations"], dtype=np.float64)
    else:
        rotations = np.array(
            [transform_utils.quat2mat(quat) for quat in quaternions],
            dtype=np.float64,
        )

    if rotations.shape != (num_frames, 3, 3):
        raise RuntimeError(
            f"rotations must have shape ({num_frames}, 3, 3), got {rotations.shape}"
        )

    timestamps = (
        np.asarray(data["timestamps_sec"], dtype=np.float64)
        if "timestamps_sec" in data
        else np.full(num_frames, np.nan, dtype=np.float64)
    )
    gripper_widths = (
        np.asarray(data["gripper_widths_m"], dtype=np.float64)
        if "gripper_widths_m" in data
        else np.full(num_frames, np.nan, dtype=np.float64)
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
        position = positions[index]
        quat = quaternions[index]
        rot = rotations[index]
        if not (
            np.all(np.isfinite(position))
            and np.all(np.isfinite(quat))
            and np.all(np.isfinite(rot))
        ):
            skipped_invalid += 1
            continue
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-8:
            skipped_invalid += 1
            continue
        quat = quat / quat_norm
        traj.append(
            _make_traj_point(
                frame_index=int(index),
                position=position,
                quaternion=quat,
                rotation=rot,
                orientation_source="npz:quaternions_xyzw",
                gripper_width=gripper_widths[index],
                timestamp_sec=timestamps[index],
            )
        )

    payload = {
        "coordinate_frame": "franka_base",
        "trajectory_type": "ee_pose_npz",
        "num_frames": int(num_frames),
        "npz_keys": sorted(data.files),
        "source_path": str(traj_npz),
    }
    logger.info(
        "Loaded %d EE pose targets from %s, skipped %d invalid rows",
        len(traj),
        traj_npz,
        skipped_invalid,
    )
    return payload, traj


def load_joint_controller_cfg(controller_type, controller_cfg_arg):
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


def send_gripper_move(robot_interface, width, speed=0.1):
    width = float(np.clip(width, 0.0, 0.08))
    gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()
    move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
    move_msg.width = width
    move_msg.speed = float(speed)
    gripper_control_msg.control_msg.Pack(move_msg)
    robot_interface._gripper_publisher.send(gripper_control_msg.SerializeToString())


def send_gripper_grasp(
    robot_interface,
    force,
    speed=0.5,
    width=0.0,
    epsilon_inner=0.08,
    epsilon_outer=0.08,
):
    gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()
    grasp_msg = franka_controller_pb2.FrankaGripperGraspMessage()
    grasp_msg.width = float(width)
    grasp_msg.speed = float(speed)
    grasp_msg.force = float(force)
    grasp_msg.epsilon_inner = float(epsilon_inner)
    grasp_msg.epsilon_outer = float(epsilon_outer)
    gripper_control_msg.control_msg.Pack(grasp_msg)
    robot_interface._gripper_publisher.send(gripper_control_msg.SerializeToString())


def send_gripper_stop(robot_interface):
    gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()
    stop_msg = franka_controller_pb2.FrankaGripperStopMessage()
    stop_msg.stop = True
    gripper_control_msg.control_msg.Pack(stop_msg)
    robot_interface._gripper_publisher.send(gripper_control_msg.SerializeToString())


def should_send_initial_gripper_command(args):
    return bool(args.initial_gripper_command and not args.no_initial_gripper_command)


def resolve_gripper_grasp_width(args, point):
    if args.gripper_grasp_width is not None:
        width = args.gripper_grasp_width
    else:
        width = point.get("gripper_width")
        if width is None:
            width = 0.0
    return float(np.clip(width, 0.0, args.gripper_open_width))


def resolve_gripper_mode(gripper_mode, traj):
    if gripper_mode != "auto":
        return gripper_mode
    has_width = any(point.get("gripper_width") is not None for point in traj)
    return "width-events" if has_width else "action"


def width_to_gripper_state(width, previous_state, close_threshold, open_threshold):
    if width is None:
        return previous_state
    if previous_state == "closed":
        return "open" if width >= open_threshold else "closed"
    if previous_state == "open":
        return "closed" if width <= close_threshold else "open"
    if width <= close_threshold:
        return "closed"
    if width >= open_threshold:
        return "open"
    return "open"


def get_gripper_width_event_plan(traj, args):
    previous_state = None
    events = []
    for point_idx, point in enumerate(traj):
        next_state = width_to_gripper_state(
            point.get("gripper_width"),
            previous_state,
            close_threshold=args.gripper_close_threshold,
            open_threshold=args.gripper_open_threshold,
        )
        if next_state is None:
            continue
        if previous_state is None and point_idx == 0 and not should_send_initial_gripper_command(args):
            previous_state = next_state
            continue
        if next_state != previous_state or (
            point_idx == 0 and should_send_initial_gripper_command(args)
        ):
            if (
                args.hold_closed_after_first_grasp
                and previous_state == "closed"
                and next_state == "open"
            ):
                continue
            events.append(
                {
                    "point_idx": point_idx,
                    "frame_index": point["frame_index"],
                    "timestamp_sec": point.get("timestamp_sec"),
                    "width": point.get("gripper_width"),
                    "state": next_state,
                }
            )
        previous_state = next_state
    return events


def maybe_send_gripper_width_event(
    robot_interface,
    point,
    previous_state,
    args,
    force_initial=False,
    run_logger=None,
):
    next_state = width_to_gripper_state(
        point.get("gripper_width"),
        previous_state,
        close_threshold=args.gripper_close_threshold,
        open_threshold=args.gripper_open_threshold,
    )
    if next_state is None:
        return previous_state
    if previous_state is None and not force_initial:
        return next_state
    if (
        args.hold_closed_after_first_grasp
        and previous_state == "closed"
        and next_state == "open"
    ):
        return "closed"
    if next_state == previous_state and not force_initial:
        return next_state

    if next_state == "closed":
        grasp_width = resolve_gripper_grasp_width(args, point)
        logger.info(
            "Gripper event frame=%s width=%s -> grasp width=%.3f",
            point["frame_index"],
            point.get("gripper_width"),
            grasp_width,
        )
        if run_logger is not None:
            run_logger.event(
                "gripper_event",
                command="grasp",
                frame_index=point["frame_index"],
                timestamp_sec=point.get("timestamp_sec"),
                recorded_width=point.get("gripper_width"),
                previous_state=previous_state,
                next_state=next_state,
                measured_width_before=_get_gripper_width(robot_interface),
            )
        if args.gripper_stop_before_command:
            if run_logger is not None:
                run_logger.event("gripper_command", command="stop", reason="before_grasp")
            send_gripper_stop(robot_interface)
            if args.gripper_command_delay > 0:
                time.sleep(args.gripper_command_delay)
        if run_logger is not None:
            run_logger.event(
                "gripper_command",
                command="grasp",
                target_width=grasp_width,
                force=args.gripper_grasp_force,
                speed=args.gripper_grasp_speed,
                measured_width_before=_get_gripper_width(robot_interface),
            )
        send_gripper_grasp(
            robot_interface,
            force=args.gripper_grasp_force,
            speed=args.gripper_grasp_speed,
            width=grasp_width,
        )
    elif next_state == "open":
        logger.info(
            "Gripper event frame=%s width=%s -> open width=%.3f",
            point["frame_index"],
            point.get("gripper_width"),
            args.gripper_open_width,
        )
        if run_logger is not None:
            run_logger.event(
                "gripper_event",
                command="open",
                frame_index=point["frame_index"],
                timestamp_sec=point.get("timestamp_sec"),
                recorded_width=point.get("gripper_width"),
                previous_state=previous_state,
                next_state=next_state,
                measured_width_before=_get_gripper_width(robot_interface),
            )
        if args.gripper_stop_before_command:
            if run_logger is not None:
                run_logger.event("gripper_command", command="stop", reason="before_open")
            send_gripper_stop(robot_interface)
            if args.gripper_command_delay > 0:
                time.sleep(args.gripper_command_delay)
        if run_logger is not None:
            run_logger.event(
                "gripper_command",
                command="open",
                target_width=args.gripper_open_width,
                speed=args.gripper_move_speed,
                measured_width_before=_get_gripper_width(robot_interface),
            )
        send_gripper_move(
            robot_interface,
            width=args.gripper_open_width,
            speed=args.gripper_move_speed,
        )
    return next_state


def sleep_until_timestamp(point, first_timestamp, replay_start_time, time_scale):
    timestamp = point.get("timestamp_sec")
    if timestamp is None or first_timestamp is None:
        return
    target_time = replay_start_time + (timestamp - first_timestamp) * time_scale
    remaining = target_time - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def follow_robot_eval_ee_pose_traj(
    robot_interface,
    controller_type,
    controller_cfg,
    traj,
    num_steps=5,
    hold_steps=0,
    ee_to_center=None,
    should_stop=None,
    args=None,
    run_logger=None,
):
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)

    ee_to_center = np.array(
        [0.0, 0.0, 0.0] if ee_to_center is None else ee_to_center,
        dtype=np.float64,
    ).reshape(3, 1)

    logger.info("Using trajectory-provided orientation for every replay point")
    logger.info("EE-to-center compensation xyz: %s", ee_to_center.flatten().tolist())
    gripper_mode = resolve_gripper_mode(args.gripper_mode, traj) if args is not None else "action"
    logger.info("Gripper replay mode: %s", gripper_mode)
    if gripper_mode == "width-events":
        logger.info(
            "Width event thresholds: close <= %.4f m, open >= %.4f m",
            args.gripper_close_threshold,
            args.gripper_open_threshold,
        )
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
    if args is not None and args.respect_timestamps:
        logger.info("Respecting trajectory timestamps with time_scale=%.3f", args.time_scale)

    first_timestamp = None
    if args is not None and args.respect_timestamps:
        for point in traj:
            if point.get("timestamp_sec") is not None:
                first_timestamp = point["timestamp_sec"]
                break
    replay_start_time = time.monotonic()

    original_has_gripper = robot_interface.has_gripper
    original_automatic_gripper_reset = robot_interface.automatic_gripper_reset
    if gripper_mode in ("width-events", "none"):
        robot_interface.has_gripper = False
        robot_interface.automatic_gripper_reset = False
    previous_gripper_state = None

    try:
        for point_idx, point in enumerate(traj):
            if should_stop is not None and should_stop():
                logger.warning("Stop requested before trajectory replay completed")
                break

            if args is not None and args.respect_timestamps:
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

            target_center = point["position"]
            target_rot = point["rotation"]
            target_quat = point["quaternion"].copy()
            current_pose_before = robot_interface.last_eef_pose.copy()
            current_quat = transform_utils.mat2quat(current_pose_before[:3, :3])
            if np.dot(target_quat, current_quat) < 0.0:
                target_quat = -target_quat
            target_pos = target_center - target_rot @ ee_to_center
            gripper_action = point["gripper_action"]
            current_pos_before = current_pose_before[:3, 3:]
            pos_error_before = float(np.linalg.norm(target_pos - current_pos_before))
            measured_gripper_before = _get_gripper_width(robot_interface)

            logger.info(
                "Replay frame=%s orientation=%s target_center=%s target_ee=%s target_quat_xyzw=%s gripper_action=%.3f gripper_width=%s timestamp=%s",
                point["frame_index"],
                point["orientation_source"],
                target_center.flatten().tolist(),
                target_pos.flatten().tolist(),
                target_quat.tolist(),
                gripper_action,
                point.get("gripper_width"),
                point.get("timestamp_sec"),
            )
            osc_move(
                robot_interface,
                controller_type,
                controller_cfg,
                (target_pos, target_quat),
                num_steps=num_steps,
                gripper_action=gripper_action,
            )
            if hold_steps > 0:
                osc_move(
                    robot_interface,
                    controller_type,
                    controller_cfg,
                    (target_pos, target_quat),
                    num_steps=hold_steps,
                    gripper_action=gripper_action,
                )
            if (
                run_logger is not None
                and args is not None
                and args.log_frame_stride > 0
                and point_idx % args.log_frame_stride == 0
            ):
                current_pose_after = robot_interface.last_eef_pose.copy()
                current_pos_after = current_pose_after[:3, 3:]
                pos_error_after = float(np.linalg.norm(target_pos - current_pos_after))
                current_quat_after = transform_utils.mat2quat(
                    current_pose_after[:3, :3]
                )
                run_logger.event(
                    "frame_executed",
                    point_idx=point_idx,
                    frame_index=point["frame_index"],
                    timestamp_sec=point.get("timestamp_sec"),
                    target_pos=target_pos.flatten().tolist(),
                    target_quat=target_quat.tolist(),
                    current_pos_before=current_pos_before.flatten().tolist(),
                    current_pos_after=current_pos_after.flatten().tolist(),
                    current_quat_after=current_quat_after.tolist(),
                    pos_error_before=pos_error_before,
                    pos_error_after=pos_error_after,
                    gripper_action=gripper_action,
                    recorded_gripper_width=point.get("gripper_width"),
                    measured_gripper_width_before=measured_gripper_before,
                    measured_gripper_width_after=_get_gripper_width(robot_interface),
                    gripper_state_memory=previous_gripper_state,
                )
    finally:
        robot_interface.has_gripper = original_has_gripper
        robot_interface.automatic_gripper_reset = original_automatic_gripper_reset


def wait_for_robot_state(robot_interface):
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)


def observe_current_robot_state(
    robot_interface,
    should_stop=None,
    args=None,
    run_logger=None,
    state_recorder=None,
):
    if args is None:
        raise ValueError("observe_current_robot_state requires args")
    if state_recorder is None:
        raise ValueError("observe_current_robot_state requires state_recorder")

    wait_for_robot_state(robot_interface)

    sample_interval = 1.0 / args.observe_fps
    start_time = time.monotonic()
    next_sample_time = start_time
    missing_gripper_widths = 0
    missing_eef_poses = 0

    logger.info(
        "Observe-only recording started at %.3f Hz; no robot commands will be sent",
        args.observe_fps,
    )
    logger.info("Press Ctrl+C to stop observe recording and save outputs")
    if run_logger is not None:
        run_logger.event(
            "observe_recording_started",
            observe_fps=args.observe_fps,
            observe_duration=args.observe_duration,
            observe_max_frames=args.observe_max_frames,
        )

    try:
        while True:
            if should_stop is not None and should_stop():
                logger.warning("Stop requested before observe recording completed")
                if run_logger is not None:
                    run_logger.event("observe_stop_requested", reason="camera_recorder")
                break

            now = time.monotonic()
            elapsed = now - start_time
            if args.observe_duration is not None and elapsed >= args.observe_duration:
                if run_logger is not None:
                    run_logger.event("observe_stop_requested", reason="duration")
                break
            if (
                args.observe_max_frames is not None
                and len(state_recorder.records) >= args.observe_max_frames
            ):
                if run_logger is not None:
                    run_logger.event("observe_stop_requested", reason="max_frames")
                break

            if now < next_sample_time:
                time.sleep(min(next_sample_time - now, sample_interval))
                continue

            snapshot = robot_interface.last_robot_state_snapshot()
            if snapshot is None:
                next_sample_time += sample_interval
                continue
            current_q, current_dq, current_eef_pose = snapshot
            current_q = np.asarray(current_q, dtype=np.float64).reshape(7)
            if not np.all(np.isfinite(current_q)):
                logger.warning("Skipping non-finite observed joint state")
                next_sample_time += sample_interval
                continue

            current_dq = (
                None
                if current_dq is None
                else np.asarray(current_dq, dtype=np.float64).reshape(7)
            )
            if current_eef_pose is None:
                missing_eef_poses += 1
            else:
                current_eef_pose = np.asarray(current_eef_pose, dtype=np.float64).reshape(4, 4)
                if not np.all(np.isfinite(current_eef_pose)):
                    missing_eef_poses += 1
                    current_eef_pose = None
            measured_gripper_width = _get_gripper_width(robot_interface)
            if measured_gripper_width is None:
                missing_gripper_widths += 1
            gripper_width_for_qpos = (
                measured_gripper_width
                if measured_gripper_width is not None
                else args.gripper_open_width
            )
            point_idx = len(state_recorder.records)
            sample_time = time.monotonic()
            point = {
                "frame_index": int(point_idx),
                "timestamp_sec": float(sample_time - start_time),
                "gripper_width": gripper_width_for_qpos,
            }

            state_recorder.record(
                point_idx=point_idx,
                point=point,
                target_q=current_q,
                current_q_before=current_q,
                current_q_after=current_q,
                current_dq_after=current_dq,
                measured_gripper_before=measured_gripper_width,
                measured_gripper_after=measured_gripper_width,
                gripper_state_memory=None,
                gripper_action=0.0,
                control_steps=0,
                max_abs_joint_error_before=0.0,
                max_abs_joint_error_after=0.0,
                current_eef_pose_after=current_eef_pose,
            )

            if (
                run_logger is not None
                and args.log_frame_stride > 0
                and point_idx % args.log_frame_stride == 0
            ):
                run_logger.event(
                    "observe_frame_recorded",
                    point_idx=point_idx,
                    frame_index=point["frame_index"],
                    timestamp_sec=point["timestamp_sec"],
                    observed_joint_positions=current_q.tolist(),
                    observed_joint_velocities=None
                    if current_dq is None
                    else current_dq.tolist(),
                    measured_gripper_width=measured_gripper_width,
                    gripper_width_for_qpos=gripper_width_for_qpos,
                    observed_ee_pose_matrix=None
                    if current_eef_pose is None
                    else current_eef_pose.tolist(),
                )

            if (
                args.observe_print_every > 0
                and len(state_recorder.records) % args.observe_print_every == 0
            ):
                logger.info(
                    "Observe frames=%d elapsed=%.3fs q=%s gripper_width=%s",
                    len(state_recorder.records),
                    point["timestamp_sec"],
                    np.round(current_q, 4).tolist(),
                    measured_gripper_width,
                )

            next_sample_time += sample_interval
            if next_sample_time < now - sample_interval:
                next_sample_time = now + sample_interval
    except KeyboardInterrupt:
        logger.info("Observe recording stopped by Ctrl+C")
        if run_logger is not None:
            run_logger.event("observe_stop_requested", reason="keyboard_interrupt")

    if missing_gripper_widths > 0:
        logger.warning(
            "Gripper width was unavailable for %d observe samples; used open width %.4f for qpos8",
            missing_gripper_widths,
            args.gripper_open_width,
        )
        if run_logger is not None:
            run_logger.event(
                "observe_missing_gripper_widths",
                count=missing_gripper_widths,
                fallback_width=args.gripper_open_width,
            )
    if missing_eef_poses > 0:
        logger.warning(
            "EE pose was unavailable for %d observe samples; saved NaN EE pose fields",
            missing_eef_poses,
        )
        if run_logger is not None:
            run_logger.event(
                "observe_missing_eef_poses",
                count=missing_eef_poses,
            )


def move_to_initial_joint_target(
    robot_interface,
    target_q,
    gripper_action,
    timeout,
    tolerance,
    run_logger=None,
):
    controller_cfg = load_joint_controller_cfg("JOINT_POSITION", "joint-position-controller.yml")
    target_q = np.asarray(target_q, dtype=np.float64).reshape(7)
    action = target_q.tolist() + [float(gripper_action)]
    start_time = time.monotonic()

    logger.info("Resetting to first recorded joint target: %s", target_q.tolist())
    while True:
        wait_for_robot_state(robot_interface)
        current_q = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
        max_abs_error = float(np.max(np.abs(current_q - target_q)))
        if max_abs_error <= tolerance:
            logger.info("Initial joint reset reached tolerance %.4f", max_abs_error)
            if run_logger is not None:
                run_logger.event(
                    "reset_completed",
                    target_joint_positions=target_q.tolist(),
                    max_abs_joint_error=max_abs_error,
                )
            return
        if time.monotonic() - start_time > timeout:
            logger.warning(
                "Initial joint reset timed out with max_abs_joint_error=%.4f",
                max_abs_error,
            )
            if run_logger is not None:
                run_logger.event(
                    "reset_timeout",
                    target_joint_positions=target_q.tolist(),
                    max_abs_joint_error=max_abs_error,
                    timeout=timeout,
                )
            return
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
        )


def follow_robot_eval_joint_traj(
    robot_interface,
    controller_type,
    controller_cfg,
    traj,
    hold_steps=0,
    should_stop=None,
    args=None,
    run_logger=None,
    state_recorder=None,
):
    wait_for_robot_state(robot_interface)

    gripper_mode = resolve_gripper_mode(args.gripper_mode, traj) if args is not None else "action"
    logger.info("Joint replay controller: %s", controller_type)
    logger.info("Joint replay advance mode: %s", args.joint_advance_mode)
    logger.info("Gripper replay mode: %s", gripper_mode)
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
    if args is not None and args.respect_timestamps and args.joint_advance_mode == "time":
        for point in traj:
            if point.get("timestamp_sec") is not None:
                first_timestamp = point["timestamp_sec"]
                break
        logger.info("Respecting trajectory timestamps with time_scale=%.3f", args.time_scale)
    elif args is not None and args.respect_timestamps and args.joint_advance_mode == "reached":
        logger.info("Ignoring --respect-timestamps because reached mode waits for each target.")
    replay_start_time = time.monotonic()

    original_has_gripper = robot_interface.has_gripper
    original_automatic_gripper_reset = robot_interface.automatic_gripper_reset
    if gripper_mode in ("width-events", "none"):
        robot_interface.has_gripper = False
        robot_interface.automatic_gripper_reset = False

    previous_gripper_state = None
    try:
        for point_idx, point in enumerate(traj):
            if should_stop is not None and should_stop():
                logger.warning("Stop requested before joint trajectory replay completed")
                break

            if (
                args is not None
                and args.respect_timestamps
                and args.joint_advance_mode == "time"
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

            if args is not None and args.joint_advance_mode == "reached":
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

            current_q_after = np.asarray(robot_interface.last_q, dtype=np.float64).reshape(7)
            current_dq_after = (
                None
                if robot_interface.last_dq is None
                else np.asarray(robot_interface.last_dq, dtype=np.float64).reshape(7)
            )
            current_eef_pose_after = robot_interface.last_eef_pose
            if current_eef_pose_after is not None:
                current_eef_pose_after = np.asarray(
                    current_eef_pose_after,
                    dtype=np.float64,
                ).reshape(4, 4)
            joint_error_after = float(np.max(np.abs(target_q - current_q_after)))
            measured_gripper_after = _get_gripper_width(robot_interface)

            if state_recorder is not None:
                state_recorder.record(
                    point_idx=point_idx,
                    point=point,
                    target_q=target_q,
                    current_q_before=current_q_before,
                    current_q_after=current_q_after,
                    current_dq_after=current_dq_after,
                    measured_gripper_before=measured_gripper_before,
                    measured_gripper_after=measured_gripper_after,
                    gripper_state_memory=previous_gripper_state,
                    gripper_action=gripper_action,
                    control_steps=control_steps,
                    max_abs_joint_error_before=joint_error_before,
                    max_abs_joint_error_after=joint_error_after,
                    current_eef_pose_after=current_eef_pose_after,
                )

            if (
                run_logger is not None
                and args is not None
                and args.log_frame_stride > 0
                and point_idx % args.log_frame_stride == 0
            ):
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
                    measured_gripper_width_after=measured_gripper_after,
                    gripper_state_memory=previous_gripper_state,
                )
    finally:
        robot_interface.has_gripper = original_has_gripper
        robot_interface.automatic_gripper_reset = original_automatic_gripper_reset


def main():
    args = parse_args()

    observe_mode = args.replay_mode == "observe"
    if observe_mode:
        if args.traj_json is not None or args.traj_npz is not None:
            raise ValueError("--replay-mode observe does not take --traj-json or --traj-npz")
    elif (args.traj_json is None) == (args.traj_npz is None):
        raise ValueError("Specify exactly one of --traj-json or --traj-npz")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.hold_steps < 0:
        raise ValueError("--hold-steps cannot be negative")
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    if args.target_tolerance <= 0:
        raise ValueError("--target-tolerance must be positive")
    if args.target_timeout <= 0:
        raise ValueError("--target-timeout must be positive")
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
    if args.observe_fps <= 0:
        raise ValueError("--observe-fps must be positive")
    if args.observe_duration is not None and args.observe_duration <= 0:
        raise ValueError("--observe-duration must be positive when provided")
    if args.observe_max_frames is not None and args.observe_max_frames <= 0:
        raise ValueError("--observe-max-frames must be positive when provided")
    if args.observe_print_every < 0:
        raise ValueError("--observe-print-every cannot be negative")

    selected_replay_mode = args.replay_mode
    payload = None
    traj = None
    ee_load_error = None
    if observe_mode:
        traj_source = "observe://current_franka_state"
        payload = {
            "schema": "deoxys_observe_only_recording_v1",
            "coordinate_frame": "franka_joint_space",
            "trajectory_type": "observed_joint_positions",
            "stereo_mode": "dual_realsense",
            "observe_fps": args.observe_fps,
            "observe_duration": args.observe_duration,
            "observe_max_frames": args.observe_max_frames,
        }
        selected_replay_mode = "observe"
    else:
        traj_source = args.traj_npz if args.traj_npz is not None else args.traj_json

    if selected_replay_mode in ("ee-pose", "auto"):
        try:
            if args.traj_npz is not None:
                payload, traj = load_npz_ee_pose_traj(
                    traj_npz=args.traj_npz,
                    stride=args.stride,
                    start_frame=args.start_frame,
                    max_frames=args.max_frames,
                )
            else:
                payload, traj = load_robot_eval_ee_pose_traj(
                    traj_json=args.traj_json,
                    stride=args.stride,
                    start_frame=args.start_frame,
                    max_frames=args.max_frames,
                    open_gripper_action=args.open_gripper_action,
                    closed_gripper_action=args.closed_gripper_action,
                )
            if traj:
                selected_replay_mode = "ee-pose"
        except Exception as exc:
            ee_load_error = exc
            if selected_replay_mode == "ee-pose":
                raise

    if selected_replay_mode == "joint" or (args.replay_mode == "auto" and not traj):
        if args.traj_npz is not None:
            payload, traj = load_npz_joint_traj(
                traj_npz=args.traj_npz,
                args=args,
                stride=args.stride,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
            )
        else:
            payload, traj = load_json_joint_traj(
                traj_json=args.traj_json,
                args=args,
                stride=args.stride,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
            )
        selected_replay_mode = "joint"

    if selected_replay_mode != "observe" and not traj:
        if ee_load_error is not None:
            raise RuntimeError("No valid trajectory points found") from ee_load_error
        raise RuntimeError("No valid trajectory points found")
    if args.respect_timestamps is None:
        args.respect_timestamps = selected_replay_mode == "joint"
    if selected_replay_mode == "observe":
        effective_gripper_mode = "none"
    else:
        effective_gripper_mode = resolve_gripper_mode(args.gripper_mode, traj)
    active_controller_type = (
        args.joint_controller_type
        if selected_replay_mode == "joint"
        else ("PASSIVE_OBSERVE" if selected_replay_mode == "observe" else args.controller_type)
    )

    if selected_replay_mode == "observe":
        logger.info(
            "Observe-only mode configured from %s (coordinate_frame=%s, stereo_mode=%s)",
            traj_source,
            payload.get("coordinate_frame"),
            payload.get("stereo_mode"),
        )
        if not args.record_video:
            logger.warning(
                "--replay-mode observe was started without --record-video; only joint states will be saved"
            )
    else:
        logger.info(
            "Loaded %d %s trajectory points from %s (coordinate_frame=%s, stereo_mode=%s)",
            len(traj),
            selected_replay_mode,
            traj_source,
            payload.get("coordinate_frame"),
            payload.get("stereo_mode"),
        )

    resolved_camera_output_dir = None
    if args.record_video:
        resolved_camera_output_dir = resolve_camera_output_dir(args.camera_output_root)
        args.resolved_camera_output_dir = str(resolved_camera_output_dir)
        logger.info("Camera recording output_dir=%s", resolved_camera_output_dir)

    run_logger = None
    if not args.disable_replay_log:
        run_logger = ReplayRunLogger(
            args=args,
            traj_source=traj_source,
            payload=payload,
            effective_gripper_mode=effective_gripper_mode,
            log_prefix=(
                "joint_replay"
                if selected_replay_mode == "joint"
                else ("observe_recording" if selected_replay_mode == "observe" else "ee_pose_replay")
            ),
        )
        run_logger.event(
            "trajectory_loaded",
            num_points=0 if traj is None else len(traj),
            traj_source=traj_source,
            replay_mode=selected_replay_mode,
            controller_type=active_controller_type,
            effective_gripper_mode=effective_gripper_mode,
        )

    robot_interface = None
    use_action_gripper = effective_gripper_mode not in ("width-events", "none")
    interface_cfg_path = Path(args.interface_cfg).expanduser()
    if not interface_cfg_path.is_absolute():
        interface_cfg_path = Path(config_root) / args.interface_cfg
    if selected_replay_mode == "observe":
        robot_interface = PassiveFrankaStateReader(str(interface_cfg_path))
    else:
        robot_interface = FrankaInterface(
            str(interface_cfg_path),
            use_visualizer=False,
            has_gripper=use_action_gripper,
            automatic_gripper_reset=use_action_gripper,
        )
    if effective_gripper_mode in ("width-events", "none"):
        if run_logger is not None:
            run_logger.event(
                "robot_interface_config",
                has_gripper=robot_interface.has_gripper,
                automatic_gripper_reset=robot_interface.automatic_gripper_reset,
            )
    if selected_replay_mode == "joint":
        controller_cfg = load_joint_controller_cfg(
            args.joint_controller_type,
            args.joint_controller_cfg,
        )
    elif selected_replay_mode == "observe":
        controller_cfg = None
    else:
        controller_cfg = get_default_controller_config(args.controller_type)
    camera_recorder = None
    if args.record_video:
        camera_high_serial = args.left_camera_serial or args.camera_high_serial
        camera_wrist_serial = args.right_camera_serial or args.camera_wrist_serial
        camera_recorder = DualRealSenseVideoRecorder(
            output_root=args.camera_output_root,
            camera_high_serial=camera_high_serial,
            camera_wrist_serial=camera_wrist_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            align_mode=args.camera_align,
            show_preview=args.show_camera_preview,
            output_dir=resolved_camera_output_dir,
        )
    state_recorder = None
    if selected_replay_mode in ("joint", "observe"):
        if camera_recorder is not None:
            state_recording_dir = camera_recorder.output_dir
        elif run_logger is not None:
            state_recording_dir = run_logger.output_dir
        elif selected_replay_mode == "observe":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_recording_dir = (
                Path(args.camera_output_root).expanduser().resolve()
                / f"observe_state_{timestamp}"
            )
        else:
            state_recording_dir = Path(traj_source).expanduser().resolve().parent
        state_recorder = ReplayStateRecorder(
            output_dir=state_recording_dir,
            args=args,
            source_traj_path=traj_source,
        )
        if run_logger is not None:
            run_logger.event(
                "state_recording_started",
                output_dir=str(state_recorder.output_dir),
                json_path=str(state_recorder.json_path),
                npz_path=str(state_recorder.npz_path),
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
                    camera_high_video_path=str(camera_recorder.camera_high_video_path),
                    camera_wrist_video_path=str(camera_recorder.camera_wrist_video_path),
                    metadata_path=str(camera_recorder.metadata_path),
                    camera_high_serial=camera_recorder.camera_high_serial,
                    camera_wrist_serial=camera_recorder.camera_wrist_serial,
                    width=camera_recorder.width,
                    height=camera_recorder.height,
                    fps=camera_recorder.fps,
                    align_mode=camera_recorder.align_mode,
                    show_preview=camera_recorder.show_preview,
                )
        if selected_replay_mode == "observe":
            if run_logger is not None:
                run_logger.event("reset_skipped", reason="observe_mode")
        elif not args.skip_reset and selected_replay_mode == "joint":
            move_to_initial_joint_target(
                robot_interface,
                target_q=traj[0]["joint_positions"],
                gripper_action=traj[0]["gripper_action"],
                timeout=args.reset_timeout,
                tolerance=args.reset_tolerance,
                run_logger=run_logger,
            )
        elif not args.skip_reset:
            reset_joint_positions = [
                0.09162008114028396,
                -0.19826458111314524,
                -0.01990020486871322,
                -2.4732269941140346,
                -0.01307073642274261,
                2.30396583422025,
                0.8480939705504309,
            ]
            reset_gripper_open = False
            reset_joints_to(
                robot_interface,
                reset_joint_positions,
                gripper_open=reset_gripper_open,
            )
            if run_logger is not None:
                run_logger.event(
                    "reset_completed",
                    reset_joint_positions=reset_joint_positions,
                    gripper_open=reset_gripper_open,
                )
        elif run_logger is not None:
            run_logger.event("reset_skipped")

        if selected_replay_mode == "joint":
            follow_robot_eval_joint_traj(
                robot_interface,
                args.joint_controller_type,
                controller_cfg,
                traj=traj,
                hold_steps=args.hold_steps,
                should_stop=None if camera_recorder is None else camera_recorder.should_stop,
                args=args,
                run_logger=run_logger,
                state_recorder=state_recorder,
            )
        elif selected_replay_mode == "observe":
            observe_current_robot_state(
                robot_interface,
                should_stop=None if camera_recorder is None else camera_recorder.should_stop,
                args=args,
                run_logger=run_logger,
                state_recorder=state_recorder,
            )
        else:
            follow_robot_eval_ee_pose_traj(
                robot_interface,
                args.controller_type,
                controller_cfg,
                traj=traj,
                num_steps=args.num_steps,
                hold_steps=args.hold_steps,
                ee_to_center=args.ee_to_center,
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
                        camera_high_video_path=str(camera_recorder.camera_high_video_path),
                        camera_wrist_video_path=str(camera_recorder.camera_wrist_video_path),
                        metadata_path=str(camera_recorder.metadata_path),
                        camera_high_frame_count=camera_recorder.left_frame_count,
                        camera_wrist_frame_count=camera_recorder.right_frame_count,
                    )
            except Exception as exc:
                recorder_stop_error = exc
                run_status = "error"
                run_error = repr(exc)
                if run_logger is not None:
                    run_logger.event("camera_recording_stop_error", error=run_error)
        if state_recorder is not None:
            saved_paths = state_recorder.save()
            if saved_paths is not None and run_logger is not None:
                json_path, npz_path = saved_paths
                run_logger.event(
                    "state_recording_saved",
                    json_path=str(json_path),
                    npz_path=str(npz_path),
                    num_frames=len(state_recorder.records),
                )
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
