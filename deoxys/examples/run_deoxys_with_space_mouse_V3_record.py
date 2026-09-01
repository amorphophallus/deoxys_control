"""SpaceMouse teleoperation with FurnitureBench-compatible raw data capture."""

import argparse
import concurrent.futures
import os
import pickle
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.furniture_bench_utils import (
    DEFAULT_FRONT_SERIAL,
    DEFAULT_WRIST_SERIAL,
    DualRealSenseSnapshotter,
    deoxys_delta_to_furniture_bench_action,
    eepose_from_wrist_pose,
    resolve_eepose_frame,
    wrist_pose_to_tip_pose,
)
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.machine_time import (
    MachineTimeSchedule,
    monotonic_target_to_wall_time_ns,
)
from deoxys.utils.prompt_depth_anything import (
    PromptDepthAnythingEstimator,
    PromptDepthWorker,
    colorize_depth,
    depth_display_bounds,
)
from deoxys.utils.panda_kinematics import PandaKinematics
from deoxys.utils.video_utils import H264VideoWriter


logger = get_deoxys_example_logger()

# Original reset joint positions (original).
# RESET_JOINT_POSITIONS = [
#     0.09162008114028396,
#     -0.19826458111314524,
#     -0.01990020486871322,
#     -2.4732269941140346,
#     -0.01307073642274261,
#     2.30396583422025,
#     0.8480939705504309,
# ]

# Reset pose with the end effector lowered 10 cm along the robot-base -Z axis
# while preserving the original XY position and orientation (下降 10 cm).
RESET_JOINT_POSITIONS = [
    0.0916502534874562,
    0.006205358472252432,
    -0.02085815329544379,
    -2.552429972459778,
    -0.010695882435351968,
    2.587622772050635,
    0.8472435743003388,
]

PREVIEW_WINDOW_NAME = "FurnitureBench SpaceMouse data collection"
TASK_PART_NAMES = {
    "one_leg": {0: "tabletop", 4: "movable_leg"},
    "round_table": {
        0: "round_table_top",
        1: "round_table_leg",
        2: "round_table_base",
    },
    "lamp": {
        0: "lamp_base",
        1: "lamp_bulb",
        2: "lamp_hood",
    },
}
PROMPT_DEPTH_FIELDS = {
    "wrist": "depth_image1",
    "front": "depth_image2",
}
SPACEMOUSE_PRODUCT_IDS = {
    "wireless": 50770,  # 0xc652 Universal Receiver
    "wired": 50746,  # 0xc63a wired connection
}
REAL_ANNOTATION_SOURCE = "real_skill_annotation_util"
REAL_ANNOTATION_OUTPUT_FIELDS = (
    "skill_state",
    "assembly_step",
    "guidance_point",
    "guidance_point_clean",
    "guidance_pose",
    "guidance_pose_clean",
    "guidance_gripper_width",
    "guidance_point_2d",
    "grasp_annotation_2d",
    "real_annotation_debug",
)
RAW_EPISODE_GLOB = "????-??-??T??-??-??.??????.pkl"
PANDA_KINEMATICS = PandaKinematics()


def _create_real_skill_annotation_session(task_name, camera_info):
    try:
        from src.eval.real_skill_annotation_util import (
            RealSkillAnnotationSession,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Could not import robust-rearrangement real annotation util. "
            "Run `source ~/.bashrc` and make sure "
            "/home/hz/code/robust-rearrangement-custom is on PYTHONPATH."
        ) from exc
    return RealSkillAnnotationSession(task_name, camera_info, mode="online")


def _clear_real_skill_annotation(observation):
    observation["skill"] = None
    observation["guidance"] = None
    for key in REAL_ANNOTATION_OUTPUT_FIELDS:
        observation.pop(key, None)


class NonBlockingKeyReader:
    def __init__(self):
        self._fd = None
        self._old_settings = None
        self.enabled = False

    def __enter__(self):
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY; keyboard controls are disabled")
            return self
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self.enabled = False

    def read_keys(self):
        keys = []
        if not self.enabled:
            return keys
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if not char:
                break
            keys.append(char.lower())
        return keys


def _array_field(message, *names, size):
    for name in names:
        if hasattr(message, name):
            value = np.asarray(getattr(message, name), dtype=np.float64).reshape(-1)
            if value.size == size:
                return value
    return np.full(size, np.nan, dtype=np.float64)


def _message_time_seconds(message):
    value = getattr(message, "time", None)
    if value is None:
        return None
    if hasattr(value, "toSec"):
        return float(value.toSec)
    return float(value)


def _gripper_width(robot_interface):
    value = robot_interface.last_gripper_q
    if value is None:
        return float("nan")
    return float(np.asarray(value).reshape(-1)[0])


def build_observation(robot_interface, camera_sample, eepose_frame="robot-base"):
    if not robot_interface._state_buffer or camera_sample is None:
        return None

    timestamped_robot_states = robot_interface.timestamped_robot_state_buffer(
        max_records=1
    )
    timestamped_gripper_states = robot_interface.timestamped_gripper_state_buffer(
        max_records=1
    )
    if not timestamped_robot_states:
        return None
    robot_record = timestamped_robot_states[-1]
    gripper_record = (
        timestamped_gripper_states[-1] if timestamped_gripper_states else None
    )
    state = robot_record["message"]
    raw_pose = np.asarray(state.O_T_EE, dtype=np.float64)
    if raw_pose.size != 16:
        return None
    wrist_pose = raw_pose.reshape(4, 4).transpose()
    tip_pose = wrist_pose_to_tip_pose(wrist_pose)
    ee_pose = eepose_from_wrist_pose(wrist_pose, eepose_frame)
    ee_quaternion = transform_utils.mat2quat(ee_pose[:3, :3])
    tip_quaternion = transform_utils.mat2quat(tip_pose[:3, :3])
    joint_positions = _array_field(state, "q", size=7)
    joint_velocities = _array_field(state, "dq", size=7)
    if not (
        np.all(np.isfinite(joint_positions))
        and np.all(np.isfinite(joint_velocities))
    ):
        return None
    ee_velocity = PANDA_KINEMATICS.ee_twist(
        joint_positions,
        joint_velocities,
        wrist_pose[:3, 3],
    )
    tip_offset_in_base = tip_pose[:3, 3] - wrist_pose[:3, 3]
    tip_linear_velocity = ee_velocity[:3] + np.cross(
        ee_velocity[3:],
        tip_offset_in_base,
    )
    ee_linear_velocity = (
        ee_velocity[:3]
        if resolve_eepose_frame(eepose_frame) == "robot-base"
        else tip_linear_velocity
    )

    observation = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in camera_sample.items()
    }
    observation.update(
        {
            "control_wall_time_ns": time.time_ns(),
            "observation_ready_wall_time_ns": time.time_ns(),
            "robot_state_receive_wall_time_ns": robot_record[
                "receive_wall_time_ns"
            ],
            "robot_state_source_time": _message_time_seconds(state),
            "robot_state_frame": getattr(state, "frame", None),
            "gripper_state_receive_wall_time_ns": (
                None
                if gripper_record is None
                else gripper_record["receive_wall_time_ns"]
            ),
            "gripper_state_source_time": (
                None
                if gripper_record is None
                else _message_time_seconds(gripper_record["message"])
            ),
            "robot_state": {
                "ee_pos": ee_pose[:3, 3].copy(),
                "ee_quat": ee_quaternion.copy(),
                "ee_pose": ee_pose.copy(),
                "ee_pos_original": tip_pose[:3, 3].copy(),
                "ee_quat_original": tip_quaternion.copy(),
                "ee_pose_original": tip_pose.copy(),
                "wrist_pose": wrist_pose.copy(),
                "ee_pos_vel": ee_linear_velocity,
                "ee_ori_vel": ee_velocity[3:],
                "joint_positions": joint_positions,
                "joint_velocities": joint_velocities,
                "joint_torques": _array_field(
                    state,
                    "tau_J",
                    "tau_J_d",
                    size=7,
                ),
                "gripper_width": _gripper_width(robot_interface),
            },
            "ee_pos_sim": None,
            "ee_quat_sim": None,
            "point_cloud": None,
            "skill": None,
            "guidance": None,
        }
    )
    return observation


def _camera_sample_with_prompt_depth(prompt_result, cameras):
    if not prompt_result:
        return None
    source_sample = prompt_result.get("camera_sample")
    enhanced_depths = prompt_result.get("depths") or {}
    if source_sample is None:
        return None

    required_depth_keys = [PROMPT_DEPTH_FIELDS[name] for name in cameras]
    if any(depth_key not in enhanced_depths for depth_key in required_depth_keys):
        return None

    sample = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in source_sample.items()
    }
    for depth_key in required_depth_keys:
        raw_depth = np.asarray(source_sample[depth_key])
        enhanced_depth = np.asarray(enhanced_depths[depth_key], dtype=np.float32)
        if enhanced_depth.shape != raw_depth.shape:
            raise ValueError(
                f"PromptDA {depth_key} shape mismatch: "
                f"{enhanced_depth.shape} vs {raw_depth.shape}"
            )
        sample[f"{depth_key}_realsense"] = raw_depth.copy()
        sample[depth_key] = enhanced_depth.astype(np.float16)
    sample["prompt_depth_source_wall_time_ns"] = source_sample.get(
        "camera_capture_wall_time_ns"
    )
    sample["prompt_depth_submitted_wall_time_ns"] = prompt_result.get(
        "submitted_wall_time_ns"
    )
    sample["prompt_depth_started_wall_time_ns"] = prompt_result.get(
        "processing_started_wall_time_ns"
    )
    sample["prompt_depth_ready_wall_time_ns"] = prompt_result.get(
        "ready_wall_time_ns"
    )
    return sample


def _record_intrinsics_matrix(record_intrinsics):
    return np.array(
        [
            [record_intrinsics["fx"], 0.0, record_intrinsics["ppx"]],
            [0.0, record_intrinsics["fy"], record_intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _draw_front_part_poses(
    front_bgr,
    camera_sample,
    record_intrinsics,
    part_names=None,
    axis_length=0.035,
):
    """Draw task-specific part poses without modifying the recorded RGB image."""
    if part_names is None:
        part_names = TASK_PART_NAMES["one_leg"]
    preview = front_bgr.copy()
    camera_to_april = camera_sample.get("camera_to_april")
    parts_poses = camera_sample.get("parts_poses")
    valid = camera_sample.get("parts_pose_valid")
    if camera_to_april is None or parts_poses is None or valid is None:
        return preview

    camera_to_april = np.asarray(camera_to_april, dtype=np.float64)
    parts_poses = np.asarray(parts_poses, dtype=np.float64).reshape(-1, 7)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if (
        camera_to_april.shape != (4, 4)
        or not np.all(np.isfinite(camera_to_april))
    ):
        return preview

    found = np.asarray(
        camera_sample.get("parts_founds", np.zeros(len(valid), dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    age_ms = np.asarray(
        camera_sample.get(
            "parts_pose_age_ms",
            np.full(len(valid), np.inf, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(-1)
    april_to_camera = np.linalg.inv(camera_to_april)
    intrinsics_matrix = _record_intrinsics_matrix(record_intrinsics)
    distortion = np.zeros(5, dtype=np.float64)

    for part_index, part_name in part_names.items():
        if part_index >= len(valid) or not valid[part_index]:
            continue
        pose = parts_poses[part_index]
        if not np.all(np.isfinite(pose)):
            continue

        part_to_april = np.eye(4, dtype=np.float64)
        part_to_april[:3, :3] = transform_utils.quat2mat(pose[3:7])
        part_to_april[:3, 3] = pose[:3]
        part_to_camera = april_to_camera @ part_to_april
        if part_to_camera[2, 3] <= 0.0:
            continue

        rotation_vector, _ = cv2.Rodrigues(part_to_camera[:3, :3])
        translation_vector = part_to_camera[:3, 3]
        cv2.drawFrameAxes(
            preview,
            intrinsics_matrix,
            distortion,
            rotation_vector,
            translation_vector,
            float(axis_length),
            2,
        )
        origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),
            rotation_vector,
            translation_vector,
            intrinsics_matrix,
            distortion,
        )
        origin_x, origin_y = np.rint(origin.reshape(2)).astype(int)
        is_found = part_index < len(found) and bool(found[part_index])
        color = (0, 255, 0) if is_found else (0, 191, 255)
        if is_found:
            state_text = "FOUND"
        else:
            age = age_ms[part_index] if part_index < len(age_ms) else np.inf
            state_text = (
                f"STALE {age:.0f}ms" if np.isfinite(age) else "STALE"
            )
        cv2.circle(preview, (origin_x, origin_y), 4, color, -1)
        cv2.putText(
            preview,
            f"P{part_index} {part_name} {state_text}",
            (origin_x + 6, origin_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return preview


def _draw_real_skill_annotation(front_bgr, observation):
    """Draw the live guidance point and skill without mutating recorded RGB."""
    preview = front_bgr.copy()
    if not observation:
        return preview

    point = None
    projections = observation.get("guidance_point_2d")
    if isinstance(projections, dict):
        point = projections.get("color_image2")
    if point is not None:
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        if point.size == 2 and np.all(np.isfinite(point)):
            source = observation.get("color_image2")
            source_height, source_width = preview.shape[:2]
            if isinstance(source, np.ndarray) and source.ndim >= 2:
                source_height, source_width = source.shape[:2]
            x = int(round(point[0] * preview.shape[1] / source_width))
            y = int(round(point[1] * preview.shape[0] / source_height))
            if 0 <= x < preview.shape[1] and 0 <= y < preview.shape[0]:
                cv2.drawMarker(
                    preview,
                    (x, y),
                    (255, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=18,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.circle(preview, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)

    skill = observation.get("skill")
    skill_state = observation.get("skill_state")
    if isinstance(skill, bytes):
        skill = skill.decode("utf-8")
    if isinstance(skill_state, bytes):
        skill_state = skill_state.decode("utf-8")
    if skill is not None:
        label = f"annotation={skill}"
        if skill_state is not None:
            label += f"/{skill_state}"
        cv2.putText(
            preview,
            label,
            (8, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def _build_camera_preview(
    camera_sample,
    camera_info,
    episode_state,
    draw_part_poses,
    task_name="one_leg",
    annotation_observation=None,
    prompt_depth_result=None,
    depth_min_m=0.05,
    depth_max_m=3.0,
    depth_colormap="viridis",
):
    if camera_sample is None:
        return None
    if "color_image1" not in camera_sample or "color_image2" not in camera_sample:
        return None

    wrist = cv2.cvtColor(camera_sample["color_image1"], cv2.COLOR_RGB2BGR)
    front = cv2.cvtColor(camera_sample["color_image2"], cv2.COLOR_RGB2BGR)
    if draw_part_poses:
        front = _draw_front_part_poses(
            front,
            camera_sample,
            camera_info["front"]["record_intrinsics"],
            part_names=TASK_PART_NAMES[task_name],
        )
    front = _draw_real_skill_annotation(front, annotation_observation)

    valid = np.asarray(
        camera_sample.get("parts_pose_valid", np.zeros(6, dtype=bool)),
        dtype=bool,
    )
    found = np.asarray(
        camera_sample.get("parts_founds", np.zeros(6, dtype=bool)),
        dtype=bool,
    )
    valid_text = "".join("1" if value else "0" for value in valid)
    found_text = "".join("1" if value else "0" for value in found)
    base_samples = int(camera_sample.get("camera_pose_samples", 0))
    base_required = int(camera_sample.get("camera_pose_samples_required", 0))
    cv2.putText(
        wrist,
        f"WRIST  state={episode_state}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        front,
        f"FRONT  pose={'ON' if draw_part_poses else 'OFF'}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        front,
        f"base={base_samples}/{base_required} found={found_text} valid={valid_text}",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if prompt_depth_result and prompt_depth_result.get("depths"):
        depths = prompt_depth_result["depths"]
        wrist_min, wrist_max = depth_display_bounds(
            camera_sample["depth_image1"], depth_min_m, depth_max_m
        )
        front_min, front_max = depth_display_bounds(
            camera_sample["depth_image2"], depth_min_m, depth_max_m
        )
        wrist_raw = colorize_depth(
            camera_sample["depth_image1"],
            wrist_min,
            wrist_max,
            depth_colormap,
        )
        front_raw = colorize_depth(
            camera_sample["depth_image2"],
            front_min,
            front_max,
            depth_colormap,
        )
        wrist_enhanced = colorize_depth(
            depths.get("depth_image1", camera_sample["depth_image1"]),
            wrist_min,
            wrist_max,
            depth_colormap,
        )
        front_enhanced = colorize_depth(
            depths.get("depth_image2", camera_sample["depth_image2"]),
            front_min,
            front_max,
            depth_colormap,
        )
        for panel, text in (
            (wrist_raw, f"WRIST RealSense {wrist_min:.2f}-{wrist_max:.2f}m"),
            (wrist_enhanced, f"WRIST PromptDA {wrist_min:.2f}-{wrist_max:.2f}m"),
            (front_raw, f"FRONT RealSense {front_min:.2f}-{front_max:.2f}m"),
            (front_enhanced, f"FRONT PromptDA {front_min:.2f}-{front_max:.2f}m"),
        ):
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
        wrist_ms = prompt_depth_result.get("stats", {}).get("wrist", {}).get(
            "inference_ms"
        )
        front_ms = prompt_depth_result.get("stats", {}).get("front", {}).get(
            "inference_ms"
        )
        timing = " / ".join(
            text
            for text in (
                f"wrist {wrist_ms:.0f} ms" if wrist_ms is not None else "",
                f"front {front_ms:.0f} ms" if front_ms is not None else "",
            )
            if text
        )
        if timing:
            cv2.putText(
                wrist_enhanced,
                timing,
                (8, wrist_enhanced.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return cv2.vconcat(
            [
                cv2.hconcat([wrist, wrist_raw, wrist_enhanced]),
                cv2.hconcat([front, front_raw, front_enhanced]),
            ]
        )

    combined = cv2.hconcat([wrist, front])
    return cv2.resize(
        combined,
        (combined.shape[1] * 2, combined.shape[0] * 2),
        interpolation=cv2.INTER_LINEAR,
    )


def _write_video_atomic(output_path, observations, fps):
    frames = []
    for observation in observations:
        if "color_image1" not in observation or "color_image2" not in observation:
            continue
        wrist = cv2.cvtColor(observation["color_image1"], cv2.COLOR_RGB2BGR)
        front = cv2.cvtColor(observation["color_image2"], cv2.COLOR_RGB2BGR)
        frames.append(cv2.hconcat([wrist, front]))
    if not frames:
        return

    temporary_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    height, width = frames[0].shape[:2]
    writer = H264VideoWriter(temporary_path, fps, frames[0].shape)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {temporary_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    os.replace(temporary_path, output_path)


def _write_episode(output_path, payload, video_fps, save_video):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".pkl.tmp")
    with temporary_path.open("wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_path)
    if save_video:
        _write_video_atomic(
            output_path.with_suffix(".mp4"),
            payload["observations"],
            video_fps,
        )
    return output_path


def _raw_episode_counts(outcomes_root):
    outcomes_root = Path(outcomes_root)
    return tuple(
        sum(
            1
            for path in (outcomes_root / outcome).glob(RAW_EPISODE_GLOB)
            if path.is_file()
        )
        for outcome in ("success", "failure")
    )


class EpisodeWriter:
    def __init__(self, video_fps=10, save_video=True):
        self.video_fps = int(video_fps)
        self.save_video = bool(save_video)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="episode_writer",
        )
        self.futures = []

    def submit(self, output_path, payload):
        future = self.executor.submit(
            _write_episode,
            output_path,
            payload,
            self.video_fps,
            self.save_video,
        )
        self.futures.append(future)
        future.add_done_callback(self._report_result)

    @staticmethod
    def _report_result(future):
        try:
            output_path = future.result()
            success_count, failure_count = _raw_episode_counts(
                output_path.parent.parent
            )
            logger.info(
                "Saved raw episode to %s; files: success=%d fail=%d",
                output_path,
                success_count,
                failure_count,
            )
        except Exception:
            logger.exception("Episode writer failed")

    def close(self):
        self.executor.shutdown(wait=True)
        for future in self.futures:
            future.result()


class RawEpisodeRecorder:
    def __init__(
        self,
        data_root,
        task_name,
        randomness,
        camera_info,
        writer,
        prompt_depth_config=None,
        annotation_session_factory=None,
        eepose_frame="robot-base",
        record_fps=10.0,
        robot_action_latency_ms=10.0,
        gripper_action_latency_ms=10.0,
        action_stale_guard_ms=10.0,
        max_command_lateness_ms=10.0,
        robot_observation_latency_ms=0.0,
        gripper_observation_latency_ms=0.0,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.task_name = task_name
        self.randomness = randomness
        self.camera_info = camera_info
        self.writer = writer
        self.prompt_depth_config = prompt_depth_config
        self.annotation_session_factory = annotation_session_factory
        self.eepose_frame = resolve_eepose_frame(eepose_frame)
        self.record_fps = float(record_fps)
        if self.record_fps <= 0:
            raise ValueError("record_fps must be positive")
        self.machine_time_schedule = MachineTimeSchedule.from_milliseconds(
            robot_action_ms=robot_action_latency_ms,
            gripper_action_ms=gripper_action_latency_ms,
            stale_guard_ms=action_stale_guard_ms,
            dispatch_tolerance_ms=max_command_lateness_ms,
        )
        for name, value in (
            ("robot_observation_latency_ms", robot_observation_latency_ms),
            ("gripper_observation_latency_ms", gripper_observation_latency_ms),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.robot_observation_latency_ms = float(robot_observation_latency_ms)
        self.gripper_observation_latency_ms = float(gripper_observation_latency_ms)
        self.annotation_session = None
        self.annotation_error = None
        self.observations = []
        self.actions = []
        self.actions_original = []
        self.actions_absolute = []
        self.action_timing = []
        self.command_attempts = []
        self.started_at = None
        self.stopped_at = None
        self.state = "idle"
        self.last_recorded_gripper = None
        self.gripper_hold_remaining = 0

    def _parts_ready(self, observation):
        if observation is None:
            return False
        valid = observation.get("parts_pose_valid")
        if valid is None:
            return False
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        required_indices = TASK_PART_NAMES[self.task_name]
        return all(
            part_index < len(valid) and bool(valid[part_index])
            for part_index in required_indices
        )

    def begin(self, initial_observation):
        if self.state == "recording":
            logger.warning("An episode is already recording")
            return False
        if self.state == "pending_save":
            logger.warning("Save or discard the previous episode first")
            return False
        if not self._parts_ready(initial_observation):
            required_parts = ", ".join(TASK_PART_NAMES[self.task_name].values())
            logger.warning(
                "Recording not started: required %s poses have not all been detected",
                required_parts,
            )
            return False
        annotation_session = None
        if self.annotation_session_factory is not None:
            try:
                annotation_session = self.annotation_session_factory(
                    self.task_name, self.camera_info
                )
                annotation_session.annotate_observation(initial_observation)
            except Exception:
                logger.exception(
                    "Recording not started: real-time skill annotation failed"
                )
                return False
        self.observations = [initial_observation]
        self.actions = []
        self.actions_original = []
        self.actions_absolute = []
        self.action_timing = []
        self.command_attempts = []
        self.started_at = datetime.now().isoformat(timespec="milliseconds")
        self.stopped_at = None
        self.last_recorded_gripper = None
        self.gripper_hold_remaining = 0
        self.annotation_session = annotation_session
        self.annotation_error = None
        self.state = "recording"
        logger.info("Recording started")
        return True

    def should_record(self, action, no_op_threshold, gripper_hold_frames):
        gripper = float(np.sign(action[-1]))
        gripper_changed = (
            self.last_recorded_gripper is None
            or gripper != self.last_recorded_gripper
        )
        if gripper_changed:
            self.gripper_hold_remaining = int(gripper_hold_frames)
        motion = float(np.linalg.norm(action[:6])) > no_op_threshold
        return motion or gripper_changed or self.gripper_hold_remaining > 0

    def append(
        self,
        observation,
        action,
        action_original=None,
        *,
        action_absolute=None,
        action_timing=None,
    ):
        if self.state != "recording" or observation is None:
            return
        if len(self.observations) == len(self.actions):
            self._annotate_observation(observation)
            self.observations.append(observation)
        if len(self.observations) != len(self.actions) + 1:
            raise RuntimeError("invalid observation/action alignment")
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        if action_original is None:
            action_original = action
        self.actions_original.append(
            np.asarray(action_original, dtype=np.float32).copy()
        )
        if action_absolute is None:
            action_absolute = np.full(8, np.nan, dtype=np.float32)
        self.actions_absolute.append(
            np.asarray(action_absolute, dtype=np.float32).copy()
        )
        timing = dict(action_timing or {})
        timing.setdefault(
            "action_target_wall_time_ns",
            timing.get(
                "action_wall_time_ns",
                observation.get("control_wall_time_ns"),
            ),
        )
        timing["action_wall_time_ns"] = timing["action_target_wall_time_ns"]
        timing.setdefault("status", "executed")
        self.action_timing.append(timing)
        self.command_attempts.append(dict(timing))
        gripper = float(np.sign(action[-1]))
        if self.last_recorded_gripper == gripper and self.gripper_hold_remaining > 0:
            self.gripper_hold_remaining -= 1
        self.last_recorded_gripper = gripper

    def record_dropped_command(self, timing):
        """Keep a stale/failed command in audit metadata, never in training arrays."""

        if self.state != "recording":
            return
        attempt = dict(timing)
        attempt.setdefault("status", "dropped")
        self.command_attempts.append(attempt)

    def stop(self, final_observation):
        if self.state != "recording":
            logger.warning("No episode is recording")
            return
        if self.actions and len(self.observations) == len(self.actions):
            if final_observation is None:
                logger.warning("Waiting for a terminal observation before stopping")
                return
            self._annotate_observation(final_observation)
            self.observations.append(final_observation)
        self.stopped_at = datetime.now().isoformat(timespec="milliseconds")
        self.state = "pending_save"
        logger.info(
            "Recording stopped with %d actions. Press s=success, f=failure, d=discard",
            len(self.actions),
        )

    def _output_path(self, outcome):
        output_dir = (
            self.data_root
            / "raw"
            / "osc"
            / "real"
            / self.task_name
            / "teleop"
            / self.randomness
            / outcome
        )
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        return output_dir / f"{timestamp}.pkl"

    def save(self, success):
        if self.state != "pending_save":
            logger.warning("There is no stopped episode waiting to be saved")
            return None
        if not self.actions:
            logger.warning("The episode contains no action; discarding it")
            self.discard()
            return None
        if len(self.observations) != len(self.actions) + 1:
            raise RuntimeError(
                "raw episode must contain N+1 observations and N actions"
            )

        outcome = "success" if success else "failure"
        output_path = self._output_path(outcome)
        payload = {
            "observations": self.observations,
            "actions": self.actions,
            "actions_original": self.actions_original,
            "actions_absolute": self.actions_absolute,
            "action_timing": self.action_timing,
            "action_timestamps_ns": [
                timing.get("action_target_wall_time_ns")
                for timing in self.action_timing
            ],
            "action_target_timestamps_ns": [
                timing.get("action_target_wall_time_ns")
                for timing in self.action_timing
            ],
            "command_attempts": self.command_attempts,
            "rewards": [0.0] * len(self.actions),
            "camera_info": self.camera_info,
            "success": bool(success),
            "task": self.task_name,
            "furniture": self.task_name,
            "action_type": "delta",
            "eepose_frame": self.eepose_frame,
            "eepose_original_frame": "real-tip",
            "eepose_schema_version": 2,
            "metadata": {
                "schema": "deoxys_furniturebench_raw_v5_target_time",
                "timebase": "unix_epoch_ns",
                "controller_observation_alignment": (
                    "raw observation stream plus action target-time master"
                ),
                "recording_frequency_hz": self.record_fps,
                "action_period_ns": int(round(1e9 / self.record_fps)),
                "episode_grid_start_wall_time_ns": self.action_timing[0].get(
                    "episode_grid_start_wall_time_ns"
                ),
                "recording_includes_noop_actions": True,
                "timing_contract": (
                    "camera sensor source, PromptDA submit/start/ready, robot/gripper "
                    "state receive, and robot/gripper command send times are stored "
                    "separately; action_target_timestamps_ns is the alignment "
                    "master and action_timestamps_ns is its compatibility alias"
                ),
                "robot_action_latency_ms": (
                    self.machine_time_schedule.robot_action_latency_ns / 1e6
                ),
                "gripper_action_latency_ms": (
                    self.machine_time_schedule.gripper_action_latency_ns / 1e6
                ),
                "action_stale_guard_ms": (
                    self.machine_time_schedule.stale_guard_ns / 1e6
                ),
                "max_command_lateness_ms": (
                    self.machine_time_schedule.dispatch_tolerance_ns / 1e6
                ),
                "action_latency_source": "estimated",
                "action_latency_basis": "Deoxys/UMI command_latency=0.01s",
                "robot_observation_latency_ms": self.robot_observation_latency_ms,
                "gripper_observation_latency_ms": (
                    self.gripper_observation_latency_ms
                ),
                "parts_poses_frame": "furniture_bench_april_tag",
                "action_translation_unit": "meter",
                "action_quaternion_order": "xyzw",
                "action_rotation_semantics": (
                    "right_multiply_local_wrist_delta"
                    if self.eepose_frame == "robot-base"
                    else "right_multiply_local_tip_delta"
                ),
                "eepose_frame": self.eepose_frame,
                "eepose_original_frame": "real-tip",
                "eepose_schema_version": 2,
                "ee_velocity_source": "geometric_jacobian_times_measured_dq",
                "ee_velocity_frame": "robot-base",
                "ee_velocity_point": (
                    "franka_O_T_EE"
                    if self.eepose_frame == "robot-base"
                    else "real-tip"
                ),
                "action_frame": (
                    "robot-base/wrist"
                    if self.eepose_frame == "robot-base"
                    else "robot-base/real-tip"
                ),
                "action_original_frame": "robot-base/real-tip",
                "randomness": self.randomness,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "num_observations": len(self.observations),
                "num_actions": len(self.actions),
                "prompt_depth_anything": self.prompt_depth_config,
            },
        }
        if self.annotation_session is not None:
            self.annotation_session.update_trajectory_metadata(payload)
        elif self.annotation_session_factory is not None:
            for observation in payload["observations"]:
                _clear_real_skill_annotation(observation)
            payload["metadata"]["real_skill_annotation"] = {
                "source": REAL_ANNOTATION_SOURCE,
                "mode": "online",
                "complete": False,
                "error": self.annotation_error,
            }
        self.writer.submit(output_path, payload)
        self.discard(log=False)
        return output_path

    def discard(self, log=True):
        action_count = len(self.actions)
        self.observations = []
        self.actions = []
        self.actions_original = []
        self.actions_absolute = []
        self.action_timing = []
        self.command_attempts = []
        self.started_at = None
        self.stopped_at = None
        self.state = "idle"
        self.last_recorded_gripper = None
        self.gripper_hold_remaining = 0
        self.annotation_session = None
        self.annotation_error = None
        if log:
            logger.info("Discarded episode with %d actions", action_count)

    def _annotate_observation(self, observation):
        if self.annotation_session is None:
            return
        try:
            self.annotation_session.annotate_observation(observation)
        except Exception as exc:
            self.annotation_error = f"{type(exc).__name__}: {exc}"
            self.annotation_session = None
            logger.exception(
                "Real-time annotation failed; the raw episode will remain savable "
                "and can be annotated offline"
            )


def wait_for_robot_state(robot_interface, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            robot_interface.received_states
            and robot_interface.check_nonzero_configuration()
        ):
            return True
        time.sleep(0.05)
    return False


def move_to_reset_joint_positions(
    robot_interface,
    joint_controller_cfg,
    timeout,
    tolerance,
    gripper_open,
):
    if not wait_for_robot_state(robot_interface):
        logger.warning("Robot state not received before reset request")
        return False
    target = np.asarray(RESET_JOINT_POSITIONS, dtype=np.float64)
    action = target.tolist() + [-1.0 if gripper_open else 1.0]
    deadline = time.monotonic() + timeout
    max_error = float("inf")
    while time.monotonic() < deadline:
        current_q = robot_interface.last_q
        if current_q is not None:
            max_error = float(np.max(np.abs(np.asarray(current_q) - target)))
            if max_error < tolerance:
                logger.info("Reset target reached; max error %.6f", max_error)
                return True
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=joint_controller_cfg,
        )
    logger.warning("Reset timed out; max error %.6f", max_error)
    return False


def scaled_deoxys_action(action, controller_cfg):
    scaled = np.asarray(action, dtype=np.float64).copy()
    scaled[:3] *= controller_cfg.action_scale.translation
    scaled[3:6] *= controller_cfg.action_scale.rotation
    return scaled


def delta_action_to_absolute(action, robot_state):
    """Return ``[absolute xyz, absolute quat_xyzw, gripper]`` for validation."""
    delta = np.asarray(action, dtype=np.float64).reshape(8)
    current_pose = np.asarray(robot_state["ee_pose"], dtype=np.float64).reshape(4, 4)
    target_pose = current_pose.copy()
    target_pose[:3, 3] = current_pose[:3, 3] + delta[:3]
    target_pose[:3, :3] = current_pose[:3, :3] @ transform_utils.quat2mat(
        delta[3:7]
    )
    return np.concatenate(
        [
            target_pose[:3, 3],
            transform_utils.mat2quat(target_pose[:3, :3]),
            [np.sign(delta[-1])],
        ]
    ).astype(np.float32)


def parse_args():
    default_data_root = os.environ.get("DATA_DIR_RAW")
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", default="config/charmander.yml")
    parser.add_argument("--controller-type", default="OSC_POSE")
    parser.add_argument(
        "--eepose-frame",
        default="robot-base",
        help=(
            "EE pose/action representation: robot-base (default), or original/"
            "real-tip for the legacy 10.34 cm/-45 degree virtual tip"
        ),
    )
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument(
        "--spacemouse-connection",
        choices=tuple(SPACEMOUSE_PRODUCT_IDS),
        default="wired",
    )
    parser.add_argument(
        "--product-id",
        type=int,
        default=None,
        help="override the product ID selected by --spacemouse-connection",
    )
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument(
        "--task-name",
        choices=tuple(TASK_PART_NAMES),
        default="one_leg",
    )
    parser.add_argument("--randomness", choices=("low",), default="low")
    parser.add_argument(
        "--front-camera-serial",
        "--camera-high-serial",
        dest="front_camera_serial",
        default=DEFAULT_FRONT_SERIAL,
    )
    parser.add_argument(
        "--wrist-camera-serial",
        dest="wrist_camera_serial",
        default=DEFAULT_WRIST_SERIAL,
    )
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
    parser.add_argument("--record-image-width", type=int, default=320)
    parser.add_argument("--record-image-height", type=int, default=240)
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument(
        "--robot-action-latency-ms",
        type=float,
        default=10.0,
        help="estimated arm command-to-effect latency used for target-time scheduling",
    )
    parser.add_argument(
        "--gripper-action-latency-ms",
        type=float,
        default=10.0,
        help="estimated gripper command-to-effect latency",
    )
    parser.add_argument(
        "--action-stale-guard-ms",
        type=float,
        default=10.0,
        help="future margin required before accepting a target-time action",
    )
    parser.add_argument(
        "--max-command-lateness-ms",
        type=float,
        default=10.0,
        help="maximum scheduler wake-up lateness before dropping a command",
    )
    parser.add_argument(
        "--robot-observation-latency-ms",
        type=float,
        default=0.0,
        help="estimated robot receive-time correction used by offline alignment",
    )
    parser.add_argument(
        "--gripper-observation-latency-ms",
        type=float,
        default=0.0,
        help="estimated gripper receive-time correction used by offline alignment",
    )
    # Deprecated compatibility flags. Timestamped v4 recording deliberately
    # keeps every fixed-rate no-op and therefore does not use either value.
    parser.add_argument(
        "--no-op-threshold", type=float, default=1e-5, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--gripper-hold-frames", type=int, default=8, help=argparse.SUPPRESS
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--draw-part-poses", action="store_true")
    parser.add_argument(
        "--real-skill-annotation",
        action="store_true",
        help=(
            "run the robust-rearrangement one_leg annotator online, draw its "
            "skill/guidance point, and save annotations in the pickle"
        ),
    )
    parser.add_argument("--no-camera-preview", action="store_true")
    parser.add_argument(
        "--prompt-depth-anything",
        action="store_true",
        help=(
            "enhance recorded depth in a background thread and show "
            "RGB/raw/enhanced depth"
        ),
    )
    parser.add_argument(
        "--prompt-depth-model",
        choices=("vits", "vitl", "vits-transparent"),
        default="vits",
    )
    parser.add_argument("--prompt-depth-device", default="cuda")
    parser.add_argument("--prompt-depth-max-size", type=int, default=448)
    parser.add_argument(
        "--prompt-depth-cameras",
        choices=("both", "front", "wrist"),
        default="both",
    )
    parser.add_argument("--prompt-depth-min-m", type=float, default=0.05)
    parser.add_argument("--prompt-depth-max-m", type=float, default=5.0)
    parser.add_argument("--prompt-depth-display-max-m", type=float, default=3.0)
    parser.add_argument(
        "--prompt-depth-colormap",
        choices=("viridis", "turbo", "inferno", "jet"),
        default="viridis",
    )
    parser.add_argument("--reset-timeout", type=float, default=7.0)
    parser.add_argument("--reset-tolerance", type=float, default=1e-3)
    parser.add_argument("--keep-gripper-closed-during-reset", action="store_true")
    args = parser.parse_args()
    if args.product_id is None:
        args.product_id = SPACEMOUSE_PRODUCT_IDS[args.spacemouse_connection]
    if not args.data_root:
        parser.error("--data-root is required when DATA_DIR_RAW is not set")
    if args.record_fps <= 0:
        parser.error("--record-fps must be greater than zero")
    for name in (
        "robot_action_latency_ms",
        "gripper_action_latency_ms",
        "action_stale_guard_ms",
        "max_command_lateness_ms",
        "robot_observation_latency_ms",
        "gripper_observation_latency_ms",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    try:
        resolve_eepose_frame(args.eepose_frame)
    except ValueError as exc:
        parser.error(str(exc))
    if args.real_skill_annotation and args.task_name != "one_leg":
        parser.error(
            "--real-skill-annotation currently supports only --task-name one_leg"
        )
    return args


def main():
    args = parse_args()
    camera = None
    prompt_depth_worker = None
    prompt_depth_cameras = ()
    prompt_depth_config = None
    annotation_session_factory = None
    preview_annotation_session = None
    preview_annotation_error = None
    preview_annotation_capture_ns = None
    last_annotation_observation = None
    episode = None
    writer = EpisodeWriter(
        video_fps=args.record_fps,
        save_video=not args.no_video,
    )
    device = None
    robot_interface = None
    controller_cfg = None
    try:
        camera = DualRealSenseSnapshotter(
            front_serial=args.front_camera_serial,
            wrist_serial=args.wrist_camera_serial,
            record_width=args.record_image_width,
            record_height=args.record_image_height,
            furniture_task=args.task_name,
            front_width=args.front_color_width,
            front_height=args.front_color_height,
            front_fps=args.front_color_fps,
            front_depth_width=args.front_depth_width,
            front_depth_height=args.front_depth_height,
            front_depth_fps=args.front_depth_fps,
            wrist_width=args.wrist_color_width,
            wrist_height=args.wrist_color_height,
            wrist_fps=args.wrist_color_fps,
            wrist_depth_width=args.wrist_depth_width,
            wrist_depth_height=args.wrist_depth_height,
            wrist_depth_fps=args.wrist_depth_fps,
        )
        camera.start()
        camera_info = camera.metadata()
        if args.real_skill_annotation:
            annotation_session_factory = _create_real_skill_annotation_session
            if not args.no_camera_preview:
                preview_annotation_session = annotation_session_factory(
                    args.task_name, camera_info
                )
        if args.prompt_depth_anything:
            prompt_depth_cameras = (
                ("wrist", "front")
                if args.prompt_depth_cameras == "both"
                else (args.prompt_depth_cameras,)
            )
            prompt_depth_worker = PromptDepthWorker(
                PromptDepthAnythingEstimator(
                    model=args.prompt_depth_model,
                    device=args.prompt_depth_device,
                    max_size=args.prompt_depth_max_size,
                    min_depth_m=args.prompt_depth_min_m,
                    max_depth_m=args.prompt_depth_max_m,
                ),
                cameras=prompt_depth_cameras,
            )
            prompt_depth_worker.start()
            prompt_depth_config = {
                "online": True,
                "model": args.prompt_depth_model,
                "max_size": args.prompt_depth_max_size,
                "prompt_size": [256, 192],
                "cameras": list(prompt_depth_cameras),
                "canonical_depth_fields": [
                    PROMPT_DEPTH_FIELDS[name] for name in prompt_depth_cameras
                ],
                "original_depth_suffix": "_realsense",
            }
        episode = RawEpisodeRecorder(
            data_root=args.data_root,
            task_name=args.task_name,
            randomness=args.randomness,
            camera_info=camera_info,
            writer=writer,
            prompt_depth_config=prompt_depth_config,
            annotation_session_factory=annotation_session_factory,
            eepose_frame=args.eepose_frame,
            record_fps=args.record_fps,
            robot_action_latency_ms=args.robot_action_latency_ms,
            gripper_action_latency_ms=args.gripper_action_latency_ms,
            action_stale_guard_ms=args.action_stale_guard_ms,
            max_command_lateness_ms=args.max_command_lateness_ms,
            robot_observation_latency_ms=args.robot_observation_latency_ms,
            gripper_observation_latency_ms=args.gripper_observation_latency_ms,
        )

        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
        device.start_control()
        robot_interface = FrankaInterface(
            args.interface_cfg,
            control_freq=args.record_fps,
            use_visualizer=False,
        )
        controller_cfg = get_default_controller_config(args.controller_type)
        joint_controller_cfg = get_default_controller_config("JOINT_POSITION")
        robot_interface.reset()
        if not wait_for_robot_state(robot_interface):
            raise RuntimeError("robot state was not received")
        # Warm the controller before an episode so its one-time dummy-message
        # handshake cannot consume the first target-time action's deadline.
        robot_interface.control(
            controller_type=args.controller_type,
            action=np.array([0.0] * 6 + [-1.0], dtype=np.float64),
            controller_cfg=controller_cfg,
            control_gripper=False,
            enforce_control_frequency=False,
        )

        logger.info(
            "Keys: b=begin, e=end, s=save success, f=save failure, "
            "d=discard, r=reset joints, p=toggle part poses, q=quit"
        )
        record_period = 1.0 / args.record_fps
        record_period_ns = int(round(1e9 / args.record_fps))
        sample_lead_s = max(
            0.01,
            episode.machine_time_schedule.common_admission_lead_ns / 1e9
            - record_period
            + 0.005,
        )
        grid_start_monotonic = None
        grid_start_wall_time_ns = None
        grid_index = 0
        pending_command = None
        draw_part_poses = bool(args.draw_part_poses)
        prompt_depth_error = None

        def cancel_pending_command(reason):
            nonlocal pending_command
            if pending_command is None:
                return
            timing = dict(pending_command["timing"])
            timing.update(
                status="dropped",
                drop_reason=reason,
                dropped_wall_time_ns=time.time_ns(),
            )
            episode.record_dropped_command(timing)
            pending_command = None

        def dispatch_pending_command(now_monotonic):
            nonlocal pending_command
            if pending_command is None:
                return
            schedule = episode.machine_time_schedule
            target_wall_time_ns = pending_command["target_wall_time_ns"]
            channels = sorted(
                ("robot", "gripper"),
                key=lambda channel: pending_command[f"{channel}_deadline_monotonic"],
            )
            for channel in channels:
                if pending_command[f"{channel}_sent"]:
                    continue
                deadline_monotonic = pending_command[
                    f"{channel}_deadline_monotonic"
                ]
                if now_monotonic < deadline_monotonic:
                    continue
                dispatch_wall_time_ns = time.time_ns()
                if schedule.dispatch_expired(
                    target_wall_time_ns,
                    dispatch_wall_time_ns,
                    channel,
                ):
                    partial = bool(
                        pending_command["robot_sent"]
                        or pending_command["gripper_sent"]
                    )
                    timing = dict(pending_command["timing"])
                    timing.update(
                        status="partial_dispatch_failure" if partial else "dropped",
                        drop_reason=f"stale_{channel}_deadline",
                        dropped_wall_time_ns=dispatch_wall_time_ns,
                    )
                    episode.record_dropped_command(timing)
                    pending_command = None
                    if partial:
                        raise RuntimeError(
                            "coordinated command became stale after one channel sent"
                        )
                    return
                if channel == "robot":
                    result = robot_interface.control(
                        controller_type=args.controller_type,
                        action=pending_command["deoxys_action"].copy(),
                        controller_cfg=controller_cfg,
                        control_gripper=False,
                        enforce_control_frequency=False,
                    )
                    send_wall_time_ns = result["robot_command_wall_time_ns"]
                else:
                    robot_interface.gripper_control(
                        float(pending_command["deoxys_action"][-1])
                    )
                    send_wall_time_ns = (
                        robot_interface.last_gripper_command_wall_time_ns
                    )
                pending_command[f"{channel}_sent"] = True
                pending_command["timing"][
                    f"{channel}_command_wall_time_ns"
                ] = send_wall_time_ns
                pending_command["timing"][f"{channel}_send_residual_ms"] = (
                    send_wall_time_ns - target_wall_time_ns
                ) / 1e6
                now_monotonic = time.monotonic()

            if pending_command is not None and all(
                pending_command[f"{channel}_sent"]
                for channel in ("robot", "gripper")
            ):
                pending_command["timing"]["status"] = "executed"
                episode.append(
                    pending_command["observation"],
                    pending_command["saved_action"],
                    pending_command["saved_action_original"],
                    action_absolute=pending_command["saved_action_absolute"],
                    action_timing=pending_command["timing"],
                )
                pending_command = None

        with NonBlockingKeyReader() as key_reader:
            running = True
            while running:
                camera_sample = camera.latest()
                prompt_result = None
                observation_camera_sample = camera_sample
                if prompt_depth_worker is not None:
                    prompt_depth_worker.submit(camera_sample)
                    prompt_result = prompt_depth_worker.latest()
                    observation_camera_sample = _camera_sample_with_prompt_depth(
                        prompt_result,
                        prompt_depth_cameras,
                    )
                    if prompt_result is not None and prompt_result.get("error"):
                        if prompt_result["error"] != prompt_depth_error:
                            prompt_depth_error = prompt_result["error"]
                            logger.error(
                                "PromptDA processing failed: %s",
                                prompt_depth_error,
                            )
                observation = build_observation(
                    robot_interface,
                    observation_camera_sample,
                    args.eepose_frame,
                )
                annotation_observation = None
                if (
                    args.real_skill_annotation
                    and not args.no_camera_preview
                    and observation is not None
                ):
                    capture_ns = observation.get("camera_capture_wall_time_ns")
                    if (
                        capture_ns is None
                        or capture_ns != preview_annotation_capture_ns
                    ):
                        if preview_annotation_session is None:
                            preview_annotation_session = annotation_session_factory(
                                args.task_name, camera_info
                            )
                        try:
                            preview_annotation_session.annotate_observation(observation)
                            last_annotation_observation = observation
                            preview_annotation_capture_ns = capture_ns
                            preview_annotation_error = None
                        except Exception as exc:
                            message = f"{type(exc).__name__}: {exc}"
                            if message != preview_annotation_error:
                                logger.warning(
                                    "Live annotation is waiting for usable poses: %s",
                                    message,
                                )
                                preview_annotation_error = message
                            preview_annotation_session = None
                            last_annotation_observation = None
                    annotation_observation = last_annotation_observation
                keys = key_reader.read_keys()
                if not args.no_camera_preview:
                    preview_sample = camera_sample
                    if prompt_result is not None and prompt_result.get("depths"):
                        preview_sample = prompt_result["camera_sample"]
                    preview = _build_camera_preview(
                        preview_sample,
                        camera_info,
                        episode.state,
                        draw_part_poses,
                        task_name=args.task_name,
                        annotation_observation=annotation_observation,
                        prompt_depth_result=prompt_result,
                        depth_min_m=args.prompt_depth_min_m,
                        depth_max_m=args.prompt_depth_display_max_m,
                        depth_colormap=args.prompt_depth_colormap,
                    )
                    if preview is not None:
                        cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                        window_key = cv2.waitKey(1) & 0xFF
                        if window_key in map(ord, "besfdrpq"):
                            keys.append(chr(window_key))
                for key in keys:
                    if key == "b":
                        if episode.begin(observation):
                            if (
                                annotation_session_factory is not None
                                and not args.no_camera_preview
                            ):
                                preview_annotation_session = annotation_session_factory(
                                    args.task_name, camera_info
                                )
                                preview_annotation_capture_ns = None
                                last_annotation_observation = None
                            grid_start_monotonic = time.monotonic()
                            grid_start_wall_time_ns = time.time_ns()
                            grid_index = 0
                            pending_command = None
                    elif key == "e":
                        cancel_pending_command("operator_stopped_episode")
                        episode.stop(observation)
                    elif key == "s":
                        episode.save(success=True)
                        if (
                            annotation_session_factory is not None
                            and not args.no_camera_preview
                        ):
                            preview_annotation_session = annotation_session_factory(
                                args.task_name, camera_info
                            )
                            preview_annotation_capture_ns = None
                            last_annotation_observation = None
                    elif key == "f":
                        episode.save(success=False)
                        if (
                            annotation_session_factory is not None
                            and not args.no_camera_preview
                        ):
                            preview_annotation_session = annotation_session_factory(
                                args.task_name, camera_info
                            )
                            preview_annotation_capture_ns = None
                            last_annotation_observation = None
                    elif key == "d":
                        cancel_pending_command("operator_discarded_episode")
                        episode.discard()
                        if (
                            annotation_session_factory is not None
                            and not args.no_camera_preview
                        ):
                            preview_annotation_session = annotation_session_factory(
                                args.task_name, camera_info
                            )
                            preview_annotation_capture_ns = None
                            last_annotation_observation = None
                    elif key == "p":
                        draw_part_poses = not draw_part_poses
                        logger.info(
                            "Front part-pose overlay %s",
                            "enabled" if draw_part_poses else "disabled",
                        )
                    elif key == "q":
                        cancel_pending_command("operator_quit")
                        running = False
                    elif key == "r":
                        if episode.state == "recording":
                            logger.warning("Joint reset is disabled while recording")
                        else:
                            move_to_reset_joint_positions(
                                robot_interface,
                                joint_controller_cfg,
                                timeout=args.reset_timeout,
                                tolerance=args.reset_tolerance,
                                gripper_open=(
                                    not args.keep_gripper_closed_during_reset
                                ),
                            )
                            device.start_control()
                if not running:
                    break

                now = time.monotonic()
                if episode.state == "recording":
                    dispatch_pending_command(now)
                    cycle_end = grid_start_monotonic + (grid_index + 1) * record_period
                    sample_time = cycle_end - sample_lead_s
                    target_monotonic = cycle_end + record_period
                    if now < sample_time or pending_command is not None:
                        continue
                    while now >= sample_time + record_period:
                        grid_index += 1
                        cycle_end += record_period
                        sample_time += record_period
                        target_monotonic += record_period

                    action, _ = input2action(
                        device=device,
                        controller_type=args.controller_type,
                    )
                    if action is None:
                        cancel_pending_command("spacemouse_stop")
                        episode.stop(observation)
                        try:
                            device.start_control(preserve_gripper=True)
                        except TypeError:
                            device.start_control()
                        continue
                    if observation is None:
                        grid_index += 1
                        continue

                    scaled_action = scaled_deoxys_action(action, controller_cfg)
                    saved_action = deoxys_delta_to_furniture_bench_action(
                        scaled_action,
                        observation["robot_state"]["wrist_pose"],
                        args.eepose_frame,
                    )
                    saved_action_original = deoxys_delta_to_furniture_bench_action(
                        scaled_action,
                        observation["robot_state"]["wrist_pose"],
                        "original",
                    )
                    saved_action_absolute = delta_action_to_absolute(
                        saved_action,
                        observation["robot_state"],
                    )
                    target_wall_time_ns = monotonic_target_to_wall_time_ns(
                        target_monotonic,
                        monotonic_now_s=grid_start_monotonic,
                        wall_now_ns=grid_start_wall_time_ns,
                    )
                    generated_wall_time_ns = time.time_ns()
                    schedule = episode.machine_time_schedule
                    timing = {
                        "action_target_wall_time_ns": target_wall_time_ns,
                        "action_generated_wall_time_ns": generated_wall_time_ns,
                        "episode_grid_start_wall_time_ns": grid_start_wall_time_ns,
                        "grid_index": grid_index + 2,
                        "action_period_ns": record_period_ns,
                        "robot_command_deadline_wall_time_ns": schedule.deadline_ns(
                            target_wall_time_ns, "robot"
                        ),
                        "gripper_command_deadline_wall_time_ns": schedule.deadline_ns(
                            target_wall_time_ns, "gripper"
                        ),
                        "observation_ready_wall_time_ns": observation.get(
                            "observation_ready_wall_time_ns"
                        ),
                    }
                    if not schedule.admit(
                        target_wall_time_ns,
                        generated_wall_time_ns,
                    ):
                        timing.update(
                            status="dropped",
                            drop_reason="stale_at_admission",
                            dropped_wall_time_ns=generated_wall_time_ns,
                        )
                        episode.record_dropped_command(timing)
                        grid_index += 1
                        continue
                    pending_command = {
                        "target_wall_time_ns": target_wall_time_ns,
                        "robot_deadline_monotonic": target_monotonic
                        - schedule.robot_action_latency_ns / 1e9,
                        "gripper_deadline_monotonic": target_monotonic
                        - schedule.gripper_action_latency_ns / 1e9,
                        "robot_sent": False,
                        "gripper_sent": False,
                        "deoxys_action": np.asarray(
                            action, dtype=np.float64
                        ).copy(),
                        "saved_action": saved_action,
                        "saved_action_original": saved_action_original,
                        "saved_action_absolute": saved_action_absolute,
                        "observation": observation,
                        "timing": timing,
                    }
                    grid_index += 1
                    continue

                action, _ = input2action(
                    device=device,
                    controller_type=args.controller_type,
                )
                if action is None:
                    break
                robot_interface.control(
                    controller_type=args.controller_type,
                    action=np.asarray(action, dtype=np.float64).copy(),
                    controller_cfg=controller_cfg,
                )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        if episode is not None and episode.state != "idle":
            logger.warning("Unsaved in-memory episode was discarded on exit")
        try:
            if robot_interface is not None:
                try:
                    if controller_cfg is not None:
                        robot_interface.control(
                            controller_type=args.controller_type,
                            action=np.array([0.0] * 6 + [1.0]),
                            controller_cfg=controller_cfg,
                            termination=True,
                        )
                finally:
                    robot_interface.close()
        finally:
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    logger.warning("Failed to close SpaceMouse cleanly: %s", exc)
            if prompt_depth_worker is not None:
                prompt_depth_worker.stop()
            if camera is not None:
                camera.stop()
            if not args.no_camera_preview:
                cv2.destroyAllWindows()
            writer.close()


if __name__ == "__main__":
    main()
