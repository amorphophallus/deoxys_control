"""SpaceMouse teleoperation with FurnitureBench-compatible raw data capture."""

import argparse
import concurrent.futures
import json
import os
import pickle
import select
import sys
import termios
import threading
import time
import traceback
import tty
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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
)
from deoxys.utils.prompt_depth_anything import (
    PromptDepthAnythingEstimator,
    colorize_depth,
    depth_display_bounds,
    has_usable_depth,
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
BUFFERED_SCHEMA = "deoxys_furniturebench_raw_v6_offline_buffered"
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


def _create_real_skill_annotation_session(task_name, camera_info, mode="online"):
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
    return RealSkillAnnotationSession(task_name, camera_info, mode=mode)


def _clear_real_skill_annotation(observation):
    observation["skill"] = None
    observation["guidance"] = None
    for key in REAL_ANNOTATION_OUTPUT_FIELDS:
        observation.pop(key, None)


def _contains_vlm_metadata(value):
    if isinstance(value, dict):
        return any(
            "vlm" in str(key).lower() or _contains_vlm_metadata(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_vlm_metadata(child) for child in value)
    return False


def _assert_matching_projection(stored, expected, path):
    if stored is None or expected is None:
        if stored is not expected:
            raise RuntimeError(f"{path} nullability disagrees with reprojection")
        return
    stored = np.asarray(stored, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if stored.shape != (2,) or expected.shape != (2,):
        raise RuntimeError(f"{path} must be a 2-D pixel")
    if not np.all(np.isfinite(stored)) or not np.allclose(
        stored, expected, atol=1.0, rtol=0.0
    ):
        raise RuntimeError(
            f"{path} does not match same-frame calibrated reprojection: "
            f"stored={stored.tolist()} expected={expected.tolist()}"
        )


def validate_buffered_payload(payload, annotation_session):
    """Fail closed unless a saved v6 episode satisfies its training contract."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("buffered payload metadata must be a mapping")
    expected_top_level = {
        "env": "FurnitureBench",
        "annotation_source": "scripted",
        "image_annotation_mode": "none",
    }
    for key, expected in expected_top_level.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"buffered payload requires {key}={expected!r}, got "
                f"{payload.get(key)!r}"
            )
    if metadata.get("schema") != BUFFERED_SCHEMA:
        raise RuntimeError(f"buffered payload requires metadata.schema={BUFFERED_SCHEMA}")
    if _contains_vlm_metadata(payload):
        raise RuntimeError("buffered payload must not contain VLM metadata")

    observations = payload.get("observations", [])
    actions = payload.get("actions", [])
    timing = payload.get("action_timing", [])
    target_times = np.asarray(
        payload.get("action_target_timestamps_ns", []), dtype=np.int64
    ).reshape(-1)
    aliases = np.asarray(payload.get("action_timestamps_ns", []), dtype=np.int64).reshape(-1)
    obs_valid = np.asarray(payload.get("obs_valid", []), dtype=np.bool_).reshape(-1)
    lengths = {
        "observations": len(observations),
        "actions": len(actions),
        "actions_original": len(payload.get("actions_original", [])),
        "actions_absolute": len(payload.get("actions_absolute", [])),
        "action_timing": len(timing),
        "action_target_timestamps_ns": len(target_times),
        "action_timestamps_ns": len(aliases),
        "obs_valid": len(obs_valid),
        "rewards": len(payload.get("rewards", [])),
    }
    frame_count = lengths["actions"]
    if frame_count == 0 or any(value != frame_count for value in lengths.values()):
        raise RuntimeError(f"buffered payload arrays must all have N>0 frames: {lengths}")
    if not np.all(obs_valid):
        raise RuntimeError("buffered payload requires obs_valid=true on every frame")
    if not np.array_equal(target_times, aliases):
        raise RuntimeError("action timestamp compatibility alias differs from master grid")
    expected_period_ns = int(metadata.get("action_period_ns", 0))
    if expected_period_ns <= 0 or (
        frame_count > 1
        and not np.all(np.abs(np.diff(target_times) - expected_period_ns) <= 1_000)
    ):
        raise RuntimeError("action target timestamps are not a continuous fixed-rate grid")

    annotation = metadata.get("real_skill_annotation")
    if not isinstance(annotation, dict) or not annotation.get("complete"):
        raise RuntimeError("offline scripted annotation is incomplete")
    if annotation.get("mode") != "offline":
        raise RuntimeError("buffered payload requires offline annotation mode")
    if annotation_session is None:
        raise RuntimeError("buffered payload requires an annotation session for audit")

    prompt_config = metadata.get("prompt_depth_anything")
    if not isinstance(prompt_config, dict) or prompt_config.get("online") is not False:
        raise RuntimeError("buffered payload requires offline PromptDA metadata")
    prompt_cameras = set(prompt_config.get("cameras", ()))
    if prompt_cameras != {"front", "wrist"}:
        raise RuntimeError("buffered payload requires PromptDA on both cameras")
    for index, (observation, target_ns) in enumerate(zip(observations, target_times)):
        if int(observation.get("observation_target_wall_time_ns", -1)) != int(target_ns):
            raise RuntimeError(
                f"observations[{index}] is not aligned to its action target time"
            )
        if observation.get("skill") is None:
            raise RuntimeError(f"observations[{index}].skill is missing")
        for camera_name, depth_key in PROMPT_DEPTH_FIELDS.items():
            if camera_name in prompt_cameras and f"{depth_key}_realsense" not in observation:
                raise RuntimeError(
                    f"observations[{index}] is missing preserved raw {depth_key}"
                )

        point = observation.get("guidance_point_clean")
        if point is None:
            point = observation.get("guidance_point")
        pose = observation.get("guidance_pose_clean")
        if pose is None:
            pose = observation.get("guidance_pose")
        expected_points, _ = annotation_session.annotator._camera_projections(
            observation,
            payload["camera_info"],
            point,
            pose,
            observation.get("guidance_gripper_width"),
        )
        stored_points = observation.get("guidance_point_2d")
        if not isinstance(stored_points, dict):
            raise RuntimeError(f"observations[{index}].guidance_point_2d is missing")
        for image_key in ("color_image1", "color_image2"):
            _assert_matching_projection(
                stored_points.get(image_key),
                expected_points.get(image_key),
                f"observations[{index}].guidance_point_2d.{image_key}",
            )

    return {"frames": frame_count, "period_ns": expected_period_ns}


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


def _gripper_width_from_record(gripper_record, fallback=float("nan")):
    if gripper_record is None:
        return float(fallback)
    value = _array_field(gripper_record["message"], "width", size=1)
    return float(value[0]) if np.isfinite(value[0]) else float(fallback)


def build_observation_from_records(
    robot_record,
    gripper_record,
    camera_sample,
    eepose_frame="robot-base",
    gripper_width_fallback=float("nan"),
):
    if robot_record is None or camera_sample is None:
        return None
    return _build_observation_from_records_impl(
        robot_record,
        gripper_record,
        camera_sample,
        eepose_frame,
        gripper_width_fallback,
    )


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
    return build_observation_from_records(
        robot_record,
        gripper_record,
        camera_sample,
        eepose_frame=eepose_frame,
        gripper_width_fallback=_gripper_width(robot_interface),
    )


def _build_observation_from_records_impl(
    robot_record,
    gripper_record,
    camera_sample,
    eepose_frame,
    gripper_width_fallback,
):
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
                "gripper_width": _gripper_width_from_record(
                    gripper_record, gripper_width_fallback
                ),
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


def _copy_mapping_arrays(value):
    return {
        key: item.copy() if isinstance(item, np.ndarray) else item
        for key, item in value.items()
    }


def _serializable_raw_state_records(records):
    """Preserve protobuf state bytes when alignment fails; protobufs cannot pickle."""
    serialized = []
    for record in records:
        copied = _copy_mapping_arrays(record)
        message = copied.get("message")
        if hasattr(message, "SerializeToString"):
            copied.pop("message")
            copied["message_proto_type"] = getattr(
                getattr(message, "DESCRIPTOR", None),
                "full_name",
                type(message).__name__,
            )
            try:
                copied["message_proto_bytes"] = message.SerializeToString()
            except Exception as exc:
                copied["message_serialization_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                copied["message_repr"] = repr(message)
        serialized.append(copied)
    return serialized


def _residual_summary(values_ns):
    values_ms = np.asarray(values_ns, dtype=np.float64) / 1e6
    if values_ms.size == 0:
        return {"count": 0}
    return {
        "count": int(values_ms.size),
        "min_ms": float(np.min(values_ms)),
        "p50_ms": float(np.percentile(values_ms, 50)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "max_ms": float(np.max(values_ms)),
    }


def _nearest_unique_matches(
    target_times_ns, source_items, max_residual_ns, label, *, require_unique=True
):
    """Match ordered samples; only camera frames must be consumed uniquely."""

    ordered = sorted(source_items, key=lambda item: item[0])
    source_times = np.asarray([item[0] for item in ordered], dtype=np.int64)
    if not len(source_times):
        raise RuntimeError(f"{label}: source buffer is empty")
    if np.any(np.diff(source_times) <= 0):
        raise RuntimeError(f"{label}: source timestamps are not strictly increasing")

    selected = []
    residuals = []
    previous_index = -1
    for target_time_ns in target_times_ns:
        insertion = int(np.searchsorted(source_times, int(target_time_ns)))
        first_allowed = previous_index + 1 if require_unique else 0
        candidates = {
            max(first_allowed, insertion - 1),
            max(first_allowed, insertion),
        }
        candidates = [index for index in candidates if index < len(ordered)]
        if not candidates:
            raise RuntimeError(f"{label}: no unused source sample remains")
        index = min(
            candidates,
            key=lambda candidate: abs(
                int(source_times[candidate]) - int(target_time_ns)
            ),
        )
        residual_ns = abs(int(source_times[index]) - int(target_time_ns))
        if residual_ns > int(max_residual_ns):
            raise RuntimeError(
                f"{label}: nearest residual {residual_ns / 1e6:.3f} ms exceeds "
                f"{max_residual_ns / 1e6:.3f} ms"
            )
        selected.append(ordered[index])
        residuals.append(residual_ns)
        previous_index = index
    return selected, residuals


def _nearest_joint_camera_matches(
    target_times_ns,
    front_items,
    wrist_items,
    warning_residual_ns,
    warning_skew_ns,
    hard_gap_ns=200_000_000,
):
    """Choose the nearest camera pair and allow reuse across 10 Hz steps.

    UMI-style downsampling repeats a source frame when a camera drops a frame.
    A coverage gap is reported as an incomplete-quality warning, but does not
    prevent the remaining observations and offline depth processing.
    """

    front = sorted(front_items, key=lambda item: item[0])
    wrist = sorted(wrist_items, key=lambda item: item[0])
    front_times = np.asarray([item[0] for item in front], dtype=np.int64)
    wrist_times = np.asarray([item[0] for item in wrist], dtype=np.int64)
    for name, times in (("front", front_times), ("wrist", wrist_times)):
        if not len(times):
            raise RuntimeError(f"{name}_camera: source buffer is empty")
        if np.any(np.diff(times) <= 0):
            raise RuntimeError(f"{name}_camera: source timestamps are not strictly increasing")

    selected_front = []
    selected_wrist = []
    front_residuals = []
    wrist_residuals = []
    pair_skews = []
    warning_steps = []
    for step, target in enumerate(target_times_ns):
        target = int(target)
        front_start = max(
            0,
            int(np.searchsorted(front_times, target - hard_gap_ns)),
        )
        wrist_start = max(
            0,
            int(np.searchsorted(wrist_times, target - hard_gap_ns)),
        )
        front_end = int(
            np.searchsorted(front_times, target + hard_gap_ns, side="right")
        )
        wrist_end = int(
            np.searchsorted(wrist_times, target + hard_gap_ns, side="right")
        )
        front_candidates = list(range(front_start, front_end))
        wrist_candidates = list(range(wrist_start, wrist_end))
        if not front_candidates:
            front_candidates = [int(np.argmin(np.abs(front_times - target)))]
        if not wrist_candidates:
            wrist_candidates = [int(np.argmin(np.abs(wrist_times - target)))]
        pairs = [
            (front_index, wrist_index)
            for front_index in front_candidates
            for wrist_index in wrist_candidates
        ]
        front_index, wrist_index = min(
            pairs,
            key=lambda pair: (
                abs(int(front_times[pair[0]]) - target)
                + abs(int(wrist_times[pair[1]]) - target),
                abs(int(front_times[pair[0]]) - int(wrist_times[pair[1]])),
            ),
        )
        selected_front.append(front[front_index])
        selected_wrist.append(wrist[wrist_index])
        front_residuals.append(abs(int(front_times[front_index]) - target))
        wrist_residuals.append(abs(int(wrist_times[wrist_index]) - target))
        pair_skews.append(
            abs(int(front_times[front_index]) - int(wrist_times[wrist_index]))
        )
        if (
            front_residuals[-1] > int(warning_residual_ns)
            or wrist_residuals[-1] > int(warning_residual_ns)
            or pair_skews[-1] > int(warning_skew_ns)
        ):
            warning_steps.append(step)
    return (
        selected_front,
        selected_wrist,
        front_residuals,
        wrist_residuals,
        pair_skews,
        warning_steps,
    )


def _causal_latest_matches(target_times_ns, records, effect_time_key, label):
    """Return the latest command already effective at every target time."""

    ordered = sorted(records, key=lambda item: int(item[effect_time_key]))
    if not ordered:
        raise RuntimeError(f"{label}: source buffer is empty")
    effect_times = np.asarray(
        [int(item[effect_time_key]) for item in ordered], dtype=np.int64
    )
    if np.any(np.diff(effect_times) < 0):
        raise RuntimeError(f"{label}: effect timestamps are not increasing")
    matches = []
    ages = []
    for target_time_ns in target_times_ns:
        index = int(np.searchsorted(effect_times, int(target_time_ns), side="right")) - 1
        if index < 0:
            raise RuntimeError(f"{label}: no command is effective at the first target")
        matches.append(ordered[index])
        ages.append(int(target_time_ns) - int(effect_times[index]))
    return matches, ages


def _camera_source_items(camera_samples, camera_name, camera_info):
    sensor_key = f"{camera_name}_sensor_timestamp_ms"
    receive_key = f"{camera_name}_receive_wall_time_ns"
    global_time = bool(camera_info.get(camera_name, {}).get("global_time_enabled"))
    use_sensor_time = global_time
    items = []
    for sample in camera_samples:
        receive_time_ns = int(sample[receive_key])
        sensor_time_ms = sample.get(sensor_key)
        if use_sensor_time:
            if sensor_time_ms is None or not np.isfinite(sensor_time_ms):
                use_sensor_time = False
                break
            sensor_time_ns = int(round(float(sensor_time_ms) * 1e6))
            if abs(sensor_time_ns - receive_time_ns) > 5_000_000_000:
                use_sensor_time = False
                break
    source = "realsense_global_time" if use_sensor_time else "receive_wall_time"
    for sample in camera_samples:
        source_time_ns = (
            int(round(float(sample[sensor_key]) * 1e6))
            if use_sensor_time
            else int(sample[receive_key])
        )
        items.append((source_time_ns, sample))
    return items, source


def _compose_camera_sample(front_match, wrist_match, target_time_ns):
    front_time_ns, front_sample = front_match
    wrist_time_ns, wrist_sample = wrist_match
    combined = _copy_mapping_arrays(front_sample)
    wrist_keys = (
        "color_image1",
        "depth_image1",
        "wrist_receive_wall_time_ns",
        "wrist_sensor_timestamp_ms",
        "wrist_timestamp_domain",
        "wrist_frame_number",
    )
    for key in wrist_keys:
        value = wrist_sample[key]
        combined[key] = value.copy() if isinstance(value, np.ndarray) else value
    combined["front_capture_sequence"] = front_sample.get("capture_sequence")
    combined["wrist_capture_sequence"] = wrist_sample.get("capture_sequence")
    combined["camera_alignment_target_wall_time_ns"] = int(target_time_ns)
    combined["front_alignment_source_wall_time_ns"] = int(front_time_ns)
    combined["wrist_alignment_source_wall_time_ns"] = int(wrist_time_ns)
    return combined


def _interpolated_robot_record(
    robot_records,
    target_time_ns,
    latency_ns,
    max_residual_ns,
):
    ordered = sorted(
        robot_records,
        key=lambda record: int(record["receive_wall_time_ns"]),
    )
    effective_times = np.asarray(
        [int(record["receive_wall_time_ns"]) - int(latency_ns) for record in ordered],
        dtype=np.int64,
    )
    if not len(effective_times):
        raise RuntimeError("robot_state: source buffer is empty")
    if np.any(np.diff(effective_times) <= 0):
        raise RuntimeError("robot_state: timestamps are not strictly increasing")

    target_time_ns = int(target_time_ns)
    insertion = int(np.searchsorted(effective_times, target_time_ns))
    if insertion == 0 or insertion == len(ordered):
        index = 0 if insertion == 0 else len(ordered) - 1
        residual_ns = abs(int(effective_times[index]) - target_time_ns)
        return ordered[index], {
            "mode": "nearest_edge",
            "residual_ns": residual_ns,
            "quality_warning": residual_ns > int(max_residual_ns),
            "source_receive_wall_time_ns": int(
                ordered[index]["receive_wall_time_ns"]
            ),
        }

    left_index = insertion - 1
    right_index = insertion
    left_time = int(effective_times[left_index])
    right_time = int(effective_times[right_index])
    nearest_residual_ns = min(target_time_ns - left_time, right_time - target_time_ns)
    alpha = (target_time_ns - left_time) / float(right_time - left_time)
    left_message = ordered[left_index]["message"]
    right_message = ordered[right_index]["message"]
    left_pose = np.asarray(left_message.O_T_EE, dtype=np.float64).reshape(4, 4).T
    right_pose = np.asarray(right_message.O_T_EE, dtype=np.float64).reshape(4, 4).T
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (1.0 - alpha) * left_pose[:3, 3] + alpha * right_pose[:3, 3]
    rotations = Rotation.from_matrix(
        np.stack([left_pose[:3, :3], right_pose[:3, :3]], axis=0)
    )
    pose[:3, :3] = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]

    def interpolate_field(*names, size):
        left = _array_field(left_message, *names, size=size)
        right = _array_field(right_message, *names, size=size)
        return ((1.0 - alpha) * left + alpha * right).tolist()

    left_source_time = _message_time_seconds(left_message)
    right_source_time = _message_time_seconds(right_message)
    source_time = None
    if left_source_time is not None and right_source_time is not None:
        source_time = (1.0 - alpha) * left_source_time + alpha * right_source_time
    message = SimpleNamespace(
        O_T_EE=pose.T.reshape(-1).tolist(),
        q=interpolate_field("q", size=7),
        dq=interpolate_field("dq", size=7),
        tau_J=interpolate_field("tau_J", "tau_J_d", size=7),
        time=source_time,
        frame=getattr(left_message, "frame", None),
    )
    record = {
        "message": message,
        "receive_wall_time_ns": target_time_ns + int(latency_ns),
    }
    return record, {
        "mode": "linear_translation_joint_slerp_rotation",
        "residual_ns": int(nearest_residual_ns),
        "quality_warning": nearest_residual_ns > int(max_residual_ns),
        "left_receive_wall_time_ns": int(
            ordered[left_index]["receive_wall_time_ns"]
        ),
        "right_receive_wall_time_ns": int(
            ordered[right_index]["receive_wall_time_ns"]
        ),
        "alpha": float(alpha),
    }


def _interpolated_gripper_record(
    gripper_records,
    target_time_ns,
    latency_ns,
    max_residual_ns,
):
    """Interpolate continuous gripper width on the corrected state timeline."""

    ordered = sorted(
        gripper_records,
        key=lambda record: int(record["receive_wall_time_ns"]),
    )
    effective_times = np.asarray(
        [int(record["receive_wall_time_ns"]) - int(latency_ns) for record in ordered],
        dtype=np.int64,
    )
    if not len(effective_times):
        raise RuntimeError("gripper_state: source buffer is empty")
    if np.any(np.diff(effective_times) <= 0):
        raise RuntimeError("gripper_state: timestamps are not strictly increasing")
    target_time_ns = int(target_time_ns)
    insertion = int(np.searchsorted(effective_times, target_time_ns))
    if insertion == 0 or insertion == len(ordered):
        index = 0 if insertion == 0 else len(ordered) - 1
        residual_ns = abs(int(effective_times[index]) - target_time_ns)
        return ordered[index], {
            "mode": "nearest_edge",
            "residual_ns": residual_ns,
            "quality_warning": residual_ns > int(max_residual_ns),
            "source_receive_wall_time_ns": int(
                ordered[index]["receive_wall_time_ns"]
            ),
        }

    left_index = insertion - 1
    right_index = insertion
    left_time = int(effective_times[left_index])
    right_time = int(effective_times[right_index])
    residual_ns = min(target_time_ns - left_time, right_time - target_time_ns)
    alpha = (target_time_ns - left_time) / float(right_time - left_time)
    left_width = _gripper_width_from_record(ordered[left_index])
    right_width = _gripper_width_from_record(ordered[right_index])
    width = (1.0 - alpha) * left_width + alpha * right_width
    left_message = ordered[left_index]["message"]
    right_message = ordered[right_index]["message"]
    left_source_time = _message_time_seconds(left_message)
    right_source_time = _message_time_seconds(right_message)
    source_time = None
    if left_source_time is not None and right_source_time is not None:
        source_time = (1.0 - alpha) * left_source_time + alpha * right_source_time
    record = {
        "message": SimpleNamespace(width=float(width), time=source_time),
        "receive_wall_time_ns": target_time_ns + int(latency_ns),
    }
    return record, {
        "mode": "linear_width",
        "residual_ns": int(residual_ns),
        "quality_warning": residual_ns > int(max_residual_ns),
        "left_receive_wall_time_ns": int(
            ordered[left_index]["receive_wall_time_ns"]
        ),
        "right_receive_wall_time_ns": int(
            ordered[right_index]["receive_wall_time_ns"]
        ),
        "alpha": float(alpha),
    }


def _absolute_wrist_target_to_action(absolute_wrist_action, robot_state, eepose_frame):
    """Convert an absolute wrist target into the existing 8-D RR delta action."""

    absolute_wrist_action = np.asarray(absolute_wrist_action, dtype=np.float64).reshape(7)
    target_wrist_pose = np.eye(4, dtype=np.float64)
    target_wrist_pose[:3, 3] = absolute_wrist_action[:3]
    target_wrist_pose[:3, :3] = Rotation.from_rotvec(
        absolute_wrist_action[3:6]
    ).as_matrix()
    target_pose = eepose_from_wrist_pose(target_wrist_pose, eepose_frame)
    current_pose = np.asarray(robot_state["ee_pose"], dtype=np.float64).reshape(4, 4)
    relative_rotation = current_pose[:3, :3].T @ target_pose[:3, :3]
    action = np.concatenate(
        [
            target_pose[:3, 3] - current_pose[:3, 3],
            transform_utils.mat2quat(relative_rotation),
            [float(np.sign(absolute_wrist_action[-1]))],
        ]
    )
    action_absolute = np.concatenate(
        [
            target_pose[:3, 3],
            transform_utils.mat2quat(target_pose[:3, :3]),
            [float(np.sign(absolute_wrist_action[-1]))],
        ]
    )
    return action.astype(np.float32), action_absolute.astype(np.float32)


def _materialize_buffered_episode_v6(
    action_records,
    camera_samples,
    robot_records,
    gripper_records,
    *,
    camera_info,
    eepose_frame,
    action_period_ns,
    camera_max_residual_ns,
    camera_pair_max_skew_ns,
    robot_max_residual_ns,
    gripper_max_residual_ns,
    robot_latency_ns=0,
    gripper_latency_ns=0,
):
    """Screen and reorder raw timestamp buffers onto the action target grid."""

    ordered_actions = sorted(
        action_records,
        key=lambda record: int(record["timing"]["action_target_wall_time_ns"]),
    )
    if not ordered_actions:
        raise RuntimeError("no executed actions were buffered")
    target_times = np.asarray(
        [
            int(record["timing"]["action_target_wall_time_ns"])
            for record in ordered_actions
        ],
        dtype=np.int64,
    )
    target_intervals = np.diff(target_times)
    if np.any(np.abs(target_intervals - int(action_period_ns)) > 1_000):
        raise RuntimeError(
            "action target grid is discontinuous: expected "
            f"{action_period_ns / 1e6:.3f} ms, got "
            f"{(target_intervals / 1e6).tolist()}"
        )

    front_items, front_time_source = _camera_source_items(
        camera_samples, "front", camera_info
    )
    wrist_items, wrist_time_source = _camera_source_items(
        camera_samples, "wrist", camera_info
    )
    (
        front_matches,
        wrist_matches,
        front_residuals,
        wrist_residuals,
        pair_skews,
    ) = _nearest_joint_camera_matches(
        target_times,
        front_items,
        wrist_items,
        camera_max_residual_ns,
        camera_pair_max_skew_ns,
    )

    gripper_items = [
        (
            int(record["receive_wall_time_ns"]) - int(gripper_latency_ns),
            record,
        )
        for record in gripper_records
    ]
    gripper_matches, gripper_residuals = _nearest_unique_matches(
        target_times,
        gripper_items,
        gripper_max_residual_ns,
        "gripper_state",
        require_unique=False,
    )

    observations = []
    actions = []
    actions_original = []
    actions_absolute = []
    action_timing = []
    robot_residuals = []
    for index, (record, front_match, wrist_match, gripper_match) in enumerate(
        zip(ordered_actions, front_matches, wrist_matches, gripper_matches)
    ):
        target_time_ns = int(target_times[index])
        robot_record, robot_match = _interpolated_robot_record(
            robot_records,
            target_time_ns,
            robot_latency_ns,
            robot_max_residual_ns,
        )
        robot_residuals.append(robot_match["residual_ns"])
        camera_sample = _compose_camera_sample(
            front_match,
            wrist_match,
            target_time_ns,
        )
        gripper_record = gripper_match[1]
        observation = build_observation_from_records(
            robot_record,
            gripper_record,
            camera_sample,
            eepose_frame=eepose_frame,
        )
        if observation is None:
            raise RuntimeError(f"could not build aligned observation {index}")
        observation["control_wall_time_ns"] = target_time_ns
        observation["observation_target_wall_time_ns"] = target_time_ns
        observation["offline_alignment"] = {
            "front_residual_ms": front_residuals[index] / 1e6,
            "wrist_residual_ms": wrist_residuals[index] / 1e6,
            "front_wrist_skew_ms": pair_skews[index] / 1e6,
            "gripper_residual_ms": gripper_residuals[index] / 1e6,
            "robot_state": robot_match,
        }
        scaled_action = np.asarray(record["scaled_action"], dtype=np.float64)
        action = deoxys_delta_to_furniture_bench_action(
            scaled_action,
            observation["robot_state"]["wrist_pose"],
            eepose_frame,
        )
        action_original = deoxys_delta_to_furniture_bench_action(
            scaled_action,
            observation["robot_state"]["wrist_pose"],
            "original",
        )
        observations.append(observation)
        actions.append(np.asarray(action, dtype=np.float32))
        actions_original.append(np.asarray(action_original, dtype=np.float32))
        actions_absolute.append(delta_action_to_absolute(action, observation["robot_state"]))
        timing = dict(record["timing"])
        timing["offline_alignment_index"] = index
        action_timing.append(timing)

    report = {
        "mode": "offline_target_time_buffered",
        "num_buffered_camera_pairs": len(camera_samples),
        "num_buffered_robot_states": len(robot_records),
        "num_buffered_gripper_states": len(gripper_records),
        "num_materialized_steps": len(observations),
        "target_interval": _residual_summary(target_intervals),
        "front_time_source": front_time_source,
        "wrist_time_source": wrist_time_source,
        "front_residual": _residual_summary(front_residuals),
        "wrist_residual": _residual_summary(wrist_residuals),
        "front_wrist_skew": _residual_summary(pair_skews),
        "robot_residual": _residual_summary(robot_residuals),
        "gripper_residual": _residual_summary(gripper_residuals),
        "gripper_reused_steps": sum(
            current[1] is previous[1]
            for previous, current in zip(gripper_matches, gripper_matches[1:])
        ),
    }
    return {
        "observations": observations,
        "actions": actions,
        "actions_original": actions_original,
        "actions_absolute": actions_absolute,
        "action_timing": action_timing,
        "report": report,
    }


def materialize_buffered_episode(
    arm_action_records,
    gripper_action_records,
    camera_samples,
    robot_records,
    gripper_records,
    *,
    grid_start_wall_time_ns,
    grid_end_wall_time_ns,
    camera_info,
    eepose_frame,
    action_period_ns,
    camera_max_residual_ns,
    camera_pair_max_skew_ns,
    robot_max_residual_ns,
    gripper_max_residual_ns,
    camera_hard_gap_ns=200_000_000,
    robot_latency_ns=0,
    gripper_latency_ns=0,
):
    """Materialize asynchronous raw streams on a fixed 10 Hz timeline."""

    if not arm_action_records:
        raise RuntimeError("no successful absolute arm commands were buffered")
    if not gripper_action_records:
        raise RuntimeError("no gripper command state was buffered")
    grid_start_wall_time_ns = int(grid_start_wall_time_ns)
    grid_end_wall_time_ns = int(grid_end_wall_time_ns)
    if grid_end_wall_time_ns < grid_start_wall_time_ns:
        raise RuntimeError("episode grid end precedes its start")

    front_items, front_time_source = _camera_source_items(
        camera_samples, "front", camera_info
    )
    wrist_items, wrist_time_source = _camera_source_items(
        camera_samples, "wrist", camera_info
    )
    if not front_items or not wrist_items or not robot_records or not gripper_records:
        raise RuntimeError("one or more observation buffers are empty")
    robot_effective_times = [
        int(record["receive_wall_time_ns"]) - int(robot_latency_ns)
        for record in robot_records
    ]
    gripper_state_effective_times = [
        int(record["receive_wall_time_ns"]) - int(gripper_latency_ns)
        for record in gripper_records
    ]
    coverage_start_ns = max(
        grid_start_wall_time_ns,
        min(time_ns for time_ns, _ in front_items) - int(camera_hard_gap_ns),
        min(time_ns for time_ns, _ in wrist_items) - int(camera_hard_gap_ns),
        min(robot_effective_times) - int(robot_max_residual_ns),
        min(gripper_state_effective_times) - int(gripper_max_residual_ns),
        min(int(record["predicted_effect_wall_time_ns"]) for record in arm_action_records),
        min(
            int(record["predicted_effect_wall_time_ns"])
            for record in gripper_action_records
        ),
    )
    coverage_end_ns = min(
        grid_end_wall_time_ns,
        max(time_ns for time_ns, _ in front_items) + int(camera_hard_gap_ns),
        max(time_ns for time_ns, _ in wrist_items) + int(camera_hard_gap_ns),
        max(robot_effective_times) + int(robot_max_residual_ns),
        max(gripper_state_effective_times) + int(gripper_max_residual_ns),
    )
    first_grid_index = max(
        0,
        int(
            np.ceil(
                (coverage_start_ns - grid_start_wall_time_ns)
                / float(action_period_ns)
            )
        ),
    )
    last_grid_index = int(
        np.floor(
            (coverage_end_ns - grid_start_wall_time_ns) / float(action_period_ns)
        )
    )
    if last_grid_index < first_grid_index:
        raise RuntimeError("raw streams have no common fixed-rate coverage")
    target_times = (
        grid_start_wall_time_ns
        + np.arange(first_grid_index, last_grid_index + 1, dtype=np.int64)
        * int(action_period_ns)
    )
    target_intervals = np.diff(target_times)

    (
        front_matches,
        wrist_matches,
        front_residuals,
        wrist_residuals,
        pair_skews,
        camera_warning_steps,
    ) = _nearest_joint_camera_matches(
        target_times,
        front_items,
        wrist_items,
        camera_max_residual_ns,
        camera_pair_max_skew_ns,
        camera_hard_gap_ns,
    )
    arm_matches, arm_action_ages = _causal_latest_matches(
        target_times,
        arm_action_records,
        "predicted_effect_wall_time_ns",
        "arm_action",
    )
    gripper_action_matches, gripper_action_ages = _causal_latest_matches(
        target_times,
        gripper_action_records,
        "predicted_effect_wall_time_ns",
        "gripper_action",
    )

    observations = []
    actions = []
    actions_original = []
    actions_absolute = []
    action_timing = []
    robot_residuals = []
    gripper_state_residuals = []
    robot_state_warning_steps = []
    gripper_state_warning_steps = []
    for index, (
        arm_record,
        gripper_action_record,
        front_match,
        wrist_match,
    ) in enumerate(
        zip(
            arm_matches,
            gripper_action_matches,
            front_matches,
            wrist_matches,
        )
    ):
        target_time_ns = int(target_times[index])
        robot_record, robot_match = _interpolated_robot_record(
            robot_records,
            target_time_ns,
            robot_latency_ns,
            robot_max_residual_ns,
        )
        robot_residuals.append(robot_match["residual_ns"])
        if robot_match["quality_warning"]:
            robot_state_warning_steps.append(index)
        gripper_record, gripper_state_match = _interpolated_gripper_record(
            gripper_records,
            target_time_ns,
            gripper_latency_ns,
            gripper_max_residual_ns,
        )
        gripper_state_residuals.append(gripper_state_match["residual_ns"])
        if gripper_state_match["quality_warning"]:
            gripper_state_warning_steps.append(index)
        camera_sample = _compose_camera_sample(
            front_match,
            wrist_match,
            target_time_ns,
        )
        observation = build_observation_from_records(
            robot_record,
            gripper_record,
            camera_sample,
            eepose_frame=eepose_frame,
        )
        if observation is None:
            raise RuntimeError(f"could not build aligned observation {index}")
        observation["control_wall_time_ns"] = target_time_ns
        observation["observation_target_wall_time_ns"] = target_time_ns
        observation["offline_alignment"] = {
            "front_residual_ms": front_residuals[index] / 1e6,
            "wrist_residual_ms": wrist_residuals[index] / 1e6,
            "front_wrist_skew_ms": pair_skews[index] / 1e6,
            "camera_quality_warning": index in camera_warning_steps,
            "robot_state": robot_match,
            "gripper_state": gripper_state_match,
        }

        if arm_record.get("scaled_physical_delta") is not None:
            scaled_action = np.asarray(
                arm_record["scaled_physical_delta"], dtype=np.float64
            ).copy()
            scaled_action[-1] = float(
                np.sign(gripper_action_record["gripper_action"])
            )
            action = deoxys_delta_to_furniture_bench_action(
                scaled_action,
                observation["robot_state"]["wrist_pose"],
                eepose_frame,
            )
            action_original = deoxys_delta_to_furniture_bench_action(
                scaled_action,
                observation["robot_state"]["wrist_pose"],
                "original",
            )
            action_absolute = delta_action_to_absolute(
                action,
                observation["robot_state"],
            ).astype(np.float32)
        else:
            absolute_wrist_action = np.asarray(
                arm_record["absolute_wrist_action"], dtype=np.float64
            ).copy()
            absolute_wrist_action[-1] = float(
                np.sign(gripper_action_record["gripper_action"])
            )
            action, action_absolute = _absolute_wrist_target_to_action(
                absolute_wrist_action,
                observation["robot_state"],
                eepose_frame,
            )
            action_original, _ = _absolute_wrist_target_to_action(
                absolute_wrist_action,
                observation["robot_state"],
                "original",
            )
        observations.append(observation)
        actions.append(action)
        actions_original.append(action_original)
        actions_absolute.append(action_absolute)
        action_timing.append(
            {
                "action_target_wall_time_ns": target_time_ns,
                "action_wall_time_ns": target_time_ns,
                "episode_grid_start_wall_time_ns": grid_start_wall_time_ns,
                "grid_index": first_grid_index + index,
                "action_period_ns": int(action_period_ns),
                "status": "materialized",
                "arm_source_sample_wall_time_ns": arm_record.get(
                    "sample_wall_time_ns"
                ),
                "arm_source_command_wall_time_ns": arm_record.get(
                    "command_wall_time_ns"
                ),
                "arm_source_effect_wall_time_ns": int(
                    arm_record["predicted_effect_wall_time_ns"]
                ),
                "arm_action_age_ms": arm_action_ages[index] / 1e6,
                "gripper_source_command_wall_time_ns": gripper_action_record.get(
                    "command_wall_time_ns"
                ),
                "gripper_source_effect_wall_time_ns": int(
                    gripper_action_record["predicted_effect_wall_time_ns"]
                ),
                "gripper_action_age_ms": gripper_action_ages[index] / 1e6,
                "camera_quality_warning": index in camera_warning_steps,
                "robot_state_alignment": robot_match,
                "gripper_state_alignment": gripper_state_match,
                "offline_alignment_index": index,
            }
        )

    report = {
        "mode": "umi_style_absolute_command_effect_time_alignment",
        "grid_start_wall_time_ns": grid_start_wall_time_ns,
        "grid_end_wall_time_ns": grid_end_wall_time_ns,
        "first_materialized_grid_index": first_grid_index,
        "last_materialized_grid_index": last_grid_index,
        "trimmed_leading_ms": (
            int(target_times[0]) - grid_start_wall_time_ns
        ) / 1e6,
        "trimmed_trailing_ms": (
            grid_end_wall_time_ns - int(target_times[-1])
        ) / 1e6,
        "num_raw_arm_commands": len(arm_action_records),
        "num_raw_gripper_commands": len(gripper_action_records),
        "num_buffered_camera_pairs": len(camera_samples),
        "num_buffered_robot_states": len(robot_records),
        "num_buffered_gripper_states": len(gripper_records),
        "num_materialized_steps": len(observations),
        "target_interval": _residual_summary(target_intervals),
        "front_time_source": front_time_source,
        "wrist_time_source": wrist_time_source,
        "front_residual": _residual_summary(front_residuals),
        "wrist_residual": _residual_summary(wrist_residuals),
        "front_wrist_skew": _residual_summary(pair_skews),
        "camera_warning_steps": camera_warning_steps,
        "camera_hard_gap_steps": [
            index
            for index, (front_residual, wrist_residual) in enumerate(
                zip(front_residuals, wrist_residuals)
            )
            if max(front_residual, wrist_residual) > int(camera_hard_gap_ns)
        ],
        "camera_reused_front_steps": sum(
            current[1] is previous[1]
            for previous, current in zip(front_matches, front_matches[1:])
        ),
        "camera_reused_wrist_steps": sum(
            current[1] is previous[1]
            for previous, current in zip(wrist_matches, wrist_matches[1:])
        ),
        "robot_residual": _residual_summary(robot_residuals),
        "gripper_state_residual": _residual_summary(gripper_state_residuals),
        "robot_state_warning_steps": robot_state_warning_steps,
        "gripper_state_warning_steps": gripper_state_warning_steps,
        "arm_action_age": _residual_summary(arm_action_ages),
        "gripper_action_age": _residual_summary(gripper_action_ages),
    }
    return {
        "observations": observations,
        "actions": actions,
        "actions_original": actions_original,
        "actions_absolute": actions_absolute,
        "action_timing": action_timing,
        "report": report,
    }


def apply_prompt_depth_offline(observations, estimator, cameras):
    """Enhance every selected observation after target-time materialization."""

    if estimator is None:
        return {"enabled": False, "frame_count": 0, "inference_ms": 0.0}
    camera_fields = {
        "wrist": ("color_image1", "depth_image1"),
        "front": ("color_image2", "depth_image2"),
    }
    last_usable_depth = {}
    total_inference_ms = 0.0
    for frame_index, observation in enumerate(observations):
        frame_stats = {}
        started_wall_time_ns = time.time_ns()
        for camera_name in cameras:
            color_key, depth_key = camera_fields[camera_name]
            raw_depth = np.asarray(observation[depth_key]).copy()
            prompt_depth = None
            if has_usable_depth(
                raw_depth,
                estimator.min_depth_m,
                estimator.max_depth_m,
            ):
                last_usable_depth[camera_name] = raw_depth
            else:
                prompt_depth = last_usable_depth.get(camera_name)
            try:
                enhanced, stats = estimator.enhance(
                    observation[color_key],
                    raw_depth,
                    prompt_depth_m=prompt_depth,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"PromptDA frame {frame_index} {camera_name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if enhanced.shape != raw_depth.shape:
                raise RuntimeError(
                    f"PromptDA frame {frame_index} {depth_key} shape mismatch: "
                    f"{enhanced.shape} vs {raw_depth.shape}"
                )
            observation[f"{depth_key}_realsense"] = raw_depth
            observation[depth_key] = enhanced.astype(np.float16)
            frame_stats[camera_name] = dict(stats)
            total_inference_ms += float(stats.get("inference_ms", 0.0))
        observation["prompt_depth_started_wall_time_ns"] = started_wall_time_ns
        observation["prompt_depth_ready_wall_time_ns"] = time.time_ns()
        observation["prompt_depth_stats"] = frame_stats
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(observations):
            logger.info(
                "Offline PromptDA processed %d/%d selected frames",
                frame_index + 1,
                len(observations),
            )
    return {
        "enabled": True,
        "frame_count": len(observations),
        "camera_inference_count": len(observations) * len(tuple(cameras)),
        "inference_ms": total_inference_ms,
    }


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


def _draw_real_skill_annotation(
    image_bgr, observation, image_key="color_image2", status=None, show_text=False
):
    """Draw this camera's live target and optional FSM text on a preview copy."""
    preview = image_bgr.copy()
    if not observation and status is None:
        return preview

    point = None
    projections = observation.get("guidance_point_2d") if observation else None
    if isinstance(projections, dict):
        point = projections.get(image_key)
    marker_xy = None
    if point is not None:
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        if point.size == 2 and np.all(np.isfinite(point)):
            source = observation.get(image_key)
            source_height, source_width = preview.shape[:2]
            if isinstance(source, np.ndarray) and source.ndim >= 2:
                source_height, source_width = source.shape[:2]
            x = int(round(point[0] * preview.shape[1] / source_width))
            y = int(round(point[1] * preview.shape[0] / source_height))
            if 0 <= x < preview.shape[1] and 0 <= y < preview.shape[0]:
                marker_xy = (x, y)
    if show_text:
        skill = observation.get("skill") if observation else None
        skill_state = observation.get("skill_state") if observation else None
        if isinstance(skill, bytes):
            skill = skill.decode("utf-8")
        if isinstance(skill_state, bytes):
            skill_state = skill_state.decode("utf-8")
        label = f"FSM: {skill or 'NONE'}"
        if skill_state:
            label += f" / {skill_state}"
        if not observation:
            label = f"FSM: {status or 'WAITING'}"
        cv2.putText(
            preview,
            label,
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if observation:
            def point_text(key):
                value = projections.get(key) if isinstance(projections, dict) else None
                if value is None:
                    return "--"
                value = np.asarray(value, dtype=np.float64).reshape(-1)
                if value.size != 2 or not np.all(np.isfinite(value)):
                    return "--"
                return f"{value[0]:.0f},{value[1]:.0f}"

            cv2.putText(
                preview,
                f"TARGET W=({point_text('color_image1')}) "
                f"F=({point_text('color_image2')})",
                (8, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if marker_xy is not None:
        cv2.drawMarker(
            preview,
            marker_xy,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(preview, marker_xy, 8, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


def _build_camera_preview(
    camera_sample,
    camera_info,
    episode_state,
    draw_part_poses,
    task_name="one_leg",
    annotation_observation=None,
    annotation_status=None,
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
    wrist = _draw_real_skill_annotation(
        wrist,
        annotation_observation,
        image_key="color_image1",
        status=annotation_status,
        show_text=True,
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
    quality = payload.get("save_quality", {})
    report_lines = [
        f"pickle: {output_path.name}",
        f"status: {quality.get('status', 'unknown')}",
        f"observations: {len(payload.get('observations', []))}",
        f"actions: {len(payload.get('actions', []))}",
        f"raw_camera_samples: {len(payload.get('raw_camera_samples', []))}",
        f"raw_arm_commands: {len(payload.get('raw_arm_commands_absolute', []))}",
        f"raw_gripper_commands: {len(payload.get('raw_gripper_commands', []))}",
        f"camera_sequence_gaps: {payload.get('metadata', {}).get('camera_sequence_gap_count', 0)}",
        "",
        "issues:",
    ]
    for issue in quality.get("issues", []):
        report_lines.append(
            f"- [{issue['time']}] {issue['phase']}: {issue['message']}"
        )
        if issue.get("traceback"):
            report_lines.append(issue["traceback"].rstrip())
    if not quality.get("issues"):
        report_lines.append("- none")
    report_lines.extend(
        [
            "",
            "alignment_report:",
            json.dumps(payload.get("alignment_report"), indent=2, default=str),
        ]
    )
    report_path = output_path.with_suffix(".txt")
    temporary_report_path = report_path.with_suffix(".txt.tmp")
    temporary_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(temporary_report_path, report_path)
    if save_video:
        try:
            _write_video_atomic(
                output_path.with_suffix(".mp4"),
                payload["observations"],
                video_fps,
            )
        except Exception as exc:
            with report_path.open("a", encoding="utf-8") as report_file:
                report_file.write(
                    f"\nvideo_write_error: {type(exc).__name__}: {exc}\n"
                    + traceback.format_exc()
                )
            logger.exception("Episode video failed; pickle and quality log were saved")
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


class AsyncObservationPreview:
    """Build live observations and render the dashboard off the control thread."""

    def __init__(
        self,
        *,
        camera,
        robot_interface,
        camera_info,
        eepose_frame,
        task_name,
        draw_part_poses,
        annotation_session_factory,
        enable_annotation,
        show_window,
        depth_min_m,
        depth_max_m,
        depth_colormap,
        refresh_hz=30.0,
    ):
        self.camera = camera
        self.robot_interface = robot_interface
        self.camera_info = camera_info
        self.eepose_frame = eepose_frame
        self.task_name = task_name
        self.draw_part_poses = bool(draw_part_poses)
        self.annotation_session_factory = annotation_session_factory
        self.enable_annotation = bool(enable_annotation)
        self.show_window = bool(show_window)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.depth_colormap = depth_colormap
        self.refresh_period_s = 1.0 / float(refresh_hz)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="observation_preview",
        )
        self.future = None
        self.next_submit_monotonic = 0.0
        self.latest_observation = None
        self.window_keys = []
        self.annotation_session = None
        self.annotation_capture_ns = None
        self.last_annotation_observation = None
        self.annotation_error = None
        self.annotation_generation = 0
        self.last_worker_error = None

    @staticmethod
    def _run_once(
        camera,
        robot_interface,
        camera_info,
        eepose_frame,
        task_name,
        draw_part_poses,
        preview_state,
        annotation_session_factory,
        enable_annotation,
        annotation_session,
        annotation_capture_ns,
        last_annotation_observation,
        annotation_error,
        annotation_generation,
        show_window,
        depth_min_m,
        depth_max_m,
        depth_colormap,
    ):
        camera_sample = camera.latest()
        observation = build_observation(
            robot_interface,
            camera_sample,
            eepose_frame,
        )
        if enable_annotation and observation is not None:
            capture_ns = observation.get("camera_capture_wall_time_ns")
            if capture_ns is None or capture_ns != annotation_capture_ns:
                try:
                    if annotation_session is None:
                        annotation_session = annotation_session_factory(
                            task_name, camera_info
                        )
                    annotation_session.annotate_observation(observation)
                    last_annotation_observation = observation
                    annotation_capture_ns = capture_ns
                    annotation_error = None
                except Exception as exc:
                    annotation_error = f"{type(exc).__name__}: {exc}"
                    annotation_session = None
                    last_annotation_observation = None

        window_key = None
        if show_window:
            annotation_status = None
            if not enable_annotation:
                annotation_status = "OFF: ADD --real-skill-annotation"
            elif annotation_error:
                annotation_status = f"ERROR {annotation_error[:36]}"
                last_annotation_observation = None
            elif last_annotation_observation is None:
                annotation_status = "WAITING FOR POSES"
            preview = _build_camera_preview(
                camera_sample,
                camera_info,
                preview_state,
                draw_part_poses,
                task_name=task_name,
                annotation_observation=last_annotation_observation,
                annotation_status=annotation_status,
                prompt_depth_result=None,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
                depth_colormap=depth_colormap,
            )
            if preview is not None:
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                key_code = cv2.waitKey(1) & 0xFF
                if key_code in map(ord, "besfdrpq"):
                    window_key = chr(key_code)
        return {
            "observation": observation,
            "window_key": window_key,
            "annotation_session": annotation_session,
            "annotation_capture_ns": annotation_capture_ns,
            "last_annotation_observation": last_annotation_observation,
            "annotation_error": annotation_error,
            "annotation_generation": annotation_generation,
        }

    def request_annotation_reset(self):
        self.annotation_generation += 1
        self.annotation_session = None
        self.annotation_capture_ns = None
        self.last_annotation_observation = None
        self.annotation_error = None

    def pump(self, preview_state):
        """Poll without waiting and enqueue at most one background refresh."""

        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                self.latest_observation = result["observation"]
                if result["window_key"] is not None:
                    self.window_keys.append(result["window_key"])
                if result["annotation_generation"] == self.annotation_generation:
                    previous_error = self.annotation_error
                    self.annotation_session = result["annotation_session"]
                    self.annotation_capture_ns = result["annotation_capture_ns"]
                    self.last_annotation_observation = result[
                        "last_annotation_observation"
                    ]
                    self.annotation_error = result["annotation_error"]
                    if self.annotation_error and self.annotation_error != previous_error:
                        logger.warning(
                            "Live annotation is waiting for usable poses: %s",
                            self.annotation_error,
                        )
                self.last_worker_error = None
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != self.last_worker_error:
                    logger.warning("Async observation preview failed: %s", message)
                    self.last_worker_error = message
            finally:
                self.future = None

        now = time.monotonic()
        if self.future is None and now >= self.next_submit_monotonic:
            generation = self.annotation_generation
            self.future = self.executor.submit(
                self._run_once,
                self.camera,
                self.robot_interface,
                self.camera_info,
                self.eepose_frame,
                self.task_name,
                self.draw_part_poses,
                preview_state,
                self.annotation_session_factory,
                self.enable_annotation,
                self.annotation_session,
                self.annotation_capture_ns,
                self.last_annotation_observation,
                self.annotation_error,
                generation,
                self.show_window,
                self.depth_min_m,
                self.depth_max_m,
                self.depth_colormap,
            )
            self.next_submit_monotonic = now + self.refresh_period_s

    def read_window_keys(self):
        keys = self.window_keys
        self.window_keys = []
        return keys

    def close(self):
        self.executor.shutdown(wait=True)


class AsyncCameraBufferCollector:
    """Copy raw camera history into an episode without touching the control loop."""

    def __init__(self, *, camera, episode, history_cursor, poll_s=0.01):
        self.camera = camera
        self.episode = episode
        self.history_cursor = history_cursor
        self.poll_s = float(poll_s)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="camera_episode_buffer",
            daemon=True,
        )
        self.started = False

    def start(self):
        if not self.started:
            self.started = True
            self.thread.start()
        return self

    def _drain_once(self):
        samples, self.history_cursor = self.camera.samples_since(
            self.history_cursor
        )
        self.episode.add_camera_samples(samples)

    def _run(self):
        while not self.stop_event.wait(self.poll_s):
            try:
                self._drain_once()
            except Exception as exc:
                self.episode.mark_buffer_error(
                    f"camera history drain failed: {type(exc).__name__}: {exc}"
                )
                return

    def stop(self):
        if not self.started:
            return
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            self.episode.mark_buffer_error(
                "camera history collector did not stop within 2 seconds"
            )
            return
        try:
            self._drain_once()
        except Exception as exc:
            self.episode.mark_buffer_error(
                f"final camera history drain failed: {type(exc).__name__}: {exc}"
            )


class _StrictGridActionDispatcherV6:
    """Dispatch arm and gripper commands on independent single-thread lanes."""

    def __init__(
        self,
        *,
        robot_interface,
        controller_type,
        controller_cfg,
        schedule,
        action_period_ns,
    ):
        self.robot_interface = robot_interface
        self.controller_type = controller_type
        self.controller_cfg = controller_cfg
        self.schedule = schedule
        self.action_period_ns = int(action_period_ns)
        self.cancel_event = threading.Event()
        self.executors = {
            "robot": concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="robot_action"
            ),
            "gripper": concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="gripper_action"
            ),
        }
        self.pending = []
        self.closed = False

    def _wait_for_deadline(self, deadline_monotonic):
        delay_s = float(deadline_monotonic) - time.monotonic()
        if delay_s > 0 and self.cancel_event.wait(delay_s):
            return False
        return not self.cancel_event.is_set()

    def _dispatch_channel(self, channel, command, channel_action):
        if not self._wait_for_deadline(command[f"{channel}_deadline_monotonic"]):
            return {"channel": channel, "status": "cancelled"}

        target_wall_time_ns = int(command["target_wall_time_ns"])
        started_wall_time_ns = time.time_ns()
        if self.schedule.dispatch_expired(
            target_wall_time_ns, started_wall_time_ns, channel
        ):
            return {
                "channel": channel,
                "status": "dropped",
                "drop_reason": f"stale_{channel}_deadline",
                "started_wall_time_ns": started_wall_time_ns,
            }

        started_monotonic = time.monotonic()
        try:
            if channel == "robot":
                result = self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=np.asarray(channel_action, dtype=np.float64).copy(),
                    controller_cfg=self.controller_cfg,
                    control_gripper=False,
                    enforce_control_frequency=False,
                )
                send_wall_time_ns = result["robot_command_wall_time_ns"]
            else:
                self.robot_interface.gripper_control(
                    float(channel_action[-1])
                )
                send_wall_time_ns = (
                    self.robot_interface.last_gripper_command_wall_time_ns
                )
        except Exception as exc:
            return {
                "channel": channel,
                "status": "error",
                "drop_reason": f"{channel}_send_error",
                "error": f"{type(exc).__name__}: {exc}",
                "started_wall_time_ns": started_wall_time_ns,
                "elapsed_ms": (time.monotonic() - started_monotonic) * 1e3,
            }

        elapsed_ms = (time.monotonic() - started_monotonic) * 1e3
        deadline_wall_time_ns = self.schedule.deadline_ns(
            target_wall_time_ns, channel
        )
        completion_lateness_ns = int(send_wall_time_ns) - deadline_wall_time_ns
        status = (
            "late_send"
            if completion_lateness_ns
            > self.schedule.dispatch_tolerance_ns
            else "executed"
        )
        return {
            "channel": channel,
            "status": status,
            "drop_reason": (
                f"late_{channel}_send_completion"
                if status == "late_send"
                else None
            ),
            "started_wall_time_ns": started_wall_time_ns,
            "send_wall_time_ns": int(send_wall_time_ns),
            "elapsed_ms": elapsed_ms,
            "deadline_lateness_ms": completion_lateness_ns / 1e6,
            "send_residual_ms": (
                int(send_wall_time_ns) - target_wall_time_ns
            )
            / 1e6,
        }

    def submit_channel(self, command, channel, sampled_action):
        if self.closed:
            raise RuntimeError("action dispatcher is closed")
        if channel not in ("robot", "gripper"):
            raise ValueError(f"unsupported action channel {channel!r}")
        target_wall_time_ns = int(command["target_wall_time_ns"])
        record = next(
            (
                pending
                for pending in self.pending
                if int(pending["target_wall_time_ns"]) == target_wall_time_ns
            ),
            None,
        )
        if record is None:
            record = dict(command)
            record["futures"] = {}
            record["channel_actions"] = {}
            self.pending.append(record)
            self.pending.sort(key=lambda item: int(item["target_wall_time_ns"]))
        if channel in record["futures"]:
            raise RuntimeError(
                f"duplicate {channel} action for target {target_wall_time_ns}"
            )
        channel_action = np.asarray(sampled_action, dtype=np.float64).copy()
        record["channel_actions"][channel] = channel_action
        sample_wall_time_ns = int(command[f"{channel}_sample_wall_time_ns"])
        record["timing"][f"{channel}_sample_wall_time_ns"] = sample_wall_time_ns
        record["timing"][f"{channel}_sample_lead_ms"] = (
            target_wall_time_ns - sample_wall_time_ns
        ) / 1e6
        record["futures"][channel] = self.executors[channel].submit(
            self._dispatch_channel, channel, record, channel_action
        )

    def _finalize_record(self, record, episode, cancel_reason=None):
        timing = dict(record["timing"])
        results = {
            channel: record["futures"][channel].result()
            for channel in ("robot", "gripper")
        }
        failed = []
        for channel, result in results.items():
            timing[f"{channel}_dispatch_status"] = result["status"]
            if result.get("started_wall_time_ns") is not None:
                timing[f"{channel}_command_start_wall_time_ns"] = result[
                    "started_wall_time_ns"
                ]
            if result.get("send_wall_time_ns") is not None:
                timing[f"{channel}_command_wall_time_ns"] = result[
                    "send_wall_time_ns"
                ]
                timing[f"{channel}_send_residual_ms"] = result[
                    "send_residual_ms"
                ]
            if result.get("elapsed_ms") is not None:
                timing[f"{channel}_control_elapsed_ms"] = result["elapsed_ms"]
                timing[f"{channel}_deadline_lateness_ms"] = result.get(
                    "deadline_lateness_ms"
                )
                if result["elapsed_ms"] > 5.0:
                    logger.warning(
                        "Timing probe slow async send grid=%s channel=%s "
                        "elapsed_ms=%.3f target_residual_ms=%s",
                        timing.get("grid_index"),
                        channel,
                        result["elapsed_ms"],
                        result.get("send_residual_ms"),
                    )
            if result["status"] != "executed":
                failed.append(result)

        if failed:
            reason = cancel_reason or failed[0].get(
                "drop_reason", failed[0]["status"]
            )
            timing.update(
                status=(
                    "cancelled"
                    if all(result["status"] == "cancelled" for result in failed)
                    else "partial_dispatch_failure"
                ),
                drop_reason=reason,
                dropped_wall_time_ns=time.time_ns(),
            )
            errors = [result.get("error") for result in failed if result.get("error")]
            if errors:
                timing["dispatch_errors"] = errors
            episode.record_dropped_command(
                timing,
                invalidate_continuity=cancel_reason is None,
            )
            return

        combined_action = np.zeros(7, dtype=np.float64)
        combined_action[:6] = record["channel_actions"]["robot"][:6]
        combined_action[-1] = record["channel_actions"]["gripper"][-1]
        timing["status"] = "executed"
        episode.append_buffered(
            scaled_deoxys_action(combined_action, self.controller_cfg),
            action_timing=timing,
        )

    def pump(self, episode):
        """Finalize completed commands in grid order without blocking."""

        completed = 0
        while (
            self.pending
            and set(self.pending[0]["futures"]) == {"robot", "gripper"}
            and all(
                future.done() for future in self.pending[0]["futures"].values()
            )
        ):
            record = self.pending.pop(0)
            self._finalize_record(record, episode)
            completed += 1
        return completed

    def flush_and_close(self, episode, timeout_s=2.0):
        deadline = time.monotonic() + float(timeout_s)
        while (
            any(
                set(record["futures"]) == {"robot", "gripper"}
                for record in self.pending
            )
            and time.monotonic() < deadline
        ):
            self.pump(episode)
            if self.pending:
                time.sleep(0.001)
        complete_pending = [
            record
            for record in self.pending
            if set(record["futures"]) == {"robot", "gripper"}
        ]
        if complete_pending:
            self.cancel_event.set()
            concurrent.futures.wait(
                [
                    future
                    for record in complete_pending
                    for future in record["futures"].values()
                ],
                timeout=2.0,
            )
            while (
                self.pending
                and set(self.pending[0]["futures"]) == {"robot", "gripper"}
                and all(
                    future.done() for future in self.pending[0]["futures"].values()
                )
            ):
                record = self.pending.pop(0)
                self._finalize_record(
                    record, episode, cancel_reason="action_dispatch_flush_timeout"
                )
        for record in self.pending:
            timing = dict(record["timing"])
            timing.update(
                status="tail_incomplete",
                drop_reason="operator_stopped_before_all_channel_samples",
                dropped_wall_time_ns=time.time_ns(),
            )
            episode.record_dropped_command(timing, invalidate_continuity=False)
        self.pending = []
        self.close()

    def cancel_and_close(self, episode, reason):
        self.cancel_event.set()
        concurrent.futures.wait(
            [
                future
                for record in self.pending
                for future in record["futures"].values()
            ],
            timeout=2.0,
        )
        for record in self.pending:
            if all(future.done() for future in record["futures"].values()):
                if set(record["futures"]) == {"robot", "gripper"}:
                    self._finalize_record(record, episode, cancel_reason=reason)
                else:
                    timing = dict(record["timing"])
                    timing.update(
                        status="cancelled",
                        drop_reason=reason,
                        dropped_wall_time_ns=time.time_ns(),
                    )
                    episode.record_dropped_command(
                        timing, invalidate_continuity=False
                    )
        self.pending = []
        self.close()

    @property
    def pending_count(self):
        return len(self.pending)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for executor in self.executors.values():
            executor.shutdown(wait=True)


class AsyncActionDispatcher:
    """Independent latest-wins lanes for absolute arm and gripper targets."""

    def __init__(
        self,
        *,
        robot_interface,
        controller_type,
        controller_cfg,
        arm_latency_ns,
        gripper_latency_ns,
    ):
        self.robot_interface = robot_interface
        self.controller_type = controller_type
        self.controller_cfg = controller_cfg
        self.latencies = {
            "robot": int(arm_latency_ns),
            "gripper": int(gripper_latency_ns),
        }
        self.executors = {
            channel: concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{channel}_latest_action"
            )
            for channel in ("robot", "gripper")
        }
        self.lock = threading.RLock()
        self.active = {"robot": None, "gripper": None}
        self.queued = {"robot": None, "gripper": None}
        self.completed = []
        self.accepting = True
        self.closed = False
        self.overwritten = {"robot": 0, "gripper": 0}

    def _send(self, channel, command):
        record = dict(command)
        record["channel"] = channel
        record["command_start_wall_time_ns"] = time.time_ns()
        started_monotonic = time.monotonic()
        try:
            if channel == "robot":
                result = self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=np.asarray(
                        command["absolute_wrist_action"], dtype=np.float64
                    ).copy(),
                    controller_cfg=self.controller_cfg,
                    control_gripper=False,
                )
                command_wall_time_ns = int(result["robot_command_wall_time_ns"])
            else:
                self.robot_interface.gripper_control(
                    float(command["gripper_action"])
                )
                command_wall_time_ns = int(
                    self.robot_interface.last_gripper_command_wall_time_ns
                )
        except Exception as exc:
            record.update(
                status="send_error",
                error=f"{type(exc).__name__}: {exc}",
                control_elapsed_ms=(time.monotonic() - started_monotonic) * 1e3,
            )
            return record
        record.update(
            status="executed",
            command_wall_time_ns=command_wall_time_ns,
            predicted_effect_wall_time_ns=(
                command_wall_time_ns + self.latencies[channel]
            ),
            control_elapsed_ms=(time.monotonic() - started_monotonic) * 1e3,
        )
        return record

    def _start_locked(self, channel, command):
        future = self.executors[channel].submit(self._send, channel, command)
        self.active[channel] = future
        future.add_done_callback(
            lambda completed, lane=channel: self._on_done(lane, completed)
        )

    def _on_done(self, channel, future):
        try:
            result = future.result()
        except Exception as exc:
            result = {
                "channel": channel,
                "status": "worker_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        with self.lock:
            self.completed.append(result)
            self.active[channel] = None
            queued = self.queued[channel]
            self.queued[channel] = None
            if queued is not None:
                self._start_locked(channel, queued)

    def _submit(self, channel, command):
        with self.lock:
            if not self.accepting:
                return False
            if self.active[channel] is None:
                self._start_locked(channel, dict(command))
            else:
                previous = self.queued[channel]
                if previous is not None:
                    self.overwritten[channel] += 1
                    dropped = dict(previous)
                    dropped.update(
                        channel=channel,
                        status="overwritten_before_send",
                        overwritten_wall_time_ns=time.time_ns(),
                    )
                    self.completed.append(dropped)
                self.queued[channel] = dict(command)
        return True

    def submit_arm(self, absolute_wrist_action, *, sample_wall_time_ns, sample_index):
        return self._submit(
            "robot",
            {
                "absolute_wrist_action": np.asarray(
                    absolute_wrist_action, dtype=np.float64
                ).copy(),
                "sample_wall_time_ns": int(sample_wall_time_ns),
                "sample_index": int(sample_index),
            },
        )

    def submit_gripper(self, gripper_action, *, sample_wall_time_ns, sample_index):
        return self._submit(
            "gripper",
            {
                "gripper_action": float(np.sign(gripper_action)),
                "sample_wall_time_ns": int(sample_wall_time_ns),
                "sample_index": int(sample_index),
            },
        )

    def drain(self, episode):
        with self.lock:
            completed = self.completed
            self.completed = []
        for record in completed:
            episode.record_raw_dispatch(record)
        return len(completed)

    @property
    def pending_count(self):
        with self.lock:
            return sum(
                self.active[channel] is not None
                or self.queued[channel] is not None
                for channel in ("robot", "gripper")
            )

    def close(self, episode, *, flush=True, timeout_s=3.0):
        if self.closed:
            return
        with self.lock:
            self.accepting = False
            if not flush:
                for channel in ("robot", "gripper"):
                    queued = self.queued[channel]
                    self.queued[channel] = None
                    if queued is not None:
                        queued = dict(queued)
                        queued.update(channel=channel, status="cancelled_before_send")
                        self.completed.append(queued)
        deadline = time.monotonic() + float(timeout_s)
        while self.pending_count and time.monotonic() < deadline:
            self.drain(episode)
            time.sleep(0.001)
        with self.lock:
            for channel in ("robot", "gripper"):
                queued = self.queued[channel]
                self.queued[channel] = None
                if queued is not None:
                    queued = dict(queued)
                    queued.update(channel=channel, status="flush_timeout")
                    self.completed.append(queued)
        for executor in self.executors.values():
            executor.shutdown(wait=True)
        self.drain(episode)
        self.closed = True


class AbsoluteSpaceMouseSampler:
    """Sample SpaceMouse at high rate and integrate a base-frame wrist target."""

    def __init__(
        self,
        *,
        device,
        dispatcher,
        episode,
        initial_wrist_pose,
        delta_controller_cfg,
        sample_hz=30.0,
        motion_reference_hz=10.0,
        gripper_keepalive_s=1.0,
    ):
        self.device = device
        self.dispatcher = dispatcher
        self.episode = episode
        self.delta_controller_cfg = delta_controller_cfg
        self.sample_hz = float(sample_hz)
        self.motion_scale = float(motion_reference_hz) / self.sample_hz
        self.gripper_keepalive_s = float(gripper_keepalive_s)
        self.target_pose = np.asarray(initial_wrist_pose, dtype=np.float64).copy()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="spacemouse_absolute_sampler",
            daemon=True,
        )
        self.sample_index = 0
        self.last_gripper_action = None
        self.last_gripper_submit_monotonic = float("-inf")
        self.operator_stopped = False
        self.error = None
        self.missed_ticks = 0

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        period_s = 1.0 / self.sample_hz
        next_tick = time.monotonic()
        try:
            while not self.stop_event.is_set():
                remaining_s = next_tick - time.monotonic()
                if remaining_s > 0 and self.stop_event.wait(remaining_s):
                    break
                action, _ = input2action(
                    device=self.device,
                    controller_type="OSC_POSE",
                )
                if action is None:
                    self.operator_stopped = True
                    break
                sample_wall_time_ns = time.time_ns()
                self.target_pose, physical_delta = integrate_absolute_wrist_target(
                    self.target_pose,
                    action,
                    self.delta_controller_cfg,
                    self.motion_scale,
                )
                gripper_action = float(np.sign(action[-1]))
                absolute_action = np.concatenate(
                    [
                        self.target_pose[:3, 3],
                        Rotation.from_matrix(self.target_pose[:3, :3]).as_rotvec(),
                        [gripper_action],
                    ]
                )
                self.episode.record_raw_spacemouse_sample(
                    {
                        "sample_index": self.sample_index,
                        "sample_wall_time_ns": sample_wall_time_ns,
                        "raw_deoxys_delta_action": np.asarray(
                            action, dtype=np.float64
                        ).copy(),
                        "scaled_physical_delta": physical_delta.copy(),
                        "absolute_wrist_action": absolute_action.copy(),
                    }
                )
                self.dispatcher.submit_arm(
                    absolute_action,
                    sample_wall_time_ns=sample_wall_time_ns,
                    sample_index=self.sample_index,
                )
                now = time.monotonic()
                if (
                    self.last_gripper_action is None
                    or gripper_action != self.last_gripper_action
                    or now - self.last_gripper_submit_monotonic
                    >= self.gripper_keepalive_s
                ):
                    self.dispatcher.submit_gripper(
                        gripper_action,
                        sample_wall_time_ns=sample_wall_time_ns,
                        sample_index=self.sample_index,
                    )
                    self.last_gripper_action = gripper_action
                    self.last_gripper_submit_monotonic = now
                self.sample_index += 1
                next_tick += period_s
                now = time.monotonic()
                if now > next_tick + period_s:
                    missed = int((now - next_tick) / period_s)
                    self.missed_ticks += missed
                    next_tick += missed * period_s
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("SpaceMouse sampler did not stop within 2 seconds")


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
        teleop_fps=30.0,
        robot_action_latency_ms=10.0,
        gripper_action_latency_ms=10.0,
        action_stale_guard_ms=10.0,
        max_command_lateness_ms=10.0,
        robot_observation_latency_ms=0.0,
        gripper_observation_latency_ms=0.0,
        latency_profile=None,
        output_suffix=None,
        annotation_source="scripted",
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.task_name = task_name
        self.randomness = randomness
        if annotation_source != "scripted":
            raise ValueError("annotation_source must be scripted")
        output_suffix = str(output_suffix or "").strip()
        if not output_suffix or Path(output_suffix).name != output_suffix:
            raise ValueError("output_suffix must be one nonempty path component")
        self.output_suffix = output_suffix
        self.annotation_source = annotation_source
        self.camera_info = camera_info
        self.writer = writer
        self.prompt_depth_config = prompt_depth_config
        self.annotation_session_factory = annotation_session_factory
        self.eepose_frame = resolve_eepose_frame(eepose_frame)
        self.record_fps = float(record_fps)
        if self.record_fps <= 0:
            raise ValueError("record_fps must be positive")
        self.teleop_fps = float(teleop_fps)
        if self.teleop_fps <= 0:
            raise ValueError("teleop_fps must be positive")
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
        self.latency_profile = dict(latency_profile or {})
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
        self.buffered_mode = False
        self.buffered_actions = []
        self.camera_samples = []
        self.camera_start_sequence = None
        self.robot_start_index = None
        self.gripper_start_index = None
        self.buffer_error = None
        self.quality_issues = []
        self.raw_failed_streams = None
        self.buffer_alignment_report = None
        self.prompt_depth_report = None
        self.raw_lock = threading.Lock()
        self.raw_arm_commands_absolute = []
        self.raw_gripper_commands = []
        self.raw_spacemouse_samples = []
        self.grid_start_wall_time_ns = None
        self.grid_end_wall_time_ns = None
        self.camera_sequence_gap_count = 0

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
        self.buffered_mode = False
        self.buffered_actions = []
        self.camera_samples = []
        self.buffer_error = None
        self.quality_issues = []
        self.raw_failed_streams = None
        self.buffer_alignment_report = None
        self.prompt_depth_report = None
        self.raw_arm_commands_absolute = []
        self.raw_gripper_commands = []
        self.raw_spacemouse_samples = []
        self.grid_start_wall_time_ns = None
        self.grid_end_wall_time_ns = None
        self.camera_sequence_gap_count = 0
        self.state = "recording"
        logger.info("Recording started")
        return True

    def begin_buffered(
        self,
        initial_observation,
        *,
        camera_start_sequence,
        robot_start_index,
        gripper_start_index,
        grid_start_wall_time_ns,
        initial_absolute_wrist_action,
        initial_gripper_action,
    ):
        """Start a raw-buffer episode without online depth/annotation work."""

        if self.state == "recording":
            logger.warning("An episode is already recording")
            return False
        if self.state == "pending_save":
            logger.warning("Save or discard the previous episode first")
            return False
        initial_poses_valid = self._parts_ready(initial_observation)
        self.observations = []
        self.actions = []
        self.actions_original = []
        self.actions_absolute = []
        self.action_timing = []
        self.command_attempts = []
        self.buffered_actions = []
        self.camera_samples = []
        self.camera_start_sequence = int(camera_start_sequence)
        self.robot_start_index = max(0, int(robot_start_index) - 2)
        self.gripper_start_index = max(0, int(gripper_start_index) - 2)
        self.buffer_error = None
        self.quality_issues = []
        self.raw_failed_streams = None
        self.buffer_alignment_report = None
        self.prompt_depth_report = None
        self.raw_arm_commands_absolute = []
        self.raw_gripper_commands = []
        self.raw_spacemouse_samples = []
        self.grid_start_wall_time_ns = int(grid_start_wall_time_ns)
        self.grid_end_wall_time_ns = None
        self.camera_sequence_gap_count = 0
        initial_absolute_wrist_action = np.asarray(
            initial_absolute_wrist_action, dtype=np.float64
        ).copy()
        self.raw_arm_commands_absolute.append(
            {
                "channel": "robot",
                "status": "initial_state",
                "absolute_wrist_action": initial_absolute_wrist_action,
                "sample_wall_time_ns": self.grid_start_wall_time_ns,
                "command_wall_time_ns": None,
                "predicted_effect_wall_time_ns": self.grid_start_wall_time_ns,
            }
        )
        self.raw_gripper_commands.append(
            {
                "channel": "gripper",
                "status": "initial_state",
                "gripper_action": float(np.sign(initial_gripper_action)),
                "sample_wall_time_ns": self.grid_start_wall_time_ns,
                "command_wall_time_ns": None,
                "predicted_effect_wall_time_ns": self.grid_start_wall_time_ns,
            }
        )
        self.started_at = datetime.now().isoformat(timespec="milliseconds")
        self.stopped_at = None
        self.last_recorded_gripper = None
        self.gripper_hold_remaining = 0
        self.annotation_session = None
        self.annotation_error = None
        self.buffered_mode = True
        self.state = "recording"
        if not initial_poses_valid:
            self.mark_buffer_error(
                "required part poses are not all valid at recording start",
                phase="geometry",
            )
        logger.info(
            "Buffered recording started at camera sequence %d; PromptDA and "
            "annotation are deferred until end",
            self.camera_start_sequence,
        )
        return True

    def add_camera_samples(self, samples):
        if self.state != "recording" or not self.buffered_mode:
            return
        for sample in samples:
            sequence = int(sample["capture_sequence"])
            if self.camera_samples:
                previous = int(self.camera_samples[-1]["capture_sequence"])
                if sequence <= previous:
                    self.mark_buffer_error(
                        f"camera capture sequence is not increasing: {previous}, {sequence}"
                    )
                    continue
                if sequence != previous + 1:
                    self.camera_sequence_gap_count += sequence - previous - 1
                    logger.warning(
                        "Camera capture buffer skipped sequences %d..%d; "
                        "offline alignment may reuse a nearby frame",
                        previous + 1,
                        sequence - 1,
                    )
            self.camera_samples.append(_copy_mapping_arrays(sample))

    def record_raw_spacemouse_sample(self, record):
        if self.state != "recording" or not self.buffered_mode:
            return
        with self.raw_lock:
            self.raw_spacemouse_samples.append(_copy_mapping_arrays(record))

    def record_raw_dispatch(self, record):
        if self.state != "recording" or not self.buffered_mode:
            return
        copied = _copy_mapping_arrays(record)
        with self.raw_lock:
            self.command_attempts.append(dict(copied))
            if copied.get("status") != "executed":
                return
            if copied.get("channel") == "robot":
                self.raw_arm_commands_absolute.append(copied)
            elif copied.get("channel") == "gripper":
                self.raw_gripper_commands.append(copied)

    def append_buffered(self, scaled_action, *, action_timing):
        if self.state != "recording" or not self.buffered_mode:
            return
        timing = dict(action_timing or {})
        timing.setdefault("status", "executed")
        target_time_ns = timing.get("action_target_wall_time_ns")
        if target_time_ns is None:
            self.mark_buffer_error("executed action is missing its target timestamp")
            return
        if self.buffered_actions:
            previous_time_ns = int(
                self.buffered_actions[-1]["timing"]["action_target_wall_time_ns"]
            )
            expected_period_ns = int(round(1e9 / self.record_fps))
            actual_period_ns = int(target_time_ns) - previous_time_ns
            if abs(actual_period_ns - expected_period_ns) > 1_000:
                self.mark_buffer_error(
                    "executed action target grid is discontinuous: "
                    f"{actual_period_ns / 1e6:.3f} ms"
                )
        record = {
            "scaled_action": np.asarray(scaled_action, dtype=np.float64).copy(),
            "timing": timing,
        }
        self.buffered_actions.append(record)
        self.command_attempts.append(dict(timing))

    def mark_buffer_error(self, reason, *, phase="recording", traceback_text=None):
        message = str(reason)
        self.quality_issues.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "phase": phase,
                "message": message,
                "traceback": traceback_text,
            }
        )
        if self.buffer_error is None:
            self.buffer_error = message
        logger.error("Buffered episode quality issue [%s]: %s", phase, message)

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

    def record_dropped_command(self, timing, invalidate_continuity=True):
        """Keep a stale/failed command in audit metadata, never in training arrays."""

        if self.state != "recording":
            return
        attempt = dict(timing)
        attempt.setdefault("status", "dropped")
        self.command_attempts.append(attempt)
        if self.buffered_mode and invalidate_continuity:
            self.mark_buffer_error(
                "action grid contains a non-executed command: "
                f"{attempt.get('drop_reason', attempt.get('status'))}"
            )

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

    def stop_buffered(
        self,
        robot_records,
        gripper_records,
        *,
        prompt_depth_estimator,
        prompt_depth_cameras,
        camera_max_residual_ms,
        camera_pair_max_skew_ms,
        camera_hard_gap_ms,
        robot_max_residual_ms,
        gripper_max_residual_ms,
    ):
        if self.state != "recording" or not self.buffered_mode:
            logger.warning("No buffered episode is recording")
            return False
        self.stopped_at = datetime.now().isoformat(timespec="milliseconds")
        if self.grid_end_wall_time_ns is None:
            self.grid_end_wall_time_ns = time.time_ns()
        self.state = "pending_save"
        robot_episode_records = robot_records[self.robot_start_index :]
        gripper_episode_records = gripper_records[self.gripper_start_index :]
        logger.info(
            "Buffered alignment input: arm_commands=%d gripper_commands=%d "
            "spacemouse_samples=%d camera_pairs=%d robot_states=%d gripper_states=%d",
            len(self.raw_arm_commands_absolute),
            len(self.raw_gripper_commands),
            len(self.raw_spacemouse_samples),
            len(self.camera_samples),
            len(robot_episode_records),
            len(gripper_episode_records),
        )
        try:
            materialized = materialize_buffered_episode(
                self.raw_arm_commands_absolute,
                self.raw_gripper_commands,
                self.camera_samples,
                robot_episode_records,
                gripper_episode_records,
                grid_start_wall_time_ns=self.grid_start_wall_time_ns,
                grid_end_wall_time_ns=self.grid_end_wall_time_ns,
                camera_info=self.camera_info,
                eepose_frame=self.eepose_frame,
                action_period_ns=int(round(1e9 / self.record_fps)),
                camera_max_residual_ns=int(round(camera_max_residual_ms * 1e6)),
                camera_pair_max_skew_ns=int(
                    round(camera_pair_max_skew_ms * 1e6)
                ),
                camera_hard_gap_ns=int(round(camera_hard_gap_ms * 1e6)),
                robot_max_residual_ns=int(round(robot_max_residual_ms * 1e6)),
                gripper_max_residual_ns=int(
                    round(gripper_max_residual_ms * 1e6)
                ),
                robot_latency_ns=int(
                    round(self.robot_observation_latency_ms * 1e6)
                ),
                gripper_latency_ns=int(
                    round(self.gripper_observation_latency_ms * 1e6)
                ),
            )
            self.observations = materialized["observations"]
            self.actions = materialized["actions"]
            self.actions_original = materialized["actions_original"]
            self.actions_absolute = materialized["actions_absolute"]
            self.action_timing = materialized["action_timing"]
            self.buffer_alignment_report = materialized["report"]
        except Exception as exc:
            self.raw_failed_streams = {
                "camera_samples": self.camera_samples,
                "robot_states": _serializable_raw_state_records(
                    robot_episode_records
                ),
                "gripper_states": _serializable_raw_state_records(
                    gripper_episode_records
                ),
            }
            self.mark_buffer_error(
                f"{type(exc).__name__}: {exc}",
                phase="offline_alignment",
                traceback_text=traceback.format_exc(),
            )
            logger.exception(
                "Buffered alignment failed; raw streams can still be saved"
            )
            return False

        for steps_key, summary_key, limit_ms in (
            ("camera_hard_gap_steps", "front_residual", camera_hard_gap_ms),
            ("robot_state_warning_steps", "robot_residual", robot_max_residual_ms),
            (
                "gripper_state_warning_steps",
                "gripper_state_residual",
                gripper_max_residual_ms,
            ),
        ):
            steps = self.buffer_alignment_report.get(steps_key, [])
            if steps:
                maximum_ms = self.buffer_alignment_report[summary_key]["max_ms"]
                if steps_key == "camera_hard_gap_steps":
                    maximum_ms = max(
                        maximum_ms,
                        self.buffer_alignment_report["wrist_residual"]["max_ms"],
                    )
                self.mark_buffer_error(
                    f"{steps_key}: {len(steps)} steps exceed {limit_ms:.3f} ms "
                    f"(first step={steps[0]}, max residual={maximum_ms:.3f} ms); "
                    "aligned observations were retained",
                    phase="offline_alignment",
                )

        invalid_pose_frames = [
            index
            for index, observation in enumerate(self.observations)
            if not self._parts_ready(observation)
        ]
        if invalid_pose_frames:
            self.mark_buffer_error(
                "required geometry is invalid after alignment at frames "
                f"{invalid_pose_frames}",
                phase="geometry",
            )
        try:
            self.prompt_depth_report = apply_prompt_depth_offline(
                self.observations,
                prompt_depth_estimator,
                prompt_depth_cameras,
            )
            if (
                prompt_depth_estimator is None
                or set(prompt_depth_cameras) != {"front", "wrist"}
            ):
                self.mark_buffer_error(
                    "offline PromptDA is not configured for both cameras",
                    phase="prompt_depth",
                )
        except Exception as exc:
            self.prompt_depth_report = {
                "enabled": True,
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.mark_buffer_error(
                f"{type(exc).__name__}: {exc}",
                phase="prompt_depth",
                traceback_text=traceback.format_exc(),
            )
            logger.exception("Offline PromptDA failed; RGBD will still be saved")

        if self.annotation_session_factory is not None:
            annotation_frame_index = None
            try:
                self.annotation_session = self.annotation_session_factory(
                    self.task_name,
                    self.camera_info,
                    mode="offline",
                )
                for index, observation in enumerate(self.observations):
                    annotation_frame_index = index
                    self.annotation_session.annotate_observation(observation)
                    if (index + 1) % 50 == 0 or index + 1 == len(self.observations):
                        logger.info(
                            "Offline geometry annotation processed %d/%d frames",
                            index + 1,
                            len(self.observations),
                        )
            except Exception as exc:
                self.annotation_error = (
                    f"frame {annotation_frame_index}: {type(exc).__name__}: {exc}"
                )
                self.annotation_session = None
                for observation in self.observations:
                    _clear_real_skill_annotation(observation)
                self.mark_buffer_error(
                    self.annotation_error,
                    phase="offline_annotation",
                    traceback_text=traceback.format_exc(),
                )
                logger.exception(
                    "Offline annotation failed; observations will still be saved"
                )
        else:
            self.mark_buffer_error(
                "no offline annotation session; add --real-skill-annotation",
                phase="offline_annotation",
            )
        logger.info(
            "Buffered recording materialized with %d actions and %d quality "
            "issues. Press s=success, f=failure, d=discard; issues do not block save",
            len(self.actions),
            len(self.quality_issues),
        )
        return not self.quality_issues

    def _output_path(self, outcome):
        output_dir = (
            self.data_root
            / "raw"
            / "osc"
            / "real"
            / self.task_name
            / "teleop"
            / self.randomness
            / self.output_suffix
            / outcome
        )
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        return output_dir / f"{timestamp}.pkl"

    def save(self, success):
        if self.state != "pending_save":
            logger.warning("There is no stopped episode waiting to be saved")
            return None
        if not self.actions:
            if not self.buffered_mode:
                logger.warning("The episode contains no action; discarding it")
                self.discard()
                return None
            self.mark_buffer_error(
                "no aligned actions; saving available raw streams only",
                phase="save",
            )
        valid_observation_lengths = (
            {len(self.actions)}
            if self.buffered_mode
            else {len(self.actions), len(self.actions) + 1}
        )
        if len(self.observations) not in valid_observation_lengths:
            if not self.buffered_mode:
                raise RuntimeError(
                    "legacy episodes require N or N+1 observations for N actions"
                )
            self.mark_buffer_error(
                "buffered episodes require N observations/N actions; legacy "
                "episodes may contain N or N+1 observations",
                phase="save",
            )

        outcome = "success" if success else "failure"
        output_path = self._output_path(outcome)
        payload = {
            "env": "FurnitureBench",
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
            "raw_arm_commands_absolute": self.raw_arm_commands_absolute,
            "raw_gripper_commands": self.raw_gripper_commands,
            "raw_spacemouse_samples": self.raw_spacemouse_samples,
            "alignment_report": self.buffer_alignment_report,
            "rewards": [0.0] * len(self.actions),
            "camera_info": self.camera_info,
            "annotation_source": self.annotation_source,
            "image_annotation_mode": "none",
            "obs_valid": np.asarray(
                [
                    index < len(self.observations)
                    and self._parts_ready(self.observations[index])
                    for index in range(len(self.actions))
                ],
                dtype=np.bool_,
            ),
            "success": bool(success),
            "task": self.task_name,
            "furniture": self.task_name,
            "action_type": "delta",
            "eepose_frame": self.eepose_frame,
            "eepose_original_frame": "real-tip",
            "eepose_schema_version": 2,
            "metadata": {
                "schema": (
                    BUFFERED_SCHEMA
                    if self.buffered_mode
                    else "deoxys_furniturebench_raw_v5_target_time"
                ),
                "timebase": "unix_epoch_ns",
                "controller_observation_alignment": (
                    "offline direct-delta-command effect-time alignment"
                    if self.buffered_mode
                    else "raw observation stream plus action target-time master"
                ),
                "recording_frequency_hz": self.record_fps,
                "spacemouse_sampling_frequency_hz": self.teleop_fps,
                "action_period_ns": int(round(1e9 / self.record_fps)),
                "episode_grid_start_wall_time_ns": (
                    self.action_timing[0].get("episode_grid_start_wall_time_ns")
                    if self.action_timing else self.grid_start_wall_time_ns
                ),
                "recording_includes_noop_actions": True,
                "online_action_execution": "legacy_direct_delta_no_queue",
                "timing_contract": (
                    "legacy direct Deoxys delta commands, successful combined "
                    "arm/gripper send "
                    "times, camera exposure times, and robot/gripper receive times "
                    "are buffered independently; send time plus calibrated device "
                    "latency estimates effect time; actions use causal latest-effective "
                    "hold on the fixed-rate master grid; PromptDA and annotation run "
                    "only after materialization"
                    if self.buffered_mode
                    else "camera sensor source, PromptDA submit/start/ready, "
                    "robot/gripper state receive, and robot/gripper command send "
                    "times are stored separately; action_target_timestamps_ns is "
                    "the alignment master"
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
                "action_latency_source": self.latency_profile.get(
                    "latency_source", "measured_default"
                ),
                "action_latency_basis": self.latency_profile.get(
                    "basis", "FrankaControl measured 2026-09-08 profile"
                ),
                "latency_profile": self.latency_profile or None,
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
                "output_suffix": self.output_suffix,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "num_observations": len(self.observations),
                "num_actions": len(self.actions),
                "prompt_depth_anything": self.prompt_depth_config,
                "offline_buffer_alignment": self.buffer_alignment_report,
                "offline_prompt_depth_report": self.prompt_depth_report,
                "camera_sequence_gap_count": self.camera_sequence_gap_count,
            },
        }
        if self.raw_failed_streams is not None:
            payload["raw_camera_samples"] = self.raw_failed_streams["camera_samples"]
            payload["raw_robot_states"] = self.raw_failed_streams["robot_states"]
            payload["raw_gripper_states"] = self.raw_failed_streams["gripper_states"]
            payload["metadata"]["raw_state_encoding"] = "protobuf_serialized_bytes"
        if self.annotation_session is not None:
            try:
                self.annotation_session.update_trajectory_metadata(payload)
                if self.buffered_mode:
                    payload["annotation_source"] = "scripted"
                    payload["metadata"]["annotation_provenance"] = {
                        "source": "scripted",
                        "implementation": REAL_ANNOTATION_SOURCE,
                        "stage": "after_target_time_selection",
                        "rgb_pixels_modified": False,
                    }
            except Exception as exc:
                self.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="annotation_metadata",
                    traceback_text=traceback.format_exc(),
                )
                self.annotation_session = None
        if self.annotation_session is None and self.annotation_session_factory is not None:
            for observation in payload["observations"]:
                _clear_real_skill_annotation(observation)
        if self.buffered_mode and self.annotation_session is None:
            payload["annotation_source"] = "unannotated"
            payload["metadata"]["real_skill_annotation"] = {
                "source": REAL_ANNOTATION_SOURCE,
                "mode": "offline",
                "complete": False,
                "error": self.annotation_error or "offline annotation was not configured",
            }
        elif self.annotation_session is None and self.annotation_session_factory is not None:
            payload["metadata"]["real_skill_annotation"] = {
                "source": REAL_ANNOTATION_SOURCE,
                "mode": "online",
                "complete": False,
                "error": self.annotation_error,
            }
        if self.buffered_mode:
            try:
                audit = validate_buffered_payload(payload, self.annotation_session)
            except Exception as exc:
                self.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="save_contract",
                    traceback_text=traceback.format_exc(),
                )
                payload["metadata"]["buffered_contract_audit"] = {
                    "passed": False,
                    "error": str(exc),
                }
            else:
                payload["metadata"]["buffered_contract_audit"] = {
                    **audit,
                    "passed": True,
                }
            if self.quality_issues:
                payload["metadata"]["schema"] = f"{BUFFERED_SCHEMA}_incomplete"
        payload["save_quality"] = {
            "status": "incomplete" if self.quality_issues else "complete",
            "issues": [dict(issue) for issue in self.quality_issues],
        }
        if self.buffered_mode and self.quality_issues:
            output_path = (
                output_path.parent.parent
                / "incomplete"
                / outcome
                / output_path.name
            )
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
        self.buffered_mode = False
        self.buffered_actions = []
        self.camera_samples = []
        self.camera_start_sequence = None
        self.robot_start_index = None
        self.gripper_start_index = None
        self.buffer_error = None
        self.quality_issues = []
        self.raw_failed_streams = None
        self.buffer_alignment_report = None
        self.prompt_depth_report = None
        self.raw_arm_commands_absolute = []
        self.raw_gripper_commands = []
        self.raw_spacemouse_samples = []
        self.grid_start_wall_time_ns = None
        self.grid_end_wall_time_ns = None
        self.camera_sequence_gap_count = 0
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


def integrate_absolute_wrist_target(
    target_pose, deoxys_delta_action, delta_controller_cfg, motion_scale
):
    """Apply one rate-normalized Deoxys delta to an absolute wrist target."""

    target_pose = np.asarray(target_pose, dtype=np.float64).copy()
    physical_delta = scaled_deoxys_action(
        deoxys_delta_action, delta_controller_cfg
    )
    physical_delta[:6] *= float(motion_scale)
    target_pose[:3, 3] += physical_delta[:3]
    target_pose[:3, :3] = (
        Rotation.from_rotvec(physical_delta[3:6]).as_matrix()
        @ target_pose[:3, :3]
    )
    return target_pose, physical_delta


def delta_wrist_command_to_absolute_target(scaled_action, wrist_pose):
    """Describe one legacy Deoxys delta command as an auditable pose target."""

    scaled_action = np.asarray(scaled_action, dtype=np.float64).reshape(7)
    target_pose = np.asarray(wrist_pose, dtype=np.float64).reshape(4, 4).copy()
    target_pose[:3, 3] += scaled_action[:3]
    target_pose[:3, :3] = (
        Rotation.from_rotvec(scaled_action[3:6]).as_matrix()
        @ target_pose[:3, :3]
    )
    return np.concatenate(
        [
            target_pose[:3, 3],
            Rotation.from_matrix(target_pose[:3, :3]).as_rotvec(),
            [float(np.sign(scaled_action[-1]))],
        ]
    )


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
    default_latency_profile = os.environ.get(
        "RR_LATENCY_PROFILE",
        "/home/hz/code/robust-rearrangement-custom/src/real/"
        "latency_profile.measured_20260908.json",
    )
    default_interface_cfg = (
        Path(__file__).resolve().parents[1] / "config" / "charmander.yml"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", default=str(default_interface_cfg))
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
        "--annotation-source",
        choices=("scripted",),
        required=True,
        help="required policy-target provenance; only scripted geometry is accepted",
    )
    parser.add_argument(
        "--output-suffix",
        required=True,
        help="new campaign directory name under teleop/<randomness>",
    )
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
        "--teleop-fps",
        type=float,
        default=30.0,
        help="legacy compatibility; direct control uses FrankaInterface's 20 Hz",
    )
    parser.add_argument(
        "--latency-profile",
        default=default_latency_profile,
        help=(
            "measured RR latency profile; its arm/gripper action and state "
            "observation values override the individual latency flags"
        ),
    )
    parser.add_argument(
        "--robot-action-latency-ms",
        type=float,
        default=120.0,
        help="estimated arm command-to-effect latency used for target-time scheduling",
    )
    parser.add_argument(
        "--gripper-action-latency-ms",
        type=float,
        default=642.0,
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
        default=0.067,
        help="estimated robot receive-time correction used by offline alignment",
    )
    parser.add_argument(
        "--gripper-observation-latency-ms",
        type=float,
        default=0.067,
        help="estimated gripper receive-time correction used by offline alignment",
    )
    parser.add_argument(
        "--camera-match-max-residual-ms",
        type=float,
        default=50.0,
        help="camera residual warning threshold during offline matching",
    )
    parser.add_argument(
        "--camera-pair-max-skew-ms",
        type=float,
        default=40.0,
        help="maximum front/wrist source-time skew after final matching",
    )
    parser.add_argument(
        "--camera-hard-gap-ms",
        type=float,
        default=200.0,
        help="per-camera coverage gap flagged incomplete without aborting offline matching",
    )
    parser.add_argument(
        "--robot-state-max-residual-ms",
        type=float,
        default=20.0,
        help="robot-state residual quality threshold during offline interpolation",
    )
    parser.add_argument(
        "--gripper-state-max-residual-ms",
        type=float,
        default=60.0,
        help="gripper-state residual quality threshold during final alignment",
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
            "preview RR real annotation for one_leg/round_table/lamp and "
            "recompute it offline; annotation errors are recorded without "
            "preventing an incomplete save"
        ),
    )
    parser.add_argument("--no-camera-preview", action="store_true")
    parser.add_argument(
        "--prompt-depth-anything",
        action="store_true",
        help=(
            "run PromptDA offline on every target-time-selected frame after "
            "the operator ends an episode"
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
    args.latency_profile_data = {}
    if args.latency_profile:
        latency_profile_path = Path(args.latency_profile).expanduser()
        if not latency_profile_path.is_file():
            parser.error(f"--latency-profile does not exist: {latency_profile_path}")
        try:
            latency_profile = json.loads(latency_profile_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not read --latency-profile: {exc}")
        required_latency_fields = (
            "robot_action_ms",
            "gripper_action_ms",
            "robot_observation_ms",
            "gripper_observation_ms",
        )
        missing = [
            field for field in required_latency_fields if field not in latency_profile
        ]
        if missing:
            parser.error(
                "--latency-profile is missing fields: " + ", ".join(missing)
            )
        args.robot_action_latency_ms = float(latency_profile["robot_action_ms"])
        args.gripper_action_latency_ms = float(
            latency_profile["gripper_action_ms"]
        )
        args.robot_observation_latency_ms = float(
            latency_profile["robot_observation_ms"]
        )
        args.gripper_observation_latency_ms = float(
            latency_profile["gripper_observation_ms"]
        )
        if "action_stale_guard_ms" in latency_profile:
            args.action_stale_guard_ms = float(
                latency_profile["action_stale_guard_ms"]
            )
        args.latency_profile = str(latency_profile_path.resolve())
        args.latency_profile_data = latency_profile
    if args.product_id is None:
        args.product_id = SPACEMOUSE_PRODUCT_IDS[args.spacemouse_connection]
    if not args.data_root:
        parser.error("--data-root is required when DATA_DIR_RAW is not set")
    if Path(args.output_suffix).name != args.output_suffix or not args.output_suffix.strip():
        parser.error("--output-suffix must be one nonempty path component")
    interface_cfg = Path(args.interface_cfg).expanduser()
    if not interface_cfg.is_absolute():
        interface_cfg = Path.cwd() / interface_cfg
    if not interface_cfg.is_file():
        parser.error(
            f"--interface-cfg does not exist: {interface_cfg}. "
            "Use an absolute path or run from the Deoxys checkout directory."
        )
    args.interface_cfg = str(interface_cfg.resolve())
    if args.record_fps <= 0:
        parser.error("--record-fps must be greater than zero")
    if args.teleop_fps <= 0:
        parser.error("--teleop-fps must be greater than zero")
    for name in (
        "robot_action_latency_ms",
        "gripper_action_latency_ms",
        "action_stale_guard_ms",
        "max_command_lateness_ms",
        "robot_observation_latency_ms",
        "gripper_observation_latency_ms",
        "camera_match_max_residual_ms",
        "camera_pair_max_skew_ms",
        "camera_hard_gap_ms",
        "robot_state_max_residual_ms",
        "gripper_state_max_residual_ms",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    try:
        resolve_eepose_frame(args.eepose_frame)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    camera = None
    prompt_depth_estimator = None
    prompt_depth_cameras = ()
    prompt_depth_config = None
    prompt_depth_startup_error = None
    annotation_session_factory = None
    preview_worker = None
    camera_collector = None
    episode = None
    writer = EpisodeWriter(
        video_fps=args.record_fps,
        save_video=not args.no_video,
    )
    device = None
    robot_interface = None
    delta_controller_cfg = None
    startup_stage = "dual_realsense_initialization"
    try:
        logger.info(
            "Startup stage=%s interface_cfg=%s data_root=%s",
            startup_stage,
            args.interface_cfg,
            args.data_root,
        )
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
        logger.info("Startup stage=dual_realsense_ready")
        startup_stage = "real_skill_annotation_initialization"
        if args.real_skill_annotation:
            annotation_session_factory = _create_real_skill_annotation_session
        else:
            logger.warning(
                "No real annotation session is configured for task=%s. "
                "Pickles will be saved as incomplete/unannotated. "
                "Add --real-skill-annotation to enable offline annotation.",
                args.task_name,
            )
        startup_stage = "prompt_depth_initialization"
        if args.prompt_depth_anything:
            prompt_depth_cameras = (
                ("wrist", "front")
                if args.prompt_depth_cameras == "both"
                else (args.prompt_depth_cameras,)
            )
            prompt_depth_config = {
                "online": False,
                "stage": "after_target_time_selection",
                "model": args.prompt_depth_model,
                "max_size": args.prompt_depth_max_size,
                "prompt_size": [256, 192],
                "cameras": list(prompt_depth_cameras),
                "canonical_depth_fields": [
                    PROMPT_DEPTH_FIELDS[name] for name in prompt_depth_cameras
                ],
                "original_depth_suffix": "_realsense",
            }
            try:
                prompt_depth_estimator = PromptDepthAnythingEstimator(
                    model=args.prompt_depth_model,
                    device=args.prompt_depth_device,
                    max_size=args.prompt_depth_max_size,
                    min_depth_m=args.prompt_depth_min_m,
                    max_depth_m=args.prompt_depth_max_m,
                )
            except Exception as exc:
                prompt_depth_startup_error = f"{type(exc).__name__}: {exc}"
                prompt_depth_config["initialization_error"] = (
                    prompt_depth_startup_error
                )
                logger.exception(
                    "PromptDA initialization failed; raw RGBD collection "
                    "can continue and be saved as incomplete"
                )
        if set(prompt_depth_cameras) != {"front", "wrist"}:
            logger.warning(
                "Offline PromptDA is not configured for both cameras. "
                "Pickles will be saved as incomplete. "
                "Add --prompt-depth-anything --prompt-depth-cameras both."
            )
        startup_stage = "episode_recorder_initialization"
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
            teleop_fps=20.0,
            robot_action_latency_ms=args.robot_action_latency_ms,
            gripper_action_latency_ms=args.gripper_action_latency_ms,
            action_stale_guard_ms=args.action_stale_guard_ms,
            max_command_lateness_ms=args.max_command_lateness_ms,
            robot_observation_latency_ms=args.robot_observation_latency_ms,
            gripper_observation_latency_ms=args.gripper_observation_latency_ms,
            latency_profile=args.latency_profile_data,
            output_suffix=args.output_suffix,
            annotation_source=args.annotation_source,
        )

        startup_stage = "spacemouse_initialization"
        logger.info(
            "Startup stage=%s vendor_id=%s product_id=%s",
            startup_stage,
            args.vendor_id,
            args.product_id,
        )
        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
        device.start_control()
        startup_stage = "franka_interface_initialization"
        logger.info("Startup stage=%s", startup_stage)
        robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)
        delta_controller_cfg = get_default_controller_config(args.controller_type)
        joint_controller_cfg = get_default_controller_config("JOINT_POSITION")
        robot_interface.reset()
        startup_stage = "robot_state_wait"
        logger.info("Startup stage=%s", startup_stage)
        if not wait_for_robot_state(robot_interface):
            raise RuntimeError("robot state was not received")
        # Warm the controller before an episode so its one-time dummy-message
        # handshake cannot consume the first target-time action's deadline.
        startup_stage = "controller_warmup"
        logger.info("Startup stage=%s", startup_stage)
        robot_interface.control(
            controller_type=args.controller_type,
            action=np.array([0.0] * 6 + [-1.0], dtype=np.float64),
            controller_cfg=delta_controller_cfg,
            control_gripper=False,
        )
        preview_worker = AsyncObservationPreview(
            camera=camera,
            robot_interface=robot_interface,
            camera_info=camera_info,
            eepose_frame=args.eepose_frame,
            task_name=args.task_name,
            draw_part_poses=args.draw_part_poses,
            annotation_session_factory=annotation_session_factory,
            enable_annotation=(
                args.real_skill_annotation and not args.no_camera_preview
            ),
            show_window=not args.no_camera_preview,
            depth_min_m=args.prompt_depth_min_m,
            depth_max_m=args.prompt_depth_display_max_m,
            depth_colormap=args.prompt_depth_colormap,
        )

        startup_stage = "control_loop"
        logger.info("Startup complete; stage=%s", startup_stage)
        logger.info(
            "Keys: b=begin, e=end, s=save success, f=save failure, "
            "d=discard, r=reset joints, p=toggle part poses, q=quit"
        )
        record_period_ns = int(round(1e9 / args.record_fps))
        recording_start_due_monotonic = None
        camera_preroll_s = 3.0
        camera_postroll_s = 1.0
        draw_part_poses = bool(args.draw_part_poses)
        last_loop_phase_ms = {}
        worst_loop_phase_ms = {}
        direct_action_sample_index = 0
        control_fault = False

        def stop_camera_collector():
            nonlocal camera_collector
            if camera_collector is None:
                return
            camera_collector.stop()
            camera_collector = None

        def stop_buffered_episode():
            try:
                stop_camera_collector()
            except Exception as exc:
                episode.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="camera_collector_stop",
                    traceback_text=traceback.format_exc(),
                )
            try:
                duplicate_counts = camera.duplicate_frame_counts()
                logger.info(
                    "Camera duplicate frames skipped before queue: front=%s wrist=%s",
                    duplicate_counts["front"],
                    duplicate_counts["wrist"],
                )
            except Exception as exc:
                episode.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="camera_diagnostics",
                    traceback_text=traceback.format_exc(),
                )
            try:
                robot_records = robot_interface.timestamped_robot_state_buffer()
            except Exception as exc:
                robot_records = []
                episode.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="robot_state_buffer",
                    traceback_text=traceback.format_exc(),
                )
            try:
                gripper_records = robot_interface.timestamped_gripper_state_buffer()
            except Exception as exc:
                gripper_records = []
                episode.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="gripper_state_buffer",
                    traceback_text=traceback.format_exc(),
                )
            try:
                return episode.stop_buffered(
                    robot_records,
                    gripper_records,
                    prompt_depth_estimator=prompt_depth_estimator,
                    prompt_depth_cameras=prompt_depth_cameras,
                    camera_max_residual_ms=args.camera_match_max_residual_ms,
                    camera_pair_max_skew_ms=args.camera_pair_max_skew_ms,
                    camera_hard_gap_ms=args.camera_hard_gap_ms,
                    robot_max_residual_ms=args.robot_state_max_residual_ms,
                    gripper_max_residual_ms=args.gripper_state_max_residual_ms,
                )
            except Exception as exc:
                episode.mark_buffer_error(
                    f"{type(exc).__name__}: {exc}",
                    phase="offline_processing",
                    traceback_text=traceback.format_exc(),
                )
                episode.stopped_at = datetime.now().isoformat(timespec="milliseconds")
                episode.state = "pending_save"
                episode.raw_failed_streams = {
                    "camera_samples": episode.camera_samples,
                    "robot_states": _serializable_raw_state_records(robot_records),
                    "gripper_states": _serializable_raw_state_records(gripper_records),
                }
                logger.exception("Offline processing failed; raw streams can still be saved")
                return False

        with NonBlockingKeyReader() as key_reader:
            running = True
            while running:
                loop_start_monotonic = time.monotonic()
                after_dispatch = time.monotonic()
                preview_state = episode.state
                if recording_start_due_monotonic is not None:
                    remaining_s = recording_start_due_monotonic - time.monotonic()
                    preview_state = (
                        f"arming {remaining_s:.1f}s"
                        if remaining_s > 0
                        else "waiting poses"
                    )
                preview_worker.draw_part_poses = draw_part_poses
                preview_worker.pump(preview_state)
                observation = preview_worker.latest_observation
                keys = key_reader.read_keys() + preview_worker.read_window_keys()
                after_keys_read = time.monotonic()
                for key in keys:
                    if key == "b":
                        if control_fault:
                            logger.warning("Robot control fault: save/discard, then quit")
                        elif episode.state != "idle":
                            logger.warning("Save or discard the current episode before b")
                        elif recording_start_due_monotonic is not None:
                            logger.warning("Camera preparation is already in progress")
                        else:
                            recording_start_due_monotonic = (
                                time.monotonic() + camera_preroll_s
                            )
                            logger.info(
                                "Camera preparation started; recording begins in %.1f s",
                                camera_preroll_s,
                            )
                    elif key == "e":
                        if recording_start_due_monotonic is not None:
                            recording_start_due_monotonic = None
                            logger.info("Camera preparation cancelled; no episode was recorded")
                        elif episode.state == "recording":
                            episode.grid_end_wall_time_ns = time.time_ns()
                            logger.info(
                                "Actions stopped; collecting %.1f s of camera post-roll",
                                camera_postroll_s,
                            )
                            time.sleep(camera_postroll_s)
                            stop_buffered_episode()
                            try:
                                device.start_control(preserve_gripper=True)
                            except TypeError:
                                device.start_control()
                        else:
                            logger.warning("No episode is recording")
                    elif key == "s":
                        episode.save(success=True)
                        preview_worker.request_annotation_reset()
                    elif key == "f":
                        episode.save(success=False)
                        preview_worker.request_annotation_reset()
                    elif key == "d":
                        recording_start_due_monotonic = None
                        stop_camera_collector()
                        episode.discard()
                        preview_worker.request_annotation_reset()
                    elif key == "p":
                        draw_part_poses = not draw_part_poses
                        logger.info(
                            "Front part-pose overlay %s",
                            "enabled" if draw_part_poses else "disabled",
                        )
                    elif key == "q":
                        recording_start_due_monotonic = None
                        stop_camera_collector()
                        running = False
                    elif key == "r":
                        if control_fault:
                            logger.warning("Joint reset is disabled after a control fault")
                        elif (
                            episode.state == "recording"
                            or recording_start_due_monotonic is not None
                        ):
                            logger.warning(
                                "Joint reset is disabled while recording or preparing"
                            )
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
                if control_fault:
                    time.sleep(0.02)
                    continue

                if (
                    recording_start_due_monotonic is not None
                    and time.monotonic() >= recording_start_due_monotonic
                    and observation is not None
                ):
                    recording_start_due_monotonic = None
                    camera_history_cursor = camera.history_cursor()
                    initial_wrist_pose = np.asarray(
                        observation["robot_state"]["wrist_pose"], dtype=np.float64
                    ).copy()
                    initial_input_action, _ = input2action(
                        device=device,
                        controller_type=args.controller_type,
                    )
                    if initial_input_action is None:
                        logger.warning(
                            "Recording start cancelled because SpaceMouse requested reset"
                        )
                        device.start_control()
                        continue
                    initial_gripper_action = float(
                        np.sign(initial_input_action[-1])
                    )
                    initial_absolute_wrist_action = np.concatenate(
                        [
                            initial_wrist_pose[:3, 3],
                            Rotation.from_matrix(
                                initial_wrist_pose[:3, :3]
                            ).as_rotvec(),
                            [initial_gripper_action],
                        ]
                    )
                    grid_start_wall_time_ns = time.time_ns()
                    if episode.begin_buffered(
                        observation,
                        camera_start_sequence=camera_history_cursor,
                        robot_start_index=len(
                            robot_interface.timestamped_robot_state_buffer()
                        ),
                        gripper_start_index=len(
                            robot_interface.timestamped_gripper_state_buffer()
                        ),
                        grid_start_wall_time_ns=grid_start_wall_time_ns,
                        initial_absolute_wrist_action=(
                            initial_absolute_wrist_action
                        ),
                        initial_gripper_action=initial_gripper_action,
                    ):
                        if prompt_depth_startup_error is not None:
                            episode.mark_buffer_error(
                                prompt_depth_startup_error,
                                phase="prompt_depth_initialization",
                            )
                        preview_worker.request_annotation_reset()
                        camera_collector = AsyncCameraBufferCollector(
                            camera=camera,
                            episode=episode,
                            history_cursor=camera_history_cursor,
                        ).start()
                        worst_loop_phase_ms = {}
                        logger.info(
                            "Legacy direct SpaceMouse control_hz=20.0 output_hz=%.1f "
                            "robot_latency_ms=%.3f gripper_latency_ms=%.3f preview=%s "
                            "draw_part_poses=%s live_skill=%s",
                            args.record_fps,
                            episode.machine_time_schedule.robot_action_latency_ns / 1e6,
                            episode.machine_time_schedule.gripper_action_latency_ns / 1e6,
                            not args.no_camera_preview,
                            draw_part_poses,
                            args.real_skill_annotation,
                        )

                now = time.monotonic()
                if episode.state == "recording":
                    last_loop_phase_ms = {
                        "dispatch": round(
                            (after_dispatch - loop_start_monotonic) * 1e3, 3
                        ),
                        "camera_drain": round(
                            0.0, 3
                        ),
                        "async_preview_pump_and_key_read": round(
                            (after_keys_read - after_dispatch) * 1e3, 3
                        ),
                        "key_handling": round((now - after_keys_read) * 1e3, 3),
                        "total": round((now - loop_start_monotonic) * 1e3, 3),
                    }
                    if last_loop_phase_ms["total"] > worst_loop_phase_ms.get("total", 0):
                        worst_loop_phase_ms = dict(last_loop_phase_ms)
                    if last_loop_phase_ms["total"] > 15.0:
                        logger.warning(
                            "Timing probe slow UI loop phases_ms=%s",
                            last_loop_phase_ms,
                        )

                if recording_start_due_monotonic is not None:
                    # The HID listener may update its current axes, but no
                    # SpaceMouse action is sampled or sent during preparation.
                    time.sleep(0.01)
                    continue

                action, _ = input2action(
                    device=device,
                    controller_type=args.controller_type,
                )
                if action is None:
                    if episode.state == "recording":
                        episode.grid_end_wall_time_ns = time.time_ns()
                        logger.info(
                            "SpaceMouse stopped recording; collecting %.1f s of "
                            "camera post-roll",
                            camera_postroll_s,
                        )
                        time.sleep(camera_postroll_s)
                        stop_buffered_episode()
                        try:
                            device.start_control(preserve_gripper=True)
                        except TypeError:
                            device.start_control()
                        continue
                    break
                sample_wall_time_ns = time.time_ns()
                scaled_action = scaled_deoxys_action(
                    action, delta_controller_cfg
                )
                wrist_pose = robot_interface.last_eef_pose
                if wrist_pose is None and observation is not None:
                    wrist_pose = observation["robot_state"]["wrist_pose"]
                if wrist_pose is None:
                    raise RuntimeError(
                        "robot wrist pose is unavailable for direct action audit"
                    )
                absolute_wrist_action = delta_wrist_command_to_absolute_target(
                    scaled_action,
                    wrist_pose,
                )
                if episode.state == "recording":
                    episode.record_raw_spacemouse_sample(
                        {
                            "sample_index": direct_action_sample_index,
                            "sample_wall_time_ns": sample_wall_time_ns,
                            "raw_deoxys_delta_action": np.asarray(
                                action, dtype=np.float64
                            ).copy(),
                            "scaled_physical_delta": np.asarray(
                                scaled_action, dtype=np.float64
                            ).copy(),
                            "absolute_wrist_action": absolute_wrist_action.copy(),
                        }
                    )
                control_started = time.monotonic()
                try:
                    command_timing = robot_interface.control(
                        controller_type=args.controller_type,
                        action=np.asarray(action, dtype=np.float64).copy(),
                        controller_cfg=delta_controller_cfg,
                    )
                except Exception as exc:
                    if episode.state != "recording":
                        raise
                    episode.mark_buffer_error(
                        f"{type(exc).__name__}: {exc}",
                        phase="robot_control",
                        traceback_text=traceback.format_exc(),
                    )
                    episode.grid_end_wall_time_ns = time.time_ns()
                    stop_buffered_episode()
                    control_fault = True
                    logger.exception(
                        "Robot control stopped; no further actions will be sent. "
                        "Press s/f to save the incomplete episode, then q to quit."
                    )
                    continue
                control_elapsed_ms = (
                    time.monotonic() - control_started
                ) * 1e3
                if episode.state == "recording":
                    arm_command_wall_time_ns = int(
                        command_timing["robot_command_wall_time_ns"]
                    )
                    episode.record_raw_dispatch(
                        {
                            "channel": "robot",
                            "status": "executed",
                            "sample_index": direct_action_sample_index,
                            "sample_wall_time_ns": sample_wall_time_ns,
                            "command_wall_time_ns": arm_command_wall_time_ns,
                            "predicted_effect_wall_time_ns": (
                                arm_command_wall_time_ns
                                + episode.machine_time_schedule.robot_action_latency_ns
                            ),
                            "control_elapsed_ms": control_elapsed_ms,
                            "absolute_wrist_action": absolute_wrist_action.copy(),
                            "raw_deoxys_delta_action": np.asarray(
                                action, dtype=np.float64
                            ).copy(),
                            "scaled_physical_delta": np.asarray(
                                scaled_action, dtype=np.float64
                            ).copy(),
                        }
                    )
                    gripper_command_wall_time_ns = command_timing.get(
                        "gripper_command_wall_time_ns"
                    )
                    if gripper_command_wall_time_ns is not None:
                        gripper_command_wall_time_ns = int(
                            gripper_command_wall_time_ns
                        )
                        episode.record_raw_dispatch(
                            {
                                "channel": "gripper",
                                "status": "executed",
                                "sample_index": direct_action_sample_index,
                                "sample_wall_time_ns": sample_wall_time_ns,
                                "command_wall_time_ns": (
                                    gripper_command_wall_time_ns
                                ),
                                "predicted_effect_wall_time_ns": (
                                    gripper_command_wall_time_ns
                                    + episode.machine_time_schedule.gripper_action_latency_ns
                                ),
                                "control_elapsed_ms": control_elapsed_ms,
                                "gripper_action": float(np.sign(action[-1])),
                            }
                        )
                direct_action_sample_index += 1
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception:
        logger.exception("Data collection failed during stage=%s", startup_stage)
        raise
    finally:
        if episode is not None and episode.state != "idle":
            logger.warning("Unsaved in-memory episode was discarded on exit")
        if camera_collector is not None:
            stop_camera_collector()
        if preview_worker is not None:
            preview_worker.close()
        try:
            if robot_interface is not None:
                try:
                    if delta_controller_cfg is not None:
                        robot_interface.control(
                            controller_type=args.controller_type,
                            action=np.array([0.0] * 6 + [1.0]),
                            controller_cfg=delta_controller_cfg,
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
            if camera is not None:
                camera.stop()
            if not args.no_camera_preview:
                cv2.destroyAllWindows()
            writer.close()


if __name__ == "__main__":
    main()
