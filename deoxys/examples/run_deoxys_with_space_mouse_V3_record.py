"""SpaceMouse teleoperation with on-demand dual RealSense demonstration recording."""

import argparse
import os
import pickle
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


RESET_JOINT_POSITIONS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

DEFAULT_FRONT_SERIAL = "327122071654"
DEFAULT_WRIST_SERIAL = "001622071252"


class NonBlockingKeyReader:
    def __init__(self):
        self._fd = None
        self._old_settings = None
        self.enabled = False

    def __enter__(self):
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY; keyboard controls are disabled.")
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


def _try_get_frames(pipeline, aligner):
    try:
        success, frames = pipeline.try_wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        return None
    if not success:
        return None
    return aligner.process(frames) if aligner is not None else frames


class DualRealSenseSnapshotter:
    """Capture synchronized-enough front/wrist RGB-D pairs on a background thread."""

    def __init__(
        self,
        front_serial,
        wrist_serial,
        width=640,
        height=480,
        fps=30,
        record_width=224,
        record_height=224,
        save_depth=False,
    ):
        if rs is None or cv2 is None:
            raise RuntimeError("pyrealsense2 and opencv-python are required for cameras")
        if front_serial == wrist_serial:
            raise ValueError("front and wrist camera serials must differ")

        self.front_serial = str(front_serial)
        self.wrist_serial = str(wrist_serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.record_size = (int(record_width), int(record_height))
        self.save_depth = bool(save_depth)

        self.front_pipeline = rs.pipeline()
        self.wrist_pipeline = rs.pipeline()
        self.front_aligner = rs.align(rs.stream.color)
        self.wrist_aligner = rs.align(rs.stream.color)
        self.front_depth_scale_m = None
        self.wrist_depth_scale_m = None

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._thread_error = None
        self._latest = None

    def _config(self, serial):
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        return config

    @staticmethod
    def _depth_scale(profile):
        return float(profile.get_device().first_depth_sensor().get_depth_scale())

    def start(self):
        front_profile = self.front_pipeline.start(self._config(self.front_serial))
        try:
            wrist_profile = self.wrist_pipeline.start(self._config(self.wrist_serial))
        except Exception:
            self.front_pipeline.stop()
            raise
        self.front_depth_scale_m = self._depth_scale(front_profile)
        self.wrist_depth_scale_m = self._depth_scale(wrist_profile)
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="dual_realsense_snapshotter",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Dual RealSense started: front=%s wrist=%s",
            self.front_serial,
            self.wrist_serial,
        )

    def _prepare_rgb(self, frame):
        bgr = np.asanyarray(frame.get_data())
        bgr = cv2.resize(bgr, self.record_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _prepare_depth(self, frame):
        depth = np.asanyarray(frame.get_data())
        return cv2.resize(depth, self.record_size, interpolation=cv2.INTER_NEAREST)

    def _capture_loop(self):
        try:
            while not self._stop_event.is_set():
                front_frames = _try_get_frames(
                    self.front_pipeline,
                    self.front_aligner,
                )
                wrist_frames = _try_get_frames(
                    self.wrist_pipeline,
                    self.wrist_aligner,
                )
                if front_frames is None or wrist_frames is None:
                    continue

                front_color = front_frames.get_color_frame()
                wrist_color = wrist_frames.get_color_frame()
                front_depth = front_frames.get_depth_frame()
                wrist_depth = wrist_frames.get_depth_frame()
                if not front_color or not wrist_color:
                    continue
                if self.save_depth and (not front_depth or not wrist_depth):
                    continue

                sample = {
                    "color_image1": self._prepare_rgb(wrist_color),
                    "color_image2": self._prepare_rgb(front_color),
                    "camera_capture_wall_time_ns": time.time_ns(),
                    "front_sensor_timestamp_ms": float(front_color.get_timestamp()),
                    "wrist_sensor_timestamp_ms": float(wrist_color.get_timestamp()),
                    "front_frame_number": int(front_color.get_frame_number()),
                    "wrist_frame_number": int(wrist_color.get_frame_number()),
                }
                if self.save_depth:
                    sample.update(
                        {
                            "depth_image1": self._prepare_depth(wrist_depth),
                            "depth_image2": self._prepare_depth(front_depth),
                        }
                    )
                with self._lock:
                    self._latest = sample
        except Exception as exc:
            self._thread_error = exc
            self._stop_event.set()
            logger.exception("Dual RealSense capture failed")

    def latest(self):
        if self._thread_error is not None:
            raise RuntimeError("dual RealSense capture failed") from self._thread_error
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        for pipeline in (self.front_pipeline, self.wrist_pipeline):
            try:
                pipeline.stop()
            except Exception:
                pass

    def metadata(self):
        return {
            "front_camera_serial": self.front_serial,
            "wrist_camera_serial": self.wrist_serial,
            "camera_stream_width": self.width,
            "camera_stream_height": self.height,
            "camera_fps": self.fps,
            "record_image_width": self.record_size[0],
            "record_image_height": self.record_size[1],
            "depth_enabled": self.save_depth,
            "depth_encoding": "uint16_z16",
            "front_depth_scale_m": self.front_depth_scale_m,
            "wrist_depth_scale_m": self.wrist_depth_scale_m,
        }


def _array_field(message, *names, size=None):
    for name in names:
        if hasattr(message, name):
            value = np.asarray(getattr(message, name), dtype=np.float64).reshape(-1)
            if size is None or value.size == size:
                return value
    return np.full(0 if size is None else size, np.nan, dtype=np.float64)


def _gripper_width(robot_interface):
    value = robot_interface.last_gripper_q
    if value is None:
        return np.array([np.nan], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)[:1]


def build_observation(robot_interface, camera_sample):
    if not robot_interface._state_buffer or camera_sample is None:
        return None

    state = robot_interface._state_buffer[-1]
    raw_pose = np.asarray(state.O_T_EE, dtype=np.float64)
    if raw_pose.size == 16:
        ee_pose = raw_pose.reshape(4, 4).transpose()
        ee_pos = ee_pose[:3, 3].copy()
        ee_quat = transform_utils.mat2quat(ee_pose[:3, :3])
    else:
        ee_pose = np.full((4, 4), np.nan, dtype=np.float64)
        ee_pos = np.full(3, np.nan, dtype=np.float64)
        ee_quat = np.full(4, np.nan, dtype=np.float64)

    ee_velocity = _array_field(state, "O_dP_EE_c", "O_dP_EE", size=6)
    observation = dict(camera_sample)
    observation.update(
        {
            "control_wall_time_ns": time.time_ns(),
            "robot_state": {
                "ee_pos": ee_pos,
                "ee_quat": ee_quat,
                "ee_pose": ee_pose,
                "ee_pos_vel": ee_velocity[:3],
                "ee_ori_vel": ee_velocity[3:],
                "joint_positions": _array_field(state, "q", size=7),
                "joint_velocities": _array_field(state, "dq", size=7),
                "joint_torques": _array_field(
                    state,
                    "tau_J",
                    "tau_J_d",
                    size=7,
                ),
                "gripper_width": _gripper_width(robot_interface),
            },
        }
    )
    return observation


class RawEpisodeRecorder:
    def __init__(self, output_root, task_name, randomness, camera_metadata):
        self.output_dir = (
            Path(output_root).expanduser().resolve() / randomness / task_name
        )
        self.task_name = task_name
        self.randomness = randomness
        self.camera_metadata = camera_metadata
        self.observations = []
        self.actions = []
        self.started_at = None
        self.stopped_at = None
        self.state = "idle"

    def begin(self):
        if self.state == "recording":
            logger.warning("An episode is already recording.")
            return
        if self.state == "pending_save":
            logger.warning("Save or discard the previous episode first.")
            return
        self.observations = []
        self.actions = []
        self.started_at = datetime.now().isoformat(timespec="milliseconds")
        self.stopped_at = None
        self.state = "recording"
        logger.info("Recording started.")

    def append(self, observation, action):
        if self.state != "recording" or observation is None:
            return
        self.observations.append(observation)
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def stop(self):
        if self.state != "recording":
            logger.warning("No episode is recording.")
            return
        self.stopped_at = datetime.now().isoformat(timespec="milliseconds")
        self.state = "pending_save"
        logger.info(
            "Recording stopped with %d samples. Press 's' to save or 'd' to discard.",
            len(self.actions),
        )

    def _next_path(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        used = []
        for path in self.output_dir.glob("*.pkl"):
            if path.stem.isdigit():
                used.append(int(path.stem))
        return self.output_dir / f"{max(used, default=-1) + 1:05d}.pkl"

    def save(self):
        if self.state != "pending_save":
            logger.warning("There is no stopped episode waiting to be saved.")
            return None
        if not self.actions:
            logger.warning("The episode contains no valid samples; discarding it.")
            self.discard()
            return None

        output_path = self._next_path()
        payload = {
            "furniture": self.task_name,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": [0] * len(self.actions),
            "skills": [0] * len(self.actions),
            "metadata": {
                "schema": "deoxys_furniturebench_raw_v1",
                "controller_observation_alignment": "observation_before_action",
                "randomness": self.randomness,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "num_samples": len(self.actions),
                "camera_key_mapping": {
                    "color_image1": "wrist",
                    "color_image2": "front",
                    "depth_image1": "wrist",
                    "depth_image2": "front",
                },
                **self.camera_metadata,
            },
        }
        temporary_path = output_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as output_file:
            pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)
        logger.info("Saved raw episode to %s", output_path)
        self.discard(log=False)
        return output_path

    def discard(self, log=True):
        sample_count = len(self.actions)
        self.observations = []
        self.actions = []
        self.started_at = None
        self.stopped_at = None
        self.state = "idle"
        if log:
            logger.info("Discarded episode with %d samples.", sample_count)


def wait_for_robot_state(robot_interface, timeout=5.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if robot_interface.received_states and robot_interface.check_nonzero_configuration():
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
        logger.warning("Robot state not received before reset request.")
        return False

    target = np.asarray(RESET_JOINT_POSITIONS, dtype=np.float64)
    action = target.tolist() + [-1.0 if gripper_open else 1.0]
    start_time = time.time()
    max_error = float("inf")
    while time.time() - start_time < timeout:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", default="config/charmander.yml")
    parser.add_argument("--controller-type", default="OSC_POSE")
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50746)
    parser.add_argument("--output-root", default="data/raw_spacemouse")
    parser.add_argument("--task-name", default="unnamed_task")
    parser.add_argument("--randomness", default="low")
    parser.add_argument("--front-camera-serial", default=DEFAULT_FRONT_SERIAL)
    parser.add_argument("--wrist-camera-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--record-image-width", type=int, default=224)
    parser.add_argument("--record-image-height", type=int, default=224)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--reset-timeout", type=float, default=7.0)
    parser.add_argument("--reset-tolerance", type=float, default=1e-3)
    parser.add_argument("--keep-gripper-closed-during-reset", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = None
    episode = None
    robot_interface = None
    controller_cfg = None
    try:
        if not args.no_cameras:
            camera = DualRealSenseSnapshotter(
                front_serial=args.front_camera_serial,
                wrist_serial=args.wrist_camera_serial,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                record_width=args.record_image_width,
                record_height=args.record_image_height,
                save_depth=args.save_depth,
            )
            camera.start()

        camera_metadata = (
            {"cameras_enabled": False}
            if camera is None
            else {"cameras_enabled": True, **camera.metadata()}
        )
        episode = RawEpisodeRecorder(
            output_root=args.output_root,
            task_name=args.task_name,
            randomness=args.randomness,
            camera_metadata=camera_metadata,
        )

        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
        device.start_control()
        robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)
        controller_cfg = get_default_controller_config(args.controller_type)
        joint_controller_cfg = get_default_controller_config("JOINT_POSITION")
        robot_interface._state_buffer = []

        logger.info(
            "Keys: b=begin, e=end, s=save, d=discard, r=reset joints, q=quit. "
            "SpaceMouse right button ends recording, or quits while idle."
        )

        with NonBlockingKeyReader() as key_reader:
            running = True
            while running:
                for key in key_reader.read_keys():
                    if key == "b":
                        episode.begin()
                    elif key == "e":
                        episode.stop()
                    elif key == "s":
                        episode.save()
                    elif key == "d":
                        episode.discard()
                    elif key == "q":
                        running = False
                    elif key == "r":
                        if episode.state == "recording":
                            logger.warning("Joint reset is disabled while recording.")
                        else:
                            move_to_reset_joint_positions(
                                robot_interface,
                                joint_controller_cfg,
                                timeout=args.reset_timeout,
                                tolerance=args.reset_tolerance,
                                gripper_open=not args.keep_gripper_closed_during_reset,
                            )
                            device.start_control()

                if not running:
                    break

                action, _ = input2action(
                    device=device,
                    controller_type=args.controller_type,
                )
                if action is None:
                    if episode.state == "recording":
                        episode.stop()
                        try:
                            device.start_control(preserve_gripper=True)
                        except TypeError:
                            device.start_control()
                        continue
                    running = False
                    continue

                camera_sample = (
                    {"camera_capture_wall_time_ns": time.time_ns()}
                    if camera is None
                    else camera.latest()
                )
                observation = build_observation(robot_interface, camera_sample)
                robot_interface.control(
                    controller_type=args.controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                episode.append(observation, action)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        if episode is not None and episode.state != "idle":
            logger.warning("Unsaved in-memory episode was discarded on exit.")
        try:
            if robot_interface is not None:
                try:
                    if controller_cfg is not None:
                        robot_interface.control(
                            controller_type=args.controller_type,
                            action=[0.0] * 6 + [1.0],
                            controller_cfg=controller_cfg,
                            termination=True,
                        )
                finally:
                    robot_interface.close()
        finally:
            if camera is not None:
                camera.stop()


if __name__ == "__main__":
    main()
