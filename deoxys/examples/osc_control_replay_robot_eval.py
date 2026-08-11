"""Replay robot_eval.json trajectories with fixed orientation under OSC_POSE."""

import argparse
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
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


DEFAULT_CAMERA_HIGH_SERIAL = "327122071654"  # D435IF, ACT cam_high
DEFAULT_CAMERA_WRIST_SERIAL = "001622071252"  # D435, ACT cam_left_wrist
DEFAULT_LEFT_SERIAL = DEFAULT_CAMERA_HIGH_SERIAL
DEFAULT_RIGHT_SERIAL = DEFAULT_CAMERA_WRIST_SERIAL


def _overlay_label(image, text):
    out = image.copy()
    cv2.rectangle(out, (8, 8), (440, 44), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _try_get_frames(pipeline, aligner):
    try:
        success, frames = pipeline.try_wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        return None
    if not success:
        return None
    if aligner is not None:
        frames = aligner.process(frames)
    return frames


class DualRealSenseVideoRecorder:
    def __init__(
        self,
        output_root,
        left_serial=None,
        right_serial=None,
        camera_high_serial=None,
        camera_wrist_serial=None,
        width=640,
        height=480,
        fps=30,
        align_mode="color",
        show_preview=True,
        output_dir=None,
    ):
        if camera_high_serial is None:
            camera_high_serial = left_serial or DEFAULT_CAMERA_HIGH_SERIAL
        if camera_wrist_serial is None:
            camera_wrist_serial = right_serial or DEFAULT_CAMERA_WRIST_SERIAL
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for RealSense recording")
        if show_preview and cv2 is None:
            raise RuntimeError("cv2 is required when --show-camera-preview is enabled")
        if camera_high_serial == camera_wrist_serial:
            raise ValueError("camera_high serial and camera_wrist serial cannot be the same")
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe is None:
            raise RuntimeError("ffmpeg is required for RealSense recording")

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = (
                Path(output_root).expanduser().resolve()
                / f"dual_realsense_{timestamp}"
            )
        else:
            self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.camera_high_serial = str(camera_high_serial)
        self.camera_wrist_serial = str(camera_wrist_serial)
        self.left_serial = self.camera_high_serial
        self.right_serial = self.camera_wrist_serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_mode = align_mode
        self.show_preview = bool(show_preview)
        self.window_name = "Dual RealSense Recorder"
        self.ffmpeg_exe = ffmpeg_exe

        self.camera_high_video_path = self.output_dir / f"camera_high_{self.camera_high_serial}.mp4"
        self.camera_wrist_video_path = self.output_dir / f"camera_wrist_{self.camera_wrist_serial}.mp4"
        self.left_video_path = self.camera_high_video_path
        self.right_video_path = self.camera_wrist_video_path
        self.metadata_path = self.output_dir / "metadata.json"
        self.frame_timestamps_path = self.output_dir / "camera_frame_timestamps.jsonl"

        self.left_pipeline = rs.pipeline()
        self.right_pipeline = rs.pipeline()
        self.left_align = None
        self.right_align = None
        self.left_encoder = None
        self.right_encoder = None
        self.stop_event = threading.Event()
        self.thread = None
        self.thread_error = None
        self.left_frame_count = 0
        self.right_frame_count = 0
        self.started_at = None
        self.stopped_at = None
        self.video_backend = "ffmpeg"
        self._frame_timestamps_file = None

    def _spawn_encoder(self, output_path):
        return subprocess.Popen(
            [
                self.ffmpeg_exe,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _write_frame(self, encoder, frame, side):
        if encoder is None or encoder.stdin is None:
            raise RuntimeError(f"{side} ffmpeg encoder is not available")
        try:
            encoder.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            stderr = b""
            if encoder.stderr is not None:
                stderr = encoder.stderr.read()
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{side} ffmpeg encoder pipe closed unexpectedly: {message}") from exc

    def _close_encoder(self, encoder, side):
        if encoder is None:
            return
        try:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            stderr = b""
            if encoder.stderr is not None:
                stderr = encoder.stderr.read()
            return_code = encoder.wait(timeout=30.0)
        except subprocess.TimeoutExpired as exc:
            encoder.kill()
            raise RuntimeError(f"{side} ffmpeg encoder did not exit cleanly") from exc

        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{side} ffmpeg encoder failed with code {return_code}: {message}")

    def start(self):
        left_config = rs.config()
        right_config = rs.config()
        left_config.enable_device(self.camera_high_serial)
        right_config.enable_device(self.camera_wrist_serial)

        for config in (left_config, right_config):
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.left_pipeline.start(left_config)
        self.right_pipeline.start(right_config)

        if self.align_mode == "color":
            self.left_align = rs.align(rs.stream.color)
            self.right_align = rs.align(rs.stream.color)

        self.left_encoder = self._spawn_encoder(self.left_video_path)
        self.right_encoder = self._spawn_encoder(self.right_video_path)
        self._frame_timestamps_file = self.frame_timestamps_path.open("w", encoding="utf-8")

        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._write_metadata()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Dual RealSense recording started, output_dir=%s", self.output_dir)
        logger.info(
            "camera_high serial=%s camera_wrist serial=%s",
            self.camera_high_serial,
            self.camera_wrist_serial,
        )

    def _capture_loop(self):
        try:
            while not self.stop_event.is_set():
                left_frames = _try_get_frames(self.left_pipeline, self.left_align)
                right_frames = _try_get_frames(self.right_pipeline, self.right_align)
                if left_frames is None or right_frames is None:
                    continue

                left_color_frame = left_frames.get_color_frame()
                right_color_frame = right_frames.get_color_frame()
                if not left_color_frame or not right_color_frame:
                    continue

                left_color = np.asanyarray(left_color_frame.get_data())
                right_color = np.asanyarray(right_color_frame.get_data())

                self._write_frame(self.left_encoder, left_color, "left")
                self._write_frame(self.right_encoder, right_color, "right")
                capture_monotonic_sec = time.monotonic()
                capture_wall_time_ns = time.time_ns()
                pair_index = min(self.left_frame_count, self.right_frame_count)
                if self._frame_timestamps_file is not None:
                    record = {
                        "pair_index": int(pair_index),
                        "capture_monotonic_sec": float(capture_monotonic_sec),
                        "capture_wall_time_ns": int(capture_wall_time_ns),
                        "camera_high_frame_index": int(self.left_frame_count),
                        "camera_wrist_frame_index": int(self.right_frame_count),
                        "camera_high_sensor_timestamp_ms": float(left_color_frame.get_timestamp()),
                        "camera_wrist_sensor_timestamp_ms": float(right_color_frame.get_timestamp()),
                        "camera_high_frame_number": int(left_color_frame.get_frame_number()),
                        "camera_wrist_frame_number": int(right_color_frame.get_frame_number()),
                    }
                    self._frame_timestamps_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._frame_timestamps_file.flush()
                self.left_frame_count += 1
                self.right_frame_count += 1

                if self.show_preview:
                    left_vis = _overlay_label(left_color, f"camera_high {self.camera_high_serial}")
                    right_vis = _overlay_label(right_color, f"camera_wrist {self.camera_wrist_serial}")
                    preview = np.hstack((left_vis, right_vis))
                    cv2.imshow(self.window_name, preview)
                    key_code = cv2.waitKey(1) & 0xFF
                    if key_code in (ord("q"), 27):
                        logger.info("Preview requested stop, ending replay and camera recording")
                        self.stop_event.set()
                        break
        except Exception as exc:
            self.thread_error = exc
            logger.exception("Dual RealSense capture loop crashed")
            self.stop_event.set()

    def should_stop(self):
        return self.stop_event.is_set()

    def raise_if_failed(self):
        if self.thread_error is not None:
            raise RuntimeError("dual RealSense recording failed") from self.thread_error

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

        try:
            self.left_pipeline.stop()
        except Exception:
            pass
        try:
            self.right_pipeline.stop()
        except Exception:
            pass

        if self.show_preview:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                cv2.destroyAllWindows()
        if self._frame_timestamps_file is not None:
            self._frame_timestamps_file.flush()
            self._frame_timestamps_file.close()
            self._frame_timestamps_file = None

        close_errors = []
        try:
            self._close_encoder(self.left_encoder, "left")
        except Exception as exc:
            close_errors.append(exc)
        finally:
            self.left_encoder = None
        try:
            self._close_encoder(self.right_encoder, "right")
        except Exception as exc:
            close_errors.append(exc)
        finally:
            self.right_encoder = None

        self.stopped_at = datetime.now().isoformat(timespec="seconds")
        self._write_metadata()
        logger.info(
            "Dual RealSense recording stopped, camera_high_frames=%d camera_wrist_frames=%d",
            self.left_frame_count,
            self.right_frame_count,
        )
        if close_errors:
            raise close_errors[0]

    def _write_metadata(self):
        metadata = {
            "camera_high_serial": self.camera_high_serial,
            "camera_wrist_serial": self.camera_wrist_serial,
            "legacy_left_serial": self.left_serial,
            "legacy_right_serial": self.right_serial,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "align_mode": self.align_mode,
            "show_preview": self.show_preview,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "camera_high_frame_count": self.left_frame_count,
            "camera_wrist_frame_count": self.right_frame_count,
            "video_backend": self.video_backend,
            "camera_high_video_path": str(self.camera_high_video_path),
            "camera_wrist_video_path": str(self.camera_wrist_video_path),
            "legacy_left_frame_count": self.left_frame_count,
            "legacy_right_frame_count": self.right_frame_count,
            "left_video_path": str(self.left_video_path),
            "right_video_path": str(self.right_video_path),
            "frame_timestamps_path": str(self.frame_timestamps_path),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Replay robot_eval pose trajectory with fixed orientation")
    parser.add_argument("--traj-json", type=str, required=True, help="Path to robot_eval.json")
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--stride", type=int, default=1, help="Replay every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip leading trajectory points")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of points to replay")
    parser.add_argument("--num-steps", type=int, default=5, help="OSC control steps per target point")
    parser.add_argument("--hold-steps", type=int, default=0, help="Additional hold steps at each target")
    parser.add_argument("--open-gripper-action", type=float, default=-1.0)
    parser.add_argument("--closed-gripper-action", type=float, default=1.0)
    parser.add_argument(
        "--respect-timestamps",
        action="store_true",
        help="Sleep between target points according to timestamp_sec from the trajectory.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiplier for recorded timing when --respect-timestamps is enabled.",
    )
    parser.add_argument(
        "--ee-to-center",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Optional offset from EE origin to gripper center in EE frame (meters); default is [0.0, 0.0, 0.0]",
    )
    parser.add_argument("--skip-reset", action="store_true", help="Do not reset joints before replay")
    parser.add_argument(
        "--camera-output-root",
        type=str,
        default="data/dual_realsense_recordings",
        help="Directory where dual RealSense videos will be saved",
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


def osc_move(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pose,
    num_steps,
    gripper_action=-1.0,
):
    target_pos, target_quat = target_pose

    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat

        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        action_pos = np.clip((target_pos - current_pos).flatten() * 10.0, -1.0, 1.0)
        action_axis_angle = np.clip(axis_angle_diff.flatten(), -0.5, 0.5)
        action = action_pos.tolist() + action_axis_angle.tolist() + [float(gripper_action)]

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


def _gripper_action_from_state(frame_entry, open_value, closed_value):
    state = frame_entry.get("gripper_state", "open")
    return float(closed_value if state == "closed" else open_value)


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


def load_robot_eval_traj(
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
    for frame in frames:
        pos = frame.get("position_abs_m")
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        if not all(np.isfinite(float(v)) for v in pos):
            continue
        traj.append(
            {
                "frame_index": int(frame["frame_index"]),
                "position": np.array(pos, dtype=np.float64).reshape(3, 1),
                "gripper_action": _gripper_action_from_state(
                    frame,
                    open_value=open_gripper_action,
                    closed_value=closed_gripper_action,
                ),
                "timestamp_sec": _optional_float(frame.get("timestamp_sec")),
            }
        )
    return payload, traj


def sleep_until_timestamp(point, first_timestamp, replay_start_time, time_scale):
    timestamp = point.get("timestamp_sec")
    if timestamp is None or first_timestamp is None:
        return
    target_time = replay_start_time + (timestamp - first_timestamp) * time_scale
    remaining = target_time - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def follow_robot_eval_traj(
    robot_interface,
    controller_type,
    controller_cfg,
    traj,
    num_steps=5,
    hold_steps=0,
    ee_to_center=None,
    should_stop=None,
    respect_timestamps=False,
    time_scale=1.0,
):
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)

    base_pose = robot_interface.last_eef_pose
    fixed_rot = base_pose[:3, :3]
    fixed_quat = transform_utils.mat2quat(base_pose[:3, :3])
    ee_to_center = np.array(
        [0.0, 0.0, 0.0] if ee_to_center is None else ee_to_center,
        dtype=np.float64,
    ).reshape(3, 1)

    logger.info("Using fixed orientation from current robot pose")
    logger.info(f"Fixed quaternion xyzw: {fixed_quat.tolist()}")
    logger.info("EE-to-center compensation xyz: %s", ee_to_center.flatten().tolist())

    first_timestamp = None
    if respect_timestamps:
        for point in traj:
            if point.get("timestamp_sec") is not None:
                first_timestamp = point["timestamp_sec"]
                break
        if first_timestamp is None:
            logger.warning(
                "--respect-timestamps was requested, but no timestamp_sec values were found; "
                "replaying at controller-loop speed"
            )
        else:
            logger.info("Respecting trajectory timestamps with time_scale=%.3f", time_scale)
    replay_start_time = time.monotonic()

    for point in traj:
        if should_stop is not None and should_stop():
            logger.warning("Stop requested before trajectory replay completed")
            break
        if respect_timestamps:
            sleep_until_timestamp(
                point,
                first_timestamp=first_timestamp,
                replay_start_time=replay_start_time,
                time_scale=time_scale,
            )
        target_center = point["position"]
        target_pos = target_center - fixed_rot @ ee_to_center
        gripper_action = point["gripper_action"]
        logger.info(
            "Replay frame=%s target_center=%s target_ee=%s gripper_action=%.3f timestamp=%s",
            point["frame_index"],
            target_center.flatten().tolist(),
            target_pos.flatten().tolist(),
            gripper_action,
            point.get("timestamp_sec"),
        )
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, fixed_quat),
            num_steps=num_steps,
            gripper_action=gripper_action,
        )
        if hold_steps > 0:
            osc_move(
                robot_interface,
                controller_type,
                controller_cfg,
                (target_pos, fixed_quat),
                num_steps=hold_steps,
                gripper_action=gripper_action,
            )


def main():
    args = parse_args()
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")

    payload, traj = load_robot_eval_traj(
        traj_json=args.traj_json,
        stride=args.stride,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        open_gripper_action=args.open_gripper_action,
        closed_gripper_action=args.closed_gripper_action,
    )
    if not traj:
        raise RuntimeError("No valid trajectory points found in robot_eval.json")

    logger.info(
        "Loaded %d trajectory points from %s (coordinate_frame=%s, stereo_mode=%s)",
        len(traj),
        args.traj_json,
        payload.get("coordinate_frame"),
        payload.get("stereo_mode"),
    )

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
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
        )

    recorder_stop_error = None
    try:
        if camera_recorder is not None:
            camera_recorder.start()
        if not args.skip_reset:
            reset_joint_positions = [
                0.09162008114028396,
                -0.19826458111314524,
                -0.01990020486871322,
                -2.4732269941140346,
                -0.01307073642274261,
                2.30396583422025,
                0.8480939705504309,
            ]
            reset_joints_to(robot_interface, reset_joint_positions)

        follow_robot_eval_traj(
            robot_interface,
            args.controller_type,
            controller_cfg,
            traj=traj,
            num_steps=args.num_steps,
            hold_steps=args.hold_steps,
            ee_to_center=args.ee_to_center,
            should_stop=None if camera_recorder is None else camera_recorder.should_stop,
            respect_timestamps=args.respect_timestamps,
            time_scale=args.time_scale,
        )
        if camera_recorder is not None:
            camera_recorder.raise_if_failed()
    finally:
        if camera_recorder is not None:
            try:
                camera_recorder.stop()
            except Exception as exc:
                recorder_stop_error = exc
        robot_interface.close()

    if recorder_stop_error is not None:
        raise recorder_stop_error
    if camera_recorder is not None:
        camera_recorder.raise_if_failed()


if __name__ == "__main__":
    main()
