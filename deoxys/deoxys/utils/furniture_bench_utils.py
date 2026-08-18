"""FurnitureBench camera and state adapters for Deoxys real-robot scripts."""

import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from deoxys.utils import transform_utils


DEFAULT_FRONT_SERIAL = "327122071654"
DEFAULT_WRIST_SERIAL = "001622071252"
DEFAULT_OBSTACLE_POSE = np.array(
    [0.0069, 0.3629, -0.0150, -1.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)

WRIST_TO_TIP = np.eye(4, dtype=np.float64)
WRIST_TO_TIP[:3, 3] = [0.0, 0.0, 0.1034]
WRIST_TO_TIP[:3, :3] = transform_utils.quat2mat(
    np.array([0.0, 0.0, -0.3826834323650898, 0.9238795325112867])
)


def camera_intrinsics_vector(intrinsics):
    return [
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy),
    ]


def camera_intrinsics_matrix(intrinsics):
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def center_crop_resize_geometry(input_width, input_height, output_width, output_height):
    """Return a distortion-free center-crop and resize description."""
    input_width = int(input_width)
    input_height = int(input_height)
    output_width = int(output_width)
    output_height = int(output_height)
    if min(input_width, input_height, output_width, output_height) <= 0:
        raise ValueError("image dimensions must be positive")

    if input_width * output_height > input_height * output_width:
        crop_height = input_height
        crop_width = int(round(input_height * output_width / output_height))
    else:
        crop_width = input_width
        crop_height = int(round(input_width * output_height / output_width))
    crop_x = (input_width - crop_width) // 2
    crop_y = (input_height - crop_height) // 2
    return {
        "method": "center_crop_then_resize",
        "input_width": input_width,
        "input_height": input_height,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "output_width": output_width,
        "output_height": output_height,
        "scale_x": output_width / crop_width,
        "scale_y": output_height / crop_height,
    }


def center_crop_resize(image, geometry, interpolation):
    """Apply ``center_crop_resize_geometry`` without changing aspect ratio."""
    expected_shape = (geometry["input_height"], geometry["input_width"])
    if image.shape[:2] != expected_shape:
        raise ValueError(
            f"image shape {image.shape[:2]} does not match {expected_shape}"
        )
    x = geometry["crop_x"]
    y = geometry["crop_y"]
    width = geometry["crop_width"]
    height = geometry["crop_height"]
    cropped = image[y : y + height, x : x + width]
    return cv2.resize(
        cropped,
        (geometry["output_width"], geometry["output_height"]),
        interpolation=interpolation,
    )


def transformed_intrinsics(intrinsics, geometry):
    """Transform color intrinsics through a center crop and resize."""
    return {
        "fx": float(intrinsics.fx) * geometry["scale_x"],
        "fy": float(intrinsics.fy) * geometry["scale_y"],
        "ppx": (float(intrinsics.ppx) - geometry["crop_x"])
        * geometry["scale_x"],
        "ppy": (float(intrinsics.ppy) - geometry["crop_y"])
        * geometry["scale_y"],
        "width": geometry["output_width"],
        "height": geometry["output_height"],
    }


def connected_realsense_devices():
    devices = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        devices[serial] = device.get_info(rs.camera_info.name)
    return devices


class RealSenseCamera:
    """One RealSense color stream with optional aligned metric depth."""

    def __init__(
        self,
        serial,
        width=640,
        height=480,
        fps=30,
        auto_exposure=True,
        enable_depth=True,
        depth_width=None,
        depth_height=None,
        depth_fps=None,
    ):
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_width = int(width if depth_width is None else depth_width)
        self.depth_height = int(height if depth_height is None else depth_height)
        self.depth_fps = int(fps if depth_fps is None else depth_fps)
        self.auto_exposure = bool(auto_exposure)
        self.enable_depth = bool(enable_depth)
        self.pipeline = rs.pipeline()
        self.aligner = rs.align(rs.stream.color) if self.enable_depth else None
        self.started = False
        self.depth_scale_m = None
        self.intrinsics = None
        self.depth_intrinsics = None

    def start(self):
        config = rs.config()
        config.enable_device(self.serial)
        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.depth_width,
                self.depth_height,
                rs.format.z16,
                self.depth_fps,
            )
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        profile = self.pipeline.start(config)
        self.started = True
        device = profile.get_device()
        if self.enable_depth:
            self.depth_scale_m = float(
                device.first_depth_sensor().get_depth_scale()
            )
            depth_profile = profile.get_stream(
                rs.stream.depth
            ).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()
        if not self.auto_exposure:
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        return self

    def read(self, timeout_ms=1000):
        try:
            success, frames = self.pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None
        if not success:
            return None
        if self.aligner is not None:
            frames = self.aligner.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        depth_m = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_m = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32)
                * self.depth_scale_m
            )
        return {
            "bgr": np.asanyarray(color_frame.get_data()).copy(),
            "depth_m": depth_m,
            "sensor_timestamp_ms": float(color_frame.get_timestamp()),
            "frame_number": int(color_frame.get_frame_number()),
            "wall_time_ns": time.time_ns(),
        }

    def metadata(self):
        if self.intrinsics is None:
            raise RuntimeError("camera must be started before reading metadata")
        return {
            "serial": self.serial,
            "stream_width": self.width,
            "stream_height": self.height,
            "fps": self.fps,
            "depth_enabled": self.enable_depth,
            "depth_scale_m": self.depth_scale_m,
            "depth_stream_width": self.depth_width if self.enable_depth else None,
            "depth_stream_height": self.depth_height if self.enable_depth else None,
            "depth_fps": self.depth_fps if self.enable_depth else None,
            "aligned_depth_width": self.width if self.enable_depth else None,
            "aligned_depth_height": self.height if self.enable_depth else None,
            "intrinsics": {
                "fx": float(self.intrinsics.fx),
                "fy": float(self.intrinsics.fy),
                "ppx": float(self.intrinsics.ppx),
                "ppy": float(self.intrinsics.ppy),
                "width": int(self.intrinsics.width),
                "height": int(self.intrinsics.height),
            },
            "depth_intrinsics": (
                None
                if self.depth_intrinsics is None
                else {
                    "fx": float(self.depth_intrinsics.fx),
                    "fy": float(self.depth_intrinsics.fy),
                    "ppx": float(self.depth_intrinsics.ppx),
                    "ppy": float(self.depth_intrinsics.ppy),
                    "width": int(self.depth_intrinsics.width),
                    "height": int(self.depth_intrinsics.height),
                }
            ),
        }

    def stop(self):
        if self.started:
            try:
                self.pipeline.stop()
            finally:
                self.started = False


def estimate_camera_to_april_frame(color_bgr, intrinsics, detector=None):
    """Estimate the camera-to-FurnitureBench-AprilTag-frame transform."""
    from furniture_bench.config import config
    from furniture_bench.perception.apriltag import AprilTag
    from furniture_bench.utils import transform as fb_transform
    from furniture_bench.utils.pose import comp_avg_pose

    if detector is None:
        detector = AprilTag(config["furniture"]["base_tag_size"])
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    tags = detector.detect_id(color_rgb, camera_intrinsics_vector(intrinsics))
    transforms = []
    for tag_id in config["furniture"]["base_tags"]:
        tag = tags.get(tag_id)
        if tag is None:
            continue
        coordinate_to_tag = config["furniture"]["rel_pose_from_coordinate"][tag_id]
        camera_to_coordinate = fb_transform.to_homogeneous(
            tag.pose_t,
            tag.pose_R,
        ) @ np.linalg.inv(coordinate_to_tag)
        transforms.append(np.linalg.inv(camera_to_coordinate))
    if not transforms:
        return None
    return comp_avg_pose(transforms)


class OneLegPoseTracker:
    """Track the dynamic one-leg parts in the FurnitureBench AprilTag frame."""

    def __init__(self, obstacle_pose=None, camera_pose_sample_count=10):
        from furniture_bench.config import config
        from furniture_bench.furniture import furniture_factory
        from furniture_bench.perception.apriltag import AprilTag

        self.config = config
        self.furniture = furniture_factory("one_leg")
        self.base_detector = AprilTag(config["furniture"]["base_tag_size"])
        self.detector = AprilTag(self.furniture.tag_size)
        self.obstacle_pose = np.asarray(
            DEFAULT_OBSTACLE_POSE if obstacle_pose is None else obstacle_pose,
            dtype=np.float32,
        ).reshape(7)
        self.camera_to_april = None
        self.camera_pose_sample_count = int(camera_pose_sample_count)
        if self.camera_pose_sample_count <= 0:
            raise ValueError("camera_pose_sample_count must be positive")
        self._camera_pose_samples = []
        self.last_poses = np.zeros((6, 7), dtype=np.float32)
        self.valid = np.zeros(6, dtype=bool)
        self.last_seen_ns = np.zeros(6, dtype=np.int64)
        self._initialize_configured_poses()

    def _initialize_configured_poses(self):
        from furniture_bench.utils import transform as fb_transform

        for part_index in (1, 2, 3):
            part = self.furniture.parts[part_index]
            position = np.asarray(part.reset_pos[0], dtype=np.float32)
            orientation = np.asarray(part.reset_ori[0], dtype=np.float32)
            quaternion = fb_transform.mat2quat(orientation[:3, :3])
            self.last_poses[part_index] = np.concatenate([position, quaternion])
            self.valid[part_index] = True
        self.last_poses[5] = self.obstacle_pose
        self.valid[5] = True

    @property
    def ready(self):
        return bool(self.valid[0] and self.valid[4])

    def update(self, color_bgr, intrinsics):
        from furniture_bench.utils import transform as fb_transform
        from furniture_bench.utils.detection import _get_parts_pose
        from furniture_bench.utils.pose import comp_avg_pose

        now_ns = time.monotonic_ns()
        if self.camera_to_april is None:
            camera_to_april = estimate_camera_to_april_frame(
                color_bgr,
                intrinsics,
                self.base_detector,
            )
            if camera_to_april is not None:
                self._camera_pose_samples.append(camera_to_april)
            if len(self._camera_pose_samples) >= self.camera_pose_sample_count:
                self.camera_to_april = comp_avg_pose(self._camera_pose_samples)

        found = np.zeros(6, dtype=bool)
        if self.camera_to_april is not None:
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            tags = self.detector.detect_id(
                color_rgb,
                camera_intrinsics_vector(intrinsics),
            )
            for part_index in (0, 4):
                pose_in_camera = _get_parts_pose(
                    self.furniture.parts[part_index],
                    tags,
                )
                if pose_in_camera is None:
                    continue
                pose_in_april = self.camera_to_april @ pose_in_camera
                position, quaternion = fb_transform.mat2pose(pose_in_april)
                self.last_poses[part_index] = np.concatenate(
                    [position, quaternion]
                ).astype(np.float32)
                self.valid[part_index] = True
                self.last_seen_ns[part_index] = now_ns
                found[part_index] = True

        age_ms = np.zeros(6, dtype=np.float32)
        for part_index in (0, 4):
            if not self.valid[part_index]:
                age_ms[part_index] = np.inf
            else:
                age_ms[part_index] = (
                    now_ns - self.last_seen_ns[part_index]
                ) / 1_000_000.0
        return {
            "parts_poses": self.last_poses.reshape(-1).copy(),
            "parts_founds": found,
            "parts_pose_valid": self.valid.copy(),
            "parts_pose_age_ms": age_ms,
            "camera_pose_samples": len(self._camera_pose_samples),
            "camera_pose_samples_required": self.camera_pose_sample_count,
            "parts_poses_frame": "furniture_bench_april_tag",
            "camera_to_april": (
                None
                if self.camera_to_april is None
                else self.camera_to_april.copy()
            ),
        }


class DualRealSenseSnapshotter:
    """Continuously capture front/wrist RGB-D and optional one-leg state."""

    def __init__(
        self,
        front_serial=DEFAULT_FRONT_SERIAL,
        wrist_serial=DEFAULT_WRIST_SERIAL,
        width=640,
        height=480,
        fps=30,
        record_width=320,
        record_height=240,
        track_one_leg=True,
        obstacle_pose=None,
        front_width=None,
        front_height=None,
        front_fps=None,
        front_depth_width=None,
        front_depth_height=None,
        front_depth_fps=None,
        wrist_width=None,
        wrist_height=None,
        wrist_fps=None,
        wrist_depth_width=None,
        wrist_depth_height=None,
        wrist_depth_fps=None,
    ):
        if str(front_serial) == str(wrist_serial):
            raise ValueError("front and wrist camera serials must differ")
        front_width = width if front_width is None else front_width
        front_height = height if front_height is None else front_height
        front_fps = fps if front_fps is None else front_fps
        wrist_width = width if wrist_width is None else wrist_width
        wrist_height = height if wrist_height is None else wrist_height
        wrist_fps = fps if wrist_fps is None else wrist_fps
        self.front = RealSenseCamera(
            front_serial,
            front_width,
            front_height,
            front_fps,
            depth_width=front_depth_width,
            depth_height=front_depth_height,
            depth_fps=front_depth_fps,
        )
        self.wrist = RealSenseCamera(
            wrist_serial,
            wrist_width,
            wrist_height,
            wrist_fps,
            depth_width=wrist_depth_width,
            depth_height=wrist_depth_height,
            depth_fps=wrist_depth_fps,
        )
        self.record_size = (int(record_width), int(record_height))
        self.front_record_geometry = center_crop_resize_geometry(
            front_width,
            front_height,
            *self.record_size,
        )
        self.wrist_record_geometry = center_crop_resize_geometry(
            wrist_width,
            wrist_height,
            *self.record_size,
        )
        self.tracker = OneLegPoseTracker(obstacle_pose) if track_one_leg else None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._thread_error = None
        self._latest = None

    def start(self):
        devices = connected_realsense_devices()
        missing = [
            camera.serial
            for camera in (self.front, self.wrist)
            if camera.serial not in devices
        ]
        if missing:
            raise RuntimeError(
                "required RealSense camera not found: " + ", ".join(missing)
            )
        self.front.start()
        try:
            self.wrist.start()
        except Exception:
            self.front.stop()
            raise
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="dual_realsense_snapshotter",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self):
        try:
            while not self._stop_event.is_set():
                front = self.front.read()
                wrist = self.wrist.read()
                if front is None or wrist is None:
                    continue
                sample = {
                    "color_image1": cv2.cvtColor(
                        center_crop_resize(
                            wrist["bgr"],
                            self.wrist_record_geometry,
                            cv2.INTER_AREA,
                        ),
                        cv2.COLOR_BGR2RGB,
                    ),
                    "color_image2": cv2.cvtColor(
                        center_crop_resize(
                            front["bgr"],
                            self.front_record_geometry,
                            cv2.INTER_AREA,
                        ),
                        cv2.COLOR_BGR2RGB,
                    ),
                    "depth_image1": center_crop_resize(
                        wrist["depth_m"],
                        self.wrist_record_geometry,
                        cv2.INTER_NEAREST,
                    ).astype(np.float16),
                    "depth_image2": center_crop_resize(
                        front["depth_m"],
                        self.front_record_geometry,
                        cv2.INTER_NEAREST,
                    ).astype(np.float16),
                    "camera_capture_wall_time_ns": time.time_ns(),
                    "front_sensor_timestamp_ms": front["sensor_timestamp_ms"],
                    "wrist_sensor_timestamp_ms": wrist["sensor_timestamp_ms"],
                    "front_frame_number": front["frame_number"],
                    "wrist_frame_number": wrist["frame_number"],
                }
                if self.tracker is not None:
                    sample.update(
                        self.tracker.update(front["bgr"], self.front.intrinsics)
                    )
                with self._lock:
                    self._latest = sample
        except Exception as exc:
            self._thread_error = exc
            self._stop_event.set()

    def latest(self):
        if self._thread_error is not None:
            raise RuntimeError("dual RealSense capture failed") from self._thread_error
        with self._lock:
            if self._latest is None:
                return None
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._latest.items()
            }

    def metadata(self):
        return {
            "camera_key_mapping": {
                "color_image1": "wrist",
                "color_image2": "front",
                "depth_image1": "wrist",
                "depth_image2": "front",
            },
            "record_width": self.record_size[0],
            "record_height": self.record_size[1],
            "depth_encoding": "float16_positive_meters",
            "front": {
                **self.front.metadata(),
                "record_transform": self.front_record_geometry,
                "record_intrinsics": transformed_intrinsics(
                    self.front.intrinsics,
                    self.front_record_geometry,
                ),
            },
            "wrist": {
                **self.wrist.metadata(),
                "record_transform": self.wrist_record_geometry,
                "record_intrinsics": transformed_intrinsics(
                    self.wrist.intrinsics,
                    self.wrist_record_geometry,
                ),
            },
        }

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.wrist.stop()
        self.front.stop()


def wrist_pose_to_tip_pose(wrist_pose):
    return np.asarray(wrist_pose, dtype=np.float64).reshape(4, 4) @ WRIST_TO_TIP


def deoxys_delta_to_furniture_bench_action(scaled_action, wrist_pose):
    """Convert a scaled Deoxys world-left wrist delta to a local tip delta."""
    action = np.asarray(scaled_action, dtype=np.float64).reshape(7)
    current_wrist = np.asarray(wrist_pose, dtype=np.float64).reshape(4, 4)
    goal_wrist = current_wrist.copy()
    goal_wrist[:3, 3] = current_wrist[:3, 3] + action[:3]
    world_delta_rotation = transform_utils.quat2mat(
        transform_utils.axisangle2quat(action[3:6])
    )
    goal_wrist[:3, :3] = world_delta_rotation @ current_wrist[:3, :3]

    current_tip = wrist_pose_to_tip_pose(current_wrist)
    goal_tip = wrist_pose_to_tip_pose(goal_wrist)
    delta_position = goal_tip[:3, 3] - current_tip[:3, 3]
    local_delta_rotation = current_tip[:3, :3].T @ goal_tip[:3, :3]
    delta_quaternion = transform_utils.mat2quat(local_delta_rotation)
    return np.concatenate(
        [delta_position, delta_quaternion, [np.sign(action[-1])]],
    ).astype(np.float32)
