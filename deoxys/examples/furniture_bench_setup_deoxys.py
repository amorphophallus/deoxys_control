"""FurnitureBench camera calibration and read-only checks using Deoxys hardware."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.furniture_bench_utils import (
    DEFAULT_FRONT_SERIAL,
    DEFAULT_WRIST_SERIAL,
    OneLegPoseTracker,
    RealSenseCamera,
    camera_intrinsics_matrix,
    connected_realsense_devices,
    estimate_camera_to_april_frame,
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


def prepare_robot(interface_cfg, reset_timeout, lift_height):
    answer = input(
        "This will move the real robot to the reset pose and lift it "
        f"{lift_height:.3f} m. Type MOVE to continue: "
    )
    if answer != "MOVE":
        print("Robot preparation cancelled.")
        return

    robot = FrankaInterface(interface_cfg, use_visualizer=False)
    joint_cfg = get_default_controller_config("JOINT_POSITION")
    osc_cfg = get_default_controller_config("OSC_POSE")
    try:
        if not wait_for_robot_state(robot):
            raise RuntimeError("robot state was not received")
        target_joints = np.asarray(RESET_JOINT_POSITIONS, dtype=np.float64)
        deadline = time.monotonic() + reset_timeout
        while time.monotonic() < deadline:
            error = target_joints - robot.last_q
            if np.max(np.abs(error)) < 1e-3:
                break
            robot.control(
                controller_type="JOINT_POSITION",
                action=target_joints.tolist() + [-1.0],
                controller_cfg=joint_cfg,
            )
        else:
            raise RuntimeError("joint reset timed out")

        initial_pose = robot.last_eef_pose
        target_position = initial_pose[:3, 3] + np.array([0.0, 0.0, lift_height])
        deadline = time.monotonic() + reset_timeout
        while time.monotonic() < deadline:
            residual = target_position - robot.last_eef_pose[:3, 3]
            if np.max(np.abs(residual)) < 0.002:
                print("Robot preparation completed.")
                return
            step = np.clip(residual, -0.004, 0.004)
            robot.control(
                controller_type="OSC_POSE",
                action=np.concatenate([step, np.zeros(3), [-1.0]]),
                controller_cfg=osc_cfg,
            )
        raise RuntimeError("Cartesian lift timed out")
    finally:
        try:
            robot.control(
                controller_type="OSC_POSE",
                action=np.array([0.0] * 6 + [1.0]),
                controller_cfg=osc_cfg,
                termination=True,
            )
        finally:
            robot.close()


def add_error_line(image, label, value, threshold, row):
    color = (0, 200, 0) if abs(value) <= threshold else (0, 0, 255)
    cv2.putText(
        image,
        f"{label}: {value:+.4f}",
        (35, 45 + row * 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )


def run_calibration(args):
    from furniture_bench.config import config
    from furniture_bench.perception.apriltag import AprilTag
    from furniture_bench.scripts.calibration import ASSET_ROOT, avg_pose
    from furniture_bench.utils.pose import mat_to_roll_pitch_yaw

    if args.prepare_robot:
        prepare_robot(args.interface_cfg, args.reset_timeout, args.lift_height)

    reference_path = Path(ASSET_ROOT) / "calibration" / f"{args.target}.png"
    reference = cv2.imread(str(reference_path))
    if reference is None:
        raise FileNotFoundError(f"calibration reference not found: {reference_path}")

    camera = RealSenseCamera(
        args.front_camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        auto_exposure=not args.disable_auto_exposure,
        enable_depth=False,
    )
    camera.start()
    target_pose = np.asarray(avg_pose[args.target], dtype=np.float64)
    target_rpy = np.asarray(mat_to_roll_pitch_yaw(target_pose[:3, :3]))
    base_detector = AprilTag(config["furniture"]["base_tag_size"])
    print("Calibration started. Press q or Esc in the window to quit.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            color = frame["bgr"]
            if (
                reference.shape[1] * color.shape[0]
                != color.shape[1] * reference.shape[0]
            ):
                raise RuntimeError(
                    "calibration camera and reference image must have the same "
                    "aspect ratio; use the default 1280x720 profile"
                )
            resized_reference = cv2.resize(reference, (color.shape[1], color.shape[0]))
            view = cv2.addWeighted(color, 0.65, resized_reference, 0.25, 0.0)
            camera_to_april = estimate_camera_to_april_frame(
                color,
                camera.intrinsics,
                base_detector,
            )
            if camera_to_april is None:
                cv2.putText(
                    view,
                    "Base AprilTags not detected",
                    (35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                position_error = camera_to_april[:3, 3] - target_pose[:3, 3]
                rotation_error_deg = np.degrees(
                    np.asarray(
                        mat_to_roll_pitch_yaw(camera_to_april[:3, :3])
                    )
                    - target_rpy
                )
                for row, (axis, value) in enumerate(
                    zip("xyz", position_error)
                ):
                    add_error_line(
                        view,
                        f"{axis} pos [m]",
                        value,
                        args.position_threshold,
                        row,
                    )
                for offset, (axis, value) in enumerate(
                    zip("xyz", rotation_error_deg),
                    start=3,
                ):
                    add_error_line(
                        view,
                        f"{axis} rot [deg]",
                        value,
                        args.rotation_threshold,
                        offset,
                    )
                april_to_camera = np.linalg.inv(camera_to_april)
                rotation_vector, _ = cv2.Rodrigues(april_to_camera[:3, :3])
                cv2.drawFrameAxes(
                    view,
                    camera_intrinsics_matrix(camera.intrinsics),
                    np.zeros(5),
                    rotation_vector,
                    april_to_camera[:3, 3],
                    0.05,
                    4,
                )

            cv2.imshow("FurnitureBench front-camera calibration", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def depth_preview(depth_m):
    clipped = np.clip(depth_m.astype(np.float32), 0.0, 1.5)
    image = np.uint8(clipped / 1.5 * 255.0)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def print_connected_devices():
    devices = connected_realsense_devices()
    print("Connected RealSense devices:")
    for serial, name in devices.items():
        print(f"  {serial}: {name}")
    return devices


def run_front_test(args):
    devices = print_connected_devices()
    if args.front_camera_serial not in devices:
        raise RuntimeError(f"front camera not found: {args.front_camera_serial}")

    camera = RealSenseCamera(
        args.front_camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=False,
    )
    tracker = OneLegPoseTracker()
    camera.start()
    try:
        last_report = 0.0
        print(
            "Front-only AprilTag test started. PASS requires valid all 1 and "
            "tabletop/leg visible in the current frame. Press q or Esc."
        )
        while True:
            frame = camera.read()
            if frame is None:
                continue
            sample = tracker.update(frame["bgr"], camera.intrinsics)
            preview = frame["bgr"].copy()
            found = sample["parts_founds"]
            valid = sample["parts_pose_valid"]
            passed = bool(valid.all() and found[0] and found[4])
            status = (
                f"found={found.astype(int).tolist()} "
                f"valid={valid.astype(int).tolist()} "
                f"base={sample['camera_pose_samples']}/"
                f"{sample['camera_pose_samples_required']} "
                f"{'PASS' if passed else 'WAIT'}"
            )
            cv2.putText(
                preview,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if passed else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("FurnitureBench front-only valid test", preview)
            now = time.monotonic()
            if now - last_report >= 1.0:
                print(status)
                last_report = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def run_wrist_test(args):
    devices = print_connected_devices()
    if args.wrist_camera_serial not in devices:
        raise RuntimeError(f"wrist camera not found: {args.wrist_camera_serial}")

    camera = RealSenseCamera(
        args.wrist_camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=True,
    )
    camera.start()
    frame_count = 0
    report_started = time.monotonic()
    try:
        print("Wrist-only RGB-D test started. Press q or Esc.")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            frame_count += 1
            now = time.monotonic()
            elapsed = now - report_started
            measured_fps = frame_count / elapsed if elapsed > 0 else 0.0
            color = frame["bgr"]
            depth = depth_preview(frame["depth_m"])
            preview = cv2.hconcat([color, depth])
            cv2.putText(
                preview,
                f"wrist RGB-D measured_fps={measured_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if measured_fps >= 10.0 else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("FurnitureBench wrist-only RGB-D test", preview)
            if elapsed >= 2.0:
                print(f"wrist measured_fps={measured_fps:.1f}")
                frame_count = 0
                report_started = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_front(subparser, width, height, fps):
        subparser.add_argument(
            "--front-camera-serial",
            default=DEFAULT_FRONT_SERIAL,
        )
        subparser.add_argument("--width", type=int, default=width)
        subparser.add_argument("--height", type=int, default=height)
        subparser.add_argument("--fps", type=int, default=fps)

    def add_interface(subparser):
        subparser.add_argument(
            "--interface-cfg",
            default="config/charmander.yml",
        )

    calibrate = subparsers.add_parser("calibrate")
    add_front(calibrate, 1280, 720, fps=15)
    add_interface(calibrate)
    calibrate.add_argument(
        "--target",
        choices=("setup_front", "obstacle", "one_leg"),
        required=True,
    )
    calibrate.add_argument("--position-threshold", type=float, default=0.004)
    calibrate.add_argument("--rotation-threshold", type=float, default=0.8)
    calibrate.add_argument("--disable-auto-exposure", action="store_true")
    calibrate.add_argument("--prepare-robot", action="store_true")
    calibrate.add_argument("--reset-timeout", type=float, default=15.0)
    calibrate.add_argument("--lift-height", type=float, default=0.2)

    test_front = subparsers.add_parser("test-front")
    add_front(test_front, 1280, 720, fps=15)

    test_wrist = subparsers.add_parser("test-wrist")
    test_wrist.add_argument(
        "--wrist-camera-serial",
        default=DEFAULT_WRIST_SERIAL,
    )
    test_wrist.add_argument("--width", type=int, default=640)
    test_wrist.add_argument("--height", type=int, default=480)
    test_wrist.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "calibrate":
        run_calibration(args)
    elif args.command == "test-front":
        run_front_test(args)
    else:
        run_wrist_test(args)


if __name__ == "__main__":
    main()
