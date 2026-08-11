#!/usr/bin/env python3
"""Move Franka with OSC and record synchronized RealSense RGB-D + poses."""
import argparse
import os
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np
import pyrealsense2 as rs

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--motion-radius", type=float, default=0.2)
    parser.add_argument("--rot-deg", type=float, default=60.0)
    parser.add_argument("--num-osc-steps", type=int, default=10)
    parser.add_argument("--num-additional-osc-steps", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/franka_record_data.npz")
    parser.add_argument(
        "--initial-settle-time",
        type=float,
        default=1.0,
        help="Wait time in seconds after reset before capturing the initial frame.",
    )
    parser.add_argument(
        "--per-point-settle-time",
        type=float,
        default=0.5,
        help=(
            "Hold each target pose for this many seconds while continuing to send "
            "OSC commands before capturing a new frame."
        ),
    )
    parser.add_argument("--show", action="store_true", default=True, help="Show camera window")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Disable camera window")
    return parser.parse_args()


def osc_move(robot_interface, controller_type, controller_cfg, target_pose, num_steps):
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

        action_pos = (target_pos - current_pos).flatten() * 10.0
        action_axis_angle = axis_angle_diff.flatten() * 1.0
        action_pos = np.clip(action_pos, -1.0, 1.0)
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

        action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


def hold_target_pose(
    robot_interface, controller_type, controller_cfg, target_pose, settle_time
):
    settle_time = max(0.0, float(settle_time))
    settle_deadline = time.time() + settle_time
    while time.time() < settle_deadline:
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            target_pose,
            num_steps=1,
        )
    return time.time()


def wait_for_frame(frame_queue, timeout, min_timestamp=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        wait_timeout = min(0.1, deadline - time.time())
        if wait_timeout <= 0.0:
            break
        try:
            frame = frame_queue.get(timeout=wait_timeout)
        except Empty:
            continue
        if min_timestamp is None or frame[2] >= min_timestamp:
            return frame
    return None


def main():
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    color_intrinsics = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    intrinsics = {
        "fx": color_intrinsics.fx,
        "fy": color_intrinsics.fy,
        "cx": color_intrinsics.ppx,
        "cy": color_intrinsics.ppy,
    }

    align = rs.align(rs.stream.color)
    frame_queue = Queue(maxsize=1)

    def camera_thread():
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
            ts = time.time()
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put((color, depth, ts))

    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_cfg = get_default_controller_config(args.controller_type)

    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]
    reset_joints_to(robot_interface, reset_joint_positions, gripper_open=True)

    while robot_interface.state_buffer_size == 0:
        time.sleep(0.05)

    steps = args.steps
    motion_radius = float(args.motion_radius)
    rot_angle = float(args.rot_deg) * np.pi / 180.0
    per_point_settle_time = max(0.0, float(args.per_point_settle_time))

    recorded_poses = []
    recorded_rgb = []
    recorded_depth = []
    recorded_timestamps = []

    try:
        stop_requested = False
        last_frame = None
        T_init = None

        if steps > 0:
            settle_time = max(0.0, float(args.initial_settle_time))
            settle_deadline = time.time() + settle_time
            if settle_time > 0.0:
                time.sleep(settle_time)

            last_frame = wait_for_frame(
                frame_queue, timeout=5.0, min_timestamp=settle_deadline
            )
            if last_frame is None:
                raise RuntimeError(
                    "No camera frames received after robot initialization."
                )

            T_init = robot_interface.last_eef_pose.copy()
            color, depth, ts = last_frame
            if args.show:
                cv2.imshow("realsense", color)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    stop_requested = True

            recorded_rgb.append(color.copy())
            recorded_depth.append(depth.copy())
            recorded_timestamps.append(ts)
            recorded_poses.append(T_init.copy())

        for i in range(1, steps):
            if stop_requested:
                break
            dx = motion_radius * np.sin(2 * np.pi * i / steps)
            dy = motion_radius * np.cos(2 * np.pi * i / steps)

            dT = np.eye(4)
            dT[0, 3] = dx
            dT[1, 3] = dy

            theta = rot_angle * np.sin(2 * np.pi * i / steps)
            Rz = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            dT[:3, :3] = Rz

            T_target = T_init @ dT
            target_pos = T_target[:3, 3].reshape(3, 1)
            target_quat = transform_utils.mat2quat(T_target[:3, :3])
            osc_move(
                robot_interface,
                args.controller_type,
                controller_cfg,
                (target_pos, target_quat),
                num_steps=args.num_osc_steps,
            )
            if args.num_additional_osc_steps > 0:
                osc_move(
                    robot_interface,
                    args.controller_type,
                    controller_cfg,
                    (target_pos, target_quat),
                    num_steps=args.num_additional_osc_steps,
                )

            settle_complete_time = hold_target_pose(
                robot_interface,
                args.controller_type,
                controller_cfg,
                (target_pos, target_quat),
                per_point_settle_time,
            )
            last_frame = wait_for_frame(
                frame_queue,
                timeout=max(5.0, per_point_settle_time + 1.0),
                min_timestamp=settle_complete_time,
            )
            if last_frame is None:
                raise RuntimeError(
                    f"No camera frame received after settling at trajectory step {i}."
                )
            color, depth, ts = last_frame
            if args.show:
                cv2.imshow("realsense", color)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
            recorded_rgb.append(color.copy())
            recorded_depth.append(depth.copy())
            recorded_timestamps.append(ts)

            recorded_poses.append(robot_interface.last_eef_pose.copy())
            time.sleep(0.05)
    finally:
        pipeline.stop()
        robot_interface.close()
        if args.show:
            cv2.destroyAllWindows()

    output_path = args.output
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        output_path,
        poses=np.array(recorded_poses),
        rgb=np.array(recorded_rgb),
        depth=np.array(recorded_depth),
        timestamps=np.array(recorded_timestamps),
        intrinsics=intrinsics,
    )
    print(f"data saved to {output_path}")


if __name__ == "__main__":
    main()
