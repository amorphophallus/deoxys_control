"""Collect synchronized samples for eye-to-hand calibration (fixed RealSense, fixed chessboard).

This script records pairs of:
  - T_base_ee: Franka end-effector pose in base frame (4x4)
  - T_cam_board: Chessboard pose in camera frame (4x4) from solvePnP

Modes:
  - Manual: move robot yourself; press 's' to save one sample.
  - Auto-move: robot moves through random small pose deltas and saves samples automatically.

Board parameters for your setup:
  - inner corners: 6x8
  - square size: 0.01 m

Keys (when --show):
  - s: save one sample (requires chessboard detected)
  - q / ESC: quit

Safety:
  - Auto-move is disabled by default. Use --auto_move to enable robot motion.
"""

import argparse
import csv
import json
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _mat44_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _rt_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R, t


def _build_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def _detect_corners_for_size(gray: np.ndarray, pattern_size: Tuple[int, int]):
    cols, rows = pattern_size

    # Try the more robust SB detector first (if available)
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags |= cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
        if ok:
            return True, corners

    # Fallback to classic detector + subpix
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if not ok:
        return False, None

    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return True, corners


def _detect_corners(gray: np.ndarray, pattern_size: Tuple[int, int], try_swap: bool):
    ok, corners = _detect_corners_for_size(gray, pattern_size)
    if ok:
        return True, corners, pattern_size
    if try_swap:
        swapped = (pattern_size[1], pattern_size[0])
        ok2, corners2 = _detect_corners_for_size(gray, swapped)
        if ok2:
            return True, corners2, swapped
    return False, None, pattern_size


def _intrinsics_from_realsense(stream_profile: rs.video_stream_profile):
    intr = stream_profile.get_intrinsics()
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array(intr.coeffs[:5], dtype=np.float64).reshape(5, 1)
    return intr, K, dist


# Robot motion helpers: mirror osc_control_replay.py style

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

        action_pos = (target_pos - current_pos).flatten() * 10
        action_axis_angle = axis_angle_diff.flatten() * 1
        action_pos = np.clip(action_pos, -1.0, 1.0)
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

        action = action_pos.tolist() + action_axis_angle.tolist() + [float(gripper_action)]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


def move_to_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_delta_pose,
    num_steps,
    num_additional_steps,
    interpolation_method="linear",
    gripper_action=-1.0,
):
    _ = interpolation_method
    while robot_interface.state_buffer_size == 0:
        time.sleep(0.2)

    target_delta_pos, target_delta_axis_angle = (
        target_delta_pose[:3],
        target_delta_pose[3:],
    )

    current_ee_pose = robot_interface.last_eef_pose
    current_pos = current_ee_pose[:3, 3:]
    current_rot = current_ee_pose[:3, :3]
    current_quat = transform_utils.mat2quat(current_rot)
    current_axis_angle = transform_utils.quat2axisangle(current_quat)

    target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos
    target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle
    target_quat = transform_utils.axisangle2quat(target_axis_angle)

    osc_move(
        robot_interface,
        controller_type,
        controller_cfg,
        (target_pos, target_quat),
        num_steps,
        gripper_action=gripper_action,
    )
    osc_move(
        robot_interface,
        controller_type,
        controller_cfg,
        (target_pos, target_quat),
        num_additional_steps,
        gripper_action=gripper_action,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")

    parser.add_argument("--out", type=str, default="./calib_data")
    parser.add_argument("--name", type=str, default="eye_to_hand")

    parser.add_argument("--cols", type=int, default=6, help="Chessboard inner corners cols")
    parser.add_argument("--rows", type=int, default=8, help="Chessboard inner corners rows")
    parser.add_argument("--square", type=float, default=0.01, help="Square size in meters")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--draw_axes", action="store_true", help="Draw board axes when detected")
    parser.add_argument("--axis_len", type=float, default=0.05, help="Axis length (m) for visualization")

    parser.add_argument("--save_depth", action="store_true", help="Also save aligned depth PNGs")
    parser.add_argument("--align", choices=["color", "none"], default="color")

    parser.add_argument("--auto_move", action="store_true", help="Enable robot auto motion and auto sample saving.")
    parser.add_argument("--reset", action="store_true", help="Reset joints to a default pose before collecting.")

    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to collect in auto mode.")
    parser.add_argument("--max_translation", type=float, default=0.08, help="Max translation magnitude (m) for random delta poses.")
    parser.add_argument("--max_rotation", type=float, default=0.35, help="Max axis-angle magnitude (rad) for random delta poses.")
    parser.add_argument("--settle_sec", type=float, default=0.3, help="Wait time after motion before sampling (sec).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for auto mode.")

    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--move_steps", type=int, default=80)
    parser.add_argument("--move_extra_steps", type=int, default=40)

    # Detection robustness: try swapped (rows, cols) and retry across multiple frames
    parser.add_argument("--try_swap", action="store_true", help="If (cols,rows) fails, also try (rows,cols).")
    parser.add_argument(
        "--detect_timeout_sec",
        type=float,
        default=1.0,
        help="After motion, retry detection for up to this many seconds.",
    )

    return parser.parse_args()


def _try_make_sample(
    *,
    sample_idx: int,
    out_dir: str,
    color_dir: str,
    depth_dir: str,
    save_depth: bool,
    frames: rs.composite_frame,
    color_frame: rs.video_frame,
    color: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    objp: np.ndarray,
    corners: np.ndarray,
    pattern_size_used: Tuple[int, int],
    args,
    robot_interface: FrankaInterface,
) -> Tuple[bool, str, str]:
    current_pose = robot_interface.last_eef_pose
    if current_pose is None or np.shape(current_pose) != (4, 4):
        return False, "", "Robot pose not ready"

    objp_used = _build_object_points(pattern_size_used[0], pattern_size_used[1], args.square)

    success, rvec, tvec = cv2.solvePnP(objp_used, corners, K, dist)
    if not success:
        return False, "", "solvePnP failed"

    R, t = _rt_from_rvec_tvec(rvec, tvec)
    T_cam_board = _mat44_from_rt(R, t)

    color_name = f"color_{sample_idx:04d}.png"
    color_path = os.path.join(color_dir, color_name)
    cv2.imwrite(color_path, color)

    depth_rel = ""
    if save_depth:
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth = np.asanyarray(depth_frame.get_data())
            depth_name = f"depth_{sample_idx:04d}.png"
            depth_path = os.path.join(depth_dir, depth_name)
            cv2.imwrite(depth_path, depth)
            depth_rel = os.path.relpath(depth_path, out_dir)

    sample_npz = os.path.join(out_dir, f"sample_{sample_idx:04d}.npz")
    np.savez(
        sample_npz,
        T_base_ee=np.array(current_pose, dtype=np.float64),
        T_cam_board=T_cam_board,
        K=K,
        dist=dist,
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        board_cols=pattern_size_used[0],
        board_rows=pattern_size_used[1],
        square_size_m=float(args.square),
        rs_timestamp_color=float(color_frame.get_timestamp()),
        rs_frame_number_color=int(color_frame.get_frame_number()),
        timestamp_unix=time.time(),
        color_path=os.path.relpath(color_path, out_dir),
        depth_path=depth_rel,
    )
    return True, sample_npz, ""


def main():
    args = parse_args()

    pattern_size = (args.cols, args.rows)

    out_dir = os.path.join(args.out, f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    color_dir = os.path.join(out_dir, "color")
    depth_dir = os.path.join(out_dir, "depth")
    _mkdir(color_dir)
    if args.save_depth:
        _mkdir(depth_dir)

    meta_path = os.path.join(out_dir, "metadata.json")
    csv_path = os.path.join(out_dir, "samples.csv")

    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
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

    if args.reset:
        reset_joints_to(robot_interface, reset_joint_positions)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.save_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print("Starting RealSense...")
    profile = pipeline.start(config)

    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr, K, dist = _intrinsics_from_realsense(color_stream)

        depth_scale = None
        if args.save_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())

        align = None
        if args.save_depth and args.align == "color":
            align = rs.align(rs.stream.color)

        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "board": {"inner_corners": {"cols": args.cols, "rows": args.rows}, "square_size_m": float(args.square)},
            "stream": {"width": args.width, "height": args.height, "fps": args.fps},
            "camera": {"model": str(intr.model), "coeffs": [float(x) for x in intr.coeffs[:5]], "K": K.tolist()},
            "depth_scale_m_per_unit": depth_scale,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            f.write("\n")

        if args.show:
            cv2.namedWindow("calib", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        rng = np.random.default_rng(args.seed)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_idx",
                    "timestamp_unix",
                    "rs_timestamp_color",
                    "rs_frame_number_color",
                    "color_path",
                    "depth_path",
                    "chessboard_found",
                    "pattern_size_used",
                    "note",
                ],
            )
            writer.writeheader()

            sample_idx = 0

            def get_frame_once():
                frames = pipeline.wait_for_frames()
                if align is not None:
                    frames = align.process(frames)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None
                color = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                ok, corners, used = _detect_corners(gray, pattern_size, try_swap=args.try_swap)
                vis = color
                if ok and corners is not None:
                    cv2.drawChessboardCorners(vis, used, corners, True)
                    if args.draw_axes:
                        objp_used = _build_object_points(used[0], used[1], args.square)
                        success, rvec, tvec = cv2.solvePnP(objp_used, corners, K, dist)
                        if success:
                            cv2.drawFrameAxes(vis, K, dist, rvec, tvec, args.axis_len)
                return frames, color_frame, color, ok, corners, used, vis

            def get_frame_with_retry(timeout_sec: float):
                deadline = time.time() + max(0.0, timeout_sec)
                last = None
                while True:
                    out = get_frame_once()
                    if out is None:
                        continue
                    last = out
                    _, _, _, ok, corners, used, _ = out
                    if ok and corners is not None:
                        return out
                    if time.time() >= deadline:
                        return last

            if not args.auto_move:
                print(f"Output dir: {out_dir}")
                print("Manual mode. Move robot yourself.")
                print("Press 's' to save a sample; 'q'/ESC to quit.")

                while True:
                    out = get_frame_once()
                    if out is None:
                        continue
                    frames, color_frame, color, ok, corners, used, vis = out

                    if args.show:
                        cv2.imshow("calib", vis)
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        key = 255

                    if key == ord("s"):
                        if not ok or corners is None:
                            print("Chessboard not found; try again.")
                            continue
                        ok_save, sample_npz, err = _try_make_sample(
                            sample_idx=sample_idx,
                            out_dir=out_dir,
                            color_dir=color_dir,
                            depth_dir=depth_dir,
                            save_depth=args.save_depth,
                            frames=frames,
                            color_frame=color_frame,
                            color=color,
                            K=K,
                            dist=dist,
                            objp=_build_object_points(used[0], used[1], args.square),
                            corners=corners,
                            pattern_size_used=used,
                            args=args,
                            robot_interface=robot_interface,
                        )
                        if ok_save:
                            print(f"Saved sample {sample_idx} -> {sample_npz} (pattern {used[0]}x{used[1]})")
                            writer.writerow(
                                {
                                    "sample_idx": sample_idx,
                                    "timestamp_unix": time.time(),
                                    "rs_timestamp_color": float(color_frame.get_timestamp()),
                                    "rs_frame_number_color": int(color_frame.get_frame_number()),
                                    "color_path": f"color/color_{sample_idx:04d}.png",
                                    "depth_path": f"depth/depth_{sample_idx:04d}.png" if args.save_depth else "",
                                    "chessboard_found": True,
                                    "pattern_size_used": f"{used[0]}x{used[1]}",
                                    "note": "",
                                }
                            )
                            f.flush()
                            sample_idx += 1
                        else:
                            print(f"Skip: {err}")

                    if key == ord("q") or key == 27:
                        break

                return

            print(f"Output dir: {out_dir}")
            print("AUTO-MOVE enabled: robot will move and collect samples.")
            print("Make sure the chessboard is fixed and visible to the camera.")
            print("Press 'q'/ESC in the window to stop early.")

            collected = 0
            attempt = 0
            while collected < args.num_samples:
                attempt += 1

                delta_pos = rng.uniform(-1.0, 1.0, size=(3,))
                norm = float(np.linalg.norm(delta_pos))
                if norm > 1e-9:
                    delta_pos = delta_pos / norm
                delta_pos = delta_pos * float(rng.uniform(0.2, 1.0) * args.max_translation)

                delta_axis = rng.uniform(-1.0, 1.0, size=(3,))
                norm = float(np.linalg.norm(delta_axis))
                if norm > 1e-9:
                    delta_axis = delta_axis / norm
                delta_axis_angle = delta_axis * float(rng.uniform(0.2, 1.0) * args.max_rotation)

                target_delta_pose = delta_pos.tolist() + delta_axis_angle.tolist()
                print(
                    f"[auto] attempt={attempt} collected={collected}/{args.num_samples} "
                    f"dpos={np.round(delta_pos,4).tolist()} daxis={np.round(delta_axis_angle,4).tolist()}"
                )

                move_to_target_pose(
                    robot_interface,
                    controller_type=args.controller_type,
                    controller_cfg=controller_cfg,
                    target_delta_pose=target_delta_pose,
                    num_steps=args.move_steps,
                    num_additional_steps=args.move_extra_steps,
                )

                time.sleep(max(0.0, float(args.settle_sec)))

                out = get_frame_with_retry(args.detect_timeout_sec)
                if out is None:
                    print("[auto] warning: no camera frame")
                    continue
                frames, color_frame, color, ok, corners, used, vis = out

                if args.show:
                    cv2.imshow("calib", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

                if not ok or corners is None:
                    print("[auto] chessboard not found, skipping save")
                    writer.writerow(
                        {
                            "sample_idx": sample_idx,
                            "timestamp_unix": time.time(),
                            "rs_timestamp_color": float(color_frame.get_timestamp()),
                            "rs_frame_number_color": int(color_frame.get_frame_number()),
                            "color_path": "",
                            "depth_path": "",
                            "chessboard_found": False,
                            "pattern_size_used": "",
                            "note": "chessboard not found",
                        }
                    )
                    f.flush()
                    continue

                ok_save, sample_npz, err = _try_make_sample(
                    sample_idx=sample_idx,
                    out_dir=out_dir,
                    color_dir=color_dir,
                    depth_dir=depth_dir,
                    save_depth=args.save_depth,
                    frames=frames,
                    color_frame=color_frame,
                    color=color,
                    K=K,
                    dist=dist,
                    objp=_build_object_points(used[0], used[1], args.square),
                    corners=corners,
                    pattern_size_used=used,
                    args=args,
                    robot_interface=robot_interface,
                )
                if not ok_save:
                    print(f"[auto] save failed: {err}")
                    writer.writerow(
                        {
                            "sample_idx": sample_idx,
                            "timestamp_unix": time.time(),
                            "rs_timestamp_color": float(color_frame.get_timestamp()),
                            "rs_frame_number_color": int(color_frame.get_frame_number()),
                            "color_path": "",
                            "depth_path": "",
                            "chessboard_found": True,
                            "pattern_size_used": f"{used[0]}x{used[1]}",
                            "note": err,
                        }
                    )
                    f.flush()
                    continue

                print(f"Saved sample {sample_idx} -> {sample_npz} (pattern {used[0]}x{used[1]})")
                writer.writerow(
                    {
                        "sample_idx": sample_idx,
                        "timestamp_unix": time.time(),
                        "rs_timestamp_color": float(color_frame.get_timestamp()),
                        "rs_frame_number_color": int(color_frame.get_frame_number()),
                        "color_path": f"color/color_{sample_idx:04d}.png",
                        "depth_path": f"depth/depth_{sample_idx:04d}.png" if args.save_depth else "",
                        "chessboard_found": True,
                        "pattern_size_used": f"{used[0]}x{used[1]}",
                        "note": "",
                    }
                )
                f.flush()
                sample_idx += 1
                collected += 1

    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()
        robot_interface.close()


if __name__ == "__main__":
    main()
