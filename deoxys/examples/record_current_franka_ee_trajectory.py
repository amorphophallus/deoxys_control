#!/usr/bin/env python3
"""Record the current Franka end-effector trajectory for later replay."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record current Franka EE poses while the robot is moved manually. "
            "The JSON output is compatible with osc_control_replay_robot_eval_ee_pose.py."
        )
    )
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/manual_franka_ee_recordings",
        help="Directory where a timestamped recording folder will be created.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Exact output directory. If omitted, a timestamped folder under --output-root is used.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Exact output directory or JSON file path. If this ends with .json, "
            "that file is used as the replay JSON path."
        ),
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="Target recording frequency in Hz.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional maximum recording duration in seconds.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to record.",
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.04,
        help="Width in meters used to label gripper_state as open/closed for replay JSON.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print one progress line every N recorded frames. Use 0 to disable.",
    )
    return parser.parse_args()


def make_output_dir(args):
    if args.output_path is not None:
        output_path = Path(args.output_path).expanduser().resolve()
        output_dir = output_path.parent if output_path.suffix == ".json" else output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root).expanduser().resolve() / f"manual_ee_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_paths(output_dir, args):
    if args.output_path is not None:
        output_path = Path(args.output_path).expanduser().resolve()
        if output_path.suffix == ".json":
            json_path = output_path
            npz_path = output_path.with_suffix(".npz")
            joint_json_path = output_path.with_name(f"{output_path.stem}_joint_trajectory.json")
            joint_npz_path = output_path.with_name(f"{output_path.stem}_joint_trajectory.npz")
            metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
            return json_path, npz_path, joint_json_path, joint_npz_path, metadata_path

    return (
        output_dir / "robot_eval.json",
        output_dir / "franka_ee_trajectory.npz",
        output_dir / "joint_trajectory.json",
        output_dir / "franka_joint_trajectory.npz",
        output_dir / "metadata.json",
    )


def load_deoxys_modules():
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import transform_utils

    return config_root, FrankaInterface, transform_utils


def resolve_interface_cfg(interface_cfg, config_root):
    cfg_path = Path(interface_cfg).expanduser()
    if cfg_path.is_absolute():
        return str(cfg_path)
    return str(Path(config_root) / interface_cfg)


def wait_for_robot_state(robot_interface):
    while robot_interface.state_buffer_size == 0:
        print("Waiting for robot state...")
        time.sleep(0.2)


def gripper_width_or_none(robot_interface):
    width = robot_interface.last_gripper_q
    if width is None:
        return None
    return float(np.asarray(width).reshape(-1)[0])


def build_frame(
    frame_index,
    timestamp_sec,
    wall_time_ns,
    pose,
    q,
    dq,
    gripper_width,
    args,
    transform_utils,
):
    position = pose[:3, 3].astype(np.float64)
    rotation = pose[:3, :3].astype(np.float64)
    quat = transform_utils.mat2quat(rotation).astype(np.float64)
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm > 1e-8:
        quat = quat / quat_norm

    frame = {
        "frame_index": int(frame_index),
        "timestamp_sec": float(timestamp_sec),
        "wall_time_ns": int(wall_time_ns),
        "position_abs_m": position.tolist(),
        "orientation_quat_xyzw": quat.tolist(),
        "rotation_matrix": rotation.tolist(),
        "T_base_ee": pose.astype(np.float64).tolist(),
        "joint_positions": None
        if q is None
        else np.asarray(q, dtype=np.float64).tolist(),
        "joint_velocities": None
        if dq is None
        else np.asarray(dq, dtype=np.float64).tolist(),
        "gripper_width_m": gripper_width,
    }
    if gripper_width is not None:
        frame["gripper_state"] = (
            "open" if gripper_width >= args.gripper_open_threshold else "closed"
        )
    return frame


def save_recording(output_dir, frames, args, started_at, stopped_at):
    json_path, npz_path, joint_json_path, joint_npz_path, metadata_path = get_output_paths(output_dir, args)
    for path in (json_path, npz_path, joint_json_path, joint_npz_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema": "manual_franka_ee_recording_v1",
        "coordinate_frame": "franka_base",
        "trajectory_type": "ee_pose",
        "contains_joint_trajectory": True,
        "replay_script": "examples/osc_control_replay_robot_eval_ee_pose.py",
        "joint_replay_command": (
            "python examples/osc_control_replay_robot_eval_ee_pose.py "
            f"--replay-mode joint --traj-npz {joint_npz_path}"
        ),
        "interface_cfg": args.interface_cfg,
        "record_fps": args.record_fps,
        "duration_arg": args.duration,
        "max_frames_arg": args.max_frames,
        "gripper_open_threshold": args.gripper_open_threshold,
        "started_at": started_at,
        "stopped_at": stopped_at,
        "num_frames": len(frames),
        "ee_json_path": str(json_path),
        "ee_npz_path": str(npz_path),
        "joint_json_path": str(joint_json_path),
        "joint_npz_path": str(joint_npz_path),
    }
    payload = {
        **metadata,
        "frames": frames,
    }
    joint_payload = {
        **metadata,
        "schema": "manual_franka_joint_recording_v1",
        "coordinate_frame": "franka_joint_space",
        "trajectory_type": "joint_positions",
        "source_ee_json_path": str(json_path),
        "source_ee_npz_path": str(npz_path),
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "timestamp_sec": frame["timestamp_sec"],
                "wall_time_ns": frame["wall_time_ns"],
                "joint_positions": frame["joint_positions"],
                "joint_velocities": frame["joint_velocities"],
                "gripper_width_m": frame["gripper_width_m"],
                **({"gripper_state": frame["gripper_state"]} if "gripper_state" in frame else {}),
            }
            for frame in frames
        ],
    }

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    joint_json_path.write_text(
        json.dumps(joint_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    poses = np.array([frame["T_base_ee"] for frame in frames], dtype=np.float64)
    positions = np.array([frame["position_abs_m"] for frame in frames], dtype=np.float64)
    rotations = np.array([frame["rotation_matrix"] for frame in frames], dtype=np.float64)
    quats = np.array(
        [frame["orientation_quat_xyzw"] for frame in frames],
        dtype=np.float64,
    )
    timestamps = np.array([frame["timestamp_sec"] for frame in frames], dtype=np.float64)
    wall_times = np.array([frame["wall_time_ns"] for frame in frames], dtype=np.int64)

    joints = np.array(
        [
            np.full(7, np.nan, dtype=np.float64)
            if frame["joint_positions"] is None
            else np.asarray(frame["joint_positions"], dtype=np.float64)
            for frame in frames
        ],
        dtype=np.float64,
    )
    dqs = np.array(
        [
            np.full(7, np.nan, dtype=np.float64)
            if frame["joint_velocities"] is None
            else np.asarray(frame["joint_velocities"], dtype=np.float64)
            for frame in frames
        ],
        dtype=np.float64,
    )
    gripper_widths = np.array(
        [
            np.nan if frame["gripper_width_m"] is None else frame["gripper_width_m"]
            for frame in frames
        ],
        dtype=np.float64,
    )

    np.savez(
        npz_path,
        timestamps_sec=timestamps,
        wall_time_ns=wall_times,
        T_base_ee=poses,
        positions_abs_m=positions,
        rotations=rotations,
        quaternions_xyzw=quats,
        joint_positions=joints,
        joint_velocities=dqs,
        gripper_widths_m=gripper_widths,
    )
    np.savez(
        joint_npz_path,
        timestamps_sec=timestamps,
        wall_time_ns=wall_times,
        joint_positions=joints,
        joint_velocities=dqs,
        gripper_widths_m=gripper_widths,
    )

    return json_path, npz_path, joint_json_path, joint_npz_path, metadata_path


def main():
    args = parse_args()
    if args.record_fps <= 0:
        raise ValueError("--record-fps must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be positive when provided")

    output_dir = make_output_dir(args)
    config_root, FrankaInterface, transform_utils = load_deoxys_modules()
    robot_interface = FrankaInterface(
        resolve_interface_cfg(args.interface_cfg, config_root),
        use_visualizer=False,
    )

    frames = []
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.monotonic()
    next_sample_time = start_time
    sample_interval = 1.0 / args.record_fps

    try:
        wait_for_robot_state(robot_interface)
        print(f"Recording to {output_dir}")
        print("Press Ctrl+C to stop and save.")

        while True:
            now = time.monotonic()
            if now < next_sample_time:
                time.sleep(min(next_sample_time - now, sample_interval))
                continue

            timestamp_sec = now - start_time
            pose = robot_interface.last_eef_pose
            if pose is None or not np.all(np.isfinite(pose)):
                next_sample_time += sample_interval
                continue

            frame = build_frame(
                frame_index=len(frames),
                timestamp_sec=timestamp_sec,
                wall_time_ns=time.time_ns(),
                pose=pose,
                q=robot_interface.last_q,
                dq=robot_interface.last_dq,
                gripper_width=gripper_width_or_none(robot_interface),
                args=args,
                transform_utils=transform_utils,
            )
            frames.append(frame)

            if args.print_every > 0 and len(frames) % args.print_every == 0:
                pos = frame["position_abs_m"]
                print(
                    "frames={:d} t={:.3f}s pos=[{:.4f}, {:.4f}, {:.4f}]".format(
                        len(frames),
                        timestamp_sec,
                        pos[0],
                        pos[1],
                        pos[2],
                    )
                )

            if args.duration is not None and timestamp_sec >= args.duration:
                break
            if args.max_frames is not None and len(frames) >= args.max_frames:
                break

            next_sample_time += sample_interval
            if next_sample_time < now - sample_interval:
                next_sample_time = now + sample_interval

    except KeyboardInterrupt:
        print("\nStop requested. Saving recording...")
    finally:
        stopped_at = datetime.now().isoformat(timespec="seconds")
        robot_interface.close()

    if not frames:
        raise RuntimeError("No valid frames recorded; nothing was saved.")

    json_path, npz_path, joint_json_path, joint_npz_path, metadata_path = save_recording(
        output_dir=output_dir,
        frames=frames,
        args=args,
        started_at=started_at,
        stopped_at=stopped_at,
    )
    print(f"Saved {len(frames)} frames")
    print(f"Replay JSON: {json_path}")
    print(f"Raw NPZ: {npz_path}")
    print(f"Joint replay JSON: {joint_json_path}")
    print(f"Joint replay NPZ: {joint_npz_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
