"""Example script of moving robot joint positions."""
import argparse
import os
import re
import time

import numpy as np

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--first-move-duration", type=float, default=5.0)
    args = parser.parse_args()
    return args


def compute_errors(pose_1, pose_2):

    pose_a = (
        pose_1[:3]
        + transform_utils.quat2axisangle(np.array(pose_1[3:]).flatten()).tolist()
    )
    pose_b = (
        pose_2[:3]
        + transform_utils.quat2axisangle(np.array(pose_2[3:]).flatten()).tolist()
    )
    return np.abs(np.array(pose_a) - np.array(pose_b))


def osc_move(
    robot_interface,
    controller_type,
    controller_cfg,
    target_pose,
    num_steps,
    gripper_action=-1.0,
):
    target_pos, target_quat = target_pose
    target_axis_angle = transform_utils.quat2axisangle(target_quat)
    current_rot, current_pos = robot_interface.last_eef_rot_and_pos
    action = None

    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        action_pos = (target_pos - current_pos).flatten() * 10
        action_axis_angle = axis_angle_diff.flatten() * 1
        action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_action]
        logger.info(f"Axis angle action {action_axis_angle.tolist()}")
        # print(np.round(action, 2))
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
    return action


def move_to_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_delta_pose,
    num_steps,
    num_additional_steps,
    interpolation_method,
    gripper_action=-1.0,
):
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

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

    logger.info(f"Before conversion {target_axis_angle}")
    target_quat = transform_utils.axisangle2quat(target_axis_angle)
    target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

    if np.dot(target_quat, current_quat) < 0.0:
        current_quat = -current_quat
    target_axis_angle = transform_utils.quat2axisangle(target_quat)
    logger.info(f"After conversion {target_axis_angle}")
    current_axis_angle = transform_utils.quat2axisangle(current_quat)

    start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

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


def load_hand_traj(npz_path, hand="right", stride=1):
    data = np.load(npz_path, allow_pickle=True)
    joints = data[f"{hand}_joints"]
    if stride > 1:
        joints = joints[::stride]
    return joints


def _transform_points(points, cam2base):
    flat = points.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
    hom = np.concatenate([flat, ones], axis=1)
    transformed = (cam2base @ hom.T).T[:, :3]
    return transformed.reshape(points.shape)


def _quat_from_two_vectors(src, dst):
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -0.999999:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return transform_utils.axisangle2quat(axis * np.pi)
    axis = np.cross(src, dst)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / axis_norm
    angle = np.arccos(dot)
    return transform_utils.axisangle2quat(axis * angle)


def convert_npz_cam_to_base(npz_path, cam2base):
    data = np.load(npz_path, allow_pickle=True)
    converted = {}
    for key in data.files:
        arr = data[key]
        if key.endswith("_joints") and arr.ndim >= 2 and arr.shape[-1] == 3:
            converted[key] = _transform_points(arr, cam2base)
        else:
            converted[key] = arr
    base_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "npz__trans_output",
    )
    parent_name = os.path.basename(os.path.dirname(npz_path))
    match = re.match(r"realsense_(\d{8})_(\d{6})$", parent_name)
    if match:
        timestamp = f"{match.group(1)}_{match.group(2)}"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hand_joints_21_base_mano.npz")
    np.savez(output_path, **converted)
    return output_path


def select_wilor_npz(
    base_dir=os.environ.get(
        "WILOR_OUTPUT_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "wilor",
        ),
    ),
    filename="hand_joints_21_cam_mano.npz",
):
    default_path = os.path.join(base_dir, "realsense_20251226_101703", filename)
    try:
        entries = [
            entry
            for entry in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, entry))
        ]
    except FileNotFoundError:
        return default_path

    candidates = []
    for entry in entries:
        match = re.match(r"realsense_(\d{8})_(\d{6})$", entry)
        if not match:
            continue
        value = match.group(1) + match.group(2)
        path = os.path.join(base_dir, entry, filename)
        if os.path.isfile(path):
            candidates.append((value, path))
    if not candidates:
        return default_path
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_wrist_traj(npz_path, hand="right", wrist_index=0, stride=1):
    joints = load_hand_traj(npz_path, hand=hand, stride=stride)
    return joints[:, wrist_index, :]


def compute_hand_rotations(
    hand_joints,
    hand,
    index_root_index,
    middle_index,
    wrist_index,
):
    rotations = []
    last_rot = np.eye(3)
    eps = 1e-6
    for joints in hand_joints:
        if not np.isfinite(
            joints[[wrist_index, middle_index, index_root_index]]
        ).all():
            rotations.append(last_rot)
            continue
        index_root = joints[index_root_index]
        middle_root = joints[middle_index]
        wrist = joints[wrist_index]

        x_axis = index_root - middle_root
        if hand == "left":
            y_axis = wrist - middle_root
        else:
            y_axis = middle_root - wrist
        x_norm = np.linalg.norm(x_axis)
        y_norm = np.linalg.norm(y_axis)
        if x_norm < eps or y_norm < eps:
            rotations.append(last_rot)
            continue

        x_axis = x_axis / x_norm
        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < eps:
            rotations.append(last_rot)
            continue

        z_axis = z_axis / z_norm
        y_axis = np.cross(z_axis, x_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)
        rotations.append(rot)
        last_rot = rot
    return rotations


def follow_wrist_traj(
    robot_interface,
    controller_type,
    controller_cfg,
    hand_joints,
    hand,
    wrist_index=0,
    index_root_index=5,
    middle_index=9,
    pinch_a_index=4,
    pinch_b_index=8,
    max_range=0.05,
    num_steps=20,
    num_additional_steps=10,
    interpolation_method="linear",
    first_move_duration=5.0,
):
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    base_pose = robot_interface.last_eef_pose
    base_rot = base_pose[:3, :3]
    pinch_a = hand_joints[:, pinch_a_index, :]
    pinch_b = hand_joints[:, pinch_b_index, :]
    index_root = hand_joints[:, index_root_index, :]
    pinch_dist = np.linalg.norm(pinch_a - pinch_b, axis=1)
    max_width = 0.08
    grip_actions = np.empty_like(pinch_dist)
    for i, w in enumerate(pinch_dist):
        if not np.isfinite(w):
            grip_actions[i] = -1.0
            continue
        grip_actions[i] = -np.clip(w / max_width, 0.0, 1.0)
    required_indices = [wrist_index, pinch_a_index, pinch_b_index, index_root_index]
    valid_mask = np.isfinite(hand_joints[:, required_indices, :]).all(axis=(1, 2))
    if not np.any(valid_mask):
        logger.warn("No valid frames for required indices; aborting.")
        return
    wrist_xyz = hand_joints[:, wrist_index, :]
    hand_centers = (pinch_a + pinch_b) * 0.5
    hand_dirs = pinch_b - pinch_a
    ee_to_center = np.array([0.0, 0.0, 0.105], dtype=np.float64)
    executed = 0
    first_move = True
    first_move_dt = 0.02
    for idx, (target_center, gripper_action, hand_dir) in enumerate(
        zip(hand_centers, grip_actions, hand_dirs)
    ):
        if not np.isfinite(hand_joints[idx, required_indices, :]).all():
            continue
        if not np.isfinite(target_center).all():
            continue
        if (
            not np.isfinite(hand_dir).all()
            or not np.isfinite(wrist_xyz[idx]).all()
            or not np.isfinite(index_root[idx]).all()
        ):
            continue

        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        hand_dir_norm = np.linalg.norm(hand_dir)
        if hand_dir_norm < 1e-9:
            continue
        hand_dir = hand_dir / hand_dir_norm
        desired_y = hand_dir if hand == "right" else -hand_dir
        z_dir = hand_centers[idx] - index_root[idx]
        z_norm = np.linalg.norm(z_dir)
        if z_norm < 1e-9:
            continue
        z_dir = z_dir / z_norm
        # Orthogonalize Z to Y, keep Y fixed.
        z_dir = z_dir - desired_y * np.dot(z_dir, desired_y)
        z_norm = np.linalg.norm(z_dir)
        if z_norm < 1e-9:
            continue
        z_dir = z_dir / z_norm
        x_dir = np.cross(desired_y, z_dir)
        x_norm = np.linalg.norm(x_dir)
        if x_norm < 1e-9:
            continue
        x_dir = x_dir / x_norm
        target_rot = np.stack([x_dir, desired_y, z_dir], axis=1)
        target_quat = transform_utils.mat2quat(target_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            target_quat = -target_quat

        target_pos = target_center.reshape(3) - target_rot @ ee_to_center

        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        target_delta_pos = (target_pos.reshape(3, 1) - current_pos).flatten()
        target_delta_axis_angle = (target_axis_angle - current_axis_angle).flatten()
        target_delta_pose = (
            target_delta_pos.tolist() + target_delta_axis_angle.tolist()
        )

        if first_move:
            start_pos = current_pos.copy()
            t0 = time.time()
            while True:
                elapsed = time.time() - t0
                alpha = min(elapsed / max(first_move_duration, 1e-6), 1.0)
                interp_pos = start_pos + alpha * (target_pos.reshape(3, 1) - start_pos)

                current_pose = robot_interface.last_eef_pose
                current_pos = current_pose[:3, 3:]
                current_rot = current_pose[:3, :3]
                current_quat = transform_utils.mat2quat(current_rot)
                if np.dot(target_quat, current_quat) < 0.0:
                    current_quat = -current_quat

                quat_diff = transform_utils.quat_distance(target_quat, current_quat)
                axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
                action_pos = (interp_pos - current_pos).flatten() * 10
                action_axis_angle = axis_angle_diff.flatten() * 1
                action = (
                    action_pos.tolist()
                    + action_axis_angle.tolist()
                    + [gripper_action]
                )
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                if alpha >= 1.0:
                    break
                time.sleep(first_move_dt)
            first_move = False
        else:
            move_to_target_pose(
                robot_interface,
                controller_type,
                controller_cfg,
                target_delta_pose=target_delta_pose,
                num_steps=num_steps,
                num_additional_steps=num_additional_steps,
                interpolation_method=interpolation_method,
                gripper_action=gripper_action,
            )
        executed += 1

    logger.info(f"Executed frames: {executed}/{len(hand_dirs)}")


def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_type = args.controller_type

    controller_cfg = get_default_controller_config(controller_type)

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

    npz_path = select_wilor_npz()
    cam2base = np.array(
        [
            [-0.62883498,  0.18228344, -0.75586988,  0.99698181],
            [ 0.7518942 ,  0.39016829, -0.53143559,  0.65923717],
            [ 0.19804456, -0.9025195 , -0.38240934,  0.44097683],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
        ],
        dtype=np.float64,
    )
    base_npz_path = convert_npz_cam_to_base(npz_path, cam2base)
    hand_joints = load_hand_traj(base_npz_path, hand=args.hand, stride=2)
    required = hand_joints[:, [0, 4, 8], :]
    valid_frames = int(np.sum(np.isfinite(required).all(axis=(1, 2))))
    nan_ratio = float(np.isnan(required).mean())
    logger.info(
        f"Using npz={base_npz_path}, hand={args.hand}, "
        f"valid_frames={valid_frames}/{len(hand_joints)}, nan_ratio={nan_ratio:.6f}"
    )
    follow_wrist_traj(
        robot_interface,
        controller_type,
        controller_cfg,
        hand_joints=hand_joints,
        hand=args.hand,
        wrist_index=0,
        index_root_index=5,
        middle_index=9,
        pinch_a_index=4,
        pinch_b_index=8,
        max_range=0.05,
        num_steps=10,
        num_additional_steps=0,
        interpolation_method="linear",
        first_move_duration=args.first_move_duration,
    )

    robot_interface.close()


if __name__ == "__main__":
    main()
