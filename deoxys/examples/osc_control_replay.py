"""Example script of moving robot joint positions."""
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
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
        action_pos = np.clip(action_pos, -1.0, 1.0)
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

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


def load_wrist_traj(npz_path, hand="right", wrist_index=0, stride=1):
    joints = load_hand_traj(npz_path, hand=hand, stride=stride)
    return joints[:, wrist_index, :]


def compute_hand_rotations(
    hand_joints, wrist_index=0, index_index=1, pinky_index=7
):
    rotations = []
    last_rot = np.eye(3)
    eps = 1e-6
    for joints in hand_joints:
        wrist = joints[wrist_index]
        index_mcp = joints[index_index]
        pinky_mcp = joints[pinky_index]

        x_axis = index_mcp - wrist
        y_axis = pinky_mcp - wrist
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
    wrist_index=0,
    index_index=1,
    pinky_index=7,
    pinch_a_index=16,
    pinch_b_index=17,
    max_range=0.05,
    num_steps=2,
    num_additional_steps=0,
    interpolation_method="linear",
):
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    base_pose = robot_interface.last_eef_pose
    base_pos = base_pose[:3, 3:].reshape(3, 1)
    base_rot = base_pose[:3, :3]

    wrist_xyz = hand_joints[:, wrist_index, :]
    pinch_a = hand_joints[:, pinch_a_index, :]
    pinch_b = hand_joints[:, pinch_b_index, :]
    pinch_dist = np.linalg.norm(pinch_a - pinch_b, axis=1)
    pinch_min = float(np.min(pinch_dist))
    pinch_max = float(np.max(pinch_dist))
    if pinch_max - pinch_min < 1e-6:
        grip_actions = np.full_like(pinch_dist, -1.0)
    else:
        pinch_norm = (pinch_dist - pinch_min) / (pinch_max - pinch_min)
        grip_actions = np.clip(1.0 - 2.0 * pinch_norm, -1.0, 1.0)
    hand_rots = compute_hand_rotations(
        hand_joints,
        wrist_index=wrist_index,
        index_index=index_index,
        pinky_index=pinky_index,
    )
    ref_hand_rot = hand_rots[0]

    offsets = wrist_xyz - wrist_xyz[0]
    max_norm = np.max(np.linalg.norm(offsets, axis=1))
    if max_norm > 1e-9:
        offsets = offsets * (max_range / max_norm)

    targets = base_pos.reshape(1, 3) + offsets
    for target_pos, hand_rot, gripper_action in zip(
        targets, hand_rots, grip_actions
    ):
        rel_rot = ref_hand_rot.T @ hand_rot
        target_rot = base_rot @ rel_rot
        target_quat = transform_utils.mat2quat(target_rot)

        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_quat = transform_utils.mat2quat(current_pose[:3, :3])
        if np.dot(target_quat, current_quat) < 0.0:
            target_quat = -target_quat

        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        target_delta_pos = (target_pos.reshape(3, 1) - current_pos).flatten()
        target_delta_axis_angle = (target_axis_angle - current_axis_angle).flatten()
        axis_angle_norm = np.linalg.norm(target_delta_axis_angle)
        max_axis_angle = 0.2
        if axis_angle_norm > max_axis_angle:
            target_delta_axis_angle = (
                target_delta_axis_angle * (max_axis_angle / axis_angle_norm)
            )
        target_delta_pose = (
            target_delta_pos.tolist() + target_delta_axis_angle.tolist()
        )

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

    hand_joints = load_hand_traj(
        "hand_joints_21_latest.npz",
        hand="right",
        stride=2,
    )
    follow_wrist_traj(
        robot_interface,
        controller_type,
        controller_cfg,
        hand_joints=hand_joints,
        wrist_index=0,
        index_index=1,
        pinky_index=7,
        pinch_a_index=16,
        pinch_b_index=17,
        max_range=0.05,
        num_steps=10,
        num_additional_steps=5,
        interpolation_method="linear",
    )

    robot_interface.close()


if __name__ == "__main__":
    main()
