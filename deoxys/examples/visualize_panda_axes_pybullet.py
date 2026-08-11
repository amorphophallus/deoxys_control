#!/usr/bin/env python3
"""Visualize Panda with base/end-effector axes in PyBullet (no robot interface)."""
import time

import pybullet as p
import pybullet_data


def draw_axes(origin, rot, length=0.15, line_width=2):
    x_axis = [rot[0][0], rot[1][0], rot[2][0]]
    y_axis = [rot[0][1], rot[1][1], rot[2][1]]
    z_axis = [rot[0][2], rot[1][2], rot[2][2]]
    p.addUserDebugLine(origin, [origin[0] + length * x_axis[0], origin[1] + length * x_axis[1], origin[2] + length * x_axis[2]], [1, 0, 0], lineWidth=line_width)
    p.addUserDebugLine(origin, [origin[0] + length * y_axis[0], origin[1] + length * y_axis[1], origin[2] + length * y_axis[2]], [0, 1, 0], lineWidth=line_width)
    p.addUserDebugLine(origin, [origin[0] + length * z_axis[0], origin[1] + length * z_axis[1], origin[2] + length * z_axis[2]], [0, 0, 1], lineWidth=line_width)


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Load Panda URDF from deoxys assets.
    urdf_path = (
        "deoxys/franka_interface/robot_models/panda/panda.urdf"
    )
    robot = p.loadURDF(urdf_path, useFixedBase=True)

    # Reset to a fixed, readable pose.
    joint_positions = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7]
    for i, q in enumerate(joint_positions):
        p.resetJointState(robot, i, q)

    # Base frame axes (world frame).
    draw_axes([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], length=0.2)

    # End-effector frame axes at panda_hand.
    ee_link_index = None
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if name == "panda_hand":
            ee_link_index = i
            break
    if ee_link_index is not None:
        link_state = p.getLinkState(robot, ee_link_index, computeForwardKinematics=True)
        pos = link_state[4]
        orn = link_state[5]
        rot = p.getMatrixFromQuaternion(orn)
        rot = [rot[0:3], rot[3:6], rot[6:9]]
        draw_axes(pos, rot, length=0.15)
        # Highlight EE -Y direction with a big red arrow.
        neg_y = [-rot[0][1], -rot[1][1], -rot[2][1]]
        arrow_len = 0.25
        p.addUserDebugLine(
            pos,
            [pos[0] + arrow_len * neg_y[0], pos[1] + arrow_len * neg_y[1], pos[2] + arrow_len * neg_y[2]],
            [1, 0, 0],
            lineWidth=6,
        )

    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
