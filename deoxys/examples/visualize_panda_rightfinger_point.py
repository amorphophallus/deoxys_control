#!/usr/bin/env python3
"""Visualize Panda and highlight the right finger link origin in PyBullet."""
import time

import pybullet as p
import pybullet_data


def draw_axes(origin, rot, length=0.12, line_width=2):
    x_axis = [rot[0][0], rot[1][0], rot[2][0]]
    y_axis = [rot[0][1], rot[1][1], rot[2][1]]
    z_axis = [rot[0][2], rot[1][2], rot[2][2]]
    p.addUserDebugLine(
        origin,
        [origin[0] + length * x_axis[0], origin[1] + length * x_axis[1], origin[2] + length * x_axis[2]],
        [1, 0, 0],
        lineWidth=line_width,
    )
    p.addUserDebugLine(
        origin,
        [origin[0] + length * y_axis[0], origin[1] + length * y_axis[1], origin[2] + length * y_axis[2]],
        [0, 1, 0],
        lineWidth=line_width,
    )
    p.addUserDebugLine(
        origin,
        [origin[0] + length * z_axis[0], origin[1] + length * z_axis[1], origin[2] + length * z_axis[2]],
        [0, 0, 1],
        lineWidth=line_width,
    )


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    urdf_path = "deoxys/franka_interface/robot_models/panda/panda.urdf"
    robot = p.loadURDF(urdf_path, useFixedBase=True)

    joint_positions = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7]
    for i, q in enumerate(joint_positions):
        p.resetJointState(robot, i, q)

    # Base axes.
    draw_axes([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], length=0.2)

    # Find right finger link index.
    right_finger_idx = None
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if name == "panda_rightfinger":
            right_finger_idx = i
            break
    if right_finger_idx is None:
        raise RuntimeError("panda_rightfinger link not found in URDF.")

    # Create a small red sphere marker.
    marker_vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.01,
        rgbaColor=[1, 0, 0, 1],
    )
    marker_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=marker_vis,
        basePosition=[0, 0, 0],
    )

    while True:
        link_state = p.getLinkState(robot, right_finger_idx, computeForwardKinematics=True)
        pos = link_state[4]
        orn = link_state[5]
        p.resetBasePositionAndOrientation(marker_id, pos, [0, 0, 0, 1])
        rot = p.getMatrixFromQuaternion(orn)
        rot = [rot[0:3], rot[3:6], rot[6:9]]
        draw_axes(pos, rot, length=0.08, line_width=2)

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
