#!/usr/bin/env python3
"""Visualize panda_hand and panda_grasptarget link origins with an arrow."""
import time

import pybullet as p
import pybullet_data


def _find_link_index(robot, link_name):
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if name == link_name:
            return i
    return None


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    urdf_path = "deoxys/franka_interface/robot_models/panda/panda.urdf"
    robot = p.loadURDF(urdf_path, useFixedBase=True)

    # Set a readable pose.
    joint_positions = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7]
    for i, q in enumerate(joint_positions):
        p.resetJointState(robot, i, q)

    hand_idx = _find_link_index(robot, "panda_hand")
    grasp_idx = _find_link_index(robot, "panda_grasptarget")
    if hand_idx is None or grasp_idx is None:
        raise RuntimeError("panda_hand or panda_grasptarget link not found.")

    # Big black spheres.
    sphere_vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 0, 0, 1]
    )
    hand_marker = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=sphere_vis)
    grasp_marker = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=sphere_vis)

    while True:
        hand_state = p.getLinkState(robot, hand_idx, computeForwardKinematics=True)
        grasp_state = p.getLinkState(robot, grasp_idx, computeForwardKinematics=True)
        hand_pos = hand_state[4]
        grasp_pos = grasp_state[4]

        p.resetBasePositionAndOrientation(hand_marker, hand_pos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(grasp_marker, grasp_pos, [0, 0, 0, 1])

        # Draw black arrow (line) from panda_hand to panda_grasptarget.
        p.addUserDebugLine(
            hand_pos,
            grasp_pos,
            [0, 0, 0],
            lineWidth=6,
            lifeTime=0.1,
        )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
