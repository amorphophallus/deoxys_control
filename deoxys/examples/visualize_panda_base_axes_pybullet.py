#!/usr/bin/env python3
"""Visualize Panda base frame axes and origin in PyBullet."""
import time

import pybullet as p
import pybullet_data


def draw_axes(origin, rot, length=0.2, line_width=3):
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

    # Base frame axes at link0/origin.
    draw_axes([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], length=0.25, line_width=4)

    # Mark origin with a small sphere.
    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=marker, basePosition=[0, 0, 0])

    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
