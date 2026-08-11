#!/usr/bin/env python3
"""Read and print current Franka pose and joint positions."""
import argparse
import time

import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    return parser.parse_args()


def main():
    args = parse_args()
    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )

    try:
        while robot_interface.state_buffer_size == 0:
            time.sleep(0.05)
        pose = robot_interface.last_eef_pose
        pos = pose[:3, 3].reshape(3)
        joint_positions = robot_interface.last_q
        np.set_printoptions(precision=6, suppress=True)
        print("T_base_ee:\n", pose)
        print("ee_position (x, y, z):", pos)
        print("joint_positions_rad (q1..q7):", joint_positions)
    finally:
        robot_interface.close()


if __name__ == "__main__":
    main()
