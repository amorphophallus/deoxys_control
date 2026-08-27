"""Minimal Panda forward kinematics for measured Cartesian velocity.

The Franka state message contains measured joint positions and velocities but
does not contain a measured Cartesian twist.  This module computes the
base-frame geometric Jacobian with NumPy only, so data collection can save the
actual ``J(q) @ dq`` twist without depending on commanded Cartesian velocity.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_PANDA_URDF = (
    Path(__file__).resolve().parents[1]
    / "franka_interface"
    / "robot_models"
    / "panda"
    / "panda.urdf"
)


def _rpy_matrix(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(xyz, rpy):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = _rpy_matrix(rpy)
    value[:3, 3] = xyz
    return value


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (
        skew @ skew
    )
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    return value


class PandaKinematics:
    """Kinematic chain used to compute a Panda base-frame zero Jacobian."""

    def __init__(
        self,
        urdf_path=DEFAULT_PANDA_URDF,
        base_link="panda_link0",
        reference_link="panda_hand",
    ):
        root = ET.parse(str(urdf_path)).getroot()
        joints_by_child = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            child = element.find("child").attrib["link"]
            joints_by_child[child] = {
                "name": element.attrib["name"],
                "type": element.attrib["type"],
                "parent": element.find("parent").attrib["link"],
                "child": child,
                "xyz": np.fromstring(
                    "0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0"),
                    sep=" ",
                ),
                "rpy": np.fromstring(
                    "0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0"),
                    sep=" ",
                ),
                "axis": np.fromstring(
                    "0 0 1" if axis is None else axis.attrib.get("xyz", "0 0 1"),
                    sep=" ",
                ),
            }

        chain = []
        current = reference_link
        while current != base_link:
            if current not in joints_by_child:
                raise ValueError(
                    f"no URDF chain from {base_link!r} to {reference_link!r}"
                )
            joint = joints_by_child[current]
            chain.append(joint)
            current = joint["parent"]
        self.chain = list(reversed(chain))
        self.movable = [joint for joint in self.chain if joint["type"] != "fixed"]
        if len(self.movable) != 7:
            raise ValueError(f"expected 7 Panda joints, found {len(self.movable)}")

    def joint_geometry(self, joint_positions):
        """Return reference pose plus joint origins and axes in robot base."""
        joint_positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if joint_positions.size != 7 or not np.all(np.isfinite(joint_positions)):
            raise ValueError("joint_positions must contain 7 finite values")

        current = np.eye(4, dtype=np.float64)
        joint_origins = []
        joint_axes = []
        position_index = 0
        for joint in self.chain:
            current = current @ _origin_transform(joint["xyz"], joint["rpy"])
            if joint["type"] == "fixed":
                continue
            if joint["type"] not in {"revolute", "continuous"}:
                raise ValueError(f"unsupported Panda joint type: {joint['type']}")
            joint_origins.append(current[:3, 3].copy())
            joint_axes.append(current[:3, :3] @ joint["axis"])
            current = current @ _axis_rotation(
                joint["axis"], joint_positions[position_index]
            )
            position_index += 1
        return current, np.asarray(joint_origins), np.asarray(joint_axes)

    def zero_jacobian(self, joint_positions, ee_position=None):
        """Return a 6x7 geometric Jacobian expressed in robot base.

        ``ee_position`` selects the physical point for linear velocity.  Passing
        the measured ``O_T_EE`` translation makes this respect the active Franka
        flange-to-EE configuration without hard-coding a tool offset.
        """
        reference_pose, joint_origins, joint_axes = self.joint_geometry(
            joint_positions
        )
        if ee_position is None:
            ee_position = reference_pose[:3, 3]
        ee_position = np.asarray(ee_position, dtype=np.float64).reshape(-1)
        if ee_position.size != 3 or not np.all(np.isfinite(ee_position)):
            raise ValueError("ee_position must contain 3 finite values")

        jacobian = np.empty((6, 7), dtype=np.float64)
        for index, (origin, axis) in enumerate(zip(joint_origins, joint_axes)):
            jacobian[:3, index] = np.cross(axis, ee_position - origin)
            jacobian[3:, index] = axis
        return jacobian

    def ee_twist(self, joint_positions, joint_velocities, ee_position):
        """Return measured ``[vx, vy, vz, wx, wy, wz]`` in robot base."""
        joint_velocities = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)
        if joint_velocities.size != 7 or not np.all(np.isfinite(joint_velocities)):
            raise ValueError("joint_velocities must contain 7 finite values")
        return self.zero_jacobian(joint_positions, ee_position) @ joint_velocities
