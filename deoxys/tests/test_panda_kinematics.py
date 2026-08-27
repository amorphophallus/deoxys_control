import unittest

import numpy as np

from deoxys.utils.panda_kinematics import PandaKinematics


class PandaKinematicsTest(unittest.TestCase):
    def test_geometric_jacobian_matches_forward_pose_difference(self):
        model = PandaKinematics()
        q = np.array([0.1, -0.3, 0.2, -1.9, 0.4, 1.7, 0.6])
        dq = np.array([0.2, -0.1, 0.15, 0.05, -0.2, 0.1, 0.3])
        tool_offset = np.array([0.01, -0.02, 0.1034])

        reference, _, _ = model.joint_geometry(q)
        ee_position = reference[:3, 3] + reference[:3, :3] @ tool_offset
        predicted = model.ee_twist(q, dq, ee_position)

        epsilon = 1e-7
        advanced, _, _ = model.joint_geometry(q + epsilon * dq)
        advanced_position = (
            advanced[:3, 3] + advanced[:3, :3] @ tool_offset
        )
        linear_difference = (advanced_position - ee_position) / epsilon
        rotation_derivative = (
            advanced[:3, :3] - reference[:3, :3]
        ) / epsilon
        angular_matrix = rotation_derivative @ reference[:3, :3].T
        angular_difference = np.array(
            [angular_matrix[2, 1], angular_matrix[0, 2], angular_matrix[1, 0]]
        )

        np.testing.assert_allclose(predicted[:3], linear_difference, atol=1e-7)
        np.testing.assert_allclose(predicted[3:], angular_difference, atol=1e-7)

    def test_zero_joint_velocity_produces_zero_twist(self):
        model = PandaKinematics()
        q = np.zeros(7)
        reference, _, _ = model.joint_geometry(q)
        twist = model.ee_twist(q, np.zeros(7), reference[:3, 3])
        np.testing.assert_array_equal(twist, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
