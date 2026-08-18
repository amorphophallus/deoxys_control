import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from deoxys.utils import transform_utils
from deoxys.utils.furniture_bench_utils import (
    WRIST_TO_TIP,
    center_crop_resize,
    center_crop_resize_geometry,
    deoxys_delta_to_furniture_bench_action,
    transformed_intrinsics,
)


class FurnitureBenchActionTest(unittest.TestCase):
    def test_zero_delta_is_identity(self):
        action = deoxys_delta_to_furniture_bench_action(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            np.eye(4),
        )
        np.testing.assert_allclose(action[:3], np.zeros(3), atol=1e-7)
        np.testing.assert_allclose(
            action[3:7],
            np.array([0.0, 0.0, 0.0, 1.0]),
            atol=1e-7,
        )
        self.assertEqual(action[-1], -1.0)

    def test_world_left_wrist_delta_becomes_local_tip_delta(self):
        wrist_pose = np.eye(4)
        wrist_pose[:3, :3] = transform_utils.quat2mat(
            transform_utils.axisangle2quat(np.array([0.2, -0.1, 0.3]))
        )
        scaled_action = np.array([0.01, -0.02, 0.005, 0.04, 0.02, -0.03, 1.0])
        converted = deoxys_delta_to_furniture_bench_action(
            scaled_action,
            wrist_pose,
        )

        world_delta = transform_utils.quat2mat(
            transform_utils.axisangle2quat(scaled_action[3:6])
        )
        goal_wrist = wrist_pose.copy()
        goal_wrist[:3, 3] += scaled_action[:3]
        goal_wrist[:3, :3] = world_delta @ wrist_pose[:3, :3]
        current_tip = wrist_pose @ WRIST_TO_TIP
        goal_tip = goal_wrist @ WRIST_TO_TIP
        expected_rotation = current_tip[:3, :3].T @ goal_tip[:3, :3]

        np.testing.assert_allclose(
            converted[:3],
            goal_tip[:3, 3] - current_tip[:3, 3],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            transform_utils.quat2mat(converted[3:7]),
            expected_rotation,
            atol=1e-6,
        )
        self.assertEqual(converted[-1], 1.0)


class FurnitureBenchImageTransformTest(unittest.TestCase):
    def test_front_16_by_9_is_center_cropped_without_stretch(self):
        geometry = center_crop_resize_geometry(1280, 720, 320, 240)
        self.assertEqual(geometry["crop_x"], 160)
        self.assertEqual(geometry["crop_y"], 0)
        self.assertEqual(geometry["crop_width"], 960)
        self.assertEqual(geometry["crop_height"], 720)
        self.assertAlmostEqual(geometry["scale_x"], 1.0 / 3.0)
        self.assertAlmostEqual(geometry["scale_y"], 1.0 / 3.0)

        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        resized = center_crop_resize(image, geometry, cv2.INTER_AREA)
        self.assertEqual(resized.shape, (240, 320, 3))

    def test_wrist_4_by_3_only_scales(self):
        geometry = center_crop_resize_geometry(640, 480, 320, 240)
        self.assertEqual(geometry["crop_x"], 0)
        self.assertEqual(geometry["crop_y"], 0)
        self.assertEqual(geometry["crop_width"], 640)
        self.assertEqual(geometry["crop_height"], 480)
        self.assertEqual(geometry["scale_x"], 0.5)
        self.assertEqual(geometry["scale_y"], 0.5)

    def test_record_intrinsics_follow_crop_and_resize(self):
        intrinsics = SimpleNamespace(
            fx=900.0,
            fy=900.0,
            ppx=640.0,
            ppy=360.0,
        )
        geometry = center_crop_resize_geometry(1280, 720, 320, 240)
        record_intrinsics = transformed_intrinsics(intrinsics, geometry)
        self.assertAlmostEqual(record_intrinsics["fx"], 300.0)
        self.assertAlmostEqual(record_intrinsics["fy"], 300.0)
        self.assertAlmostEqual(record_intrinsics["ppx"], 160.0)
        self.assertAlmostEqual(record_intrinsics["ppy"], 120.0)
        self.assertEqual(record_intrinsics["width"], 320)
        self.assertEqual(record_intrinsics["height"], 240)


if __name__ == "__main__":
    unittest.main()
