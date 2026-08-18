import unittest

import numpy as np

from examples.run_deoxys_with_space_mouse_V3_record import (
    _build_camera_preview,
    _draw_front_part_poses,
)


def camera_sample(part_z=1.0):
    parts_poses = np.zeros((6, 7), dtype=np.float32)
    parts_poses[:, 6] = 1.0
    parts_poses[0, 2] = part_z
    return {
        "color_image1": np.zeros((240, 320, 3), dtype=np.uint8),
        "color_image2": np.zeros((240, 320, 3), dtype=np.uint8),
        "parts_poses": parts_poses.reshape(-1),
        "parts_founds": np.array([True, False, False, False, False, False]),
        "parts_pose_valid": np.array([True, False, False, False, False, False]),
        "parts_pose_age_ms": np.zeros(6, dtype=np.float32),
        "camera_pose_samples": 10,
        "camera_pose_samples_required": 10,
        "camera_to_april": np.eye(4, dtype=np.float64),
    }


RECORD_INTRINSICS = {
    "fx": 200.0,
    "fy": 200.0,
    "ppx": 160.0,
    "ppy": 120.0,
    "width": 320,
    "height": 240,
}


class PartPosePreviewTest(unittest.TestCase):
    def test_draws_visible_part_pose_without_mutating_input(self):
        sample = camera_sample(part_z=1.0)
        front = np.zeros((240, 320, 3), dtype=np.uint8)
        original = front.copy()

        rendered = _draw_front_part_poses(
            front,
            sample,
            RECORD_INTRINSICS,
        )

        np.testing.assert_array_equal(front, original)
        self.assertTrue(np.any(rendered != 0))
        self.assertTrue(np.any(rendered[110:131, 150:171] != 0))

    def test_skips_part_pose_behind_camera(self):
        sample = camera_sample(part_z=-1.0)
        front = np.zeros((240, 320, 3), dtype=np.uint8)

        rendered = _draw_front_part_poses(
            front,
            sample,
            RECORD_INTRINSICS,
        )

        np.testing.assert_array_equal(rendered, front)

    def test_combined_preview_uses_copies_and_expected_size(self):
        sample = camera_sample(part_z=1.0)
        wrist_before = sample["color_image1"].copy()
        front_before = sample["color_image2"].copy()
        camera_info = {"front": {"record_intrinsics": RECORD_INTRINSICS}}

        preview = _build_camera_preview(
            sample,
            camera_info,
            episode_state="recording",
            draw_part_poses=True,
        )

        self.assertEqual(preview.shape, (480, 1280, 3))
        np.testing.assert_array_equal(sample["color_image1"], wrist_before)
        np.testing.assert_array_equal(sample["color_image2"], front_before)


if __name__ == "__main__":
    unittest.main()
