import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from deoxys.utils.io_devices import spacemouse as spacemouse_module
from examples.run_deoxys_with_space_mouse_V3_record import (
    RawEpisodeRecorder,
    SPACEMOUSE_PRODUCT_IDS,
    TASK_PART_NAMES,
    _build_camera_preview,
    _camera_sample_with_prompt_depth,
    _draw_front_part_poses,
    parse_args,
)


def camera_sample(part_z=1.0, pose_count=6):
    parts_poses = np.zeros((pose_count, 7), dtype=np.float32)
    parts_poses[:, 6] = 1.0
    parts_poses[0, 2] = part_z
    return {
        "color_image1": np.zeros((240, 320, 3), dtype=np.uint8),
        "color_image2": np.zeros((240, 320, 3), dtype=np.uint8),
        "depth_image1": np.full((240, 320), 0.5, dtype=np.float32),
        "depth_image2": np.full((240, 320), 1.0, dtype=np.float32),
        "parts_poses": parts_poses.reshape(-1),
        "parts_founds": np.arange(pose_count) == 0,
        "parts_pose_valid": np.arange(pose_count) == 0,
        "parts_pose_age_ms": np.zeros(pose_count, dtype=np.float32),
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


class SpaceMouseConnectionTest(unittest.TestCase):
    def parse(self, *arguments):
        with patch.dict(os.environ, {"DATA_DIR_RAW": "/tmp"}):
            with patch.object(sys, "argv", ["record", *arguments]):
                return parse_args()

    def test_wired_is_the_default(self):
        args = self.parse()

        self.assertEqual(args.spacemouse_connection, "wired")
        self.assertEqual(args.product_id, SPACEMOUSE_PRODUCT_IDS["wired"])

    def test_wireless_connection_selects_wireless_product_id(self):
        args = self.parse("--spacemouse-connection", "wireless")

        self.assertEqual(args.product_id, SPACEMOUSE_PRODUCT_IDS["wireless"])

    def test_explicit_product_id_overrides_connection(self):
        args = self.parse(
            "--spacemouse-connection",
            "wired",
            "--product-id",
            "12345",
        )

        self.assertEqual(args.product_id, 12345)

    def test_round_table_task_is_supported(self):
        args = self.parse("--task-name", "round_table")

        self.assertEqual(args.task_name, "round_table")

    def test_lamp_task_is_supported(self):
        args = self.parse("--task-name", "lamp")

        self.assertEqual(args.task_name, "lamp")

    def test_close_stops_listener_and_closes_hid_device(self):
        hid_device = MagicMock()
        hid_device.read.return_value = []

        with patch.object(spacemouse_module.hid, "device", return_value=hid_device):
            device = spacemouse_module.SpaceMouse(
                vendor_id=9583,
                product_id=SPACEMOUSE_PRODUCT_IDS["wireless"],
            )
            device.start_control()
            device.close()

        self.assertFalse(device.thread.is_alive())
        hid_device.set_nonblocking.assert_called_once_with(1)
        hid_device.close.assert_called_once_with()


class PartPosePreviewTest(unittest.TestCase):
    def test_round_table_requires_all_three_part_poses(self):
        recorder = RawEpisodeRecorder(
            data_root="/tmp/round-table-test",
            task_name="round_table",
            randomness="low",
            camera_info={},
            writer=MagicMock(),
        )
        sample = camera_sample(pose_count=3)

        self.assertFalse(recorder._parts_ready(sample))
        sample["parts_pose_valid"][:] = True
        self.assertTrue(recorder._parts_ready(sample))

    def test_draws_all_round_table_part_poses(self):
        sample = camera_sample(part_z=1.0, pose_count=3)
        sample["parts_poses"] = np.tile(
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            3,
        )
        sample["parts_pose_valid"][:] = True
        front = np.zeros((240, 320, 3), dtype=np.uint8)

        rendered = _draw_front_part_poses(
            front,
            sample,
            RECORD_INTRINSICS,
            part_names=TASK_PART_NAMES["round_table"],
        )

        self.assertTrue(np.any(rendered != 0))

    def test_lamp_requires_all_three_part_poses(self):
        recorder = RawEpisodeRecorder(
            data_root="/tmp/lamp-test",
            task_name="lamp",
            randomness="low",
            camera_info={},
            writer=MagicMock(),
        )
        sample = camera_sample(pose_count=3)

        self.assertFalse(recorder._parts_ready(sample))
        sample["parts_pose_valid"][:] = True
        self.assertTrue(recorder._parts_ready(sample))

    def test_prompt_depth_sample_saves_enhanced_and_original_depth(self):
        sample = camera_sample(part_z=1.0)
        sample["camera_capture_wall_time_ns"] = 123
        original_wrist = sample["depth_image1"].copy()
        original_front = sample["depth_image2"].copy()
        prompt_result = {
            "camera_sample": sample,
            "depths": {
                "depth_image1": np.full((240, 320), 0.6, dtype=np.float32),
                "depth_image2": np.full((240, 320), 1.1, dtype=np.float32),
            },
        }

        enhanced = _camera_sample_with_prompt_depth(
            prompt_result,
            ("wrist", "front"),
        )

        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced["depth_image1"].shape, (240, 320))
        self.assertEqual(enhanced["depth_image2"].shape, (240, 320))
        self.assertEqual(enhanced["depth_image1"].dtype, np.float16)
        self.assertEqual(enhanced["depth_image2"].dtype, np.float16)
        np.testing.assert_allclose(enhanced["depth_image1"], 0.6, atol=1e-3)
        np.testing.assert_allclose(enhanced["depth_image2"], 1.1, atol=1e-3)
        np.testing.assert_array_equal(
            enhanced["depth_image1_realsense"],
            original_wrist,
        )
        np.testing.assert_array_equal(
            enhanced["depth_image2_realsense"],
            original_front,
        )
        self.assertEqual(enhanced["prompt_depth_source_wall_time_ns"], 123)
        np.testing.assert_array_equal(sample["depth_image1"], original_wrist)
        np.testing.assert_array_equal(sample["depth_image2"], original_front)

    def test_prompt_depth_sample_waits_for_every_selected_camera(self):
        sample = camera_sample(part_z=1.0)
        prompt_result = {
            "camera_sample": sample,
            "depths": {
                "depth_image2": np.full((240, 320), 1.1, dtype=np.float32),
            },
        }

        enhanced = _camera_sample_with_prompt_depth(
            prompt_result,
            ("wrist", "front"),
        )

        self.assertIsNone(enhanced)

    def test_episode_payload_keeps_enhanced_depth_and_prompt_metadata(self):
        class CapturingWriter:
            def __init__(self):
                self.payload = None

            def submit(self, output_path, payload):
                self.payload = payload

        sample = camera_sample(part_z=1.0)
        sample["parts_pose_valid"][4] = True
        prompt_result = {
            "camera_sample": sample,
            "depths": {
                "depth_image1": np.full((240, 320), 0.6, dtype=np.float32),
                "depth_image2": np.full((240, 320), 1.1, dtype=np.float32),
            },
        }
        enhanced = _camera_sample_with_prompt_depth(
            prompt_result,
            ("wrist", "front"),
        )
        prompt_config = {
            "online": True,
            "model": "vitl",
            "max_size": 448,
        }
        writer = CapturingWriter()
        recorder = RawEpisodeRecorder(
            data_root="/tmp/promptda-test",
            task_name="one_leg",
            randomness="low",
            camera_info={},
            writer=writer,
            prompt_depth_config=prompt_config,
        )

        self.assertTrue(recorder.begin(enhanced))
        recorder.append(enhanced, np.zeros(8, dtype=np.float32))
        recorder.stop(enhanced)
        recorder.save(success=True)

        self.assertIsNotNone(writer.payload)
        saved = writer.payload["observations"][0]
        self.assertEqual(saved["depth_image1"].dtype, np.float16)
        self.assertEqual(saved["depth_image2"].dtype, np.float16)
        self.assertIn("depth_image1_realsense", saved)
        self.assertIn("depth_image2_realsense", saved)
        self.assertEqual(
            writer.payload["metadata"]["prompt_depth_anything"],
            prompt_config,
        )

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

    def test_prompt_depth_preview_shows_two_exact_camera_rows(self):
        sample = camera_sample(part_z=1.0)
        camera_info = {"front": {"record_intrinsics": RECORD_INTRINSICS}}
        prompt_result = {
            "camera_sample": sample,
            "depths": {
                "depth_image1": np.full((240, 320), 0.6, dtype=np.float32),
                "depth_image2": np.full((240, 320), 1.1, dtype=np.float32),
            },
            "stats": {
                "wrist": {"inference_ms": 12.0},
                "front": {"inference_ms": 13.0},
            },
        }

        preview = _build_camera_preview(
            sample,
            camera_info,
            episode_state="idle",
            draw_part_poses=True,
            prompt_depth_result=prompt_result,
        )

        self.assertEqual(preview.shape, (480, 960, 3))


if __name__ == "__main__":
    unittest.main()
