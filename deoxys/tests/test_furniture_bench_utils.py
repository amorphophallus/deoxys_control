import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np

from deoxys.utils import transform_utils
from deoxys.utils.furniture_bench_utils import (
    DualRealSenseSnapshotter,
    FurniturePoseTracker,
    WRIST_TO_TIP,
    center_crop_resize,
    center_crop_resize_geometry,
    deoxys_delta_to_furniture_bench_action,
    eepose_from_wrist_pose,
    resolve_eepose_frame,
    transformed_intrinsics,
)


class DualRealSenseShutdownTest(unittest.TestCase):
    def test_capture_thread_finishes_before_native_pipelines_stop(self):
        snapshotter = DualRealSenseSnapshotter.__new__(DualRealSenseSnapshotter)
        snapshotter._stop_event = threading.Event()
        snapshotter._thread = MagicMock()
        snapshotter._thread.is_alive.return_value = False
        snapshotter.wrist = MagicMock()
        snapshotter.front = MagicMock()

        self.assertTrue(snapshotter.stop())

        self.assertIsNone(snapshotter._thread)
        snapshotter.wrist.stop.assert_called_once_with()
        snapshotter.front.stop.assert_called_once_with()


class DualRealSenseDuplicateFrameTest(unittest.TestCase):
    def test_repeated_wrist_frame_is_not_added_to_history(self):
        def frame(number):
            return {
                "bgr": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth_m": np.ones((2, 2), dtype=np.float32),
                "frame_number": number,
                "sensor_timestamp_ms": float(number),
                "timestamp_domain": "timestamp_domain.global_time",
                "wall_time_ns": number * 1_000_000,
            }

        snapshotter = DualRealSenseSnapshotter.__new__(DualRealSenseSnapshotter)
        snapshotter._stop_event = threading.Event()
        snapshotter._lock = threading.Lock()
        snapshotter._thread_error = None
        snapshotter._latest = None
        snapshotter._history = deque(maxlen=8)
        snapshotter._next_sequence = 0
        snapshotter._duplicate_frame_counts = {"front": 0, "wrist": 0}
        snapshotter.tracker = None
        snapshotter.record_size = (2, 2)
        snapshotter.front_record_geometry = center_crop_resize_geometry(2, 2, 2, 2)
        snapshotter.wrist_record_geometry = center_crop_resize_geometry(2, 2, 2, 2)
        snapshotter.front = MagicMock()
        snapshotter.wrist = MagicMock()
        snapshotter.front.read.side_effect = [frame(1), frame(2), frame(3)]
        wrist_frames = iter([frame(1), frame(1), frame(2)])

        def read_wrist():
            result = next(wrist_frames)
            if result["frame_number"] == 2:
                snapshotter._stop_event.set()
            return result

        snapshotter.wrist.read.side_effect = read_wrist
        snapshotter._capture_loop()

        samples, cursor = snapshotter.samples_since(0)
        self.assertEqual(cursor, 2)
        self.assertEqual([sample["capture_sequence"] for sample in samples], [0, 1])
        self.assertEqual([sample["wrist_frame_number"] for sample in samples], [1, 2])
        self.assertEqual(snapshotter.duplicate_frame_counts(), {"front": 0, "wrist": 1})


class DualRealSenseTrackerThreadTest(unittest.TestCase):
    def test_slow_tracker_uses_latest_frame_without_blocking_submitter(self):
        entered = threading.Event()
        release = threading.Event()
        processed = []

        class SlowTracker:
            def update(self, color_bgr, _intrinsics):
                value = int(color_bgr[0, 0, 0])
                processed.append(value)
                if value == 1:
                    entered.set()
                    release.wait(timeout=1.0)
                return {"tracker_value": value}

        snapshotter = DualRealSenseSnapshotter.__new__(DualRealSenseSnapshotter)
        snapshotter._stop_event = threading.Event()
        snapshotter._tracker_condition = threading.Condition()
        snapshotter._tracker_input = None
        snapshotter._latest_tracking = None
        snapshotter._tracker_error = None
        snapshotter.tracker = SlowTracker()
        snapshotter.front = SimpleNamespace(intrinsics=object())
        thread = threading.Thread(target=snapshotter._tracker_loop)
        thread.start()
        try:
            with snapshotter._tracker_condition:
                snapshotter._tracker_input = np.full((1, 1, 3), 1, dtype=np.uint8)
                snapshotter._tracker_condition.notify()
            self.assertTrue(entered.wait(timeout=1.0))
            with snapshotter._tracker_condition:
                snapshotter._tracker_input = np.full((1, 1, 3), 2, dtype=np.uint8)
                snapshotter._tracker_input = np.full((1, 1, 3), 3, dtype=np.uint8)
                snapshotter._tracker_condition.notify()
            release.set()
            for _ in range(100):
                with snapshotter._tracker_condition:
                    if snapshotter._latest_tracking == {"tracker_value": 3}:
                        break
                threading.Event().wait(0.005)
            self.assertEqual(processed, [1, 3])
            self.assertEqual(snapshotter._latest_tracking, {"tracker_value": 3})
        finally:
            snapshotter._stop_event.set()
            with snapshotter._tracker_condition:
                snapshotter._tracker_condition.notify_all()
            thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(snapshotter._tracker_error)


class FurniturePoseTrackerTest(unittest.TestCase):
    def test_one_leg_keeps_existing_six_pose_layout(self):
        tracker = FurniturePoseTracker("one_leg")

        self.assertEqual(tracker.last_poses.shape, (6, 7))
        self.assertEqual(tracker.tracked_part_indices, (0, 4))
        np.testing.assert_array_equal(
            tracker.valid,
            np.array([False, True, True, True, False, True]),
        )

    def test_round_table_tracks_all_three_parts(self):
        tracker = FurniturePoseTracker("round_table")

        self.assertEqual(tracker.last_poses.shape, (3, 7))
        self.assertEqual(tracker.tracked_part_indices, (0, 1, 2))
        self.assertEqual(
            tracker.part_names,
            ("round_table_top", "round_table_leg", "round_table_base"),
        )
        np.testing.assert_array_equal(tracker.valid, np.zeros(3, dtype=bool))

    def test_lamp_tracks_all_three_parts(self):
        tracker = FurniturePoseTracker("lamp")

        self.assertEqual(tracker.last_poses.shape, (3, 7))
        self.assertEqual(tracker.tracked_part_indices, (0, 1, 2))
        self.assertEqual(
            tracker.part_names,
            ("lamp_base", "lamp_bulb", "lamp_hood"),
        )
        np.testing.assert_array_equal(tracker.valid, np.zeros(3, dtype=bool))


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

    def test_world_left_wrist_delta_becomes_local_wrist_delta_by_default(self):
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
        expected_rotation = wrist_pose[:3, :3].T @ goal_wrist[:3, :3]

        np.testing.assert_allclose(
            converted[:3],
            goal_wrist[:3, 3] - wrist_pose[:3, 3],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            transform_utils.quat2mat(converted[3:7]),
            expected_rotation,
            atol=1e-6,
        )
        self.assertEqual(converted[-1], 1.0)

    def test_original_mode_exactly_preserves_local_tip_delta(self):
        wrist_pose = np.eye(4)
        scaled_action = np.array([0.01, -0.02, 0.005, 0.04, 0.02, -0.03, 1.0])
        converted = deoxys_delta_to_furniture_bench_action(
            scaled_action,
            wrist_pose,
            "original",
        )
        world_delta = transform_utils.quat2mat(
            transform_utils.axisangle2quat(scaled_action[3:6])
        )
        goal_wrist = wrist_pose.copy()
        goal_wrist[:3, 3] += scaled_action[:3]
        goal_wrist[:3, :3] = world_delta @ wrist_pose[:3, :3]
        current_tip = wrist_pose @ WRIST_TO_TIP
        goal_tip = goal_wrist @ WRIST_TO_TIP
        np.testing.assert_allclose(
            converted[:3], goal_tip[:3, 3] - current_tip[:3, 3], atol=1e-7
        )
        np.testing.assert_allclose(
            transform_utils.quat2mat(converted[3:7]),
            current_tip[:3, :3].T @ goal_tip[:3, :3],
            atol=1e-6,
        )

    def test_eepose_selection_preserves_legacy_tip(self):
        wrist = np.eye(4)
        np.testing.assert_allclose(eepose_from_wrist_pose(wrist), wrist)
        np.testing.assert_allclose(
            eepose_from_wrist_pose(wrist, "original"), wrist @ WRIST_TO_TIP
        )
        self.assertEqual(resolve_eepose_frame("original"), "real-tip")
        self.assertEqual(
            resolve_eepose_frame("original"), resolve_eepose_frame("real-tip")
        )
        with self.assertRaises(ValueError):
            resolve_eepose_frame("original=real-tip")


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
