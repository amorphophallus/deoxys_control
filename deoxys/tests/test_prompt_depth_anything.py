import time
import unittest

import numpy as np

from deoxys.utils.prompt_depth_anything import (
    PromptDepthWorker,
    _inference_shape,
    colorize_depth,
    depth_display_bounds,
    has_usable_depth,
    prepare_prompt_depth,
)


class PromptDepthAnythingUtilityTest(unittest.TestCase):
    def test_prepare_prompt_removes_invalid_and_saturated_depth(self):
        depth = np.linspace(0.2, 1.2, 320 * 240, dtype=np.float32).reshape(240, 320)
        depth[:, :80] = 0.0
        depth[10:20, 100:120] = np.nan
        depth[30:40, 150:180] = 64.438

        prompt, stats = prepare_prompt_depth(depth)

        self.assertEqual(prompt.shape, (192, 256))
        self.assertTrue(np.all(np.isfinite(prompt)))
        self.assertGreaterEqual(float(np.min(prompt)), 0.05)
        self.assertLessEqual(float(np.max(prompt)), 5.0)
        self.assertLess(stats["raw_valid_fraction"], 0.8)

    def test_inference_shape_preserves_ratio_and_patch_multiple(self):
        height, width = _inference_shape(240, 320, max_size=448)

        self.assertEqual((height, width), (336, 448))
        self.assertEqual(height % 14, 0)
        self.assertEqual(width % 14, 0)

    def test_colorize_depth_keeps_invalid_pixels_black(self):
        depth = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)

        rendered = colorize_depth(depth, min_depth_m=0.05, max_depth_m=3.0)

        self.assertEqual(rendered.shape, (2, 2, 3))
        np.testing.assert_array_equal(rendered[0, 0], np.zeros(3, dtype=np.uint8))
        np.testing.assert_array_equal(rendered[1, 1], np.zeros(3, dtype=np.uint8))
        self.assertTrue(np.any(rendered[0, 1] != 0))

    def test_constant_depth_is_not_a_usable_metric_prompt(self):
        depth = np.full((240, 320), 0.2, dtype=np.float32)

        self.assertFalse(has_usable_depth(depth))

    def test_display_bounds_ignore_holes_and_far_outliers(self):
        depth = np.linspace(0.2, 1.0, 100, dtype=np.float32).reshape(10, 10)
        depth[0, :] = 0.0
        depth[-1, :] = 64.0

        low, high = depth_display_bounds(depth, 0.05, 3.0)

        self.assertGreater(low, 0.19)
        self.assertLess(high, 1.01)
        self.assertGreater(high - low, 0.5)

    def test_worker_processes_both_cameras_without_blocking_submitter(self):
        class FakeEstimator:
            min_depth_m = 0.05
            max_depth_m = 5.0

            def enhance(self, rgb, depth, prompt_depth_m=None):
                return np.full(depth.shape, 0.7, dtype=np.float32), {
                    "inference_ms": 1.0
                }

        gradient = np.linspace(0.2, 1.0, 80, dtype=np.float32).reshape(8, 10)
        sample = {
            "camera_capture_wall_time_ns": 1,
            "color_image1": np.zeros((8, 10, 3), dtype=np.uint8),
            "color_image2": np.zeros((8, 10, 3), dtype=np.uint8),
            "depth_image1": gradient,
            "depth_image2": gradient,
        }
        worker = PromptDepthWorker(FakeEstimator())
        worker.start()
        try:
            worker.submit(sample)
            deadline = time.monotonic() + 1.0
            while worker.latest() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            result = worker.latest()
            self.assertIsNotNone(result)
            self.assertEqual(
                set(result["depths"]), {"depth_image1", "depth_image2"}
            )
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
