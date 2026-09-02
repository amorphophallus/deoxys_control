import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from examples.run_deoxys_with_space_mouse_V3_record import (
    BUFFERED_SCHEMA,
    RawEpisodeRecorder,
    apply_prompt_depth_offline,
    materialize_buffered_episode,
    validate_buffered_payload,
)


BASE_TIME_NS = 1_700_000_000_000_000_000
PERIOD_NS = 100_000_000


def camera_sample(sequence, wall_time_ns):
    poses = np.zeros((6, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    valid = np.zeros(6, dtype=bool)
    valid[[0, 4]] = True
    return {
        "capture_sequence": sequence,
        "color_image1": np.zeros((4, 5, 3), dtype=np.uint8),
        "color_image2": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth_image1": np.linspace(0.2, 0.8, 20, dtype=np.float32).reshape(4, 5),
        "depth_image2": np.linspace(0.3, 0.9, 20, dtype=np.float32).reshape(4, 5),
        "front_receive_wall_time_ns": wall_time_ns,
        "wrist_receive_wall_time_ns": wall_time_ns + 1_000_000,
        "front_sensor_timestamp_ms": wall_time_ns / 1e6,
        "wrist_sensor_timestamp_ms": (wall_time_ns + 1_000_000) / 1e6,
        "front_timestamp_domain": "system_time",
        "wrist_timestamp_domain": "system_time",
        "front_frame_number": sequence,
        "wrist_frame_number": sequence,
        "camera_capture_wall_time_ns": wall_time_ns + 2_000_000,
        "parts_poses": poses.reshape(-1),
        "parts_founds": valid.copy(),
        "parts_pose_valid": valid,
        "parts_pose_age_ms": np.zeros(6, dtype=np.float32),
        "camera_to_april": np.eye(4),
    }


def robot_record(wall_time_ns, x):
    pose = np.eye(4)
    pose[0, 3] = x
    message = SimpleNamespace(
        O_T_EE=pose.T.reshape(-1).tolist(),
        q=np.full(7, x).tolist(),
        dq=np.full(7, 0.1).tolist(),
        tau_J=np.zeros(7).tolist(),
        time=wall_time_ns / 1e9,
        frame="robot-base",
    )
    return {"message": message, "receive_wall_time_ns": wall_time_ns}


def gripper_record(wall_time_ns, width=0.07):
    message = SimpleNamespace(width=width, time=wall_time_ns / 1e9)
    return {"message": message, "receive_wall_time_ns": wall_time_ns}


class OfflineBufferedAlignmentTest(unittest.TestCase):
    def test_materializes_sorted_continuous_grid_and_interpolates_state(self):
        targets = [BASE_TIME_NS + PERIOD_NS, BASE_TIME_NS + 2 * PERIOD_NS]
        action_records = [
            {
                "scaled_action": np.array([0.0] * 6 + [-1.0]),
                "timing": {"action_target_wall_time_ns": target},
            }
            for target in reversed(targets)
        ]
        cameras = [
            camera_sample(index, target - 4_000_000)
            for index, target in enumerate(targets)
        ]
        robots = []
        for index, target in enumerate(targets):
            robots.extend(
                [
                    robot_record(target - 10_000_000, index + 0.4),
                    robot_record(target + 10_000_000, index + 0.6),
                ]
            )
        grippers = [gripper_record(target + 2_000_000) for target in targets]

        result = materialize_buffered_episode(
            action_records,
            cameras,
            robots,
            grippers,
            camera_info={
                "front": {"global_time_enabled": False},
                "wrist": {"global_time_enabled": False},
            },
            eepose_frame="robot-base",
            action_period_ns=PERIOD_NS,
            camera_max_residual_ns=45_000_000,
            camera_pair_max_skew_ns=40_000_000,
            robot_max_residual_ns=20_000_000,
            gripper_max_residual_ns=60_000_000,
        )

        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(len(result["actions"]), 2)
        self.assertEqual(
            [row["action_target_wall_time_ns"] for row in result["action_timing"]],
            targets,
        )
        np.testing.assert_allclose(
            [obs["robot_state"]["ee_pos"][0] for obs in result["observations"]],
            [0.5, 1.5],
            atol=1e-6,
        )
        self.assertEqual(
            result["report"]["robot_residual"]["max_ms"],
            10.0,
        )

    def test_rejects_discontinuous_action_grid(self):
        action_records = [
            {
                "scaled_action": np.zeros(7),
                "timing": {"action_target_wall_time_ns": BASE_TIME_NS},
            },
            {
                "scaled_action": np.zeros(7),
                "timing": {
                    "action_target_wall_time_ns": BASE_TIME_NS + 2 * PERIOD_NS
                },
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "grid is discontinuous"):
            materialize_buffered_episode(
                action_records,
                [],
                [],
                [],
                camera_info={},
                eepose_frame="robot-base",
                action_period_ns=PERIOD_NS,
                camera_max_residual_ns=45_000_000,
                camera_pair_max_skew_ns=40_000_000,
                robot_max_residual_ns=20_000_000,
                gripper_max_residual_ns=60_000_000,
            )

    def test_offline_prompt_depth_preserves_raw_depth(self):
        class Estimator:
            min_depth_m = 0.05
            max_depth_m = 5.0

            def enhance(self, rgb, depth_m, prompt_depth_m=None):
                return np.asarray(depth_m) + 0.1, {"inference_ms": 2.5}

        observation = camera_sample(0, BASE_TIME_NS)
        raw_wrist = observation["depth_image1"].copy()
        report = apply_prompt_depth_offline(
            [observation],
            Estimator(),
            ("wrist", "front"),
        )

        np.testing.assert_array_equal(
            observation["depth_image1_realsense"], raw_wrist
        )
        np.testing.assert_allclose(
            observation["depth_image1"], raw_wrist + 0.1, atol=1e-3
        )
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["camera_inference_count"], 2)

    def test_buffered_drop_refuses_save_until_discard(self):
        writer = SimpleNamespace(submit=Mock())
        recorder = RawEpisodeRecorder(
            data_root="/tmp/buffered-drop-test",
            task_name="one_leg",
            randomness="low",
            camera_info={},
            writer=writer,
            output_suffix="unit-test-v6",
        )
        initial = camera_sample(0, BASE_TIME_NS)
        self.assertTrue(
            recorder.begin_buffered(
                initial,
                camera_start_sequence=0,
                robot_start_index=0,
                gripper_start_index=0,
            )
        )
        recorder.record_dropped_command(
            {"status": "dropped", "drop_reason": "test_gap"}
        )
        self.assertFalse(
            recorder.stop_buffered(
                [],
                [],
                prompt_depth_estimator=None,
                prompt_depth_cameras=(),
                camera_max_residual_ms=45.0,
                camera_pair_max_skew_ms=40.0,
                robot_max_residual_ms=20.0,
                gripper_max_residual_ms=60.0,
            )
        )
        self.assertIsNone(recorder.save(success=True))
        self.assertEqual(recorder.state, "pending_save")
        writer.submit.assert_not_called()

    def test_v6_contract_rejects_vlm_metadata(self):
        observation = camera_sample(0, BASE_TIME_NS)
        observation.update(
            observation_target_wall_time_ns=BASE_TIME_NS,
            skill="pick",
            guidance_point=np.array([0.0, 0.0, 1.0]),
            guidance_point_clean=np.array([0.0, 0.0, 1.0]),
            guidance_pose=None,
            guidance_pose_clean=None,
            guidance_gripper_width=None,
            guidance_point_2d={"color_image1": None, "color_image2": None},
            depth_image1_realsense=observation["depth_image1"].copy(),
            depth_image2_realsense=observation["depth_image2"].copy(),
        )
        payload = {
            "env": "FurnitureBench",
            "annotation_source": "scripted",
            "image_annotation_mode": "none",
            "observations": [observation],
            "actions": [np.array([0.0] * 6 + [-1.0])],
            "actions_original": [np.array([0.0] * 6 + [-1.0])],
            "actions_absolute": [np.zeros(8)],
            "action_timing": [{"action_target_wall_time_ns": BASE_TIME_NS}],
            "action_target_timestamps_ns": [BASE_TIME_NS],
            "action_timestamps_ns": [BASE_TIME_NS],
            "obs_valid": np.ones(1, dtype=bool),
            "rewards": [0.0],
            "camera_info": {},
            "metadata": {
                "schema": BUFFERED_SCHEMA,
                "action_period_ns": PERIOD_NS,
                "real_skill_annotation": {"complete": True, "mode": "offline"},
                "prompt_depth_anything": {
                    "online": False,
                    "cameras": ["wrist", "front"],
                },
            },
        }
        annotation = SimpleNamespace(
            annotator=SimpleNamespace(
                _camera_projections=Mock(
                    return_value=(
                        {"color_image1": None, "color_image2": None},
                        {},
                    )
                )
            )
        )
        self.assertEqual(validate_buffered_payload(payload, annotation)["frames"], 1)
        payload["metadata"]["vlm_result"] = {}
        with self.assertRaisesRegex(RuntimeError, "VLM metadata"):
            validate_buffered_payload(payload, annotation)

    def test_save_emits_dense_v6_contract_and_campaign_path(self):
        writer = SimpleNamespace(submit=Mock())
        prompt_config = {
            "online": False,
            "cameras": ["wrist", "front"],
        }
        recorder = RawEpisodeRecorder(
            data_root="/tmp/v6-save-test",
            task_name="one_leg",
            randomness="low",
            camera_info={},
            writer=writer,
            prompt_depth_config=prompt_config,
            output_suffix="unit-test-v6",
        )
        observation = camera_sample(0, BASE_TIME_NS)
        observation.update(
            observation_target_wall_time_ns=BASE_TIME_NS,
            skill="pick",
            guidance_point=np.array([0.0, 0.0, 1.0]),
            guidance_point_clean=np.array([0.0, 0.0, 1.0]),
            guidance_pose=None,
            guidance_pose_clean=None,
            guidance_gripper_width=None,
            guidance_point_2d={"color_image1": None, "color_image2": None},
            depth_image1_realsense=observation["depth_image1"].copy(),
            depth_image2_realsense=observation["depth_image2"].copy(),
        )
        annotation = SimpleNamespace(
            annotator=SimpleNamespace(
                _camera_projections=Mock(
                    return_value=(
                        {"color_image1": None, "color_image2": None},
                        {},
                    )
                )
            ),
            update_trajectory_metadata=lambda payload: payload["metadata"].update(
                real_skill_annotation={"complete": True, "mode": "offline"}
            ),
        )
        action = np.array([0.0] * 6 + [1.0, -1.0])
        recorder.state = "pending_save"
        recorder.buffered_mode = True
        recorder.observations = [observation]
        recorder.actions = [action]
        recorder.actions_original = [action.copy()]
        recorder.actions_absolute = [np.zeros(8)]
        recorder.action_timing = [
            {"action_target_wall_time_ns": BASE_TIME_NS}
        ]
        recorder.annotation_session = annotation
        recorder.buffer_alignment_report = {"matched": 1}

        output_path = recorder.save(success=True)

        self.assertIn("/low/unit-test-v6/success/", str(output_path))
        writer.submit.assert_called_once()
        payload = writer.submit.call_args.args[1]
        self.assertEqual(payload["env"], "FurnitureBench")
        self.assertEqual(payload["annotation_source"], "scripted")
        self.assertEqual(payload["image_annotation_mode"], "none")
        np.testing.assert_array_equal(payload["obs_valid"], [True])
        self.assertEqual(payload["metadata"]["buffered_contract_audit"]["frames"], 1)


if __name__ == "__main__":
    unittest.main()
