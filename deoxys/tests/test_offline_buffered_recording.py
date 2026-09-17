import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from examples.run_deoxys_with_space_mouse_V3_record import (
    BUFFERED_SCHEMA,
    RawEpisodeRecorder,
    _absolute_wrist_target_to_action,
    _nearest_unique_matches,
    _nearest_joint_camera_matches,
    _serializable_raw_state_records,
    _write_episode,
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


def arm_command(effect_time_ns, x=0.5, sample_index=0):
    return {
        "channel": "robot",
        "status": "executed",
        "absolute_wrist_action": np.array(
            [x, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64
        ),
        "sample_index": sample_index,
        "sample_wall_time_ns": effect_time_ns - 120_000_000,
        "command_wall_time_ns": effect_time_ns - 120_000_000,
        "predicted_effect_wall_time_ns": effect_time_ns,
    }


def gripper_command(effect_time_ns, action=-1.0, sample_index=0):
    return {
        "channel": "gripper",
        "status": "executed",
        "gripper_action": float(action),
        "sample_index": sample_index,
        "sample_wall_time_ns": effect_time_ns - 642_000_000,
        "command_wall_time_ns": effect_time_ns - 642_000_000,
        "predicted_effect_wall_time_ns": effect_time_ns,
    }


class OfflineBufferedAlignmentTest(unittest.TestCase):
    def test_raw_franka_protobuf_state_can_be_pickled_after_conversion(self):
        from deoxys.proto.franka_interface import franka_robot_state_pb2

        message = franka_robot_state_pb2.FrankaRobotStateMessage()
        message.q.extend([0.0] * 7)
        rows = _serializable_raw_state_records(
            [{"message": message, "receive_wall_time_ns": BASE_TIME_NS}]
        )

        self.assertNotIn("message", rows[0])
        self.assertEqual(rows[0]["message_proto_type"], "FrankaRobotStateMessage")
        self.assertIsInstance(rows[0]["message_proto_bytes"], bytes)
        pickle.dumps(rows)

    def test_missing_initial_part_pose_is_reported_but_recording_starts(self):
        recorder = RawEpisodeRecorder(
            data_root="/tmp/missing-initial-pose-test",
            task_name="one_leg",
            randomness="low",
            camera_info={},
            writer=SimpleNamespace(submit=Mock()),
            output_suffix="unit-test-v6",
        )
        initial = camera_sample(0, BASE_TIME_NS)
        initial["parts_pose_valid"][4] = False

        self.assertTrue(
            recorder.begin_buffered(
                initial,
                camera_start_sequence=0,
                robot_start_index=0,
                gripper_start_index=0,
                grid_start_wall_time_ns=BASE_TIME_NS,
                initial_absolute_wrist_action=np.zeros(7),
                initial_gripper_action=-1.0,
            )
        )
        self.assertEqual(recorder.state, "recording")
        self.assertEqual(recorder.quality_issues[0]["phase"], "geometry")

    def test_absolute_target_round_trips_through_delta_action(self):
        current_pose = np.eye(4)
        current_pose[:3, 3] = [0.4, -0.1, 0.3]
        target_rotation = np.array([0.1, -0.2, 0.05])
        target = np.r_[[0.45, -0.08, 0.28], target_rotation, 1.0]
        state = {"ee_pose": current_pose}
        delta, absolute = _absolute_wrist_target_to_action(
            target, state, "robot-base"
        )
        reconstructed = current_pose.copy()
        reconstructed[:3, 3] += delta[:3]
        from scipy.spatial.transform import Rotation
        reconstructed[:3, :3] = current_pose[:3, :3] @ Rotation.from_quat(
            delta[3:7]
        ).as_matrix()
        np.testing.assert_allclose(reconstructed[:3, 3], absolute[:3], atol=1e-6)
        np.testing.assert_allclose(
            reconstructed[:3, :3],
            Rotation.from_quat(absolute[3:7]).as_matrix(),
            atol=1e-6,
        )

    def test_materializes_sorted_continuous_grid_and_interpolates_state(self):
        targets = [BASE_TIME_NS + PERIOD_NS, BASE_TIME_NS + 2 * PERIOD_NS]
        action_records = [arm_command(target, x=index + 0.75, sample_index=index)
                          for index, target in enumerate(targets)]
        gripper_actions = [gripper_command(targets[0])]
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
            gripper_actions,
            cameras,
            robots,
            grippers,
            grid_start_wall_time_ns=targets[0],
            grid_end_wall_time_ns=targets[-1],
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
        self.assertTrue(
            all(
                row["arm_source_effect_wall_time_ns"]
                <= row["action_target_wall_time_ns"]
                for row in result["action_timing"]
            )
        )

    def test_direct_delta_command_is_preserved_during_offline_alignment(self):
        target = BASE_TIME_NS + PERIOD_NS
        command = arm_command(target, x=9.0)
        command["scaled_physical_delta"] = np.array(
            [0.01, -0.02, 0.03, 0.0, 0.0, 0.1, -1.0]
        )
        result = materialize_buffered_episode(
            [command],
            [gripper_command(target)],
            [camera_sample(0, target)],
            [
                robot_record(target - 10_000_000, 0.4),
                robot_record(target + 10_000_000, 0.4),
            ],
            [gripper_record(target)],
            grid_start_wall_time_ns=target,
            grid_end_wall_time_ns=target,
            camera_info={"front": {}, "wrist": {}},
            eepose_frame="robot-base",
            action_period_ns=PERIOD_NS,
            camera_max_residual_ns=50_000_000,
            camera_pair_max_skew_ns=40_000_000,
            robot_max_residual_ns=20_000_000,
            gripper_max_residual_ns=60_000_000,
        )
        np.testing.assert_allclose(
            result["actions"][0][:3], [0.01, -0.02, 0.03], atol=1e-7
        )

    def test_missing_30hz_send_becomes_causal_target_hold(self):
        targets = [BASE_TIME_NS + index * PERIOD_NS for index in range(4)]
        actions = [
            arm_command(targets[0], x=0.5),
            arm_command(targets[3], x=0.8),
        ]
        cameras = [camera_sample(index, target) for index, target in enumerate(targets)]
        robots = [
            robot_record(target + offset, 0.4)
            for target in targets
            for offset in (-10_000_000, 10_000_000)
        ]
        grippers = [gripper_record(target) for target in targets]
        result = materialize_buffered_episode(
            actions,
            [gripper_command(targets[0])],
            cameras,
            robots,
            grippers,
            grid_start_wall_time_ns=targets[0],
            grid_end_wall_time_ns=targets[-1],
            camera_info={"front": {}, "wrist": {}},
            eepose_frame="robot-base",
            action_period_ns=PERIOD_NS,
            camera_max_residual_ns=50_000_000,
            camera_pair_max_skew_ns=40_000_000,
            robot_max_residual_ns=20_000_000,
            gripper_max_residual_ns=60_000_000,
        )
        np.testing.assert_allclose(
            [row["arm_action_age_ms"] for row in result["action_timing"]],
            [0.0, 100.0, 200.0, 0.0],
        )

    def test_gripper_state_can_be_reused_within_residual_limit(self):
        targets = [BASE_TIME_NS + index * PERIOD_NS for index in range(3)]
        actions = [arm_command(targets[0])]
        cameras = [camera_sample(index, target) for index, target in enumerate(targets)]
        robots = [
            robot_record(target + offset, 0.5)
            for target in targets
            for offset in (-10_000_000, 10_000_000)
        ]
        grippers = [
            gripper_record(targets[0], width=0.01),
            gripper_record(targets[1] + 50_000_000, width=0.02),
        ]

        result = materialize_buffered_episode(
            actions,
            [gripper_command(targets[0])],
            cameras,
            robots,
            grippers,
            grid_start_wall_time_ns=targets[0],
            grid_end_wall_time_ns=targets[-1],
            camera_info={"front": {}, "wrist": {}},
            eepose_frame="robot-base",
            action_period_ns=PERIOD_NS,
            camera_max_residual_ns=45_000_000,
            camera_pair_max_skew_ns=40_000_000,
            robot_max_residual_ns=20_000_000,
            gripper_max_residual_ns=60_000_000,
        )

        self.assertEqual(len(result["observations"]), 3)
        self.assertEqual(result["report"]["gripper_state_residual"]["max_ms"], 50.0)

    def test_gripper_state_60_161ms_gap_keeps_alignment_and_prompt_depth(self):
        targets = [BASE_TIME_NS + index * PERIOD_NS for index in range(3)]
        cameras = [camera_sample(index, target) for index, target in enumerate(targets)]
        robots = [
            robot_record(target + offset, 0.5)
            for target in targets
            for offset in (-10_000_000, 10_000_000)
        ]
        grippers = [
            gripper_record(targets[0]),
            gripper_record(targets[1] - 62_208_000),
            gripper_record(targets[1] + 60_161_000),
            gripper_record(targets[2]),
        ]
        result = materialize_buffered_episode(
            [arm_command(targets[0])],
            [gripper_command(targets[0])],
            cameras,
            robots,
            grippers,
            grid_start_wall_time_ns=targets[0],
            grid_end_wall_time_ns=targets[-1],
            camera_info={"front": {}, "wrist": {}},
            eepose_frame="robot-base",
            action_period_ns=PERIOD_NS,
            camera_max_residual_ns=50_000_000,
            camera_pair_max_skew_ns=40_000_000,
            robot_max_residual_ns=20_000_000,
            gripper_max_residual_ns=60_000_000,
        )
        self.assertEqual(len(result["observations"]), 3)
        self.assertEqual(result["report"]["gripper_state_warning_steps"], [1])
        self.assertAlmostEqual(
            result["report"]["gripper_state_residual"]["max_ms"], 60.161
        )
        self.assertTrue(
            result["observations"][1]["offline_alignment"]["gripper_state"][
                "quality_warning"
            ]
        )

        class Estimator:
            min_depth_m = 0.05
            max_depth_m = 5.0

            def enhance(self, _rgb, depth_m, prompt_depth_m=None):
                return np.asarray(depth_m) + 0.1, {"inference_ms": 1.0}

        depth_report = apply_prompt_depth_offline(
            result["observations"], Estimator(), ("front", "wrist")
        )
        self.assertEqual(depth_report["frame_count"], 3)
        self.assertIn("depth_image1_realsense", result["observations"][1])

    def test_gripper_reuse_still_rejects_stale_state(self):
        with self.assertRaisesRegex(RuntimeError, "nearest residual"):
            _nearest_unique_matches(
                [BASE_TIME_NS, BASE_TIME_NS + PERIOD_NS],
                [(BASE_TIME_NS, gripper_record(BASE_TIME_NS))],
                60_000_000,
                "gripper_state",
                require_unique=False,
            )

    def test_joint_camera_match_uses_synchronized_alternative(self):
        target = BASE_TIME_NS
        front_items = [
            (target - 40_000_000, "front_near"),
            (target + 70_000_000, "front_late"),
        ]
        wrist_items = [
            (target - 30_000_000, "wrist_synchronized"),
            (target + 4_000_000, "wrist_independently_nearest"),
        ]

        front, wrist, front_residuals, wrist_residuals, skews, warnings = (
            _nearest_joint_camera_matches(
                [target], front_items, wrist_items, 45_000_000, 40_000_000
            )
        )

        self.assertEqual(front[0][1], "front_near")
        self.assertEqual(wrist[0][1], "wrist_independently_nearest")
        self.assertEqual(front_residuals, [40_000_000])
        self.assertEqual(wrist_residuals, [4_000_000])
        self.assertEqual(skews, [44_000_000])
        self.assertEqual(warnings, [0])

    def test_joint_camera_match_warns_but_keeps_pair(self):
        target = BASE_TIME_NS
        result = _nearest_joint_camera_matches(
            [target],
            [(target - 70_000_000, "front")],
            [(target + 4_000_000, "wrist")],
            50_000_000,
            40_000_000,
        )
        self.assertEqual(result[-1], [0])

    def test_joint_camera_match_allows_reusing_a_frame(self):
        targets = [BASE_TIME_NS, BASE_TIME_NS + PERIOD_NS]
        frame = (BASE_TIME_NS + 45_000_000, "frame")
        front, wrist, *_ = _nearest_joint_camera_matches(
            targets, [frame], [frame], 50_000_000, 40_000_000
        )
        self.assertIs(front[0][1], front[1][1])
        self.assertIs(wrist[0][1], wrist[1][1])

    def test_joint_camera_hard_gap_is_recorded_without_aborting(self):
        target = BASE_TIME_NS + 300_000_000
        front, wrist, front_residuals, _, _, warnings = (
            _nearest_joint_camera_matches(
                [target],
                [(BASE_TIME_NS, "front")],
                [(target, "wrist")],
                50_000_000,
                40_000_000,
                200_000_000,
            )
        )
        self.assertEqual(front[0][1], "front")
        self.assertEqual(wrist[0][1], "wrist")
        self.assertEqual(front_residuals, [300_000_000])
        self.assertEqual(warnings, [0])

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

    def test_buffered_drop_saves_incomplete_raw_streams(self):
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
                grid_start_wall_time_ns=BASE_TIME_NS,
                initial_absolute_wrist_action=np.array(
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                ),
                initial_gripper_action=-1.0,
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
                camera_hard_gap_ms=200.0,
                robot_max_residual_ms=20.0,
                gripper_max_residual_ms=60.0,
            )
        )
        output_path = recorder.save(success=True)
        self.assertIsNotNone(output_path)
        self.assertIn("/unit-test-v6/incomplete/success/", str(output_path))
        writer.submit.assert_called_once()
        payload = writer.submit.call_args.args[1]
        self.assertEqual(payload["save_quality"]["status"], "incomplete")
        self.assertIn("raw_camera_samples", payload)
        self.assertIn("raw_robot_states", payload)
        self.assertIn("raw_gripper_states", payload)
        self.assertEqual(payload["actions"], [])
        self.assertTrue(payload["metadata"]["schema"].endswith("_incomplete"))
        self.assertEqual(recorder.state, "idle")

    def test_alignment_failure_writes_raw_franka_states_and_error_log(self):
        from deoxys.proto.franka_interface import franka_robot_state_pb2

        writer = SimpleNamespace(submit=Mock())
        recorder = RawEpisodeRecorder(
            data_root="/tmp/raw-only-v6-test",
            task_name="one_leg",
            randomness="low",
            camera_info={"front": {}, "wrist": {}},
            writer=writer,
            output_suffix="unit-test-v6",
        )
        self.assertTrue(
            recorder.begin_buffered(
                camera_sample(0, BASE_TIME_NS),
                camera_start_sequence=0,
                robot_start_index=0,
                gripper_start_index=0,
                grid_start_wall_time_ns=BASE_TIME_NS,
                initial_absolute_wrist_action=np.zeros(7),
                initial_gripper_action=-1.0,
            )
        )
        recorder.add_camera_samples(
            [camera_sample(1, BASE_TIME_NS - 1_000_000_000)]
        )
        robot_message = franka_robot_state_pb2.FrankaRobotStateMessage()
        gripper_message = franka_robot_state_pb2.FrankaGripperStateMessage()
        state_time = BASE_TIME_NS + PERIOD_NS
        recorder.grid_end_wall_time_ns = state_time
        self.assertFalse(
            recorder.stop_buffered(
                [{"message": robot_message, "receive_wall_time_ns": state_time}],
                [{"message": gripper_message, "receive_wall_time_ns": state_time}],
                prompt_depth_estimator=None,
                prompt_depth_cameras=(),
                camera_max_residual_ms=50.0,
                camera_pair_max_skew_ms=40.0,
                camera_hard_gap_ms=200.0,
                robot_max_residual_ms=20.0,
                gripper_max_residual_ms=60.0,
            )
        )
        recorder.save(success=False)
        payload = writer.submit.call_args.args[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw-only.pkl"
            _write_episode(path, payload, 10, False)
            with path.open("rb") as saved_file:
                saved = pickle.load(saved_file)
            self.assertEqual(saved["save_quality"]["status"], "incomplete")
            self.assertIsInstance(
                saved["raw_robot_states"][0]["message_proto_bytes"], bytes
            )
            self.assertEqual(len(saved["raw_camera_samples"]), 1)
            self.assertIn(
                "offline_alignment",
                path.with_suffix(".txt").read_text(encoding="utf-8"),
            )

    def test_missing_offline_annotation_still_saves_aligned_rgbd(self):
        class Estimator:
            min_depth_m = 0.05
            max_depth_m = 5.0

            def enhance(self, _rgb, depth_m, prompt_depth_m=None):
                return np.asarray(depth_m), {"inference_ms": 1.0}

        writer = SimpleNamespace(submit=Mock())
        target = BASE_TIME_NS + PERIOD_NS
        recorder = RawEpisodeRecorder(
            data_root="/tmp/unannotated-v6-test",
            task_name="one_leg",
            randomness="low",
            camera_info={"front": {}, "wrist": {}},
            writer=writer,
            output_suffix="unit-test-v6",
            prompt_depth_config={
                "online": False,
                "cameras": ["front", "wrist"],
            },
        )
        self.assertTrue(
            recorder.begin_buffered(
                camera_sample(0, target),
                camera_start_sequence=0,
                robot_start_index=0,
                gripper_start_index=0,
                grid_start_wall_time_ns=target,
                initial_absolute_wrist_action=np.zeros(7),
                initial_gripper_action=-1.0,
            )
        )
        recorder.add_camera_samples([camera_sample(1, target)])
        recorder.grid_end_wall_time_ns = target
        self.assertFalse(
            recorder.stop_buffered(
                [
                    robot_record(target - 10_000_000, 0.4),
                    robot_record(target + 10_000_000, 0.4),
                ],
                [gripper_record(target)],
                prompt_depth_estimator=Estimator(),
                prompt_depth_cameras=("front", "wrist"),
                camera_max_residual_ms=50.0,
                camera_pair_max_skew_ms=40.0,
                camera_hard_gap_ms=200.0,
                robot_max_residual_ms=20.0,
                gripper_max_residual_ms=60.0,
            )
        )
        output_path = recorder.save(success=True)
        self.assertIsNotNone(output_path)
        self.assertIn("/unit-test-v6/incomplete/success/", str(output_path))
        payload = writer.submit.call_args.args[1]
        self.assertEqual(len(payload["observations"]), 1)
        self.assertEqual(len(payload["actions"]), 1)
        self.assertIn("depth_image1_realsense", payload["observations"][0])
        self.assertEqual(payload["annotation_source"], "unannotated")
        self.assertEqual(payload["save_quality"]["status"], "incomplete")
        self.assertTrue(
            any(issue["phase"] == "offline_annotation"
                for issue in payload["save_quality"]["issues"])
        )

    def test_gripper_gap_saves_enhanced_incomplete_episode(self):
        class Estimator:
            min_depth_m = 0.05
            max_depth_m = 5.0

            def enhance(self, _rgb, depth_m, prompt_depth_m=None):
                return np.asarray(depth_m) + 0.1, {"inference_ms": 1.0}

        writer = SimpleNamespace(submit=Mock())
        targets = [BASE_TIME_NS + index * PERIOD_NS for index in range(3)]
        recorder = RawEpisodeRecorder(
            data_root="/tmp/gripper-gap-v6-test",
            task_name="one_leg",
            randomness="low",
            camera_info={"front": {}, "wrist": {}},
            writer=writer,
            output_suffix="unit-test-v6",
            prompt_depth_config={"online": False, "cameras": ["front", "wrist"]},
        )
        self.assertTrue(
            recorder.begin_buffered(
                camera_sample(0, targets[0]),
                camera_start_sequence=0,
                robot_start_index=0,
                gripper_start_index=0,
                grid_start_wall_time_ns=targets[0],
                initial_absolute_wrist_action=np.zeros(7),
                initial_gripper_action=-1.0,
            )
        )
        recorder.add_camera_samples(
            [camera_sample(index + 1, target) for index, target in enumerate(targets)]
        )
        recorder.grid_end_wall_time_ns = targets[-1]
        robots = [
            robot_record(target + offset, 0.5)
            for target in targets
            for offset in (-10_000_000, 10_000_000)
        ]
        grippers = [
            gripper_record(targets[0]),
            gripper_record(targets[1] - 62_208_000),
            gripper_record(targets[1] + 60_161_000),
            gripper_record(targets[2]),
        ]
        self.assertFalse(
            recorder.stop_buffered(
                robots,
                grippers,
                prompt_depth_estimator=Estimator(),
                prompt_depth_cameras=("front", "wrist"),
                camera_max_residual_ms=50.0,
                camera_pair_max_skew_ms=40.0,
                camera_hard_gap_ms=200.0,
                robot_max_residual_ms=20.0,
                gripper_max_residual_ms=60.0,
            )
        )
        output_path = recorder.save(success=True)
        self.assertIn("/incomplete/success/", str(output_path))
        payload = writer.submit.call_args.args[1]
        self.assertEqual(len(payload["observations"]), 3)
        self.assertEqual(payload["alignment_report"]["gripper_state_warning_steps"], [1])
        self.assertIn("depth_image1_realsense", payload["observations"][1])
        self.assertTrue(
            any(
                issue["phase"] == "offline_alignment"
                and "gripper_state_warning_steps" in issue["message"]
                for issue in payload["save_quality"]["issues"]
            )
        )

    def test_incomplete_pickle_has_same_name_quality_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "episode.pkl"
            payload = {
                "observations": [],
                "actions": [],
                "raw_camera_samples": [camera_sample(0, BASE_TIME_NS)],
                "raw_arm_commands_absolute": [arm_command(BASE_TIME_NS)],
                "raw_gripper_commands": [gripper_command(BASE_TIME_NS)],
                "alignment_report": None,
                "save_quality": {
                    "status": "incomplete",
                    "issues": [
                        {
                            "time": "2026-09-16T15:22:22",
                            "phase": "offline_alignment",
                            "message": "camera gap 200ms",
                            "traceback": None,
                        }
                    ],
                },
            }

            _write_episode(output_path, payload, 10, False)

            with output_path.open("rb") as saved_file:
                saved = pickle.load(saved_file)
            report = output_path.with_suffix(".txt").read_text(encoding="utf-8")
            self.assertEqual(saved["save_quality"]["status"], "incomplete")
            self.assertEqual(len(saved["raw_camera_samples"]), 1)
            self.assertIn("offline_alignment: camera gap 200ms", report)

    def test_video_failure_does_not_remove_saved_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "episode.pkl"
            payload = {
                "observations": [],
                "actions": [],
                "save_quality": {"status": "complete", "issues": []},
            }
            with patch(
                "examples.run_deoxys_with_space_mouse_V3_record._write_video_atomic",
                side_effect=RuntimeError("codec unavailable"),
            ):
                _write_episode(output_path, payload, 10, True)

            self.assertTrue(output_path.is_file())
            self.assertIn(
                "video_write_error: RuntimeError: codec unavailable",
                output_path.with_suffix(".txt").read_text(encoding="utf-8"),
            )

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
            update_trajectory_metadata=lambda payload: (
                payload.update(annotation_source="real_skill_annotation_util"),
                payload["metadata"].update(
                    real_skill_annotation={"complete": True, "mode": "offline"}
                ),
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
