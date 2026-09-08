import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[1] / "examples" / "calibrate_action_latency.py"
SPEC = importlib.util.spec_from_file_location("calibrate_action_latency", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_detect_motion_onset_requires_persistent_crossing():
    samples = [
        (90, [0.0, 0.0]),
        (110, [0.6, 0.0]),
        (120, [0.1, 0.0]),
        (130, [0.7, 0.0]),
        (140, [0.8, 0.0]),
    ]
    assert MODULE.detect_motion_onset_ns(
        samples,
        command_wall_time_ns=100,
        baseline=[0.0, 0.0],
        threshold=0.5,
        confirm_samples=2,
    ) == 130


def test_latency_summary_recommends_requested_statistic():
    median_summary = MODULE.summarize_latency_ms([10.0, 20.0, 30.0], "median")
    p95_summary = MODULE.summarize_latency_ms([10.0, 20.0, 30.0], "p95")
    assert median_summary["recommended_action_latency_ms"] == 20.0
    assert p95_summary["recommended_action_latency_ms"] == pytest.approx(29.0)


def test_updated_profile_only_replaces_calibrated_action_fields():
    base = {
        "schema_version": 2,
        "front_observation_ms": 1.0,
        "wrist_observation_ms": 2.0,
        "robot_observation_ms": 3.0,
        "gripper_observation_ms": 4.0,
        "robot_action_ms": 10.0,
        "gripper_action_ms": 11.0,
        "action_stale_guard_ms": 10.0,
    }
    calibration = {
        "arm": {"recommended_action_latency_ms": 17.5},
        "gripper": None,
    }
    profile = MODULE.build_updated_latency_profile(
        base, calibration, measured_at="2026-09-07T12:00:00+08:00"
    )
    assert profile["robot_action_ms"] == 17.5
    assert profile["gripper_action_ms"] == 11.0
    assert profile["front_observation_ms"] == 1.0
    assert profile["calibrated_fields"] == ["robot_action_ms"]


def test_updated_profile_replaces_all_six_timing_fields():
    base = {
        "schema_version": 2,
        "front_observation_ms": 0.0,
        "wrist_observation_ms": 0.0,
        "robot_observation_ms": 0.0,
        "gripper_observation_ms": 0.0,
        "robot_action_ms": 10.0,
        "gripper_action_ms": 10.0,
        "action_stale_guard_ms": 10.0,
    }
    calibration = {
        "front_observation": {"recommended_observation_latency_ms": 31.0},
        "wrist_observation": {"recommended_observation_latency_ms": 28.0},
        "robot_observation": {"recommended_observation_latency_ms": 0.4},
        "gripper_observation": {"recommended_observation_latency_ms": 0.4},
        "arm": {"recommended_action_latency_ms": 14.0},
        "gripper": {"recommended_action_latency_ms": 37.0},
    }
    profile = MODULE.build_updated_latency_profile(
        base, calibration, measured_at="2026-09-07T12:00:00+08:00"
    )
    assert profile["front_observation_ms"] == 31.0
    assert profile["wrist_observation_ms"] == 28.0
    assert profile["robot_observation_ms"] == 0.4
    assert profile["gripper_observation_ms"] == 0.4
    assert profile["robot_action_ms"] == 14.0
    assert profile["gripper_action_ms"] == 37.0
    assert profile["action_stale_guard_ms"] == 10.0
    assert profile["inherited_fields"] == ["action_stale_guard_ms"]


def test_action_execution_latency_subtracts_observation_path():
    assert MODULE.corrected_execution_latency_ms(17.5, 2.5) == 15.0
    with pytest.raises(ValueError, match="smaller than observation"):
        MODULE.corrected_execution_latency_ms(1.0, 2.0)


def test_clock_mapping_reports_offset_and_drift():
    result = MODULE.fit_clock_mapping(
        [1000.0, 2000.0, 3000.0],
        [1505.0, 2505.0, 3505.0],
    )
    assert result["slope"] == pytest.approx(1.0)
    assert result["drift_ppm"] == pytest.approx(0.0)
    assert result["host_epoch_offset_ms"] == pytest.approx(505.0)
    assert result["absolute_residual"]["max_ms"] == pytest.approx(0.0)


def test_parse_ping_rtts_handles_normal_and_submillisecond_output():
    output = "64 bytes: time=0.321 ms\n64 bytes: time<1 ms\n"
    assert MODULE.parse_ping_rtts_ms(output) == [0.321, 1.0]


def test_observation_only_does_not_require_motion_acknowledgement():
    args = MODULE.parse_args(["--component", "observation", "--output", "/tmp/out.json"])
    assert args.component == "observation"
    assert not args.execute
    assert args.arm_command_frequency_hz == 5.0
    assert args.controller_time_fraction == 2.0
    assert args.arm_step_m == 0.010
    assert args.arm_threshold_m == 0.0005
    assert args.gripper_command_attempts == 3


def test_full_calibration_requires_motion_acknowledgement():
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--component", "all", "--output", "/tmp/out.json"])


def test_existing_outputs_receive_a_shared_run_suffix(tmp_path):
    output = tmp_path / "full-latency-calibration-tag.json"
    profile = tmp_path / "latency_profile-tag.json"
    output.write_text("old full")
    profile.write_text("old profile")

    next_output, next_profile, run_number = (
        MODULE.choose_non_overwriting_output_paths(output, profile)
    )

    assert run_number == 2
    assert next_output.name == "full-latency-calibration-tag-run02.json"
    assert next_profile.name == "latency_profile-tag-run02.json"
    assert output.read_text() == "old full"
    assert profile.read_text() == "old profile"


def test_json_writer_refuses_to_replace_an_existing_file(tmp_path):
    path = tmp_path / "calibration.json"
    MODULE._write_json(path, {"run": 1})
    with pytest.raises(FileExistsError):
        MODULE._write_json(path, {"run": 2})
    assert path.read_text() == '{\n  "run": 1\n}\n'


def test_absolute_pose_action_preserves_pose_and_uses_rotation_vector():
    pose = MODULE.np.eye(4)
    pose[:3, 3] = [0.4, -0.1, 0.2]
    assert MODULE._absolute_pose_action(pose).tolist() == [
        0.4, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0
    ]


def test_eval_controller_settings_replace_delta_yaml_values():
    cfg = SimpleNamespace(
        is_delta=True,
        action_scale=SimpleNamespace(translation=0.05, rotation=0.5),
        traj_interpolator_cfg=SimpleNamespace(
            traj_interpolator_type="MIN_JERK_POSE", time_fraction=0.3
        ),
    )
    assert MODULE._configure_eval_absolute_controller(cfg, 2.0) is cfg
    assert cfg.is_delta is False
    assert cfg.action_scale.translation == 1.0
    assert cfg.action_scale.rotation == 1.0
    assert cfg.traj_interpolator_cfg.traj_interpolator_type == "LINEAR_POSE"
    assert cfg.traj_interpolator_cfg.time_fraction == 2.0


def test_arm_wait_resends_target_and_uses_causal_send_time():
    def message_at_x(x):
        pose = MODULE.np.eye(4)
        pose[0, 3] = x
        return SimpleNamespace(O_T_EE=pose.T.reshape(-1).tolist())

    class Robot:
        def __init__(self):
            self.resent = False
            self.calls = []

        def timestamped_robot_state_buffer(self):
            if not self.resent:
                return []
            return [
                {"receive_wall_time_ns": 210, "message": message_at_x(0.001)},
                {"receive_wall_time_ns": 220, "message": message_at_x(0.001)},
            ]

        def control(self, **kwargs):
            self.calls.append(kwargs)
            self.resent = True
            return {"robot_command_wall_time_ns": 200}

    robot = Robot()
    target = MODULE.np.asarray([0.401, 0.0, 0.2, 0.0, 0.0, 0.0, -1.0])
    onset_ns, causal_ns, sends = MODULE._wait_for_arm_motion_with_resend(
        robot, object(), target,
        first_command_wall_time_ns=100,
        baseline=MODULE.np.zeros(3), threshold=0.0008,
        confirm_samples=2, timeout_s=0.1, command_frequency_hz=1000.0,
    )
    assert onset_ns == 210
    assert causal_ns == 200
    assert sends == [100, 200]
    assert robot.calls[0]["action"].tolist() == target.tolist()


def test_gripper_calibration_retries_a_dropped_command(monkeypatch):
    class Robot:
        def __init__(self):
            self.commands = []
            self.last_gripper_command_wall_time_ns = None

        def timestamped_gripper_state_buffer(self, max_records=None):
            return [
                {
                    "receive_wall_time_ns": 50,
                    "message": SimpleNamespace(width=0.08),
                }
            ]

        def gripper_control(self, action):
            self.commands.append(action)
            self.last_gripper_command_wall_time_ns = 100 * len(self.commands)

    wait_calls = []

    def wait_for_motion(*_args, **kwargs):
        wait_calls.append(kwargs["command_wall_time_ns"])
        if len(wait_calls) == 1:
            raise TimeoutError("simulated dropped command")
        return kwargs["command_wall_time_ns"] + 1_000_000

    monkeypatch.setitem(
        MODULE.calibrate_gripper.__globals__, "_wait_for_motion", wait_for_motion
    )
    args = SimpleNamespace(
        trials=1,
        gripper_switch_width_m=0.04,
        gripper_command_attempts=2,
        gripper_threshold_m=0.002,
        confirm_samples=2,
        motion_timeout_s=2.0,
        settle_s=0.0,
        recommendation="median",
    )
    summary, trials = MODULE.calibrate_gripper(Robot(), args, 0.0)
    assert wait_calls == [100, 200]
    assert trials[0]["command_send_attempts"] == 2
    assert trials[0]["command_send_wall_times_ns"] == [100, 200]
    assert summary["recommended_action_latency_ms"] == 1.0


def test_arm_settle_resends_absolute_pose_at_eval_frequency():
    class Robot:
        def __init__(self):
            self.calls = []

        def control(self, **kwargs):
            self.calls.append(kwargs)

    robot = Robot()
    hold_action = MODULE.np.asarray([0.4, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0])
    MODULE._keep_arm_controller_alive(robot, object(), hold_action, 0.06, 20.0)
    assert len(robot.calls) >= 2
    for call in robot.calls:
        assert call["controller_type"] == "OSC_POSE"
        assert call["control_gripper"] is False
        assert call["enforce_control_frequency"] is False
        assert call["action"].tolist() == hold_action.tolist()
