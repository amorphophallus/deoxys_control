import inspect
import threading
import unittest
from unittest.mock import MagicMock

import numpy as np

from deoxys.utils.machine_time import (
    MachineTimeSchedule,
    common_future_mask,
    monotonic_target_to_wall_time_ns,
)
from deoxys.franka_interface import FrankaInterface


class MachineTimeScheduleTest(unittest.TestCase):
    def setUp(self):
        self.schedule = MachineTimeSchedule.from_milliseconds(
            robot_action_ms=10,
            gripper_action_ms=20,
            stale_guard_ms=10,
            dispatch_tolerance_ms=10,
        )

    def test_common_admission_uses_slower_channel_and_guard(self):
        now = 1_000_000_000
        targets = [1_020_000_000, 1_030_000_000, 1_040_000_000]

        mask = common_future_mask(targets, now, self.schedule)

        np.testing.assert_array_equal(mask, [False, False, True])

    def test_channel_deadlines_keep_one_common_target(self):
        target = 2_000_000_000

        self.assertEqual(self.schedule.deadline_ns(target, "robot"), 1_990_000_000)
        self.assertEqual(self.schedule.deadline_ns(target, "gripper"), 1_980_000_000)

    def test_expired_target_is_never_dispatched(self):
        target = 2_000_000_000

        self.assertTrue(
            self.schedule.dispatch_expired(target, target, "robot")
        )
        self.assertFalse(
            self.schedule.dispatch_expired(target, 1_995_000_000, "robot")
        )

    def test_monotonic_deadline_converts_to_wall_time(self):
        target = monotonic_target_to_wall_time_ns(
            11.25,
            monotonic_now_s=10.0,
            wall_now_ns=1_700_000_000_000_000_000,
        )

        self.assertEqual(target, 1_700_000_001_250_000_000)

    def test_franka_control_exposes_arm_only_external_scheduler_flags(self):
        parameters = inspect.signature(FrankaInterface.control).parameters

        self.assertTrue(parameters["control_gripper"].default)
        self.assertTrue(parameters["enforce_control_frequency"].default)

    def test_franka_close_releases_both_receivers_and_all_sockets(self):
        interface = FrankaInterface.__new__(FrankaInterface)
        interface._closed = False
        interface._stop_event = threading.Event()
        interface._state_sub_thread = MagicMock()
        interface._gripper_sub_thread = MagicMock()
        interface._subscriber = MagicMock()
        interface._gripper_subscriber = MagicMock()
        interface._publisher = MagicMock()
        interface._gripper_publisher = MagicMock()
        interface._context = MagicMock()

        interface.close()

        self.assertTrue(interface._stop_event.is_set())
        interface._state_sub_thread.join.assert_called_once_with(timeout=2.0)
        interface._gripper_sub_thread.join.assert_called_once_with(timeout=2.0)
        for socket in (
            interface._subscriber,
            interface._gripper_subscriber,
            interface._publisher,
            interface._gripper_publisher,
        ):
            socket.close.assert_called_once_with(linger=0)
        interface._context.term.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
