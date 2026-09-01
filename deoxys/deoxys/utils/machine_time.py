"""Machine-wall-time scheduling primitives shared by teleop and policy control.

The public contract intentionally mirrors UMI: an action is labelled by the
wall-clock time at which it should take effect.  Transport latency only changes
when a channel is submitted; it never changes the action's target timestamp.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True)
class MachineTimeSchedule:
    """Latency and stale-action policy for a coordinated arm/gripper action."""

    robot_action_latency_ns: int
    gripper_action_latency_ns: int
    stale_guard_ns: int = 10 * NANOSECONDS_PER_MILLISECOND
    dispatch_tolerance_ns: int = 10 * NANOSECONDS_PER_MILLISECOND

    @classmethod
    def from_milliseconds(
        cls,
        *,
        robot_action_ms: float,
        gripper_action_ms: float,
        stale_guard_ms: float = 10.0,
        dispatch_tolerance_ms: float = 10.0,
    ):
        values = (
            robot_action_ms,
            gripper_action_ms,
            stale_guard_ms,
            dispatch_tolerance_ms,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("machine-time latency values must be finite and non-negative")
        return cls(
            robot_action_latency_ns=int(round(robot_action_ms * 1e6)),
            gripper_action_latency_ns=int(round(gripper_action_ms * 1e6)),
            stale_guard_ns=int(round(stale_guard_ms * 1e6)),
            dispatch_tolerance_ns=int(round(dispatch_tolerance_ms * 1e6)),
        )

    @property
    def common_admission_lead_ns(self):
        return max(
            self.robot_action_latency_ns,
            self.gripper_action_latency_ns,
        ) + self.stale_guard_ns

    def deadline_ns(self, target_time_ns, channel):
        if channel == "robot":
            latency_ns = self.robot_action_latency_ns
        elif channel == "gripper":
            latency_ns = self.gripper_action_latency_ns
        else:
            raise ValueError(f"unsupported action channel {channel!r}")
        return int(target_time_ns) - latency_ns

    def admit(self, target_time_ns, now_ns):
        """Return true only when both channels still have scheduling margin."""

        return int(target_time_ns) > int(now_ns) + self.common_admission_lead_ns

    def dispatch_expired(self, target_time_ns, now_ns, channel):
        """Return true after a channel deadline exceeds the allowed jitter."""

        target_time_ns = int(target_time_ns)
        now_ns = int(now_ns)
        if now_ns >= target_time_ns:
            return True
        return (
            now_ns
            > self.deadline_ns(target_time_ns, channel) + self.dispatch_tolerance_ns
        )


def common_future_mask(target_times_ns: Iterable[int], now_ns: int, schedule):
    """UMI-style common stale-prefix filter for coordinated action chunks."""

    target_times = np.asarray(tuple(target_times_ns), dtype=np.int64)
    if target_times.ndim != 1:
        raise ValueError("target_times_ns must be one-dimensional")
    if len(target_times) > 1 and np.any(np.diff(target_times) <= 0):
        raise ValueError("target_times_ns must increase strictly")
    return target_times > int(now_ns) + schedule.common_admission_lead_ns


def monotonic_target_to_wall_time_ns(
    target_monotonic_s,
    *,
    monotonic_now_s,
    wall_now_ns,
):
    """Convert a local monotonic deadline to the corresponding wall-clock ns."""

    delta_s = float(target_monotonic_s) - float(monotonic_now_s)
    if not np.isfinite(delta_s):
        raise ValueError("monotonic times must be finite")
    return int(wall_now_ns) + int(round(delta_s * 1e9))
