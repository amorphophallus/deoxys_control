import argparse
import os
import select
import signal
import sys
import termios
import time
import tty

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


RESET_JOINT_POSITIONS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

OBSERVE_SCRIPT_NAME = "osc_control_replay_robot_eval_ee_pose.py"


class NonBlockingKeyReader:
    def __init__(self, reset_key):
        self.reset_key = reset_key.lower()
        self._fd = None
        self._old_settings = None
        self.enabled = False

    def __enter__(self):
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY; keyboard reset key is disabled.")
            return self

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self.enabled = False

    def reset_requested(self):
        if not self.enabled:
            return False

        pressed_key = None
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if not char:
                break
            pressed_key = char.lower()

        return pressed_key == self.reset_key


def wait_for_robot_state(robot_interface, timeout=5.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if (
            robot_interface.received_states
            and robot_interface.check_nonzero_configuration()
        ):
            return True
        time.sleep(0.05)
    return False


def move_to_reset_joint_positions(
    robot_interface,
    joint_controller_cfg,
    reset_joint_positions,
    timeout=7.0,
    tolerance=1e-3,
    gripper_open=True,
):
    if not wait_for_robot_state(robot_interface):
        logger.warning("Robot state not received before reset request.")
        return False

    gripper_action = -1.0 if gripper_open else 1.0
    action = list(reset_joint_positions) + [gripper_action]
    target = np.array(reset_joint_positions)
    start_time = time.time()
    max_error = float("inf")

    logger.info("Moving to reset joint positions with JOINT_POSITION controller.")
    while time.time() - start_time < timeout:
        current_q = robot_interface.last_q
        if current_q is not None:
            max_error = float(np.max(np.abs(np.array(current_q) - target)))
            if max_error < tolerance:
                logger.info(
                    f"Reset joint target reached. max_joint_error={max_error:.6f}"
                )
                return True

        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=joint_controller_cfg,
        )

    logger.warning(
        f"Reset joint motion timed out. max_joint_error={max_error:.6f}"
    )
    return False


def read_process_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw_cmdline = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return []

    parts = [part for part in raw_cmdline.split(b"\0") if part]
    return [part.decode("utf-8", errors="replace") for part in parts]


def is_observe_recording_process(pid, argv):
    if pid == os.getpid() or not argv:
        return False

    has_target_script = any(OBSERVE_SCRIPT_NAME in arg for arg in argv)
    if not has_target_script:
        return False

    for idx, arg in enumerate(argv):
        if arg == "--replay-mode" and idx + 1 < len(argv):
            return argv[idx + 1] == "observe"
        if arg.startswith("--replay-mode="):
            return arg.split("=", 1)[1] == "observe"
    return False


def find_observe_recording_processes():
    matches = []
    proc_root = "/proc"
    try:
        entries = os.listdir(proc_root)
    except OSError:
        logger.warning("Cannot inspect /proc; observe process detection disabled.")
        return matches

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        argv = read_process_cmdline(pid)
        if is_observe_recording_process(pid, argv):
            matches.append((pid, argv))
    return matches


def interrupt_observe_recording_processes():
    matches = find_observe_recording_processes()
    interrupted = []
    errors = []

    for pid, argv in matches:
        try:
            os.kill(pid, signal.SIGINT)
            interrupted.append((pid, argv))
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            errors.append((pid, exc))
        except OSError as exc:
            errors.append((pid, exc))

    return matches, interrupted, errors


def handle_spacemouse_right_button(device):
    matches, interrupted, errors = interrupt_observe_recording_processes()
    for pid, exc in errors:
        logger.warning(f"Failed to send Ctrl+C to observe pid={pid}: {exc}")

    if matches:
        interrupted_pids = [pid for pid, _ in interrupted]
        if interrupted_pids:
            logger.info(
                "Sent Ctrl+C to observe process(es) %s; continuing SpaceMouse control.",
                interrupted_pids,
            )
        else:
            logger.warning(
                "Observe process was found, but no process was interrupted; "
                "continuing SpaceMouse control."
            )
        device.start_control(preserve_gripper=True)
        return False

    logger.info(
        "SpaceMouse right button pressed and no observe process was found; "
        "stopping teleoperation."
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50741)
    parser.add_argument("--reset-key", type=str, default="r")
    parser.add_argument("--reset-timeout", type=float, default=7.0)
    parser.add_argument("--reset-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--keep-gripper-closed-during-reset",
        action="store_true",
        help="Close the gripper during keyboard-triggered joint reset.",
    )
    args = parser.parse_args()

    if len(args.reset_key) != 1:
        raise ValueError("--reset-key must be a single character.")

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    joint_controller_cfg = get_default_controller_config(
        controller_type="JOINT_POSITION"
    )

    robot_interface._state_buffer = []

    logger.info(
        f"SpaceMouse teleoperation started. Press '{args.reset_key}' to move to "
        "reset_joint_positions, then resume SpaceMouse control. Press the "
        "SpaceMouse right button to stop observe recording when it is running; "
        "otherwise stop this script."
    )

    with NonBlockingKeyReader(args.reset_key) as key_reader:
        try:
            while True:
                start_time = time.time_ns()

                if key_reader.reset_requested():
                    reset_succeeded = move_to_reset_joint_positions(
                        robot_interface=robot_interface,
                        joint_controller_cfg=joint_controller_cfg,
                        reset_joint_positions=RESET_JOINT_POSITIONS,
                        timeout=args.reset_timeout,
                        tolerance=args.reset_tolerance,
                        gripper_open=not args.keep_gripper_closed_during_reset,
                    )
                    if reset_succeeded:
                        logger.info("Returning to SpaceMouse teleoperation.")
                    else:
                        logger.warning(
                            "Returning to SpaceMouse teleoperation after reset timeout."
                        )
                    device.start_control()
                    continue

                action, grasp = input2action(
                    device=device,
                    controller_type=controller_type,
                )
                if action is None:
                    should_stop = handle_spacemouse_right_button(device)
                    if should_stop:
                        break
                    continue

                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                end_time = time.time_ns()
                logger.debug(
                    f"Time duration: {((end_time - start_time) / (10**9))}"
                )
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received; stopping teleoperation.")
        finally:
            try:
                robot_interface.control(
                    controller_type=controller_type,
                    action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
                    controller_cfg=controller_cfg,
                    termination=True,
                )
            finally:
                robot_interface.close()

    for (state, next_state) in zip(
        robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
    ):
        if (next_state.frame - state.frame) > 1:
            print(state.frame, next_state.frame)


if __name__ == "__main__":
    main()
