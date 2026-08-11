"""Driver class for AirExo controller. Modified based on the robosuite code.

Reference: https://github.com/AirExo/collector

"""

import threading
import time
from collections import namedtuple
from typing import List, Dict, Any

import json
import copy
import math
import serial
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

from typing import Any, Dict, Optional, Protocol

import numpy as np
# import tyro


def deg_2_rad(x):
    """
    Transform the degree into rad.
    """
    return x / 180 * math.pi


def rad_2_deg(x):
    """
    Transform the rad into degree.
    """
    return x / math.pi * 180


def deg_clip(x, w0=True):
    """
    Clip the degree into range [0, 360) or (0, 360] (specified by parameter w0: whether include 0 in the range).
    """
    x = x - 360 * math.ceil((x - 360 + 1e-8) / 360)
    if x == 0 and not w0:
        x = 360
    return x


def deg_distance(x, y, direction, w0=True):
    """
    Get the degree distance from degree x to degree y, according to the given direction.
    Here we assume that the distance is in range [0, 360) or (0, 360]
    (specified by parameter w0: whether include 0 in the range).
    """
    dis = (deg_clip(y) - deg_clip(x)) * direction
    if dis < 0 or (dis == 0 and not w0):
        dis = dis + 360
    return dis


def deg_check_range(x, xmin, xmax, direction):
    """
    Check whether the angle degree x is in the given range [xmin, xmax], according to the given direction.
    Here we assume that the range is not larger than 360 degrees.

    [Examples]
    - for x = 50, [xmin, xmax] = [10, 100], direction = 1, return: True.
    - for x = 150, [xmin, xmax] = [10, 100], direction = 1, return: False
    - for x = 50, [xmin, xmax] = [250, 100], direction = 1, return: True.
    - for x = 150, [xmin, xmax] = [250, 100], direction = 1, return: False.
    - for x = 50, [xmin, xmax] = [10, 100], direction = -1, return: False.
    - for x = 150, [xmin, xmax] = [10, 100], direction = -1, return: True.
    - for x = 50, [xmin, xmax] = [250, 100], direction = -1, return: False.
    - for x = 150, [xmin, xmax] = [250, 100], direction = -1, return: True.
    """
    if direction == -1:
        xmin, xmax = xmax, xmin
    return (
        (x >= xmin and x <= xmax)
        or (0 <= x and x <= xmax and xmax < xmin)
        or (xmax < xmin and xmin <= x and x <= 360)
    )


def deg_clip_in_range(x, xmin, xmax, direction=1):
    """
    Clip the angle degree x into the given range [xmin, xmax], according to the given direction.
    Here we assume that the range is not larger than 360 degrees.

    [Examples]
    - for x = 50, [xmin, xmax] = [10, 100], direction = 1, return: 50.
    - for x = 150, [xmin, xmax] = [10, 100], direction = 1, return: 100.
    - for x = 50, [xmin, xmax] = [250, 100], direction = 1, return: 50.
    - for x = 150, [xmin, xmax] = [250, 100], direction = 1, return: 100.
    - for x = 50, [xmin, xmax] = [10, 100], direction = -1, return: 10.
    - for x = 150, [xmin, xmax] = [10, 100], direction = -1, return: 150.
    - for x = 50, [xmin, xmax] = [250, 100], direction = -1, return: 100.
    - for x = 150, [xmin, xmax] = [250, 100], direction = -1, return: 150.
    """
    x = deg_clip(x)
    if deg_check_range(x, xmin, xmax, direction):
        return x
    dxmin = deg_distance(x, xmin, direction)
    dxmax = deg_distance(xmax, x, direction)
    return xmin if dxmin <= dxmax else xmax


def deg_percentile(x, xmin, xmax, direction=1):
    """
    Calculate the degree percentile of x given the expected range [xmin, xmax], according to the given direction. Notice the return is not necessarily in range [0, 1] because this function does not consider out-of-range situations. Refer to "deg_check_range" for more details.
    Here we assume that the range is not larger than 360 degrees.

    [Examples]
    - for x = 50, [xmin, xmax] = [10, 100], direction = 1, the percentile is 4/9.
    - for x = 50, [xmin, xmax] = [250, 100], direction = 1, the percentile is 16/21.
    """
    return deg_distance(xmin, x, direction, w0=True) / deg_distance(
        xmin, xmax, direction, w0=False
    )


def deg_zero_centered(x, xmin, xmax, xdir):
    """
    Transform the degree in [xmin, xmax] into zero-centered equivalent degree.
    Here we assume that the range is not larger than 360 degrees.
    """
    assert deg_check_range(
        x, xmin - 1e6 * xdir, xmax + 1e6 * xdir, xdir
    ) and deg_check_range(0, xmin - 1e6 * xdir, xmax + 1e6 * xdir, xdir)
    neg_bound = max(xmin, xmax)
    return x - 360 if x >= neg_bound else x


def hex2dex(e_hex):
    return int(e_hex, 16)


def hex2bin(e_hex):
    return bin(int(e_hex, 16))


def dex2bin(e_dex):
    return bin(e_dex)


def crc16(hex_num):
    """
    CRC16 verification
    :param hex_num:
    :return:
    """
    crc = "0xffff"
    crc16 = "0xA001"
    test = hex_num.split(" ")

    crc = hex2dex(crc)
    crc16 = hex2dex(crc16)
    for i in test:
        temp = f"0x{i}"
        temp = hex2dex(temp)
        crc ^= temp
        for _ in range(8):
            if dex2bin(crc)[-1] == "0":
                crc >>= 1
            elif dex2bin(crc)[-1] == "1":
                crc >>= 1
                crc ^= crc16

    crc = hex(crc)
    crc_H = crc[2:4]
    crc_L = crc[-2:]

    return crc, crc_H, crc_L


class AngleEncoder:
    """
    Angle Encoder(s) Interface: receive signals from the angle encoders.
    """

    def __init__(
        self,
        ids,
        port,
        baudrate=115200,
        sleep_gap=0.002,
        **kwargs,
    ):
        """
        Args:
        - ids: list of int, e.g., [1, 2, ..., 8], the id(s) of the desired encoder(s);
        - port, baudrate, (**kwargs): the args of the serial agents;
        - sleep_gap: float, optional, default: 0.002, the sleep gap between adjacent write options;
        - logger_name: str, optional, default: "AngleEncoder", the name of the logger;
        - frame_rate: int, optional, default: 30, the streaming frequency.
        """
        self.ids = ids
        self.ids_num = len(ids)
        self.ids_map = {}
        for i, id in enumerate(ids):
            self.ids_map[id] = i
        self.sleep_gap = sleep_gap
        self.ser = serial.Serial(port, baudrate=baudrate, **kwargs)
        if not self.ser.is_open:
            raise RuntimeError(
                "Fail to open the serial port, please check your settings again."
            )
        self.ser.flushInput()
        self.ser.flushOutput()
        self.last_angle = self.get_angles(ignore_error=False)

    def get_angles(self, ignore_error=False, **kwargs):
        """
        Get the angles of the encoder.

        Args:
        - ignore_error: bool, optional, default: False, whether to ignore the incomplete data error (if set True, then the last results will be used.)

        Returns:
        - ret: np.array, the encoder angle results corresponding to the ids array.
        """
        self.ser.flushInput()
        ids = copy.deepcopy(self.ids)
        for i in ids:
            sendbytes = f"{str(i).zfill(2)} 03 00 41 00 01"
            crc, crc_H, crc_L = crc16(sendbytes)
            sendbytes = f"{sendbytes} {crc_L} {crc_H}"
            sendbytes = bytes.fromhex(sendbytes)
            self.ser.write(sendbytes)
            time.sleep(self.sleep_gap)

        re = self.ser.read(len(ids) * 7)
        if self.ser.inWaiting() > 0:
            se = self.ser.read_all()
            re += se

        count = 0
        remains = ids.copy()
        if ignore_error:
            ret = np.copy(self.last_angle).astype(np.float32)
        else:
            ret = np.zeros(self.ids_num).astype(np.float32)
        b = 0
        while b <= len(re) - 7:
            if re[b + 1] == 3 and re[b + 2] == 2 and re[b] in remains:
                angle = 360 * (re[b + 3] * 256 + re[b + 4]) / 4096
                ret[self.ids_map[re[b]]] = angle
                count += 1
                remains.remove(re[b])
                b += 7
            else:
                b += 1
        if not ignore_error and count != len(ids):
            raise RuntimeError(
                f"Failure to receive all encoders, errors occurred in ID {remains}."
            )
        self.last_angle = ret
        return ret

    def get_circles(self, **kwargs):
        """
        Get the circles of the encoder.

        Returns:
        - ret: np.array, the encoder circle results corresponding to the ids array.
        """
        self.ser.flushInput()
        ids = copy.deepcopy(self.ids)
        for i in ids:
            sendbytes = f"{str(i).zfill(2)} 03 00 44 00 01"
            crc, crc_H, crc_L = crc16(sendbytes)
            sendbytes = f"{sendbytes} {crc_L} {crc_H}"
            sendbytes = bytes.fromhex(sendbytes)
            self.ser.write(sendbytes)
            time.sleep(self.sleep_gap)

        re = self.ser.read(len(ids) * 7)
        if self.ser.inWaiting() > 0:
            se = self.ser.read_all()
            re += se

        count = 0
        remains = ids.copy()
        ret = np.zeros(self.ids_num).astype(np.float32)
        b = 0
        while b <= len(re) - 7:
            if re[b + 1] == 3 and re[b + 2] == 2 and re[b] in remains:
                angle = re[b + 3] * 256 + re[b + 4]
                ret[self.ids_map[re[b]]] = angle
                count += 1
                remains.remove(re[b])
                b += 7
            else:
                b += 1
        if count != len(ids):
            raise RuntimeError(
                f"Failure to receive all encoders, errors occurred in ID {remains}."
            )
        return ret

    def _get_state(self, ignore_error=False, **kwargs):
        """
        Receive signals from the encoders.

        Args:
        - ignore_error: bool, optional, default: False, whether to ignore the incomplete data error (if set True, then the last results will be used.)

        Returns:
        - ret: np.array, the encoder results corresponding to the ids array.
        """
        return self.get_angles(ignore_error=ignore_error, **kwargs)

    def get_state(self):
        return self._get_state(ignore_error=False)


class AirExo:
    """
    A minimalistic driver class for AirExo controller.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        ids: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
        enc_mapping: Optional[dict] = None,
        baudrate: int = 115200,
        sleep_gap: float = 0.002,
        **kwargs,
    ):
        """
        Args:
        - port: str, the port of the AirExo controller;
        """
        print("Opening AirExo device")
        self.encoder = AngleEncoder(
            ids=ids, port=port, baudrate=baudrate, sleep_gap=sleep_gap, **kwargs
        )
        self.ids = ids

        self.enc_mapping = enc_mapping

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._grasp = False
        self._reset_state = 0
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        # print_command("Right button", "reset simulation")
        # print_command("Left button (hold)", "close gripper")
        # print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        # print_command("Move mouse vertically", "move arm vertically")
        # print_command(
        #     "Twist mouse about an axis", "rotate arm about a corresponding axis"
        # )
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset control
        self._control = np.zeros(7)
        # Reset grasp
        self._grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        raise NotImplementedError

    def mapping(self, enc, x):
        use_pct = self.enc_mapping[enc].get("use_pct", 0)
        erad = self.enc_mapping[enc].get("encoder_rad", 0)
        ecenter = self.enc_mapping[enc].get("encoder_center", 180)
        rmin = self.enc_mapping[enc].get("robot_min", -180)
        rmax = self.enc_mapping[enc].get("robot_max", 180)
        if erad:
            x = rad_2_deg(x)

        rdir = self.enc_mapping[enc].get("relative_direction", 1)
        fixed = self.enc_mapping[enc].get("fixed", 0)

        if fixed:
            x = self.enc_mapping[enc].get("fixed_value", 0)
        elif use_pct:
            emin, emax = self.enc_mapping[enc].get("encoder_min", 0), self.enc_mapping[
                enc
            ].get("encoder_max", 360)
            x = np.clip(x, emin, emax)
            return rmin + (rmax - rmin) * (x - emin) / (emax - emin)
        else:
            me, mr = (
                self.enc_mapping[enc].get("encoder_mapping", 0),
                self.enc_mapping[enc].get("robot_mapping", 0),
            )
            if ecenter > 180 and x < ecenter - 180:
                x = x + 360
            if ecenter < 180 and x > ecenter + 180:
                x = 360 - x
            x = mr + (x - me) * rdir
            x = np.clip(x, rmin, rmax)
        rzc = self.enc_mapping[enc].get("robot_zero_centered", 1)
        rrad = self.enc_mapping[enc].get("robot_rad", 1)
        if rzc:
            x = deg_zero_centered(x, rmin, rmax, rdir)
        if rrad:
            x = deg_2_rad(x)
        return x

    def transform_action(self, enc_res):
        action = []
        for i in range(len(self.ids)):
            rad = self.mapping(f"enc{self.ids[i]}", enc_res[i])
            action.append(rad)
        return action

    def run(self):
        """Listener method that keeps pulling new messages."""

        while True:
            if self._enabled:
                enc_res = self.encoder.get_state()
                if enc_res is not None:
                    action = self.transform_action(enc_res)
                    self._control = action[:7]
                    self._grasp = (
                        action[7] < 0.04
                    )  # raw gripper value range in [0, 0.08]

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        return np.array([float(self._grasp)])

    def get_action(self):
        return np.concatenate([self.control, self.control_gripper])


if __name__ == "__main__":

    airexo = AirExo(port="/dev/ttyUSB0")
    for i in range(100):
        print(airexo.get_action())
        time.sleep(0.02)
