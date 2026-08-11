"""Live preview for the front and wrist RealSense cameras."""

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_FRONT_SERIAL = "327122071654"
DEFAULT_WRIST_SERIAL = "001622071252"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show front and wrist RealSense RGB streams in one window."
    )
    parser.add_argument("--front-camera-serial", default=DEFAULT_FRONT_SERIAL)
    parser.add_argument("--wrist-camera-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
    )
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument(
        "--front-rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
    )
    parser.add_argument(
        "--wrist-rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
    )
    return parser.parse_args()


def connected_realsense_devices():
    devices = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        devices[serial] = name
    return devices


def make_color_config(serial, width, height, fps):
    config = rs.config()
    config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    return config


def try_read_color(pipeline, timeout_ms=100):
    try:
        success, frames = pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
    except RuntimeError:
        return None
    if not success:
        return None
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data()).copy()


def rotate_image(image, degrees):
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def add_label(image, camera_name, serial, fps):
    output = image.copy()
    text = f"{camera_name} | {serial} | {fps:.1f} FPS"
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 520), 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def resize_to_height(image, target_height):
    if image.shape[0] == target_height:
        return image
    scale = target_height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def resize_to_width(image, target_width):
    if image.shape[1] == target_width:
        return image
    scale = target_width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, height), interpolation=cv2.INTER_AREA)


def combine_frames(front, wrist, layout):
    if layout == "vertical":
        target_width = min(front.shape[1], wrist.shape[1])
        return cv2.vconcat(
            [
                resize_to_width(front, target_width),
                resize_to_width(wrist, target_width),
            ]
        )

    target_height = min(front.shape[0], wrist.shape[0])
    return cv2.hconcat(
        [
            resize_to_height(front, target_height),
            resize_to_height(wrist, target_height),
        ]
    )


class FrameRateCounter:
    def __init__(self):
        self.value = 0.0
        self.frame_count = 0
        self.started_at = time.monotonic()

    def update(self):
        self.frame_count += 1
        now = time.monotonic()
        elapsed = now - self.started_at
        if elapsed >= 1.0:
            self.value = self.frame_count / elapsed
            self.frame_count = 0
            self.started_at = now


def main():
    args = parse_args()
    if args.front_camera_serial == args.wrist_camera_serial:
        raise ValueError("front and wrist camera serials must be different")
    if args.display_scale <= 0:
        raise ValueError("--display-scale must be greater than zero")

    devices = connected_realsense_devices()
    print("Connected RealSense devices:")
    if devices:
        for serial, name in devices.items():
            print(f"  {serial}: {name}")
    else:
        print("  none")

    missing = [
        f"front={args.front_camera_serial}"
        if args.front_camera_serial not in devices
        else None,
        f"wrist={args.wrist_camera_serial}"
        if args.wrist_camera_serial not in devices
        else None,
    ]
    missing = [item for item in missing if item is not None]
    if missing:
        raise SystemExit("Required RealSense camera not found: " + ", ".join(missing))

    front_pipeline = rs.pipeline()
    wrist_pipeline = rs.pipeline()
    front_started = False
    wrist_started = False
    window_name = "Front + Wrist RealSense Preview"

    try:
        front_pipeline.start(
            make_color_config(
                args.front_camera_serial,
                args.width,
                args.height,
                args.fps,
            )
        )
        front_started = True
        wrist_pipeline.start(
            make_color_config(
                args.wrist_camera_serial,
                args.width,
                args.height,
                args.fps,
            )
        )
        wrist_started = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        front_fps = FrameRateCounter()
        wrist_fps = FrameRateCounter()
        latest_front = None
        latest_wrist = None

        print("Preview started. Press q or Esc in the preview window to quit.")
        while True:
            front = try_read_color(front_pipeline)
            wrist = try_read_color(wrist_pipeline)
            if front is not None:
                latest_front = front
                front_fps.update()
            if wrist is not None:
                latest_wrist = wrist
                wrist_fps.update()
            if latest_front is None or latest_wrist is None:
                continue

            front_view = add_label(
                rotate_image(latest_front, args.front_rotate),
                "FRONT",
                args.front_camera_serial,
                front_fps.value,
            )
            wrist_view = add_label(
                rotate_image(latest_wrist, args.wrist_rotate),
                "WRIST",
                args.wrist_camera_serial,
                wrist_fps.value,
            )
            preview = combine_frames(front_view, wrist_view, args.layout)
            if args.display_scale != 1.0:
                preview = cv2.resize(
                    preview,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if wrist_started:
            wrist_pipeline.stop()
        if front_started:
            front_pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
