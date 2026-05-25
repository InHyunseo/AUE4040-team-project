"""
One-shot calibration for vehicle bbox-height distance.

Closed-world: a single fixed-size car. We model distance as
    d = K / bbox_h_px        with K = bbox_h_px * d  (px * m)

This script subscribes to `/image_rectified` so the calibration is captured
in the EXACT coordinate system used at runtime (rectified + ROI-cropped).
Reading raw camera and clicking on it would yield a K that doesn't match
runtime bbox sizes — they get rectified+cropped before YOLO sees them.

Procedure:
  1. Run the autonomous stack OR rover_stereo standalone so /image_rectified
     is being published.
  2. Park the car at a known distance `d` in front of the camera.
  3. Run this script. It grabs one frame from /image_rectified and shows it.
  4. Click top-left of the car's bbox, then bottom-right.
  5. Script prints K and (with --write) updates rover_bringup params.yaml.

Usage:
    python3 calibrate_vehicle_distance.py --distance 0.5
    python3 calibrate_vehicle_distance.py --distance 0.5 --write
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


PARAMS_YAML = (Path.home() / "Personal_Research" / "AUE4040" / "main" /
               "ros2_ws" / "src" / "rover_bringup" / "config" / "params.yaml")


class FrameGrabber(Node):
    def __init__(self):
        super().__init__("calibrate_grabber")
        self.frame = None
        self.create_subscription(Image, "/image_rectified", self._on_img, 10)

    def _on_img(self, msg: Image) -> None:
        if self.frame is not None:
            return
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warn(f"unexpected encoding {msg.encoding}")
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]
        self.frame = img.copy()


def grab_frame(timeout_s: float = 10.0) -> np.ndarray:
    rclpy.init()
    node = FrameGrabber()
    deadline = time.time() + timeout_s
    while rclpy.ok() and node.frame is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()
    if frame is None:
        raise RuntimeError("no frame received on /image_rectified within "
                           f"{timeout_s}s — is rover_stereo running?")
    return frame


def pick_bbox(frame_bgr) -> tuple:
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    win = "click TL, then BR of the car's bbox  (q=abort)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_click)
    while True:
        disp = frame_bgr.copy()
        for pt in clicks:
            cv2.circle(disp, pt, 4, (0, 255, 0), -1)
        if len(clicks) == 2:
            cv2.rectangle(disp, clicks[0], clicks[1], (0, 255, 0), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if len(clicks) >= 2:
            cv2.imshow(win, disp)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            return clicks[0], clicks[1]


def update_params_yaml(K: float) -> None:
    text = PARAMS_YAML.read_text()
    new_lines = []
    written = False
    for line in text.splitlines():
        if line.strip().startswith("vehicle_dist_K:"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}vehicle_dist_K: {K:.2f}")
            written = True
        else:
            new_lines.append(line)
    if not written:
        raise RuntimeError(f"vehicle_dist_K key not found in {PARAMS_YAML}")
    PARAMS_YAML.write_text("\n".join(new_lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", required=True, type=float,
                    help="known distance to the car in meters")
    ap.add_argument("--write", action="store_true",
                    help="update vehicle_dist_K in rover_bringup params.yaml")
    args = ap.parse_args()

    frame = grab_frame()
    tl, br = pick_bbox(frame)
    bbox_h = abs(br[1] - tl[1])
    if bbox_h < 1:
        raise SystemExit("degenerate bbox height")
    K = bbox_h * args.distance
    print(f"bbox_h = {bbox_h} px  (rectified+ROI-cropped coords)")
    print(f"d      = {args.distance} m")
    print(f"K      = bbox_h * d = {K:.2f}  (px * m)")
    print(f"=> at safe_dist=0.4 m, threshold bbox_h = {K / 0.4:.1f} px")

    if args.write:
        update_params_yaml(K)
        print(f"wrote vehicle_dist_K = {K:.2f} to {PARAMS_YAML}")


if __name__ == "__main__":
    main()
