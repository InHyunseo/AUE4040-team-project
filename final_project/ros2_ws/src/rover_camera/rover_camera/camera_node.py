"""Dual CSI camera publisher.

Publishes two raw monocular streams (jetcam wrapper at
/home/ircv16/team/calibration/camera):
  /lane_image/compressed   sensor_msgs/CompressedImage  — sensor 0 (lane-seg head)
  /front_image/compressed  sensor_msgs/CompressedImage  — sensor 1 (object-detection head)

Each camera runs its own reader thread; the timer publishes the latest frame
at `fps` Hz, JPEG-encoded by OpenCV (quality 85). Color is BGR on the wire —
we read jetcam's native BGR frame (Camera.read_bgr()) and encode it as-is,
with no color conversion in the hot path.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

sys.path.insert(0, "/home/ircv16/team")
from calibration.camera import Camera  # noqa: E402


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("rover_camera")
        self.declare_parameter("lane_sensor_id", 0)
        self.declare_parameter("front_sensor_id", 1)
        self.declare_parameter("cam_width", 1280)
        self.declare_parameter("cam_height", 720)
        self.declare_parameter("fps", 15)
        self.declare_parameter("jpeg_quality", 85)

        lane_id  = int(self.get_parameter("lane_sensor_id").value)
        front_id = int(self.get_parameter("front_sensor_id").value)
        w        = int(self.get_parameter("cam_width").value)
        h        = int(self.get_parameter("cam_height").value)
        fps      = int(self.get_parameter("fps").value)
        self.jpeg_q = int(self.get_parameter("jpeg_quality").value)

        # capture_fps is set 2x publish fps so jetcam's internal buffer is fresh.
        self.cam_lane  = Camera(sensor_id=lane_id,  capture_width=w, capture_height=h, capture_fps=fps * 2)
        self.cam_front = Camera(sensor_id=front_id, capture_width=w, capture_height=h, capture_fps=fps * 2)
        self.get_logger().info(
            f"cameras up lane(sensor_id={lane_id})={self.cam_lane.running()} "
            f"front(sensor_id={front_id})={self.cam_front.running()}"
        )

        self._latest: dict[str, np.ndarray | None] = {"lane": None, "front": None}
        self._stop = False
        for cam, key in [(self.cam_lane, "lane"), (self.cam_front, "front")]:
            threading.Thread(target=self._reader, args=(cam, key), daemon=True).start()

        self.pub_lane  = self.create_publisher(CompressedImage, "/lane_image/compressed",  10)
        self.pub_front = self.create_publisher(CompressedImage, "/front_image/compressed", 10)
        self.timer = self.create_timer(1.0 / fps, self._tick)
        self._tick_i = 0

    def _reader(self, cam: Camera, key: str) -> None:
        # read_bgr() returns the raw BGR frame straight from jetcam (its
        # GStreamer pipeline already ends in BGR). We JPEG-encode BGR below,
        # so reading BGR avoids a wasteful BGR->RGB->BGR round-trip per frame.
        while not self._stop:
            try:
                f = cam.read_bgr()
                if f is not None:
                    self._latest[key] = f
            except Exception:
                time.sleep(0.01)

    def _encode(self, bgr: np.ndarray) -> CompressedImage:
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = jpg.tobytes()
        return msg

    def _tick(self) -> None:
        self._tick_i += 1
        for key, pub, frame_id in [("lane",  self.pub_lane,  "lane_camera"),
                                   ("front", self.pub_front, "front_camera")]:
            frame = self._latest[key]
            if frame is None:
                continue
            msg = self._encode(frame)
            msg.header.frame_id = frame_id
            pub.publish(msg)
        if self._tick_i % 30 == 0:
            self.get_logger().info(
                f"published lane={self._latest['lane'] is not None} "
                f"front={self._latest['front'] is not None}"
            )

    def destroy_node(self) -> bool:
        self._stop = True
        time.sleep(0.2)
        for cam in (self.cam_lane, self.cam_front):
            try: cam.stop()
            except Exception: pass
            try: cam._cam.cap.release()
            except Exception: pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
