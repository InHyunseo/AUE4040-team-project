"""Dual CSI camera publisher.

Publishes two raw monocular streams (jetcam wrapper from repo `calibration/camera`):
  /lane_image/compressed   sensor_msgs/CompressedImage  — sensor 0 (lane-seg head)
  /front_image/compressed  sensor_msgs/CompressedImage  — sensor 1 (object-detection head)

Each camera runs its own reader thread; the timer publishes the latest frame
at `fps` Hz, JPEG-encoded by OpenCV (quality 85). Color is BGR on the wire —
we read jetcam's native BGR frame (Camera.read_bgr()) and encode it as-is,
with no color conversion in the hot path.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

from rover_common.constants import FRONT_IMAGE_TOPIC, LANE_IMAGE_TOPIC
from rover_common.image_io import encode_bgr
from rover_common.paths import ensure_repo_on_path
from rover_common.qos import IMAGE_PUB_QOS

ensure_repo_on_path(__file__)
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

        # 각 키마다 (frame, capture_stamp) 를 함께 보관한다. capture_stamp 는
        # reader 스레드가 프레임을 잡은 직후 찍은 ROS 시각으로, publish 시
        # header.stamp 에 그대로 넣어 두 카메라 간 정합 기준을 '캡처 시각'으로
        # 만든다(인코딩/송신 순서 지연이 stamp 에 누적되지 않게).
        self._latest: dict[str, tuple[np.ndarray, "rclpy.time.Time"] | None] = {
            "lane": None, "front": None}
        self._stop = False
        for cam, key in [(self.cam_lane, "lane"), (self.cam_front, "front")]:
            threading.Thread(target=self._reader, args=(cam, key), daemon=True).start()

        self.pub_lane  = self.create_publisher(CompressedImage, LANE_IMAGE_TOPIC,  IMAGE_PUB_QOS)
        self.pub_front = self.create_publisher(CompressedImage, FRONT_IMAGE_TOPIC, IMAGE_PUB_QOS)
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
                    # 프레임을 받은 '직후'의 시각이 캡처 시각의 최선의 근사다.
                    self._latest[key] = (f, self.get_clock().now())
            except Exception:
                time.sleep(0.01)

    def _tick(self) -> None:
        self._tick_i += 1
        for key, pub, frame_id in [("lane",  self.pub_lane,  "lane_camera"),
                                   ("front", self.pub_front, "front_camera")]:
            latest = self._latest[key]
            if latest is None:
                continue
            frame, stamp = latest
            # 캡처 시각을 그대로 싣는다(publish/인코딩 시각이 아니라). 추출 단계가
            # 이 stamp 로 lane↔front 를 매칭하면 두 카메라 정합이 캡처 기준이 된다.
            header = Header(stamp=stamp.to_msg(), frame_id=frame_id)
            pub.publish(encode_bgr(frame, header, quality=self.jpeg_q))
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
