"""
Stereo capture + rectify node.

Opens both CSI cameras via jetcam (same path the recorder notebook uses) and
publishes /image_rectified (left, rectified, ROI-cropped) at `fps` Hz.

Publishes: /image_rectified (sensor_msgs/Image) — single source of truth for
           downstream nodes (yolo, lane).

Distance to the vehicle is computed from its bbox height in rover_decision —
NOT from disparity. We rectify the right image too (kept for future depth
work) but discard it at this stage.
"""
import array
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from rover_stereo.rectify import StereoCalib, StereoRectifier

# jetcam wrapper, same as record_and_label.ipynb
sys.path.insert(0, str(Path.home() / "team"))
from calibration.camera import Camera  # noqa: E402


class StereoNode(Node):
    def __init__(self):
        super().__init__("rover_stereo")
        self.declare_parameter("calib_path",
                               str(Path.home() / "team/main/ros2_ws/src/"
                                   "rover_stereo/config/stereo_calib.yaml"))
        self.declare_parameter("cam_width", 1280)
        self.declare_parameter("cam_height", 720)
        self.declare_parameter("fps", 15)

        try:
            calib = StereoCalib.load(self.get_parameter("calib_path").value)
            self.rectifier = StereoRectifier(calib)
        except Exception as e:
            self.get_logger().error(f"calibration not loaded: {e}")
            self.rectifier = None

        w = int(self.get_parameter("cam_width").value)
        h = int(self.get_parameter("cam_height").value)
        fps = int(self.get_parameter("fps").value)

        self.cam_l = Camera(sensor_id=0, capture_width=w, capture_height=h, capture_fps=fps * 2)
        self.cam_r = Camera(sensor_id=1, capture_width=w, capture_height=h, capture_fps=fps * 2)
        self.get_logger().info(f"cameras up L={self.cam_l.running()} R={self.cam_r.running()}")

        self.latest = {"L": None, "R": None}
        self.stop = False
        for cam, key in [(self.cam_l, "L"), (self.cam_r, "R")]:
            threading.Thread(target=self._reader, args=(cam, key), daemon=True).start()

        self.rect_pub = self.create_publisher(Image, "/image_rectified", 10)
        self.timer = self.create_timer(1.0 / fps, self._tick)

    def _reader(self, cam, key):
        while not self.stop:
            try:
                f = cam.read()
                if f is not None:
                    self.latest[key] = f
            except Exception:
                time.sleep(0.01)

    def _tick(self):
        if self.rectifier is None:
            return
        L, R = self.latest["L"], self.latest["R"]
        if L is None or R is None:
            return
        t0 = time.time()
        # jetcam returns RGB; convert to BGR to match downstream bgr8 expectation
        Lb = L[:, :, ::-1]
        Rb = R[:, :, ::-1]
        t1 = time.time()
        lr, _rr = self.rectifier.rectify_pair(Lb, Rb)
        t2 = time.time()
        lr_roi = self.rectifier.crop_to_roi(lr)
        t3 = time.time()
        self.rect_pub.publish(self._numpy_to_image(lr_roi))
        t4 = time.time()
        self._tick_i = getattr(self, "_tick_i", 0) + 1
        if self._tick_i % 15 == 0:
            self.get_logger().info(
                f"_tick: bgr={1000*(t1-t0):.1f}ms rectify={1000*(t2-t1):.1f}ms "
                f"crop={1000*(t3-t2):.1f}ms pub={1000*(t4-t3):.1f}ms "
                f"total={1000*(t4-t0):.1f}ms"
            )

    def _numpy_to_image(self, img: np.ndarray) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_left"
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        # Bypass sensor_msgs.Image.data setter — its Python-level per-byte
        # validation (`all(0 <= v < 256 for v in value)`) costs ~500 ms for a
        # 1280x720 BGR frame and pins publish rate at <2 Hz. Assigning via the
        # _data slot with array.array('B', ...) skips the check; rclpy still
        # serializes correctly because the field type is uint8[].
        buf = np.ascontiguousarray(img).tobytes()
        msg._data = array.array("B", buf)
        return msg

    def destroy_node(self):
        self.stop = True
        time.sleep(0.2)
        for cam in (self.cam_l, self.cam_r):
            try: cam.stop()
            except Exception: pass
        super().destroy_node()


def main():
    rclpy.init()
    node = StereoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
