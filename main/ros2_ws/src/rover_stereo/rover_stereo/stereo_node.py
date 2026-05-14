"""
Stereo node.

Subscribes: /camera/left/image_raw, /camera/right/image_raw (sensor_msgs/Image)
Publishes:  /image_rectified  (sensor_msgs/Image)  — left rectified, ROI-cropped
            /vehicle_distance (std_msgs/Float32)   — meters; inf if no vehicle

The rectified-left stream is the *single source of truth* for downstream
nodes (yolo, pilotnet, recorder). They must not consume /camera/*/image_raw.
"""
from pathlib import Path

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from rover_stereo.rectify import StereoCalib, StereoRectifier, disparity_to_distance


class StereoNode(Node):
    def __init__(self):
        super().__init__("rover_stereo")
        self.declare_parameter("calib_path",
                               str(Path.home() / "team/main/ros2_ws/src/"
                                   "rover_stereo/config/stereo_calib.yaml"))
        self.declare_parameter("baseline_m", 0.06)
        self.declare_parameter("sync_slop_s", 0.03)

        try:
            calib = StereoCalib.load(self.get_parameter("calib_path").value)
            self.rectifier = StereoRectifier(calib)
            self.fx = float(calib.P1[0, 0]) if calib.P1 is not None else float(calib.K1[0, 0])
        except Exception as e:
            self.get_logger().warn(f"calibration not loaded: {e}")
            self.rectifier = None
            self.fx = 0.0

        # StereoBM is the cheapest matcher; tune block size + numDisparities in-situ.
        self.matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        self.rect_pub = self.create_publisher(Image, "/image_rectified", 10)
        self.dist_pub = self.create_publisher(Float32, "/vehicle_distance", 10)

        left_sub = Subscriber(self, Image, "/camera/left/image_raw")
        right_sub = Subscriber(self, Image, "/camera/right/image_raw")
        self.sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10,
            slop=self.get_parameter("sync_slop_s").value,
        )
        self.sync.registerCallback(self.on_pair)

    @staticmethod
    def _image_to_numpy(msg: Image) -> np.ndarray:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    @staticmethod
    def _numpy_to_image(img: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        return msg

    def on_pair(self, left_msg: Image, right_msg: Image) -> None:
        if self.rectifier is None:
            return
        left = self._image_to_numpy(left_msg)
        right = self._image_to_numpy(right_msg)
        lr, rr = self.rectifier.rectify_pair(left, right)
        lr_roi = self.rectifier.crop_to_roi(lr)
        rr_roi = self.rectifier.crop_to_roi(rr)

        self.rect_pub.publish(self._numpy_to_image(lr_roi, left_msg.header))

        # Distance: median disparity in a centered ROI around the assumed vehicle.
        gray_l = cv2.cvtColor(lr_roi, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rr_roi, cv2.COLOR_BGR2GRAY)
        disp = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        h, w = disp.shape
        cy, cx = h // 2, w // 2
        win = disp[cy - 40:cy + 40, cx - 60:cx + 60]
        valid = win[win > 0.0]
        if valid.size > 200:
            dist = disparity_to_distance(
                float(np.median(valid)),
                self.fx,
                self.get_parameter("baseline_m").value,
            )
        else:
            dist = float("inf")
        self.dist_pub.publish(Float32(data=float(dist)))


def main():
    rclpy.init()
    rclpy.spin(StereoNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
