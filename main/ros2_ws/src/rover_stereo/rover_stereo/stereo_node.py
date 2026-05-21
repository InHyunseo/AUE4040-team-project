"""
Stereo rectify node.

Subscribes: /camera/left/image_raw, /camera/right/image_raw (sensor_msgs/Image)
Publishes:  /image_rectified  (sensor_msgs/Image)  — left rectified, ROI-cropped

The rectified-left stream is the *single source of truth* for downstream
nodes (yolo, lane, recorder). They must not consume /camera/*/image_raw.

Distance to the vehicle is computed from its bbox height in rover_decision —
NOT from disparity. We rectify the right image too (kept for future depth
work) but discard it at this stage.
"""
from pathlib import Path

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

from rover_stereo.rectify import StereoCalib, StereoRectifier


class StereoNode(Node):
    def __init__(self):
        super().__init__("rover_stereo")
        self.declare_parameter("calib_path",
                               str(Path.home() / "team/main/ros2_ws/src/"
                                   "rover_stereo/config/stereo_calib.yaml"))
        self.declare_parameter("sync_slop_s", 0.03)

        try:
            calib = StereoCalib.load(self.get_parameter("calib_path").value)
            self.rectifier = StereoRectifier(calib)
        except Exception as e:
            self.get_logger().warn(f"calibration not loaded: {e}")
            self.rectifier = None

        self.rect_pub = self.create_publisher(Image, "/image_rectified", 10)

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
        lr, _rr = self.rectifier.rectify_pair(left, right)
        lr_roi = self.rectifier.crop_to_roi(lr)
        self.rect_pub.publish(self._numpy_to_image(lr_roi, left_msg.header))


def main():
    rclpy.init()
    rclpy.spin(StereoNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
