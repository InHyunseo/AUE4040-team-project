"""
Center-regression ROS wrapper.

Subscribes: /image_rectified (sensor_msgs/Image, bgr8),
            /active_model (std_msgs/String)
Publishes:  /road_center (rover_msgs/RoadCenter)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from rover_msgs.msg import RoadCenter


class LaneNode(Node):
    def __init__(self):
        super().__init__("rover_lane")
        self.declare_parameter("model_common", "models/center_common.engine")
        self.declare_parameter("model_left", "models/center_left.engine")
        self.declare_parameter("model_right", "models/center_right.engine")
        self.declare_parameter("smoothing_alpha", 0.7)

        try:
            from rover_lane.model_manager import ModelManager
            self.manager = ModelManager({
                "common": self.get_parameter("model_common").value,
                "left": self.get_parameter("model_left").value,
                "right": self.get_parameter("model_right").value,
            })
        except Exception as e:
            self.get_logger().warn(f"models not loaded: {e}")
            self.manager = None

        self.prev_x = 0.0
        self.center_pub = self.create_publisher(RoadCenter, "/road_center", 10)
        self.create_subscription(Image, "/image_rectified", self.on_image, 10)
        self.create_subscription(String, "/active_model", self.on_active, 10)

    def on_active(self, msg: String) -> None:
        if self.manager is None:
            return
        try:
            self.manager.set_active(msg.data)
        except KeyError as e:
            self.get_logger().warn(f"ignoring unknown model tag: {e}")

    def on_image(self, msg: Image) -> None:
        if self.manager is None:
            return
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warn(
                f"unexpected encoding {msg.encoding!r}; expected bgr8/rgb8")
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]

        x, y = self.manager.infer(img)

        alpha = float(self.get_parameter("smoothing_alpha").value)
        x = alpha * self.prev_x + (1.0 - alpha) * x
        self.prev_x = x

        out = RoadCenter()
        out.header = msg.header
        out.x = float(x)
        out.y = float(y)
        out.model_tag = self.manager.active
        self.center_pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(LaneNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
