"""
E2E BC ROS wrapper.

Subscribes: /image_rectified (sensor_msgs/Image, bgr8/rgb8),
            /active_model (std_msgs/String)
Publishes:  /bc_cmd (geometry_msgs/Twist) — linear.x = speed, angular.z = steer
            (decision_node gates this into /cmd_vel)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class LaneNode(Node):
    def __init__(self):
        super().__init__("rover_lane")
        self.declare_parameter("model_common", "models/e2e_common.engine")
        self.declare_parameter("model_left", "models/e2e_left.engine")
        self.declare_parameter("model_right", "models/e2e_right.engine")
        self.declare_parameter("smoothing_alpha", 0.7)
        # speed boost during turns: speed *= (1 + k * |steer|). 0 disables.
        self.declare_parameter("turn_speed_boost", 0.0)
        # multiplicative gains for model output before publishing
        self.declare_parameter("steer_gain", 1.0)
        self.declare_parameter("speed_gain", 1.0)

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

        self.prev_steer = 0.0
        self.cmd_pub = self.create_publisher(Twist, "/bc_cmd", 10)
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

        steer, speed = self.manager.infer(img)

        steer *= float(self.get_parameter("steer_gain").value)
        speed *= float(self.get_parameter("speed_gain").value)

        alpha = float(self.get_parameter("smoothing_alpha").value)
        steer = alpha * self.prev_steer + (1.0 - alpha) * steer
        self.prev_steer = steer

        boost = float(self.get_parameter("turn_speed_boost").value)
        if boost > 0.0:
            speed = speed * (1.0 + boost * abs(steer))

        out = Twist()
        out.linear.x = float(speed)
        out.angular.z = float(steer)
        self.cmd_pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(LaneNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
