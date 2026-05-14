"""
Center-regression ROS wrapper.

Subscribes: /image_rectified (left rectified from rover_stereo),
            /active_model (std_msgs/String)
Publishes:  /road_center (rover_msgs/RoadCenter)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class PilotnetNode(Node):
    def __init__(self):
        super().__init__("rover_pilotnet")
        self.declare_parameter("model_common", "models/center_common.engine")
        self.declare_parameter("model_left", "models/center_left.engine")
        self.declare_parameter("model_right", "models/center_right.engine")
        self.declare_parameter("smoothing_alpha", 0.7)

        try:
            from rover_pilotnet.model_manager import ModelManager
            self.manager = ModelManager({
                "common": self.get_parameter("model_common").value,
                "left": self.get_parameter("model_left").value,
                "right": self.get_parameter("model_right").value,
            })
        except Exception as e:
            self.get_logger().warn(f"models not loaded: {e}")
            self.manager = None

        self.prev_x = 0.0
        self.create_subscription(Image, "/image_rectified", self.on_image, 10)
        self.create_subscription(String, "/active_model", self.on_active, 10)

    def on_active(self, msg: String) -> None:
        if self.manager is not None:
            self.manager.set_active(msg.data)

    def on_image(self, msg: Image) -> None:
        if self.manager is None:
            return
        # TODO: numpy view of msg.data, call manager.infer, EMA-smooth, publish.


def main():
    rclpy.init()
    rclpy.spin(PilotnetNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
