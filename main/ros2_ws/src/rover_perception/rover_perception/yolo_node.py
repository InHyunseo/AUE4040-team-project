"""
YOLO ROS wrapper.

Subscribes: /image_rectified (sensor_msgs/Image) — left rectified, from rover_stereo
Publishes:  /detections (rover_msgs/DetectionArray)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class YoloNode(Node):
    def __init__(self):
        super().__init__("rover_perception")
        self.declare_parameter("engine_path", "models/yolov8n.engine")
        self.declare_parameter("conf_threshold", 0.4)
        self.declare_parameter("iou_threshold", 0.5)

        try:
            from rover_perception.yolo_inference import YoloInference
            self.engine = YoloInference(
                self.get_parameter("engine_path").value,
                self.get_parameter("conf_threshold").value,
                self.get_parameter("iou_threshold").value,
            )
        except Exception as e:
            self.get_logger().warn(f"YOLO engine not loaded: {e}")
            self.engine = None

        self.create_subscription(Image, "/image_rectified", self.on_image, 10)
        # Publisher hooked up once rover_msgs is built.

    def on_image(self, msg: Image) -> None:
        if self.engine is None:
            return
        # cv_bridge-free numpy view (bgr8 expected)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        dets = self.engine.infer(img)
        self.get_logger().debug(f"detections: {len(dets)}")


def main():
    rclpy.init()
    rclpy.spin(YoloNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
