"""
YOLO ROS wrapper.

Subscribes: /image_rectified (sensor_msgs/Image) — left rectified, from rover_stereo
Publishes:  /detections (rover_msgs/DetectionArray)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from rover_msgs.msg import Detection, DetectionArray


class YoloNode(Node):
    def __init__(self):
        super().__init__("rover_perception")
        self.declare_parameter("engine_path", "models/yolov8n.engine")
        self.declare_parameter("conf_threshold", 0.4)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("detect_every_n", 2)
        self.declare_parameter("class_names", [
            "traffic_light_red", "traffic_light_green", "traffic_light_yellow",
            "stop_sign", "vehicle", "turn_left_sign", "turn_right_sign",
        ])

        self._frame_i = 0

        try:
            from rover_perception.yolo_inference import YoloInference
            self.engine = YoloInference(
                self.get_parameter("engine_path").value,
                self.get_parameter("conf_threshold").value,
                self.get_parameter("iou_threshold").value,
                self.get_parameter("class_names").value,
            )
        except Exception as e:
            self.get_logger().warn(f"YOLO engine not loaded: {e}")
            self.engine = None

        self.det_pub = self.create_publisher(DetectionArray, "/detections", 10)
        self.create_subscription(Image, "/image_rectified", self.on_image, 10)

    def on_image(self, msg: Image) -> None:
        if self.engine is None:
            return
        self._frame_i += 1
        n = int(self.get_parameter("detect_every_n").value)
        if n > 1 and (self._frame_i % n) != 0:
            return

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        dets = self.engine.infer(img)

        out = DetectionArray()
        out.header = msg.header
        for d in dets:
            det = Detection()
            det.header = msg.header
            det.class_id = int(d.class_id)
            det.class_name = str(d.class_name)
            det.score = float(d.score)
            det.x1 = float(d.x1)
            det.y1 = float(d.y1)
            det.x2 = float(d.x2)
            det.y2 = float(d.y2)
            out.detections.append(det)
        self.det_pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(YoloNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
