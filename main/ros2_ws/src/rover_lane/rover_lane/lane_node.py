"""
E2E BC ROS wrapper.

Subscribes: /image_rectified (sensor_msgs/Image, bgr8/rgb8),
            /active_model (std_msgs/String)
Publishes:  /bc_cmd (geometry_msgs/Twist) — linear.x = speed, angular.z = steer
            (decision_node gates this into /cmd_vel)
            /common_done (std_msgs/Bool) — True once the common BC engine's
            internal step counter has reached step_max. Latched True from that
            point on; decision_node uses this to trigger the auto-stop and
            mission-branch swap.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist


class LaneNode(Node):
    def __init__(self):
        super().__init__("rover_lane")
        self.declare_parameter("model_common", "models/e2e_common.engine")
        self.declare_parameter("model_left", "models/e2e_left.engine")
        self.declare_parameter("model_right", "models/e2e_right.engine")
        # 0 = no smoothing (notebook teleop doesn't smooth either). With
        # action-classification BC, smoothing kills sparse LEFT/RIGHT spikes
        # before they reach the wheels — turns visibly "jerky and short."
        self.declare_parameter("smoothing_alpha", 0.0)
        # speed boost during turns: speed *= (1 + k * |steer|). 0 disables.
        self.declare_parameter("turn_speed_boost", 0.0)
        # multiplicative gains for model output before publishing
        self.declare_parameter("steer_gain", 1.0)
        self.declare_parameter("speed_gain", 1.0)
        # Branch-entry freeze: zero out steer AND speed for the first N frames
        # after switching to a left/right model. The training sessions begin
        # with ~2s of "drive straight + accelerate" before the human pressed
        # left/right, so the model copies that and delays the turn. We freeze
        # the rover in place for that window so it doesn't roll past the turn.
        self.declare_parameter("branch_init_frames", 30)

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
        self._common_done_sent = False
        # Counts frames since the most recent switch to a left/right model.
        # Reset to 0 in on_active when the tag changes to a branch tag.
        self._branch_frames_since_switch = None  # None = not in branch mode
        self.cmd_pub = self.create_publisher(Twist, "/bc_cmd", 10)
        self.done_pub = self.create_publisher(Bool, "/common_done", 10)
        self.create_subscription(Image, "/image_rectified", self.on_image, 10)
        self.create_subscription(String, "/active_model", self.on_active, 10)

    def on_active(self, msg: String) -> None:
        if self.manager is None:
            return
        try:
            prev = self.manager.active
            self.manager.set_active(msg.data)
        except KeyError as e:
            self.get_logger().warn(f"ignoring unknown model tag: {e}")
            return
        # Detect first switch INTO a branch model and arm the freeze.
        if msg.data in ("left", "right") and prev != msg.data:
            self._branch_frames_since_switch = 0
            self.get_logger().info(
                f"branch entry: freezing (steer=0, speed=0) for "
                f"{self.get_parameter('branch_init_frames').value} frames"
            )

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

        # Branch-entry freeze: hold the rover still for the first N frames
        # after a branch. Bypasses the model's initial "drive straight" output
        # so we don't roll past the turn before the model decides to turn.
        if self._branch_frames_since_switch is not None:
            n_max = int(self.get_parameter("branch_init_frames").value)
            if self._branch_frames_since_switch < n_max:
                steer = 0.0
                speed = 0.0
                self._branch_frames_since_switch += 1
            else:
                self._branch_frames_since_switch = None  # release control

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

        # Signal end-of-common only while the common model is active. Once sent,
        # don't spam — decision_node latches the value. Cleared on next run.
        if (not self._common_done_sent
                and self.manager.active == "common"
                and self.manager.step_done()):
            self.done_pub.publish(Bool(data=True))
            self._common_done_sent = True
            self.get_logger().info("common BC step_done — published /common_done")


def main():
    rclpy.init()
    rclpy.spin(LaneNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
