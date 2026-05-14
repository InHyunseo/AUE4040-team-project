"""
Decision node — wires perception + pilotnet -> /cmd_vel + active model tag.

Subscribes: /detections, /road_center
Publishes:  /cmd_vel, /active_model, /fsm_state
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String

from rover_decision.fsm import Fsm, FsmInputs
from rover_decision.safety import Stabilizer, vehicle_close
from rover_decision.steering import center_to_steering


class DecisionNode(Node):
    def __init__(self):
        super().__init__("rover_decision")
        self.declare_parameter("mission", "left")
        self.declare_parameter("steering_gain_k", 1.2)
        self.declare_parameter("stable_frames_sign", 4)
        self.declare_parameter("stable_frames_light", 4)
        self.declare_parameter("stable_frames_roundabout", 5)
        self.declare_parameter("safe_dist_m", 0.4)
        self.declare_parameter("vehicle_dist_K", 180.0)   # fallback K (px*m)

        self.fsm = Fsm(self.get_parameter("mission").value)
        self.last_vehicle_dist_m = float("inf")
        self.stab = Stabilizer({
            "stop_sign": self.get_parameter("stable_frames_sign").value,
            "traffic_light_red": self.get_parameter("stable_frames_light").value,
            "traffic_light_green": self.get_parameter("stable_frames_light").value,
            "roundabout_sign": self.get_parameter("stable_frames_roundabout").value,
        })

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.model_pub = self.create_publisher(String, "/active_model", 10)
        self.create_subscription(Float32, "/vehicle_distance", self.on_vehicle_dist, 10)
        # /detections + /road_center subscriptions wired after rover_msgs build.

    def on_vehicle_dist(self, msg: Float32) -> None:
        self.last_vehicle_dist_m = float(msg.data)

    # Placeholder tick — real version is driven by the /road_center callback.
    def tick(self, x_center: float, present_classes: dict,
             vehicle_bbox_h_px: float = 0.0) -> None:
        stable = self.stab.update(present_classes)
        safe_dist = self.get_parameter("safe_dist_m").value
        v_close = self.last_vehicle_dist_m < safe_dist
        if not v_close and vehicle_bbox_h_px > 0.0:
            # Fallback when stereo distance is unavailable / inf.
            v_close = vehicle_close(
                vehicle_bbox_h_px,
                self.get_parameter("vehicle_dist_K").value,
                safe_dist,
            )
        state = self.fsm.step(FsmInputs(
            stop_sign_stable=stable.get("stop_sign", False),
            red_light_stable=stable.get("traffic_light_red", False),
            green_light_seen=stable.get("traffic_light_green", False),
            roundabout_trigger=stable.get("roundabout_sign", False),
            vehicle_close=v_close,
        ))
        steering = center_to_steering(
            x_center, self.get_parameter("steering_gain_k").value,
        )
        twist = Twist()
        twist.linear.x = 1.0   # placeholder — control_node scales to throttle
        twist.angular.z = steering
        self.cmd_pub.publish(twist)
        self.model_pub.publish(String(data=self.fsm.active_model()))
        self.get_logger().debug(f"state={state} steer={steering:.2f}")


def main():
    rclpy.init()
    rclpy.spin(DecisionNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
