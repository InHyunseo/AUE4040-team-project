"""
Decision node — wires perception + lane -> /cmd_vel + active model tag.

Subscribes: /detections (rover_msgs/DetectionArray)
            /road_center (rover_msgs/RoadCenter)
Publishes:  /cmd_vel (geometry_msgs/Twist)
            /active_model (std_msgs/String)
            /fsm_state (rover_msgs/FSMState)

Distance to the vehicle comes from its bbox height — see safety.vehicle_close.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from rover_msgs.msg import DetectionArray, RoadCenter, FSMState

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
        self.declare_parameter("stable_frames_turn", 5)
        self.declare_parameter("safe_dist_m", 0.4)
        self.declare_parameter("vehicle_dist_K", 180.0)   # K = bbox_h * d (px*m)
        self.declare_parameter("det_score_min", 0.4)

        mission = self.get_parameter("mission").value
        self.fsm = Fsm(mission)
        self.stab = Stabilizer({
            "stop_sign": self.get_parameter("stable_frames_sign").value,
            "traffic_light_red": self.get_parameter("stable_frames_light").value,
            "traffic_light_green": self.get_parameter("stable_frames_light").value,
            "turn_left_sign": self.get_parameter("stable_frames_turn").value,
            "turn_right_sign": self.get_parameter("stable_frames_turn").value,
        })
        self.last_dets: list = []

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.model_pub = self.create_publisher(String, "/active_model", 10)
        self.fsm_pub = self.create_publisher(FSMState, "/fsm_state", 10)
        self.create_subscription(DetectionArray, "/detections", self.on_detections, 10)
        self.create_subscription(RoadCenter, "/road_center", self.on_road_center, 10)

    def on_detections(self, msg: DetectionArray) -> None:
        self.last_dets = list(msg.detections)

    def on_road_center(self, msg: RoadCenter) -> None:
        score_min = float(self.get_parameter("det_score_min").value)

        present = {}
        vehicle_h_px = 0.0
        for d in self.last_dets:
            if d.score < score_min:
                continue
            present[d.class_name] = True
            if d.class_name == "vehicle":
                h = max(0.0, float(d.y2) - float(d.y1))
                if h > vehicle_h_px:
                    vehicle_h_px = h

        stable = self.stab.update(present)

        # Turn-sign trigger: only fires when the stable detection matches mission.
        # If the wrong turn sign is stable we log + ignore.
        mission = self.fsm.mission
        want_class = f"turn_{mission}_sign"
        other_class = "turn_right_sign" if mission == "left" else "turn_left_sign"
        turn_stable = bool(stable.get(want_class, False))
        if stable.get(other_class, False) and not turn_stable:
            self.get_logger().warn(
                f"saw {other_class} but mission={mission}; ignoring trigger")

        v_close = vehicle_close(
            vehicle_h_px,
            self.get_parameter("vehicle_dist_K").value,
            self.get_parameter("safe_dist_m").value,
        )

        state = self.fsm.step(FsmInputs(
            stop_sign_stable=bool(stable.get("stop_sign", False)),
            red_light_stable=bool(stable.get("traffic_light_red", False)),
            green_light_seen=bool(stable.get("traffic_light_green", False)),
            turn_sign_stable=turn_stable,
            vehicle_close=v_close,
        ))

        steering = center_to_steering(
            float(msg.x), self.get_parameter("steering_gain_k").value,
        )
        twist = Twist()
        twist.linear.x = 1.0   # control_node scales to throttle
        twist.angular.z = float(steering)
        self.cmd_pub.publish(twist)
        self.model_pub.publish(String(data=self.fsm.active_model()))

        fsm_msg = FSMState()
        fsm_msg.header = msg.header
        fsm_msg.state = state
        fsm_msg.mission = mission
        fsm_msg.active_model = self.fsm.active_model()
        self.fsm_pub.publish(fsm_msg)


def main():
    rclpy.init()
    rclpy.spin(DecisionNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
