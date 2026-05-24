"""
Decision node — wires perception + E2E BC -> /cmd_vel + active model tag.

Subscribes: /detections (rover_msgs/DetectionArray)
            /bc_cmd (geometry_msgs/Twist) — raw model (steer, speed)
Publishes:  /cmd_vel (geometry_msgs/Twist) — gated by FSM + safety
            /active_model (std_msgs/String)
            /fsm_state (rover_msgs/FSMState)

Distance to the vehicle comes from its bbox height — see safety.vehicle_close.
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from rover_msgs.msg import DetectionArray, FSMState

from rover_decision.fsm import Fsm, FsmInputs
from rover_decision.safety import Stabilizer, vehicle_close


class DecisionNode(Node):
    def __init__(self):
        super().__init__("rover_decision")
        self.declare_parameter("mission", "left")
        self.declare_parameter("stable_frames_sign", 4)
        self.declare_parameter("stable_frames_light", 4)
        self.declare_parameter("stable_frames_turn", 5)
        self.declare_parameter("safe_dist_m", 0.4)
        self.declare_parameter("vehicle_dist_K", 180.0)   # K = bbox_h * d (px*m)
        self.declare_parameter("det_score_min", 0.4)
        # Turn-sign "reached" threshold: bbox h_px ≥ this → trigger stop+swap
        self.declare_parameter("turn_sign_close_h_px", 80.0)
        # Stop duration before resuming into TURNING state
        self.declare_parameter("stop_duration_s", 2.0)

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
        self.stopped_since: float | None = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.model_pub = self.create_publisher(String, "/active_model", 10)
        self.fsm_pub = self.create_publisher(FSMState, "/fsm_state", 10)
        self.create_subscription(DetectionArray, "/detections", self.on_detections, 10)
        self.create_subscription(Twist, "/bc_cmd", self.on_bc_cmd, 10)

    def on_detections(self, msg: DetectionArray) -> None:
        self.last_dets = list(msg.detections)

    def on_bc_cmd(self, msg: Twist) -> None:
        score_min = float(self.get_parameter("det_score_min").value)

        present = {}
        vehicle_h_px = 0.0
        turn_sign_h_px = {"turn_left_sign": 0.0, "turn_right_sign": 0.0}
        for d in self.last_dets:
            if d.score < score_min:
                continue
            present[d.class_name] = True
            h = max(0.0, float(d.y2) - float(d.y1))
            if d.class_name == "vehicle" and h > vehicle_h_px:
                vehicle_h_px = h
            elif d.class_name in turn_sign_h_px and h > turn_sign_h_px[d.class_name]:
                turn_sign_h_px[d.class_name] = h

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

        close_h = float(self.get_parameter("turn_sign_close_h_px").value)
        turn_close = turn_sign_h_px[want_class] >= close_h

        # Stop-timer bookkeeping: arms when we enter STOPPED, fires after stop_duration_s.
        now = time.time()
        was_stopped = self.fsm.state == "STOPPED"
        timer_elapsed = False
        if was_stopped and self.stopped_since is not None:
            timer_elapsed = (now - self.stopped_since) >= float(
                self.get_parameter("stop_duration_s").value
            )

        state = self.fsm.step(FsmInputs(
            stop_sign_stable=bool(stable.get("stop_sign", False)),
            red_light_stable=bool(stable.get("traffic_light_red", False)),
            green_light_seen=bool(stable.get("traffic_light_green", False)),
            turn_sign_stable=turn_stable,
            turn_sign_close=turn_close,
            vehicle_close=v_close,
            stop_timer_elapsed=timer_elapsed,
        ))

        if state == "STOPPED" and self.stopped_since is None:
            self.stopped_since = now
        elif state != "STOPPED":
            self.stopped_since = None

        twist = Twist()
        twist.linear.x = float(msg.linear.x)   # model's speed; control_node zeros it in SAFE states
        twist.angular.z = float(msg.angular.z)
        self.cmd_pub.publish(twist)
        self.model_pub.publish(String(data=self.fsm.active_model()))

        fsm_msg = FSMState()
        fsm_msg.header.stamp = self.get_clock().now().to_msg()
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
