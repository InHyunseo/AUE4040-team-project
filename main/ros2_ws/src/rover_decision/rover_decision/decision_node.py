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
        self.declare_parameter("stable_frames_sign", 4)
        self.declare_parameter("stable_frames_light", 4)
        self.declare_parameter("stable_frames_turn", 5)
        self.declare_parameter("stable_frames_person", 4)
        self.declare_parameter("safe_dist_m", 0.4)
        self.declare_parameter("vehicle_dist_K", 180.0)   # K = bbox_h * d (px*m)
        self.declare_parameter("det_score_min", 0.4)
        # Turn-sign "reached" threshold: bbox h_px ≥ this → trigger stop+swap
        self.declare_parameter("turn_sign_close_h_px", 80.0)
        # Stop duration before resuming into TURNING state
        self.declare_parameter("stop_duration_s", 2.0)
        # Person → SLOW duration
        self.declare_parameter("slow_duration_s", 3.0)

        # Class names follow best.pt training: car/green/left/person/red/right/stop.
        # Mission (left/right) is latched at runtime by the FSM from whichever
        # turn-sign first stabilizes — the course direction isn't known ahead.
        self.fsm = Fsm()
        self.stab = Stabilizer({
            "stop": self.get_parameter("stable_frames_sign").value,
            "red": self.get_parameter("stable_frames_light").value,
            "green": self.get_parameter("stable_frames_light").value,
            "left": self.get_parameter("stable_frames_turn").value,
            "right": self.get_parameter("stable_frames_turn").value,
            "person": self.get_parameter("stable_frames_person").value,
        })
        self.last_dets: list = []
        self.stopped_since: float | None = None
        self.slow_since: float | None = None

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
        turn_sign_h_px = {"left": 0.0, "right": 0.0}
        for d in self.last_dets:
            if d.score < score_min:
                continue
            present[d.class_name] = True
            h = max(0.0, float(d.y2) - float(d.y1))
            if d.class_name == "car" and h > vehicle_h_px:
                vehicle_h_px = h
            elif d.class_name in turn_sign_h_px and h > turn_sign_h_px[d.class_name]:
                turn_sign_h_px[d.class_name] = h

        stable = self.stab.update(present)

        left_stable = bool(stable.get("left", False))
        right_stable = bool(stable.get("right", False))

        v_close = vehicle_close(
            vehicle_h_px,
            self.get_parameter("vehicle_dist_K").value,
            self.get_parameter("safe_dist_m").value,
        )

        close_h = float(self.get_parameter("turn_sign_close_h_px").value)
        # "Reached the turn sign" — once mission is latched, only that side's
        # bbox counts; before latch, neither side is close yet.
        if self.fsm.mission == "left":
            turn_close = turn_sign_h_px["left"] >= close_h
        elif self.fsm.mission == "right":
            turn_close = turn_sign_h_px["right"] >= close_h
        else:
            turn_close = False

        # Timer bookkeeping: arm when entering STOPPED/SLOW, fire when duration elapses.
        now = time.time()
        was_stopped = self.fsm.state == "STOPPED"
        was_slow = self.fsm.state == "SLOW"
        stop_elapsed = False
        slow_elapsed = False
        if was_stopped and self.stopped_since is not None:
            stop_elapsed = (now - self.stopped_since) >= float(
                self.get_parameter("stop_duration_s").value
            )
        if was_slow and self.slow_since is not None:
            slow_elapsed = (now - self.slow_since) >= float(
                self.get_parameter("slow_duration_s").value
            )

        state = self.fsm.step(FsmInputs(
            stop_sign_stable=bool(stable.get("stop", False)),
            red_light_stable=bool(stable.get("red", False)),
            green_light_seen=bool(stable.get("green", False)),
            left_sign_stable=left_stable,
            right_sign_stable=right_stable,
            turn_sign_close=turn_close,
            vehicle_close=v_close,
            stop_timer_elapsed=stop_elapsed,
            person_stable=bool(stable.get("person", False)),
            slow_timer_elapsed=slow_elapsed,
        ))

        if state == "STOPPED" and self.stopped_since is None:
            self.stopped_since = now
        elif state != "STOPPED":
            self.stopped_since = None
        if state == "SLOW" and self.slow_since is None:
            self.slow_since = now
        elif state != "SLOW":
            self.slow_since = None

        twist = Twist()
        twist.linear.x = float(msg.linear.x)   # model's speed; control_node zeros it in SAFE states
        twist.angular.z = float(msg.angular.z)
        self.cmd_pub.publish(twist)
        self.model_pub.publish(String(data=self.fsm.active_model()))

        fsm_msg = FSMState()
        fsm_msg.header.stamp = self.get_clock().now().to_msg()
        fsm_msg.state = state
        fsm_msg.mission = self.fsm.mission if self.fsm.mission is not None else ""
        fsm_msg.active_model = self.fsm.active_model()
        self.fsm_pub.publish(fsm_msg)


def main():
    rclpy.init()
    rclpy.spin(DecisionNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
