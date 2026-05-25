"""
Decision node — wires perception + E2E BC -> /cmd_vel + active model tag.

Subscribes: /detections (rover_msgs/DetectionArray)
            /bc_cmd (geometry_msgs/Twist) — raw model (steer, speed)
            /common_done (std_msgs/Bool) — common BC reached step_max
Publishes:  /cmd_vel (geometry_msgs/Twist) — gated by FSM + safety
            /active_model (std_msgs/String)
            /fsm_state (rover_msgs/FSMState)

Distance to the vehicle comes from its bbox height — see safety.vehicle_close.
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

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
        self.declare_parameter("stable_frames_vehicle", 4)
        self.declare_parameter("safe_dist_m", 0.4)
        self.declare_parameter("vehicle_dist_K", 180.0)   # K = bbox_h * d (px*m)
        self.declare_parameter("det_score_min", 0.4)
        # Stop duration before resuming (used for stop sign + turn auto-stop).
        self.declare_parameter("stop_duration_s", 2.0)
        # Person → SLOW duration
        self.declare_parameter("slow_duration_s", 3.0)
        # Restrict which turn-signs may latch the mission. Use ["right"] when
        # only the right model is trained/loaded so a stray left-sign detection
        # doesn't latch an unusable mission.
        self.declare_parameter("allowed_missions", ["left", "right"])

        # Class names follow best.pt training: car/green/left/person/red/right/stop.
        # Mission (left/right) is latched at runtime by the FSM from whichever
        # turn-sign first stabilizes — the course direction isn't known ahead.
        allowed = tuple(self.get_parameter("allowed_missions").value)
        self.fsm = Fsm(allowed_missions=allowed)
        self.stab = Stabilizer({
            "stop": self.get_parameter("stable_frames_sign").value,
            "red": self.get_parameter("stable_frames_light").value,
            "green": self.get_parameter("stable_frames_light").value,
            "left": self.get_parameter("stable_frames_turn").value,
            "right": self.get_parameter("stable_frames_turn").value,
            "person": self.get_parameter("stable_frames_person").value,
            "car": self.get_parameter("stable_frames_vehicle").value,
        })
        self.last_dets: list = []
        self.stopped_since: float | None = None
        self.slow_since: float | None = None
        self._common_done = False
        self._log_i = 0
        self._last_state = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.model_pub = self.create_publisher(String, "/active_model", 10)
        self.fsm_pub = self.create_publisher(FSMState, "/fsm_state", 10)
        self.create_subscription(DetectionArray, "/detections", self.on_detections, 10)
        self.create_subscription(Twist, "/bc_cmd", self.on_bc_cmd, 10)
        self.create_subscription(Bool, "/common_done", self.on_common_done, 10)

    def on_common_done(self, msg: Bool) -> None:
        # Latches True; cleared once we leave COMMON (handled below).
        if msg.data:
            self._common_done = True

    def on_detections(self, msg: DetectionArray) -> None:
        self.last_dets = list(msg.detections)

    def on_bc_cmd(self, msg: Twist) -> None:
        score_min = float(self.get_parameter("det_score_min").value)

        # The model often draws both "left" and "right" boxes on the SAME
        # physical turn sign at near-identical coords (NMS is per-class). Pick
        # the higher-score one per frame so the stabilizer doesn't accumulate
        # both counters in parallel and latch the wrong mission.
        present = {}
        vehicle_h_px = 0.0
        turn_best = None  # (class_name, score)
        for d in self.last_dets:
            if d.score < score_min:
                continue
            if d.class_name in ("left", "right"):
                if turn_best is None or d.score > turn_best[1]:
                    turn_best = (d.class_name, d.score)
                continue
            present[d.class_name] = True
            h = max(0.0, float(d.y2) - float(d.y1))
            if d.class_name == "car" and h > vehicle_h_px:
                vehicle_h_px = h
        if turn_best is not None:
            present[turn_best[0]] = True

        stable = self.stab.update(present)

        left_stable = bool(stable.get("left", False))
        right_stable = bool(stable.get("right", False))

        v_close = vehicle_close(
            vehicle_h_px,
            self.get_parameter("vehicle_dist_K").value,
            self.get_parameter("safe_dist_m").value,
        )

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
            car_stable=bool(stable.get("car", False)),
            vehicle_close=v_close,
            stop_timer_elapsed=stop_elapsed,
            person_stable=bool(stable.get("person", False)),
            slow_timer_elapsed=slow_elapsed,
            common_step_done=self._common_done,
        ))

        if state == "STOPPED" and self.stopped_since is None:
            self.stopped_since = now
        elif state != "STOPPED":
            self.stopped_since = None
        if state == "SLOW" and self.slow_since is None:
            self.slow_since = now
        elif state != "SLOW":
            self.slow_since = None
        # Once we've consumed common_step_done into STOPPED(entered_by="turn"),
        # clear the latch so it doesn't keep re-triggering.
        if state != "COMMON":
            self._common_done = False

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

        # Stream decision to the launch console: every tick if state changed,
        # otherwise every 10th tick so we still see live cmd values.
        self._log_i += 1
        state_changed = state != self._last_state
        self._last_state = state
        if state_changed or self._log_i % 10 == 0:
            seen = ",".join(k for k, v in stable.items() if v) or "-"
            self.get_logger().info(
                f"state={state} mission={self.fsm.mission or '-'} "
                f"model={self.fsm.active_model()} "
                f"cmd v={msg.linear.x:+.3f} w={msg.angular.z:+.3f} "
                f"stable={seen} v_close={v_close} common_done={self._common_done}"
            )


def main():
    rclpy.init()
    rclpy.spin(DecisionNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
