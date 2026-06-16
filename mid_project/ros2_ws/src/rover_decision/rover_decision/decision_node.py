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
        # Person bbox-height threshold (px) on /image_rectified frame. SLOW
        # triggers only when person_stable AND person bbox h ≥ this. Stops the
        # rover from braking for distant pedestrian signs across the room.
        self.declare_parameter("person_close_h_px", 200.0)
        self.declare_parameter("det_score_min", 0.4)
        # Stop duration before resuming (used for stop sign + turn auto-stop).
        self.declare_parameter("stop_duration_s", 2.0)
        # Person → SLOW duration
        self.declare_parameter("slow_duration_s", 3.0)
        # Restrict which turn-signs may latch the mission. Use ["right"] when
        # only the right model is trained/loaded so a stray left-sign detection
        # doesn't latch an unusable mission.
        self.declare_parameter("allowed_missions", ["left", "right"])
        # Per-label cooldown (seconds). Once a label triggers its action, the
        # same label is suppressed for this long even if still stably detected.
        # Prevents one stop sign / red light / person / car from firing twice
        # as the rover lingers near it.
        self.declare_parameter("label_cooldown_s", 10.0)
        # Minimum seconds the rover must drive in COMMON before any branch
        # transition is allowed. The mission may latch from a sign detection
        # earlier than this; the latch is remembered, but STOPPED→TURNING
        # waits until the grace window elapses. Guards against branching on a
        # sign that was visible at launch / before the rover started moving.
        self.declare_parameter("common_grace_s", 8.0)

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
        self._log_i = 0
        self._last_state = None
        # Per-label cooldown bookkeeping: label -> wall time when its cooldown
        # expires. While now < that time, force `stable[label] = False`.
        self._cooldown_until: dict[str, float] = {}
        # Red light is intentionally NOT in cooldown — we want to keep
        # STOPPED for as long as red is visible, not auto-release after N s.
        self._cooldown_labels = ("stop", "person", "car")
        # Wall time when COMMON started — used to gate branch transitions.
        self._common_started_at: float = time.time()
        # Latched once /branch_done fires (cleared on STATE leaving TURNING).
        self._branch_done = False

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.model_pub = self.create_publisher(String, "/active_model", 10)
        self.fsm_pub = self.create_publisher(FSMState, "/fsm_state", 10)
        self.create_subscription(DetectionArray, "/detections", self.on_detections, 10)
        self.create_subscription(Twist, "/bc_cmd", self.on_bc_cmd, 10)
        self.create_subscription(Bool, "/branch_done", self.on_branch_done, 10)

    def on_branch_done(self, msg: Bool) -> None:
        if msg.data:
            self._branch_done = True

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
        person_h_px = 0.0
        turn_best = None  # (class_name, score)
        for d in self.last_dets:
            if d.score < score_min:
                continue
            h = max(0.0, float(d.y2) - float(d.y1))
            if d.class_name in ("left", "right"):
                # Per-class NMS can stamp BOTH left and right boxes on the same
                # physical sign. Keep the highest-score one per frame so the
                # stabilizer doesn't latch the wrong mission.
                if turn_best is None or d.score > turn_best[1]:
                    turn_best = (d.class_name, d.score)
                continue
            present[d.class_name] = True
            if d.class_name == "car" and h > vehicle_h_px:
                vehicle_h_px = h
            elif d.class_name == "person" and h > person_h_px:
                person_h_px = h
        if turn_best is not None:
            present[turn_best[0]] = True

        stable = self.stab.update(present)

        # While the BC is actively cornering (TURNING + large |steer|), drop
        # ALL perception triggers. A one-frame person/car/sign detection
        # mid-corner would yank the FSM into SLOW/WAITING, which momentarily
        # flips active_model back to common and (when TURNING resumes) makes
        # lane_node re-trigger its branch_init freeze — visible in logs as
        # repeated "branch entry: freezing" lines during a single right turn.
        cornering_steer = 0.5
        if (self.fsm.state == "TURNING"
                and abs(float(msg.angular.z)) >= cornering_steer):
            stable = {k: False for k in stable}

        # Per-label cooldown: once a label has triggered its FSM action we
        # suppress it for label_cooldown_s so the same sign / light / person
        # doesn't fire twice while we linger. Turn signs are NOT in this set
        # (mission latches once and ignores further detections on its own).
        now = time.time()
        cooldown_s = float(self.get_parameter("label_cooldown_s").value)
        for lbl in self._cooldown_labels:
            if not stable.get(lbl, False):
                continue
            until = self._cooldown_until.get(lbl, 0.0)
            if now < until:
                # In cooldown — pretend not stable so FSM ignores it.
                stable[lbl] = False
            else:
                # First stable frame in a while — arm the cooldown so further
                # frames in this trigger window get suppressed.
                self._cooldown_until[lbl] = now + cooldown_s

        left_stable = bool(stable.get("left", False))
        right_stable = bool(stable.get("right", False))

        # Common-mode grace: branch only after the rover has been driving in
        # COMMON for at least common_grace_s seconds.
        common_grace_s = float(self.get_parameter("common_grace_s").value)
        common_grace_elapsed = (now - self._common_started_at) >= common_grace_s

        v_close = vehicle_close(
            vehicle_h_px,
            self.get_parameter("vehicle_dist_K").value,
            self.get_parameter("safe_dist_m").value,
        )
        person_close = person_h_px >= float(
            self.get_parameter("person_close_h_px").value
        )

        # Timer bookkeeping: arm when entering STOPPED/SLOW, fire when duration elapses.
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
            # person triggers SLOW only when both stable AND close (bbox big).
            person_stable=bool(stable.get("person", False)) and person_close,
            slow_timer_elapsed=slow_elapsed,
            common_grace_elapsed=common_grace_elapsed,
            branch_step_done=self._branch_done,
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

        # Stream decision to the launch console: every tick if state changed,
        # otherwise every 10th tick so we still see live cmd values.
        self._log_i += 1
        state_changed = state != self._last_state
        self._last_state = state
        if state_changed or self._log_i % 10 == 0:
            seen = ",".join(k for k, v in stable.items() if v) or "-"
            # Show TURNING with its mission inline (TURNING_RIGHT/_LEFT) so the
            # branch direction is visible at a glance, without changing the
            # underlying FSM state string used by other code.
            display_state = (
                f"TURNING_{self.fsm.mission.upper()}"
                if state == "TURNING" and self.fsm.mission else state
            )
            slow_age = (now - self.slow_since) if self.slow_since else -1
            self.get_logger().info(
                f"state={display_state} prev={self.fsm.prev_state} "
                f"mission={self.fsm.mission or '-'} "
                f"model={self.fsm.active_model()} "
                f"cmd v={msg.linear.x:+.3f} w={msg.angular.z:+.3f} "
                f"stable={seen} v_close={v_close} "
                f"grace={common_grace_elapsed} "
                f"slow_age={slow_age:.1f} slow_elapsed={slow_elapsed}"
            )


def main():
    rclpy.init()
    rclpy.spin(DecisionNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
