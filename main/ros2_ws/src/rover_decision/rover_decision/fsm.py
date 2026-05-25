"""
Pure FSM — no ROS dependency. Unit-testable.

States: COMMON, TURNING, SLOW, WAITING, STOPPED, ARRIVED.
Inputs: stable-detection flags + BC step-done signal. Mission (left/right) is
        latched at runtime from whichever turn sign is first stably detected;
        subsequent turn-sign detections are ignored.
Outputs: next state + active-model tag.

Stop trigger taxonomy (STOPPED.entered_by):
  - "stop": stop sign → released by stop_timer_elapsed
  - "red" : red light  → released by green_light_seen OR red no longer stable
  - "turn": common BC step_done (reached end of common segment) →
            released by stop_timer_elapsed; resumes into TURNING with
            mission-branch model.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


COMMON, TURNING, SLOW, WAITING, STOPPED, ARRIVED = (
    "COMMON", "TURNING", "SLOW", "WAITING", "STOPPED", "ARRIVED",
)

ALL_MISSIONS: Tuple[str, ...] = ("left", "right")


@dataclass
class FsmInputs:
    stop_sign_stable: bool = False
    red_light_stable: bool = False
    car_stable: bool = False          # 4-frame stable car detection
    vehicle_close: bool = False       # bbox distance gate (used together with car_stable)
    left_sign_stable: bool = False
    right_sign_stable: bool = False
    lane_lost: bool = False
    green_light_seen: bool = False
    stop_timer_elapsed: bool = False
    person_stable: bool = False
    slow_timer_elapsed: bool = False
    common_grace_elapsed: bool = False  # ≥common_grace_s since launch → eligible to branch


class Fsm:
    def __init__(self, allowed_missions: Tuple[str, ...] = ALL_MISSIONS):
        # Restrict which turn-signs can latch the mission. Set this to
        # ("right",) when only the right-branch model is loaded so a stray
        # left-sign detection doesn't latch an unusable mission.
        self.allowed_missions = tuple(allowed_missions)
        self.mission: Optional[str] = None   # latched on first stable turn-sign
        self.state = COMMON
        self.prev_state = COMMON
        self.entered_by: Optional[str] = None   # what caused current STOPPED

    def step(self, inp: FsmInputs) -> str:
        # Latch mission from first stable turn-sign observation, restricted to
        # allowed_missions. Once latched, subsequent left/right detections are
        # ignored — the mission only fires once per run.
        if self.mission is None:
            if inp.left_sign_stable and "left" in self.allowed_missions:
                self.mission = "left"
            elif inp.right_sign_stable and "right" in self.allowed_missions:
                self.mission = "right"

        # Safety overrides first.
        if inp.stop_sign_stable:
            if self.state != STOPPED:
                self.prev_state = self.state
                self.state = STOPPED
                self.entered_by = "stop"
            return self.state
        if inp.red_light_stable:
            if self.state != STOPPED:
                self.prev_state = self.state
                self.state = STOPPED
                self.entered_by = "red"
            return self.state
        # Car: stable detection + close → WAITING. If stable but far,
        # pass through (state stays whatever it was).
        if inp.car_stable and inp.vehicle_close:
            if self.state != WAITING:
                self.prev_state = self.state
                self.state = WAITING
            return self.state
        # Person seen → SLOW for slow_duration_s. Lower priority than stop/car,
        # higher than normal driving transitions. Don't re-arm if already in SLOW.
        if inp.person_stable and self.state != SLOW:
            self.prev_state = self.state if self.state != SLOW else self.prev_state
            self.state = SLOW
            return self.state

        # Resume from non-normal states.
        if self.state == STOPPED:
            released = False
            if self.entered_by == "stop":
                released = inp.stop_timer_elapsed
            elif self.entered_by == "red":
                released = inp.green_light_seen or (not inp.red_light_stable)
            elif self.entered_by == "turn":
                # After the auto-stop at end-of-common, resume into TURNING
                # with the mission-branch model.
                if inp.stop_timer_elapsed:
                    self.state = TURNING
                    self.entered_by = None
                    return self.state
            if released:
                self.state = self.prev_state
                self.entered_by = None
        if self.state == WAITING and not (inp.car_stable and inp.vehicle_close):
            self.state = self.prev_state
        if self.state == SLOW and inp.slow_timer_elapsed:
            self.state = self.prev_state

        # Normal driving transitions.
        # Branch trigger: mission was latched at some earlier point (from a
        # stable left/right sign detection — possibly long before now) AND
        # the post-launch grace window has elapsed. The detection is
        # "remembered" via self.mission; we don't require the sign to still be
        # visible at branch time. The grace window prevents branching on a
        # sign detected before the rover actually starts moving.
        if (self.state == COMMON and self.mission is not None
                and inp.common_grace_elapsed):
            self.prev_state = TURNING   # restored after stop_timer
            self.state = STOPPED
            self.entered_by = "turn"
        elif self.state == TURNING and inp.lane_lost:
            self.state = ARRIVED
        return self.state

    def active_model(self) -> str:
        if self.state == TURNING and self.mission is not None:
            return self.mission  # "left" or "right"
        return "common"
