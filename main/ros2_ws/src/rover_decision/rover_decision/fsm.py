"""
Pure FSM — no ROS dependency. Unit-testable.

States: COMMON, TURNING, SLOW, WAITING, STOPPED, ARRIVED.
Inputs: stable-detection flags. Mission (left/right) is latched at runtime
        from whichever turn sign is first stably detected.
Outputs: next state + active-model tag.
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
    vehicle_close: bool = False
    left_sign_stable: bool = False
    right_sign_stable: bool = False
    turn_sign_close: bool = False     # bbox big enough → reached the sign
    lane_lost: bool = False
    green_light_seen: bool = False
    stop_timer_elapsed: bool = False
    person_stable: bool = False
    slow_timer_elapsed: bool = False


class Fsm:
    def __init__(self, allowed_missions: Tuple[str, ...] = ALL_MISSIONS):
        # Restrict which turn-signs can latch the mission. Set this to
        # ("right",) when only the right-branch model is loaded so a stray
        # left-sign detection doesn't latch an unusable mission.
        self.allowed_missions = tuple(allowed_missions)
        self.mission: Optional[str] = None   # latched on first stable turn-sign
        self.state = COMMON
        self.prev_state = COMMON

    def step(self, inp: FsmInputs) -> str:
        # Latch mission from first stable turn-sign observation, restricted to
        # allowed_missions. If both fire in the same tick (shouldn't happen
        # with stabilizer thresholds), left wins per caller-side tiebreak.
        if self.mission is None:
            if inp.left_sign_stable and "left" in self.allowed_missions:
                self.mission = "left"
            elif inp.right_sign_stable and "right" in self.allowed_missions:
                self.mission = "right"

        turn_sign_stable = (
            (self.mission == "left" and inp.left_sign_stable)
            or (self.mission == "right" and inp.right_sign_stable)
        )

        # Safety overrides first.
        if inp.stop_sign_stable or inp.red_light_stable:
            if self.state != STOPPED:
                self.prev_state = self.state
                self.state = STOPPED
            return self.state
        if inp.vehicle_close:
            if self.state != WAITING:
                self.prev_state = self.state
                self.state = WAITING
            return self.state
        # Person seen → SLOW for slow_duration_s. Lower priority than stop/vehicle,
        # higher than normal driving transitions. Don't re-arm if already in SLOW.
        if inp.person_stable and self.state != SLOW:
            self.prev_state = self.state if self.state not in (SLOW,) else self.prev_state
            self.state = SLOW
            return self.state

        # Resume from non-normal states.
        if self.state == STOPPED and (inp.green_light_seen or inp.stop_timer_elapsed):
            self.state = self.prev_state
        if self.state == WAITING and not inp.vehicle_close:
            self.state = self.prev_state
        if self.state == SLOW and inp.slow_timer_elapsed:
            self.state = self.prev_state

        # Normal driving transitions.
        if self.state == COMMON and turn_sign_stable and inp.turn_sign_close:
            # Reached the turn sign: stop, swap model on resume.
            self.prev_state = TURNING
            self.state = STOPPED
        elif self.state == TURNING and inp.lane_lost:
            self.state = ARRIVED
        return self.state

    def active_model(self) -> str:
        if self.state == TURNING and self.mission is not None:
            return self.mission  # "left" or "right"
        return "common"
