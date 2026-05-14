"""
Pure FSM — no ROS dependency. Unit-testable.

States: COMMON, TURNING, WAITING, STOPPED, ARRIVED.
Inputs: stable-detection flags + mission tag.
Outputs: next state + active-model tag.
"""
from dataclasses import dataclass


COMMON, TURNING, WAITING, STOPPED, ARRIVED = (
    "COMMON", "TURNING", "WAITING", "STOPPED", "ARRIVED",
)


@dataclass
class FsmInputs:
    stop_sign_stable: bool = False
    red_light_stable: bool = False
    vehicle_close: bool = False
    roundabout_trigger: bool = False
    lane_lost: bool = False
    green_light_seen: bool = False
    stop_timer_elapsed: bool = False


class Fsm:
    def __init__(self, mission: str):
        assert mission in ("left", "right")
        self.mission = mission
        self.state = COMMON
        self.prev_state = COMMON

    def step(self, inp: FsmInputs) -> str:
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

        # Resume from safety states.
        if self.state == STOPPED and (inp.green_light_seen or inp.stop_timer_elapsed):
            self.state = self.prev_state
        if self.state == WAITING and not inp.vehicle_close:
            self.state = self.prev_state

        # Normal driving transitions.
        if self.state == COMMON and inp.roundabout_trigger:
            self.state = TURNING
        elif self.state == TURNING and inp.lane_lost:
            self.state = ARRIVED
        return self.state

    def active_model(self) -> str:
        if self.state == TURNING:
            return self.mission  # "left" or "right"
        return "common"
