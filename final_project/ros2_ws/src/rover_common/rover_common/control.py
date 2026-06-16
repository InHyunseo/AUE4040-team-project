"""Shared control-contract helpers.

The teleop node and the E2E inference node independently re-implemented the same
steer→cmd_vel mapping and the same low-pass smoothing. Keeping them here means
the verified vehicle behavior lives in exactly one place; if BASE_V/TURN_V/
MAX_OMEGA ever change (constants.py), both paths follow without drift.

Sign contract (see constants.py / extract_labels.py):
  linear.x  < 0  = forward
  angular.z > 0  = right turn
  steer ∈ [-1, 1]: |steer| sets speed (coupled), steer*MAX_OMEGA sets yaw.
"""
from __future__ import annotations

from .constants import BASE_V, MAX_OMEGA, TURN_V


def steer_to_cmd_vel(steer: float) -> tuple[float, float]:
    """Normalized steer ∈ [-1, 1] → (linear.x, angular.z).

    Speed is coupled to |steer| (teleop never used a separate throttle output):
      linear.x  = -(BASE_V + |steer| * (TURN_V - BASE_V))   # neg = forward
      angular.z = steer * MAX_OMEGA
    Used by both teleop (steer = sign·TURN_FRAC[level]) and inference
    (steer = ControlHead/pursuit output), so the published cmd_vel distribution
    is identical across data collection and autonomous driving.
    """
    steer = max(-1.0, min(1.0, steer))
    lin = -(BASE_V + abs(steer) * (TURN_V - BASE_V))
    ang = steer * MAX_OMEGA
    return lin, ang


def approach(cur: float, target: float, alpha: float) -> float:
    """One step of first-order low-pass toward target (exponential smoothing).

    Replayed at TICK_HZ on both teleop and the inference watchdog so the motor
    sees a steady, smoothed command rather than per-frame jumps.
    """
    return cur + (target - cur) * alpha
