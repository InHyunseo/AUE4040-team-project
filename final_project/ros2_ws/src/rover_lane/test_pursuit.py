"""waypoint 조향 모드 부호/대칭 단위 테스트 (ROS 불필요).

부호 규약(중요): 이 로버는 angular.z(=steer) **양수가 우회전**(teleop d키 /
motor_bridge mix). 우회전 목표는 waypoint y<0(우). 따라서 모든 모드가
  steer = -gain · (곡률 또는 heading)
로 마이너스를 붙여야 GT steer 와 같은 방향. 빠뜨리면 코너/회피에서 반대로
꺾어 사고나므로 이 테스트로 고정한다.

  python3 test_pursuit.py
"""
from __future__ import annotations

import math
import numpy as np


def waypoint_steer(wp, mode, lookahead_idx, gain, idx_lo=2, idx_hi=4):
    """e2e_infer_node.RoverE2EInfer.waypoint_steer 와 동일 로직(검증용 복제)."""
    if wp is None or len(wp) == 0:
        return 0.0
    n = len(wp)
    lo = max(0, min(int(idx_lo), n - 1))
    hi = max(lo, min(int(idx_hi), n - 1))
    i = max(0, min(int(lookahead_idx), n - 1))
    if mode == "pursuit":
        gx, gy = float(wp[i, 0]), float(wp[i, 1])
        L2 = gx * gx + gy * gy
        if L2 < 1e-6 or gx <= 0.0:
            return 0.0
        raw = 2.0 * gy / L2
    elif mode == "heading":
        gx, gy = float(wp[i, 0]), float(wp[i, 1])
        if gx <= 0.0:
            return 0.0
        raw = math.atan2(gy, gx)
    elif mode == "max_y":
        seg = wp[lo:hi + 1]
        j = int(np.argmax(np.abs(seg[:, 1])))
        gx, gy = float(seg[j, 0]), float(seg[j, 1])
        if gx <= 0.0:
            return 0.0
        raw = math.atan2(gy, gx)
    elif mode == "mean":
        seg = wp[lo:hi + 1]
        gx, gy = float(seg[:, 0].mean()), float(seg[:, 1].mean())
        if gx <= 0.0:
            return 0.0
        raw = math.atan2(gy, gx)
    else:
        return 0.0
    return float(max(-1.0, min(1.0, -gain * raw)))


MODES = ("pursuit", "heading", "max_y", "mean")


def _wp(ys):
    return np.array([[0.1 * (i + 1), ys[i]] for i in range(len(ys))], np.float32)


def test_straight_zero_all_modes():
    for m in MODES:
        assert abs(waypoint_steer(_wp([0, 0, 0, 0, 0]), m, 3, 0.5)) < 1e-6, m


def test_right_turn_positive_all_modes():
    wp = _wp([-0.02, -0.08, -0.18, -0.30, -0.44])   # 우회전 = y<0
    for m in MODES:
        assert waypoint_steer(wp, m, 3, 0.5) > 0, m


def test_left_turn_negative_all_modes():
    wp = _wp([0.02, 0.08, 0.18, 0.30, 0.44])         # 좌회전 = y>0
    for m in MODES:
        assert waypoint_steer(wp, m, 3, 0.5) < 0, m


def test_symmetry_all_modes():
    l = _wp([0.02, 0.08, 0.18, 0.30, 0.44])
    r = _wp([-0.02, -0.08, -0.18, -0.30, -0.44])
    for m in MODES:
        assert abs(waypoint_steer(l, m, 3, 0.5) + waypoint_steer(r, m, 3, 0.5)) < 1e-6, m


def test_clamp_and_safety():
    for m in MODES:
        assert -1.0 <= waypoint_steer(_wp([0, 0, 0, 0.5, 0.5]), m, 3, 10.0) <= 1.0, m
        back = np.array([[-0.1, 0.2]] * 5, np.float32)
        assert waypoint_steer(back, m, 3, 0.5) == 0.0, m
        assert waypoint_steer(None, m, 3, 0.5) == 0.0, m


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all waypoint_steer mode tests passed")
