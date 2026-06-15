"""pure pursuit 조향 부호/스케일 단위 테스트 (ROS 불필요).

부호 규약(중요): 이 로버는 angular.z(=steer) **양수가 우회전**(teleop d키 /
motor_bridge mix). 우회전 목표는 waypoint y<0(우). 따라서
  steer = -gain · (2·gy/L²)
가 GT steer 와 같은 방향(실측 상관 r=+0.79). 부호를 빠뜨리면 코너에서 반대로
꺾어 사고나므로 반드시 이 테스트로 고정한다.

  python3 test_pursuit.py
"""
from __future__ import annotations

import numpy as np


def pursuit_steer(wp, lookahead_idx, gain):
    """e2e_infer_node.RoverE2EInfer.pursuit_steer 와 동일 로직(검증용 복제)."""
    if wp is None or len(wp) == 0:
        return 0.0
    i = max(0, min(int(lookahead_idx), len(wp) - 1))
    gx, gy = float(wp[i, 0]), float(wp[i, 1])
    L2 = gx * gx + gy * gy
    if L2 < 1e-6 or gx <= 0.0:
        return 0.0
    kappa = 2.0 * gy / L2
    return float(max(-1.0, min(1.0, -gain * kappa)))


def _wp(ys):
    return np.array([[0.1 * (i + 1), ys[i]] for i in range(len(ys))], np.float32)


def test_straight_zero():
    assert abs(pursuit_steer(_wp([0, 0, 0, 0, 0]), 3, 0.25)) < 1e-6


def test_right_turn_positive():
    # 우회전 = wp.y<0 → steer 양수 (이 로버 규약: +가 우)
    wp = _wp([-0.02, -0.08, -0.18, -0.30, -0.44])
    assert pursuit_steer(wp, 3, 0.25) > 0


def test_left_turn_negative():
    # 좌회전 = wp.y>0 → steer 음수
    wp = _wp([0.02, 0.08, 0.18, 0.30, 0.44])
    assert pursuit_steer(wp, 3, 0.25) < 0


def test_symmetry():
    wl = pursuit_steer(_wp([0.02, 0.08, 0.18, 0.30, 0.44]), 3, 0.25)
    wr = pursuit_steer(_wp([-0.02, -0.08, -0.18, -0.30, -0.44]), 3, 0.25)
    assert abs(wl + wr) < 1e-6


def test_clamp_and_backward_safe():
    # 큰 곡률 + 큰 gain → [-1,1] clamp
    assert -1.0 <= pursuit_steer(_wp([0, 0, 0, 0.5, 0]), 3, 10.0) <= 1.0
    # 전방거리 0 / 후방 목표 → 0
    back = np.array([[-0.1, 0.2]] * 5, np.float32)
    assert pursuit_steer(back, 3, 0.25) == 0.0
    assert pursuit_steer(None, 3, 0.25) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all pursuit tests passed")
