"""waypoint_gt 부호/방향 단위 테스트 (bag 불필요, 합성 cmd_vel).

이 로버의 cmd_vel 관례(teleop_node / motor_bridge_node 확인):
  linear.x  < 0 = 전진,  angular.z > 0 = 우회전.
waypoint_gt 는 표준 로봇 프레임(x=forward+, y=left+)으로 정규화해야 한다:
  직진      → x>0, y≈0
  좌회전(w<0) → x>0, y>0 (좌)
  우회전(w>0) → x>0, y<0 (우)

  python3 test_waypoint_gt.py    # 또는 pytest
"""
from __future__ import annotations

import numpy as np

from extract_labels import CmdSample, waypoint_gt, WP_HORIZON_S, WP_N


def _const_cmd(v: float, w: float, dt_s: float = 0.05, pad_s: float = 0.5):
    """horizon + 여유를 덮는 일정 cmd_vel 시퀀스를 t0=0 부터 생성."""
    total = WP_HORIZON_S + pad_s
    n = int(total / dt_s) + 1
    cmds = [CmdSample(int(k * dt_s * 1e9), v, w) for k in range(n)]
    cmd_ts = [c.t_ns for c in cmds]
    return cmds, cmd_ts


def _wp(v, w):
    cmds, cmd_ts = _const_cmd(v, w)
    out = waypoint_gt(cmds, cmd_ts, t0_ns=0)
    assert out is not None, "horizon 커버리지 부족(테스트 cmd 생성 오류)"
    assert out.shape == (WP_N, 2)
    return out


def test_straight_forward():
    # 전진(linear.x<0), 직진(w=0) → x 증가(전방), y≈0
    wp = _wp(v=-0.20, w=0.0)
    xs, ys = wp[:, 0], wp[:, 1]
    assert (xs > 0).all(), f"전진인데 x가 전방(+)이 아님: {xs}"
    assert np.all(np.diff(xs) > 0), f"x가 단조 증가해야 함(전진): {xs}"
    assert np.allclose(ys, 0, atol=1e-6), f"직진인데 y가 0이 아님: {ys}"
    assert wp[-1, 0] > 0.3, f"2.5s 전진 끝점이 너무 짧음: {wp[-1,0]}"


def test_left_turn():
    # 이 로버: angular.z<0 = 좌회전 → y>0(좌)
    wp = _wp(v=-0.20, w=-0.8)
    assert (wp[:, 0] > 0).all(), f"전진인데 x가 전방(+)이 아님: {wp[:,0]}"
    assert wp[-1, 1] > 0, f"좌회전인데 끝점 y가 좌(+)가 아님: {wp[-1,1]}"


def test_right_turn():
    # angular.z>0 = 우회전 → y<0(우)
    wp = _wp(v=-0.20, w=+0.8)
    assert (wp[:, 0] > 0).all(), f"전진인데 x가 전방(+)이 아님: {wp[:,0]}"
    assert wp[-1, 1] < 0, f"우회전인데 끝점 y가 우(-)가 아님: {wp[-1,1]}"


def test_left_right_mirror():
    # 좌/우 회전은 같은 |w| 면 y 부호만 반대, x는 동일(대칭)
    wl = _wp(v=-0.20, w=-0.8)
    wr = _wp(v=-0.20, w=+0.8)
    assert np.allclose(wl[:, 0], wr[:, 0], atol=1e-6), "좌/우 x가 대칭이 아님"
    assert np.allclose(wl[:, 1], -wr[:, 1], atol=1e-6), "좌/우 y가 반대부호 대칭이 아님"


def test_insufficient_horizon_returns_none():
    # horizon 을 못 덮으면 None
    cmds = [CmdSample(0, -0.2, 0.0), CmdSample(int(0.1 * 1e9), -0.2, 0.0)]
    cmd_ts = [c.t_ns for c in cmds]
    assert waypoint_gt(cmds, cmd_ts, t0_ns=0) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all waypoint_gt sign tests passed")
