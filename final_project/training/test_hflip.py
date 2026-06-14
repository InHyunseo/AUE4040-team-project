"""hflip_sample 부호/채널교환/대칭 단위 테스트 (bag·torch 불필요).

flip 은 라벨 6종(seg 채널교환 포함)을 일관 반전해야 하고, 하나라도 틀리면
학습이 조용히 망가진다(loss 는 돌아감). flip(flip(x))==x 와 방향성을 검증한다.

  python3 test_hflip.py
"""
from __future__ import annotations

import numpy as np

# dataset 은 torch 의존이라, 검증 대상 함수만 가볍게 재현하지 않고 직접 import.
# (torch 가 없는 환경이면 ImportError → 그 환경에선 학습도 못 하므로 무방.)
from dataset import hflip_sample


def _sample():
    rng = np.random.default_rng(0)
    lane = rng.integers(0, 255, (224, 224, 3), np.uint8)
    front = rng.integers(0, 255, (224, 224, 3), np.uint8)
    seg = np.zeros((3, 224, 224), np.uint8)
    seg[0, :, 10:20] = 255    # 좌실선(ch0): 왼쪽
    seg[1, :, 200:210] = 255  # 우실선(ch1): 오른쪽
    seg[2, 100:110, :] = 255  # 중앙(ch2)
    det = np.array([30, 50, 40, 60, 0.9], np.float32)   # 좌측 bbox
    steer = np.float32(0.7)
    thr = np.float32(-0.22)
    wp = np.array([[0.1 * i, 0.05 * i] for i in range(1, 6)], np.float32)
    return lane, front, seg, det, steer, thr, wp


def test_involution():
    s = _sample()
    a = hflip_sample(*s)
    b = hflip_sample(*a)
    assert np.array_equal(b[0], s[0]) and np.array_equal(b[1], s[1])
    assert np.array_equal(b[2], s[2])            # seg
    assert np.allclose(b[3], s[3])               # det
    assert b[4] == s[4] and b[5] == s[5]         # steer, throttle
    assert np.allclose(b[6], s[6])               # wp


def test_seg_channel_swap():
    s = _sample()
    _, _, sf, *_ = hflip_sample(*s)
    # flip 후 ch0(좌실선 라벨)에는 원본 우실선이 좌로 와야 → 왼쪽 절반
    cols = np.where(sf[0].any(axis=0))[0]
    assert cols.min() < 112, f"좌실선 채널이 flip 후 왼쪽에 없음: {cols}"


def test_sign_flips():
    s = _sample()
    _, _, _, df, st, th, wpf = hflip_sample(*s)
    assert st == -s[4]                           # steer 부호 반전
    assert th == s[5]                            # throttle 불변
    assert np.allclose(wpf[:, 0], s[6][:, 0])    # wp x(전방) 불변
    assert np.allclose(wpf[:, 1], -s[6][:, 1])   # wp y(좌우) 반전
    assert df[0] == 224 - (30 + 40)              # bbox x 미러


def test_no_car_bbox_untouched():
    s = list(_sample())
    s[3] = np.zeros(5, np.float32)               # 차 없음
    _, _, _, df, *_ = hflip_sample(*s)
    assert np.allclose(df, 0)                     # conf<=0 이면 bbox 그대로


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all hflip tests passed")
