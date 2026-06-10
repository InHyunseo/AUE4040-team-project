"""E2E 학습용 Dataset — labels_cache.h5 에서 오버레이 입력을 합성한다.

H5 는 raw lane/front + seg/det 를 따로 저장하고(추출 시점에 합성하지 않음),
여기서 매 샘플 오버레이를 합성한다. 그래야 재추출 없이 합성 계약을 튜닝할 수
있다(PHASE2.md 4단계 명세). 합성 픽셀은 추론 노드(rover_lane)와 **동일**해야
하므로 아래 두 함수가 학습/추론 공용 계약이다:

  composite_lane(lane_bgr, seg)   : raw lane(BGR) + seg 3채널 alpha-blend
  composite_front(front_bgr, det) : raw front(BGR) + car bbox 사각형

합성 규칙은 extract_labels.save_debug / visualize_labels.overlay_seg 와 동일:
  - seg ch0=left-solid→red(0,0,255), ch1=right-solid→green, ch2=center-dashed→blue
  - blend = base*0.4 + color*0.6 (마스크 픽셀만)
  - bbox = det[4]>0 일 때만 초록 사각형 (디버그 텍스트/waypoint 는 그리지 않음)

색공간/정규화:
  H5 의 lane/front 는 BGR(uint8). 합성도 BGR 로 한 뒤, 텐서화 직전에 RGB 로
  변환하고 ImageNet mean/std 로 정규화한다. ResNet18 ImageNet pretrained 가
  RGB 기준이므로. 추론 노드도 반드시 BGR→RGB→ImageNet 동일 순서를 써야 한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 합성 계약 상수는 추출 스크립트와 한 소스를 공유한다.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from extract_labels import SEG_N_CLASSES  # noqa: E402

# seg 채널 → BGR 색 (extract_labels.save_debug 와 동일)
#   ch0=left-solid red, ch1=right-solid green, ch2=center-dashed blue
SEG_COLORS_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
SEG_ALPHA = 0.6   # color 비중 (base 0.4)

BBOX_COLOR_BGR = (0, 255, 0)
BBOX_THICK = 2

# ImageNet 정규화 (RGB 기준)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def composite_lane(lane_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """raw lane(BGR uint8) 에 seg 3채널을 색으로 alpha-blend. 반환 BGR uint8."""
    out = lane_bgr.copy()
    for c in range(SEG_N_CLASSES):
        m = seg[c] > 0
        if not m.any():
            continue
        color = np.array(SEG_COLORS_BGR[c], dtype=np.float32)
        out[m] = (out[m] * (1.0 - SEG_ALPHA) + color * SEG_ALPHA).astype(np.uint8)
    return out


def composite_front(front_bgr: np.ndarray, det: np.ndarray) -> np.ndarray:
    """raw front(BGR uint8) 에 car bbox 사각형. det=[x,y,w,h,conf], conf<=0 이면 무변경."""
    out = front_bgr.copy()
    if det[4] > 0:
        x, y, w, h, _ = det
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                      BBOX_COLOR_BGR, BBOX_THICK)
    return out


def to_input_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """합성된 BGR uint8 (H,W,3) → RGB ImageNet-정규화 텐서 (3,H,W) float32.

    추론 노드는 이 함수와 픽셀 단위로 동일한 변환을 써야 한다."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def _color_jitter(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """보수적 색상 jitter (밝기/대비/채도). 기하 변형 없음 — steer/waypoint GT 보존.

    HSV 채도 + 밝기/대비만 건드린다. flip/회전/crop 은 좌표계를 깨므로 제외."""
    # 밝기·대비: out = img*contrast + brightness
    contrast = 1.0 + rng.uniform(-0.15, 0.15)
    brightness = rng.uniform(-15, 15)
    out = img_bgr.astype(np.float32) * contrast + brightness
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 채도
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= (1.0 + rng.uniform(-0.20, 0.20))
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


class E2EDataset(Dataset):
    """labels_cache.h5 (또는 여러 개) → (lane_tensor, front_tensor, steer, throttle, wp).

    Args:
      h5_paths   : 하나 또는 여러 H5 경로. 여러 bag 을 합쳐 학습할 때 리스트로.
      indices    : 사용할 전역 인덱스 부분집합 (train/val split 용). None=전체.
      augment    : True 면 합성된 입력에 색상 jitter (val 은 False).
      seed       : jitter 재현용.

    여러 H5 는 길이를 이어붙여 하나의 평탄한 인덱스 공간으로 노출한다.
    h5py.File 은 워커별로 lazy open (멀티프로세싱 안전)."""

    def __init__(self, h5_paths, indices=None, augment=False, seed=0):
        if isinstance(h5_paths, (str, Path)):
            h5_paths = [h5_paths]
        self.h5_paths = [str(p) for p in h5_paths]
        self.augment = augment
        self.seed = seed
        self._files = None  # 워커별 lazy

        # 각 파일 길이로 누적 오프셋 구성
        self._lengths = []
        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                self._lengths.append(f["lane"].shape[0])
        self._cumsum = np.cumsum([0] + self._lengths)
        total = int(self._cumsum[-1])

        self.indices = list(range(total)) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def _files_handle(self):
        if self._files is None:
            self._files = [h5py.File(p, "r") for p in self.h5_paths]
        return self._files

    def _locate(self, gidx):
        """전역 인덱스 → (파일 인덱스, 파일 내 로컬 인덱스)."""
        fi = int(np.searchsorted(self._cumsum, gidx, side="right") - 1)
        return fi, gidx - int(self._cumsum[fi])

    def __getitem__(self, i):
        gidx = self.indices[i]
        fi, li = self._locate(gidx)
        f = self._files_handle()[fi]

        lane  = f["lane"][li]      # (224,224,3) BGR uint8
        front = f["front"][li]
        seg   = f["seg"][li]       # (3,224,224) {0,255}
        det   = f["det"][li]       # (5,)
        steer = float(f["steer"][li])
        thr   = float(f["throttle"][li])
        wp    = f["waypoint"][li].astype(np.float32)  # (5,2)

        lane_c  = composite_lane(lane, seg)
        front_c = composite_front(front, det)

        if self.augment:
            rng = np.random.default_rng(self.seed + gidx)
            lane_c  = _color_jitter(lane_c, rng)
            front_c = _color_jitter(front_c, rng)

        lane_t  = to_input_tensor(lane_c)
        front_t = to_input_tensor(front_c)

        return (lane_t, front_t,
                torch.tensor(steer, dtype=torch.float32),
                torch.tensor(thr, dtype=torch.float32),
                torch.from_numpy(wp))

    def close(self):
        if self._files is not None:
            for f in self._files:
                f.close()
            self._files = None


def make_splits(h5_paths, val_frac=0.15, seed=0):
    """전역 인덱스를 train/val 로 무작위 분할. 같은 seed 면 재현."""
    if isinstance(h5_paths, (str, Path)):
        h5_paths = [h5_paths]
    total = 0
    for p in h5_paths:
        with h5py.File(str(p), "r") as f:
            total += f["lane"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    n_val = int(total * val_frac)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_idx, val_idx
