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

# steer 정규화 스케일 = teleop MAX_OMEGA. extract_labels 는 angular.z(∈[-1.2,1.2])를
# raw 로 저장하는데 ControlHead 끝이 Tanh(∈[-1,1]) 라, raw 를 그대로 타깃으로 쓰면
# |steer|>1 인 급코너 프레임이 도달 불가 → tanh 포화로 steer loss 에 영구 floor.
# steer/STEER_SCALE 로 [-1,1] 에 맞춘 뒤 학습한다(추론 노드는 angular.z=steer*1.2 로
# 역변환하므로 일관). 옛 체크포인트와는 타깃 스케일이 달라 resume 불가(재학습 필요).
STEER_SCALE = 1.2


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


def hflip_sample(lane, front, seg, det, steer, throttle, wp):
    """좌우(수평) flip — 한 샘플의 이미지·라벨을 일관되게 반전한다.

    합성 **전** raw 단계에서 호출해야 한다(합성 후 flip 하면 오버레이 색까지
    꼬임). 반전 규칙(좌우 대칭만, 전방 성분은 불변):
      lane/front : 좌우 반전 (가로축)
      seg        : 좌실선(ch0)↔우실선(ch1) **채널 교환** + 각 채널 좌우 반전.
                   중앙점선(ch2)은 교환 없이 좌우 반전만.
      det bbox   : x → W-(x+w) 로 미러 (y,w,h,conf 불변). 차 없으면(conf<=0) 그대로.
      steer(ang.z): 부호 반전 (좌회전↔우회전)
      throttle   : 불변 (전진속도는 좌우 무관)
      waypoint   : y 부호 반전(좌↔우), x(전방) 불변

    flip(flip(x)) == x 가 성립한다(단위 테스트로 보장)."""
    W = lane.shape[1]
    lane_f  = lane[:, ::-1].copy()
    front_f = front[:, ::-1].copy()

    # seg: 좌/우 실선 채널 교환 후, 모든 채널 가로 반전.
    seg_f = seg[:, :, ::-1].copy()
    seg_f[[0, 1]] = seg_f[[1, 0]]   # ch0(좌실선) ↔ ch1(우실선)

    det_f = det.copy()
    if det_f[4] > 0:
        x, w = det_f[0], det_f[2]
        det_f[0] = W - (x + w)      # 좌상단 x 미러; w,h,conf 그대로

    steer_f = -steer
    throttle_f = throttle
    wp_f = wp.copy()
    wp_f[:, 1] *= -1.0              # y(좌우) 반전, x(전방) 유지

    return lane_f, front_f, seg_f, det_f, steer_f, throttle_f, wp_f


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
      steer_smooth : >0 이면 steer GT 를 H5(세션) 내부에서 ±k 프레임 이동평균.
                     teleop raw 조향의 순간 떨림(같은 장면 다른 라벨)을 완화한다.
                     H5 행은 시간순(extract_labels 가 정렬 저장)이라 행 이동평균이
                     곧 시간 스무딩이고, 세션 경계는 절대 넘지 않는다(파일별 계산).

    여러 H5 는 길이를 이어붙여 하나의 평탄한 인덱스 공간으로 노출한다.
    h5py.File 은 워커별로 lazy open (멀티프로세싱 안전)."""

    def __init__(self, h5_paths, indices=None, augment=False, seed=0,
                 steer_smooth=0, wp_fix_sign=False, hflip=False):
        if isinstance(h5_paths, (str, Path)):
            h5_paths = [h5_paths]
        self.h5_paths = [str(p) for p in h5_paths]
        self.augment = augment
        # hflip: 50% 확률 좌우 flip (이미지+seg채널교환+det+steer+wp 일관 반전).
        # train 만 켜고 val 은 끈다(평가는 원본 분포). 데이터 2배 + 좌우 균형.
        self.hflip = bool(hflip)
        self.seed = seed
        self.steer_smooth = int(steer_smooth)
        # wp_fix_sign: 부호 버그 시절 추출된 옛 H5 의 waypoint 를 즉석 보정.
        # extract_labels 부호 수정(v_fwd=-c.v, yaw_rate=-c.w)의 순효과는
        # (x,y) → (-x, +y) 다 (cos 짝함수 → x 만 반전, sin 은 heading 반전과
        # -v 가 상쇄돼 y 불변). 단순 -wp 가 아니라 x 만 뒤집어야 맞는다.
        # 재추출한 새 부호 H5 에는 False 로 둔다(이중 반전 방지).
        self.wp_fix_sign = bool(wp_fix_sign)
        self._files = None  # 워커별 lazy

        # 각 파일 길이로 누적 오프셋 구성
        self._lengths = []
        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                self._lengths.append(f["lane"].shape[0])
        self._cumsum = np.cumsum([0] + self._lengths)
        total = int(self._cumsum[-1])

        self.indices = list(range(total)) if indices is None else list(indices)

        # steer 스무딩 룩업: 파일별로 전체 steer 를 읽어 ±k 이동평균(경계 클램프).
        # 미리 계산해 두면 __getitem__ 이 O(1) 로 조회한다.
        self._steer_lut = None
        if self.steer_smooth > 0:
            self._steer_lut = []
            k = self.steer_smooth
            for p in self.h5_paths:
                with h5py.File(p, "r") as f:
                    s = f["steer"][:].astype(np.float32)
                # 경계를 복제(edge)해 패딩하면 세션 밖 값이 안 섞인다.
                pad = np.pad(s, k, mode="edge")
                kernel = np.ones(2 * k + 1, dtype=np.float32) / (2 * k + 1)
                self._steer_lut.append(np.convolve(pad, kernel, mode="valid"))

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
        if self._steer_lut is not None:
            steer = float(self._steer_lut[fi][li])   # 세션 내부 ±k 이동평균
        else:
            steer = float(f["steer"][li])
        # raw angular.z(∈[-1.2,1.2]) → tanh 범위 [-1,1] 로 정규화 + 안전 clamp.
        # (smoothing/전이로 |값|>1.2 가 드물게 나와도 clamp 로 막는다.)
        steer = float(np.clip(steer / STEER_SCALE, -1.0, 1.0))
        thr   = float(f["throttle"][li])
        wp    = f["waypoint"][li].astype(np.float32)  # (5,2)
        if self.wp_fix_sign:
            wp = wp.copy()
            wp[:, 0] *= -1.0   # 옛 부호 보정: x(전방)만 반전, y(좌우)는 그대로

        # 좌우 flip: 합성 **전** raw 단계에서 모든 라벨 일관 반전 (50% 확률).
        # epoch 마다 다시 동전을 던져야(고정 seed X) 같은 샘플을 원본/거울상 둘 다
        # 본다 → 유효 다양성 2배. np.random(전역, 워커별 시드됨)을 그대로 쓴다.
        if self.hflip and np.random.random() < 0.5:
            lane, front, seg, det, steer, thr, wp = hflip_sample(
                lane, front, seg, det, steer, thr, wp)

        lane_c  = composite_lane(lane, seg)
        front_c = composite_front(front, det)

        if self.augment:
            # epoch 마다 다른 jitter (전역 np.random, 워커별 재시드됨) — flip 과 동일 정책.
            lane_c  = _color_jitter(lane_c, np.random)
            front_c = _color_jitter(front_c, np.random)

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
    """H5(세션) 단위로 train/val 분할. 같은 seed 면 재현.

    프레임 단위 무작위 분할은 연속 주행(15fps)의 인접 프레임이 거의 동일해
    train↔val 사이에 data leakage 가 생긴다(val 이 train 의 '쌍둥이'를 봐서
    val loss 가 일반화를 과대평가). 그래서 **H5 한 개를 통째로** train 또는 val
    에 배정한다 — val 로 빠진 세션의 프레임은 train 에 절대 안 섞인다.

    크기가 제각각인 H5 들을 무작위 순서로 보며, 누적 val 샘플이 val_frac 비율을
    넘기 직전까지 val 로 배정한다(비율을 대략 맞춤). h5_paths 가 1개뿐이면
    세션 분리가 불가능하므로 그 안에서 시간 블록(앞=train, 뒤=val)으로 나눈다."""
    if isinstance(h5_paths, (str, Path)):
        h5_paths = [h5_paths]

    lengths = []
    for p in h5_paths:
        with h5py.File(str(p), "r") as f:
            lengths.append(f["lane"].shape[0])
    cumsum = np.cumsum([0] + lengths)
    total = int(cumsum[-1])
    n_val_target = int(total * val_frac)

    # H5 가 1개뿐: 세션 분리 불가 → 시간 블록 split (뒤쪽 val_frac 을 val 로).
    if len(h5_paths) == 1:
        n_val = n_val_target
        train_idx = list(range(0, total - n_val))
        val_idx = list(range(total - n_val, total))
        return train_idx, val_idx

    # H5 단위 배정: 무작위 순서로 보되 작은 H5부터 채워 val 비율을 안정화한다
    # (큰 H5 하나가 통째로 들어가 목표를 크게 초과하는 것을 줄임). 같은 seed 면
    # 무작위성은 동률 H5 간 순서에만 작용해 재현된다.
    rng = np.random.default_rng(seed)
    order = sorted(range(len(h5_paths)),
                   key=lambda fi: (lengths[fi], rng.random()))
    val_files, n_val = set(), 0
    for fi in order:
        # 아직 val 이 비었거나, 이 파일을 넣어도 목표를 안 넘으면 val 로.
        if n_val == 0 or (n_val + lengths[fi] <= n_val_target):
            val_files.add(int(fi))
            n_val += lengths[fi]
        if n_val >= n_val_target:
            break

    train_idx, val_idx = [], []
    for fi in range(len(h5_paths)):
        rng_idx = range(int(cumsum[fi]), int(cumsum[fi + 1]))
        (val_idx if fi in val_files else train_idx).extend(rng_idx)
    return train_idx, val_idx


def oversample_avoidance(h5_paths, indices, factor=3, lat_thresh=0.15):
    """train 인덱스에서 '회피' 프레임(차 감지 det>0 + waypoint 측면이동 큰)을 factor
    배로 복제해 비중을 키운다. 재수집 없이 클래스 불균형(회피가 ~18%)을 완화 —
    모델이 다수인 차선주행에 맞춰 회피를 약하게(평균쳐서) 학습하는 걸 막는다.

    val 인덱스에는 쓰지 말 것(평가는 원본 분포). factor=1 이면 무변경.
    회피 판정은 det[4]>0(차 있음) AND |waypoint 끝점 y|>lat_thresh(실제 비킴).
    wp_fix_sign 과 무관하게 |y| 만 보므로 옛/새 H5 모두 동작."""
    if factor <= 1:
        return list(indices)
    if isinstance(h5_paths, (str, Path)):
        h5_paths = [h5_paths]
    h5_paths = [str(p) for p in h5_paths]

    # 파일별 누적 오프셋 + det/waypoint 통째로 읽어 회피 마스크 구성
    lengths, dets, wp_lat = [], [], []
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            lengths.append(f["lane"].shape[0])
            dets.append(f["det"][:, 4])              # conf
            wp_lat.append(np.abs(f["waypoint"][:, -1, 1]))   # |끝점 y|
    cumsum = np.cumsum([0] + lengths)
    det_all = np.concatenate(dets)
    lat_all = np.concatenate(wp_lat)
    is_avoid = (det_all > 0) & (lat_all > lat_thresh)   # 전역 인덱스 기준

    out = list(indices)
    extra = [g for g in indices if is_avoid[g]]
    for _ in range(factor - 1):
        out.extend(extra)
    return out
