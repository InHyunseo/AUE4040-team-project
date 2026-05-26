# AUE4040 자동차임베디드AI — E2E MTL 자율주행 설계문서

> 한양대학교 미래자동차공학과 자동차임베디드AI 수업  
> 기존 BC 모델 → Dual-View E2E Multi-Task Learning 구조로 개선

---

## 1. 프로젝트 개요

### 주제
**BEV 시점과 정면 시점을 함께 입력으로 사용하는 단일 E2E Multi-Task Learning 자율주행 네트워크**

기존 Behavior Cloning(BC) 모델의 구조적 한계를 극복하고, UniAD에서 영감받은 경량 MTL 구조로 인지-판단-제어를 단일 end-to-end computational graph 안에서 학습한다.

### 단일 네트워크의 정의
"단일 신경망"의 기준은 backbone이 하나냐가 아니라:

```
하나의 end-to-end computational graph 안에서
입력 → feature 추출 → fusion → waypoint/control 출력까지
loss로 같이 학습되느냐
```

따라서 아래 구조도 단일 네트워크다:

```
BEV image ── Encoder A ─┐
                        ├── Fusion Neck ── Waypoint Head ── Control Head
Front image ─ Encoder B ─┘
```

이는 "두 개의 모델"이 아니라 **dual-input single network** 또는
**multi-view E2E network**다.

스테레오 깊이 추정 X → **dual-view monocular fusion** O

### 핵심 아이디어
- 학습 시: Seg/Det 헤드(보조 인지)가 encoder를 풍부하게 학습시킴
- 추론 시: 보조 인지 헤드(Seg, Det)만 제거, Waypoint/Control 헤드는 유지
- Waypoint Head는 critic이 아니라 **planning output** — 제어에 직접 연결

### 파이프라인이 아닌 이유
기존 파이프라인:
```
YOLO 검출 → 후처리 → FSM 판단 → BC 제어
(중간에 규칙/threshold/FSM 개입, gradient 단절)
```

제안 구조:
```
BEV + Front 이미지 → Encoder → Fusion → Waypoint → Control
(전체가 한 번에 학습, gradient 끊김 없음)
```

---

## 2. 기존 BC 모델 대비 차별점

| 항목 | 기존 BC | 제안 구조 |
|------|---------|----------|
| 구조 | 파이프라인 (YOLO→FSM→BC) | 단일 E2E 네트워크 |
| 입력 | 단일 카메라 + step 스칼라 | BEV + Front dual-view |
| 출력 | 6-class 분류 | steer/throttle 회귀 |
| step 입력 | 있음 (추가학습 어려움) | 없음 |
| 차량 회피 | 불가 | 가능 |
| Gradient | 단절 | encoder까지 전부 연결 |
| 해석 가능성 | 블랙박스 | waypoint 시각화 가능 |
| 추가학습 | 어려움 | 자유로움 |

---

## 3. 시스템 아키텍처

### 하드웨어
```
NVIDIA Jetson Orin Nano
듀얼 CSI 카메라
  - 카메라 A: 바닥/BEV 시점 (마운트 아래로 꺾음)
  - 카메라 B: 정면 시점
ROS2 Humble
```

> 카메라 마운트 변경이 어려울 경우: 단일 카메라(정면)로 fallback 가능.
> 단일 카메라 시 Encoder B만 사용, 나머지 구조 동일.

### 네트워크 구조

```
BEV image (224×224×3)        Front image (224×224×3)
        ↓                              ↓
━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━
Encoder A (ResNet18)          Encoder B (ResNet18)
차선/근거리 특화               차량/원거리 특화
high-res feature              mid-res feature
━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━
        ↓                              ↓
        └──────────────┬───────────────┘
                       ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fusion Neck (feature-level fusion)
  - BEV mid feature + Front mid feature
    → channel-align 후 concat
  - 양쪽 GAP → FC fusion → global feature
  - Seg: BEV high-res feature 사용
  - Det: Front mid-res feature 사용
  - Waypoint/Control: fused global feature 사용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ↓
┌──────────┬──────────┬──────────────────────┐
Seg Head   Det Head   Waypoint Head           Control Head
(보조 인지) (보조 인지) fused global feature   waypoint +
BEV        Front      사용                    fused feature
high-res   mid-res
좌/우      차량       미래 위치 5개            steer
실선+점선  bbox       (x,y)×5 (m 단위)        throttle
마스크     +confidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
학습 시에만  학습 시에만  학습+추론 모두 사용    학습+추론
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 학습 시 vs 추론 시

```
학습 시 (전체 헤드 활성):
  Seg Head   → BEV feature → 차선 마스크 (보조 인지)
  Det Head   → Front feature → 차량 bbox (보조 인지)
  Waypoint   → fused feature → 미래 경로 5개
  Control    → waypoint + fused → steer/throttle

추론 시 (보조 인지 제거):
  Encoder A + Encoder B
  Fusion Neck
  Waypoint Head   ← 유지 (Control Head 입력으로 필요)
  Control Head    ← 유지
  → steer/throttle 출력
```

### 각 카메라의 역할 분담
```
BEV 카메라 (Encoder A):
  차선 위치 정밀 파악 (근거리)
  Seg Head에 주로 기여
  차선 중앙 오프셋, 곡률 정보

Front 카메라 (Encoder B):
  전방 차량 감지 (원거리)
  Det Head에 주로 기여
  차량 거리, 위치 정보
```

### Cascade 연결
```
Front encoder feature (front_mid)
    ↓ GAP → FC (차량 위치 정보 압축)
Waypoint Head (obstacle feature 반영해서 경로 생성)
    ↓ waypoint 전달 (detach 없이)
Control Head (경로 보고 제어값 결정)

→ Control loss가 Waypoint Head까지 역전파
→ Det Head는 bbox loss만 제공, 추론 시 제거해도 무관
→ Waypoint Head는 Front encoder feature에만 의존 (Det output X)
```

**핵심**: Waypoint Head는 Det Head output(bbox)에 의존하지 않는다.
Det Head는 Front encoder feature가 차량 위치를 잘 담도록 학습시키는
보조 loss 역할만 한다. 추론 시 Det Head를 제거해도 Waypoint 생성에 영향 없음.

### Gradient 흐름
```
Seg loss      → Seg Head → Fusion → Encoder A
Det loss      → Det Head → Fusion → Encoder B
Waypoint loss → Waypoint Head → Fusion → Encoder A, B
Control loss  → Control Head → Waypoint Head → Fusion → Encoder A, B

모든 loss가 양쪽 encoder까지 역전파
loss.backward() 한 번으로 전체 업데이트
```

---

## 4. 각 헤드 상세 구조

### Seg Head (보조 인지, 학습 시에만)
```python
# BEV high-res feature 받아서 원본 해상도로 복원
bev_feat → Conv3×3 → BN → ReLU
         → Upsample×4 (bilinear)
         → Conv1×1
         → (224×224×4)  # 실선좌/실선우/점선좌/점선우
```

### Det Head (보조 인지, 학습 시에만)
```python
# Front mid-res feature 받아서 bbox 예측
# mask는 제거 — bbox + confidence만
front_feat → Conv3×3 × 3
           → bbox 브랜치: FC → (N, 5)  # x,y,w,h,conf
```

> mask 예측은 과함 (closed-world, 차량 1대 환경).
> 필요 시 MobileSAM 기반 mask label을 추가할 수 있으나,
> 본 구현에서는 bbox 중심으로 전방 차량 위치를 학습한다.

### Fusion Neck
```python
# BEV + Front feature fusion
bev_mid   = encoder_a.layer3_out   # (B, 256, 14, 14)
front_mid = encoder_b.layer3_out   # (B, 256, 14, 14)

# channel-align 후 concat
fused_spatial = concat([bev_mid, front_mid], dim=1)  # (B, 512, 14, 14)
fused_spatial = Conv1x1(fused_spatial)               # (B, 256, 14, 14)

# global feature
bev_global   = GAP(encoder_a.layer4_out)   # (B, 512)
front_global = GAP(encoder_b.layer4_out)   # (B, 512)
fused_global = FC(concat([bev_global, front_global]))  # (B, 256)
```

### Waypoint Head (추론 시 유지)
```python
# fused_global + Front obstacle feature concat
# Det Head output(bbox)에 의존하지 않음
# → 추론 시 Det Head 제거해도 Waypoint 생성 가능

obstacle_feat = GAP(front_mid)       # Front encoder mid feature
obstacle_feat = FC(obstacle_feat)    # 차량 위치 정보 압축
# Det loss가 front_mid를 차량 위치 잘 담도록 학습시킴

planning_input = concat(fused_global, obstacle_feat)
planning_input → FC(256) → ReLU → FC(128) → ReLU → FC(10)
→ reshape → (5, 2)  # 5개 waypoint (x,y) 미터 단위
```

### Control Head (추론 시 유지)
```python
# fused_global + waypoint concat (detach 없음 → gradient 유지)
control_input = concat(fused_global, waypoints.flatten())
control_input → FC(128) → ReLU → FC(64) → ReLU → FC(2)
→ (steer, throttle)
```

---

## 5. Loss 구성

```python
L = 1.0 * DiceLoss(seg_pred, seg_gt)           # Seg Head (보조)
  + 0.5 * FocalLoss(det_pred, det_gt)           # Det Head (보조)
  + 2.0 * SmoothL1Loss(waypoint_pred, wp_gt)    # Waypoint Head
  + 1.0 * MSELoss(steer_pred, steer_gt)         # Control Head
  + 0.5 * MSELoss(throttle_pred, throttle_gt)   # Control Head

loss.backward()  # 한 번에 전체 역전파
optimizer.step()
```

waypoint loss 가중치를 높게 — 판단 품질이 제어에 직접 영향.
초기 가중치는 waypoint loss를 높게 두되,
실험 중 control 안정성과 waypoint 정확도를 보며 λ를 조정한다.

### 3주 구현 우선순위
```
1순위: 단일 Front 또는 BEV 기반 control 성공
2순위: Dual-view fusion 성공
3순위: Waypoint 시각화
4순위: Seg/Det 보조학습 추가
5순위: 차량 회피 검증
```

처음부터 모든 헤드를 동시에 완성하려 하지 말 것.
Phase 1 (단일 카메라 + Control Head만)이 동작하면 그다음 확장.

---

## 6. 입출력 데이터

### 입력
```
BEV 이미지   (224×224×3)  카메라 A → perspective warp
Front 이미지 (224×224×3)  카메라 B → rectify
```

### 학습 시 출력 (전체 헤드)
```
차선 세그 마스크  (224×224×4)  실선좌/실선우/점선좌/점선우
차량 bbox        (N, 5)        x,y,w,h,conf
waypoint         (5, 2)        현재 기준 상대좌표 (미터)
steer            float
throttle         float
```

### 추론 시 출력
```
waypoint  (5, 2)   경로 시각화용 (옵션)
steer     float
throttle  float
```

---

## 7. 데이터 파이프라인

### rosbag 개념
rosbag은 이미지 파일을 개별로 저장하지 않는다.
모든 토픽 메시지를 시간순으로 직렬화해서 **파일 하나(db3)**에 저장한다.

```
ros2 bag record ...
→ rosbag2_data/
    ├── metadata.yaml
    └── rosbag2_data_0.db3   ← 이미지/제어값 전부 여기에
```

이미지 파일 수만 개가 생기는 게 아니라 db3 파일 하나.
로컬에 이미지 파일을 따로 추출할 필요 없음.

### 수집
```bash
# compressed image로 저장 → 용량 1/3 수준
ros2 bag record \
  /bev_image/compressed \
  /front_image/compressed \
  /cmd_vel \
  /fsm_state

# 용량 예상
# 무압축 bgr8:     ~20GB (4시간)
# jpeg compressed: ~5~7GB (4시간)
```

```
1단계: 단독 주행 (차량 없음)
  다양한 속도, 조명, 트랙 구간

2단계: 전방 차량 포함
  좌측/우측/중앙 다양한 위치
  회피 상황 의도적으로 많이 수집
```

### 전체 데이터 흐름
```
Jetson 주행
  ↓ ros2 bag record
rosbag2_data.db3 (파일 하나)
  ↓ Google Drive 업로드 (파일 하나라 빠름)
Google Drive
  ↓ Colab Drive 마운트
Colab 학습
  (이미지 파일 추출 없이 on-the-fly 파싱)
```

### Colab에서 rosbag 직접 읽기
```python
# pip install rosbags (ROS2 설치 없이 순수 Python)
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2, numpy as np

from google.colab import drive
drive.mount('/content/drive')

with Reader('/content/drive/MyDrive/rosbag2_data') as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/bev_image/compressed':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            img = cv2.imdecode(
                np.frombuffer(msg.data, np.uint8),
                cv2.IMREAD_COLOR
            )  # 바로 numpy array → tensor 변환
```

### 라벨 자동 생성

| 라벨 | 생성 방법 | 수동 작업 |
|------|----------|----------|
| 차선 세그 (4채널) | HSV 자동 마스킹 + Zone 검증 | 오검출만 보정 |
| 차량 bbox | GroundingDINO 자동 | 없음 |
| waypoint GT | bicycle model 적분 | 없음 |
| steer/throttle | rosbag /cmd_vel 추출 | 없음 |

라벨은 첫 epoch 전에 한 번만 캐싱:
```python
if not os.path.exists('labels_cache.h5'):
    build_label_cache(rosbag_path, 'labels_cache.h5')
```

### BEV 전처리
```
체커보드로 perspective transform 캘리브 (OpenCV)
pixels_per_meter 측정
waypoint: BEV 픽셀 좌표 → 미터 변환 적용

Zone 사전 할당 (robustness):
  근거리 Zone: 차선 반드시 존재해야 함
  중거리 Zone: 차선 존재해야 함
  원거리 Zone: 차선 없어도 허용
  → Zone 미검출 시 이전 프레임 값 유지
```

### waypoint GT 자동 추출
```python
# t 시점 기준 미래 5개 상대좌표 (bicycle model 적분)
for i, t in enumerate(timestamps):
    x, y, theta = 0, 0, 0
    for j in range(1, 6):
        dt = timestamps[i+j] - timestamps[i+j-1]
        v = throttles[i+j] * max_speed
        steer = steers[i+j]
        x += v * cos(theta) * dt
        y += v * sin(theta) * dt
        theta += v * tan(steer) / L * dt  # L: 실제 wheelbase 측정값
        waypoints.append((x, y))
    waypoints_m = [(x/ppm, y/ppm) for x, y in waypoints]
```

### 최종 저장 구조
```
Google Drive/
├── rosbag2_data.db3      원본 rosbag (파일 하나)
└── labels_cache.h5       전처리된 라벨 캐시
    ├── seg/              차선 마스크 (4채널)
    ├── det/              차량 bbox
    ├── waypoint/         waypoint GT
    ├── steer/            조향값
    └── throttle/         속도값
```

---

## 8. 학습 전략

### 단계적 학습 (안정성 확보)
```
Phase 1: Seg + Det 헤드만
  양쪽 encoder + fusion 안정화
  인지 먼저 충분히 학습
  lr: 1e-4, epoch: 30

Phase 2: Waypoint 헤드 추가
  encoder lr 낮추기 (1e-5)
  waypoint 헤드 lr: 1e-3
  epoch: 20

Phase 3: Control 헤드 추가 + 전체 fine-tuning
  전체 lr: 1e-4
  epoch: 30
```

### 학습 환경
```
Colab GPU
배치 사이즈: 16 (메모리 부족 시 8)
gradient accumulation: 4 steps
옵티마이저: AdamW
encoder: ImageNet pretrained 재사용
```

---

## 9. 배포 파이프라인

```
학습 완료
  ↓
보조 인지 헤드만 제거 (Seg Head, Det Head)
  ↓
Encoder A + Encoder B + Fusion + Waypoint + Control Head 유지
  ↓
ONNX export (opset 12)
  ↓
trtexec --fp16 → Jetson TRT engine
  ↓
rover_lane_mtl 노드에서 로드
  ↓
/bev_image + /front_image → waypoint + steer/throttle 출력
```

### 추론 속도 예상
```
Dual Encoder + Fusion + Waypoint + Control
TRT FP16: ~8~10ms → 30Hz 가능
INT8 필요 시: ~5ms

단일 카메라 fallback:
TRT FP16: ~5ms → 30Hz 충분
```

---

## 10. ROS2 노드 변경사항

### 제거
```
rover_perception  YOLO 노드
FSM 복잡한 로직
step 카운터
```

### 유지
```
rover_stereo    rectify + BEV warp 추가
rover_control   UART 그대로
rover_msgs      메시지 타입 그대로
```

### 수정/추가
```
rover_lane_mtl  새 노드 (기존 rover_lane 대체)
  구독: /bev_image/compressed, /front_image/compressed
  발행: /cmd_vel (steer, throttle)
        /waypoints (시각화 옵션)
  threading:
    Thread 1: 이미지 수신 + 전처리
    Thread 2: TRT 추론
    Thread 3: 제어 명령 발행
```

---

## 11. 레포 구조

```
AUE4040/
├── main/                      기존 BC 코드 (건드리지 않음)
│   └── ros2_ws/
│
├── [프로젝트명]/               새로 추가
│   ├── README.md
│   ├── ros2_ws/
│   │   └── src/
│   │       └── rover_lane_mtl/
│   │           ├── rover_lane_mtl_node.py
│   │           ├── trt_inference.py
│   │           └── image_preprocess.py
│   ├── train/
│   │   ├── model.py           DualView E2E MTL 모델
│   │   ├── dataset.py         rosbag 기반 데이터 로더
│   │   ├── train.py           단계적 학습 루프
│   │   └── export.py          ONNX/TRT 변환 (Seg/Det 제거)
│   ├── data_pipeline/
│   │   ├── extract_rosbag.py
│   │   ├── bev_warp.py        perspective transform
│   │   ├── hsv_lane_mask.py
│   │   ├── zone_validator.py  Zone 기반 오검출 필터
│   │   ├── grounding_dino_det.py
│   │   └── waypoint_gt.py     bicycle model 적분
│   └── models/                TRT engine (gitignored)
│
└── calibration/               그대로
```

기존 main/ 완전히 건드리지 않음 → 실패 시 rollback 가능

---

## 12. 구현 일정

| 기간 | 작업 |
|------|------|
| Week 1 | rosbag 수집, BEV 캘리브, HSV 마스킹, GroundingDINO 라벨, waypoint GT |
| Week 2 | Dual Encoder + Fusion + 헤드 구현, Colab 단계적 학습, loss 튜닝 |
| Week 3 | TRT 변환 (Seg/Det 제거), rover_lane_mtl 노드, 실차 테스트, 비교 실험 |

---

## 13. 예상 문제 및 해결

### 데이터
| 문제 | 해결 |
|------|------|
| BEV 캘리브 흔들림 | 마운트 고정 또는 단일 카메라 fallback |
| HSV 마스킹 실패 | Zone 기반 오검출 감지, 오검출만 수동 보정 |
| waypoint GT 오차 | horizon 짧게 (0.5초 이내), wheelbase 정확히 측정 |
| 회피 데이터 불균형 | 회피 상황 의도적으로 많이 수집, 가중치 샘플링 |

### 학습
| 문제 | 해결 |
|------|------|
| MTL loss 불균형 | 단계적 학습, λ 실험적 튜닝 |
| cascade 불안정 | Phase 1에서 인지 먼저 안정화 후 추가 |
| encoder 과적합 | 밝기/대비 augmentation, 다양한 조명 수집 |
| Colab 메모리 부족 | 배치 8~16, gradient accumulation 4 steps |

### 배포
| 문제 | 해결 |
|------|------|
| TRT 변환 실패 | opset 12, ONNX netron으로 확인 |
| 추론 속도 미달 | INT8 변환, 입력 해상도 낮추기 |
| steer 튀는 현상 | EMA 필터 (α=0.7) |

### 실차
| 문제 | 해결 |
|------|------|
| 차선 못 잡음 | 실차 이미지로 fine-tuning |
| 회피 성공률 낮음 | DAgger로 실패 구간 추가 학습 |
| 카메라 마운트 실패 | 단일 카메라 fallback |

---

## 14. 발표 표현

```
스테레오 깊이 추정은 사용하지 않고,
두 카메라를 각각 BEV 주행 시점과 정면 장애물 시점으로 활용한다.
두 영상은 독립 encoder에서 feature로 변환된 뒤
fusion neck에서 통합되며,
최종 waypoint와 steer/throttle은
하나의 end-to-end network에서 예측된다.
```

### 발표 스토리
```
문제 정의:
  기존 BC — step 의존, gradient 단절
  차량 회피 불가, 추가학습 어려움

제안:
  UniAD에서 영감받은 경량 Dual-View E2E MTL
  BEV(차선) + Front(차량) dual-view 입력
  인지(보조) → 판단(waypoint) → 제어
  보조 인지로 encoder 풍부하게 학습
  추론 시 Seg/Det 제거, Waypoint/Control 유지

실험:
  차선 이탈률 비교 (기존 BC vs 제안)
  차량 회피 성공률
  waypoint 시각화
  단일 vs dual-view ablation (카메라 마운트 성공 시)
```

---

## 15. 구현 가능성 평가

```
차선 추종:               높음
BEV + Front 동기화:      중간
정지 차량 1대 회피:       중간~높음
차량 위치 변동 대응:      중간
전체 MTL 안정 학습:      중간
3주 내 데모:             가능
```

---

## 16. 참고

- **UniAD** (CVPR 2023 Best Paper): 인지-판단-제어 단일 네트워크 원형
- **NVIDIA DAVE-2** (2016): 소규모 E2E 자율주행 원조
- **Mask R-CNN**: ResNet+FPN+헤드 구조 표준
- **DAgger**: 분포 이탈 문제 해결 IL 방법론
- **rosbags**: ROS2 없이 Python에서 rosbag 파싱 라이브러리

---

*작성: 2026년 5월 26일*

---

## 17. 데이터 파이프라인 구현 (data_pipeline/)

### 파일

```
final_project/data_pipeline/
├── bev_calibration.py    체커보드 → M, pixels_per_meter → calib/calib.json
├── extract_labels.py     rosbag2 .db3 → labels_cache.h5  (+ debug_samples/)
└── visualize_labels.py   labels_cache.h5 한 샘플 시각화
```

### main/ 와의 연동 (중요)

**`final_project/`의 어떤 파일도 `main/`을 import 하지 않는다.** 한 방향 런타임
계약만 존재:

| `final_project/` 파일 | `main/` import | 런타임 의존 |
|---|---|---|
| `bev_calibration.py`  | 없음 | 로컬에서 체커보드 JPEG 한 장 입력 |
| `extract_labels.py`   | 없음 | rosbag2 디렉토리 (`.db3` + `metadata.yaml`) |
| `visualize_labels.py` | 없음 | `labels_cache.h5` |

연동은 **rosbag 토픽 스키마**로만 일어난다. 향후 `main/ros2_ws/src/rover_recorder/`에
구현될 ROS2 recorder 노드는 아래 스펙을 반드시 발행해야 `extract_labels.py`가
읽을 수 있다:

| 토픽 | 메시지 타입 | 단위/주기 |
|---|---|---|
| `/bev_image/compressed`   | `sensor_msgs/CompressedImage` | JPEG, ~15 Hz |
| `/front_image/compressed` | `sensor_msgs/CompressedImage` | JPEG, ~15 Hz |
| `/cmd_vel`                | `geometry_msgs/Twist`         | `linear.x` m/s, `angular.z` rad/s, ≥30 Hz |

**왜 이 분리인가**:
- `main/calibration.camera.Camera`, `control.base_ctrl.BaseController`는 Jetson + UART + jetcam 의존이라 Colab에서 import 자체가 깨진다.
- `extract_labels.py`는 Colab/로컬에서 ROS2/카메라 하드웨어 없이 돌아야 한다는 제약이 있어, `rosbags` + `opencv` + `numpy` + `h5py` + `transformers` 만 사용.
- 향후 ROS recorder가 위 토픽 스펙만 지키면 양쪽이 독립적으로 발전 가능.

### waypoint GT 모델 (설계문서 §7과의 차이)

설계문서 §7는 자전거 모델 (`θ̇ = v·tan(δ)/L`, wheelbase `L` 필요)을 명시했지만,
실차는 differential-drive (L/R wheel speed mixing in `rover_control/control_node.py`).
Twist 메시지의 `angular.z`가 이미 yaw rate이므로 wheelbase 없이 직접 적분:

```
θ̇ = ω         (= cmd_vel.angular.z)
ẋ = v·cos(θ)  (v = cmd_vel.linear.x, true m/s)
ẏ = v·sin(θ)
```

horizon = 0.5 s, 5점 (= 0.1 s 간격), 20 ms substep ZOH integration.
robot frame: `+x` 전방, `+y` 좌측. 단위 미터.

### 사용 순서

```bash
# 1) BEV calib (한 번만, 마운트 변경 시 재실행)
python final_project/data_pipeline/bev_calibration.py \
    --image path/to/checker_on_floor.jpg \
    --rows 6 --cols 9 --square_m 0.025
# → final_project/calib/calib.json
#   final_project/calib/calib_{warped,source}.png

# 2) rosbag → labels (Colab/로컬, ROS2 불필요)
pip install rosbags opencv-python h5py transformers torch
python final_project/data_pipeline/extract_labels.py \
    --bag /path/to/rosbag2_data \
    --calib final_project/calib/calib.json \
    --device cuda
# → final_project/labels_cache.h5
#   final_project/debug_samples/frame_*.png  (100장 자동 저장)

# 3) 단일 샘플 시각화
python final_project/data_pipeline/visualize_labels.py \
    --cache final_project/labels_cache.h5 --idx 0
```

### labels_cache.h5 레이아웃

```
/bev          (N, 224, 224, 3) uint8     BGR, M으로 warp된 BEV
/front        (N, 224, 224, 3) uint8     BGR, 224로 resize
/seg          (N, 4, 224, 224) uint8     0/255, [solidL, solidR, dashedL, dashedR]
/det          (N, 5)           float32   [x, y, w, h, conf]  front pixels
/waypoint     (N, 5, 2)        float32   robot frame meters, +x 전방 +y 좌측
/steer        (N,)             float32   = cmd_vel.angular.z (rad/s)
/throttle     (N,)             float32   = cmd_vel.linear.x  (m/s)
/timestamp_ns (N,)             int64

attrs: pixels_per_meter, bev_size, wp_horizon_s, wp_n, bag
```
