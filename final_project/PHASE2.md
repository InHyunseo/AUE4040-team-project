# Phase 2 — 대량 수집 & E2E 본 학습 → 실차 배포

> Phase 1에서 freeze한 SegFormer/YOLO를 써서 **라벨링 없이** 대량 데이터를
> 자동 라벨링하고, ResNet18×2 + ControlHead/WaypointHead를 E2E로 학습해
> ONNX→TensorRT로 변환, 실차에 올리는 단계.
>
> Phase 1 상세는 [PHASE1.md](PHASE1.md), 모델 구조는 [model.py](model.py) 참고.

---

## 전체 흐름

```
1. 대량 rosbag 수집 (라벨링 X)   → final_project/rover_data/<session>_<ts>/bag
2. 라벨 자동 추출                → labels_cache.h5  (SegFormer/YOLO freeze + cmd_vel)
3. 라벨 검증 (눈 확인)           → debug_samples/*.png
4. E2E 학습 (Colab)             → E2ENet best 체크포인트   [미구현 — 명세만]
5. ONNX export                  → e2e.onnx
6. Jetson TensorRT 변환          → e2e.engine  (trtexec --fp16, Jetson에서만)
7. rover_lane 추론 노드 교체      → 실차 테스트
```

Phase 2 수집 데이터는 **라벨링 0** — cmd_vel은 텔레옵에서, 차선 세그/차량 bbox는
freeze된 SegFormer/YOLO가 자동 생성한다.

---

## 1단계 — 대량 rosbag 수집

Phase 1과 동일한 스택 사용:

```bash
# 터미널 A — 카메라 + 모터 + bag 레코더
ros2 launch rover_recorder record.launch.py session_name:=phase2

# 터미널 B — 키보드 텔레옵 (별도 SSH TTY)
cd /home/ircv16/team/final_project/ros2_ws && source install/setup.bash
ros2 run rover_teleop teleop_node
```

`r`로 녹화 토글 → `final_project/rover_data/phase2_<ts>/bag/`에 저장.
(노트북으로 띄우려면 [data_pipeline/launch_and_record.ipynb](data_pipeline/launch_and_record.ipynb).)

수집 시나리오 — 모델이 배워야 할 행동을 다양하게:
- 차선 주행: 직선 / 코너 / S자 (좌우 복귀 포함)
- 정지 차량: **접근 → 회피 → 복귀** 시퀀스를 좌/우/중앙, 가까이/멀리 반복
- Phase 1보다 훨씬 많이 (수십 분~시간 단위). 다양성이 양보다 중요.

---

## 2단계 — 라벨 자동 추출

```bash
cd final_project/data_pipeline
python3 extract_labels.py \
    --bag ../rover_data/phase2_<ts>/bag \
    --segformer_ckpt <segformer_lane 폴더> \
    --yolo_weights ../../main/best.pt \
    --out ../labels_cache.h5 \
    --debug_dir ../debug_samples \
    --device cuda
```

- `--segformer_ckpt` **필수** (Phase 1 산출물). 없으면 에러.
- `--yolo_weights` 기본값 `main/best.pt` (Phase 1 YOLO26 산출물).
- `--skip_det` : YOLO 생략 (차량 없는 차선 전용 bag일 때).
- `--limit N` : 앞 N 프레임만 (빠른 디버그용, 0=전체).
- bag 여러 개면 각각 다른 `--out`으로 추출 후 학습 때 합치거나, 스크립트를
  확장해 누적. (현재는 1 bag → 1 h5.)

### labels_cache.h5 스키마

각 lane 프레임 t마다 가장 가까운 front/cmd_vel을 ±50 ms로 동기화해 1 샘플 생성:

| 데이터셋 | shape | dtype | 내용 |
|---|---|---|---|
| `lane` | (224,224,3) | uint8 | lane 이미지, 상단 ROI 크롭 후 resize (**BGR**) |
| `front` | (224,224,3) | uint8 | raw front 이미지, resize (**BGR**) |
| `seg` | (3,224,224) | uint8 | SegFormer 마스크 {0,255}. ch0=좌실선/ch1=우실선/ch2=중앙점선 |
| `det` | (5,) | float32 | YOLO car bbox `[x,y,w,h,conf]` (front 픽셀). 차 없으면 0 |
| `waypoint` | (5,2) | float32 | cmd_vel 적분 미래 0.5초 궤적 (로봇 프레임, 미터) |
| `steer` | () | float32 | angular.z at t |
| `throttle` | () | float32 | linear.x at t |
| `timestamp_ns` | () | int64 | lane 프레임 타임스탬프 |

> **색공간 주의**: lane/front는 **BGR**로 저장된다(추출이 wire JPEG를 BGR로 디코드 후
> resize). 학습 데이터로더/오버레이 합성/추론 전처리가 모두 BGR로 일관되면 된다.
>
> **lane ROI 크롭**: lane 경로는 224 resize **직전에** `crop_lane_roi`로 상단
> `LANE_CROP_TOP`(extract_labels.py 상수, 기본 0.0=크롭 없음) 비율을 잘라낸다. 차선
> 카메라 상단이 도로 밖 배경이면 라벨링 **전에** 이 값을 정한다(예: 0.30). 한 번 정하면
> 라벨링·추출·추론이 **모두 같은 값**을 써야 좌표계가 맞는다(라벨링 후 변경 금지).
> front/YOLO는 크롭하지 않는다.

---

## 3단계 — 라벨 검증 (눈으로 확인)

추출이 자동이라 **반드시** 샘플을 눈으로 검증한다 (세그·bbox가 엉망이면 학습 무의미):

```bash
cd final_project/data_pipeline
python3 visualize_labels.py --cache ../labels_cache.h5 --idx 0 \
    --out ../debug_samples/viz_000.png
```

- seg 오버레이(좌=빨강/우=초록/중앙=파랑), car bbox, waypoint 점이 그려진 패널 출력.
- `extract_labels.py`가 `--debug_dir`에 자동 저장한 `frame_*.png`도 함께 확인.
- 세그가 차선을 못 잡거나 bbox가 헛돌면 → Phase 1 모델 재학습(라벨 추가)로 되돌아감.

---

## 4단계 — E2E 학습 (Colab) [미구현 — 명세만]

> 데이터로더 / 학습 루프 / Colab 노트북은 **아직 없음 (TODO)**. 아래는 구현 명세.

모델: [model.py](model.py) 의 `E2ENet`
- `LaneEncoder` / `FrontEncoder` (ResNet18×2, ImageNet pretrained) → 각 256-d
- concat(512) → `ControlHead`(steer, throttle Tanh) + `WaypointHead`(5×2 미터)

### 오버레이 합성 계약 (학습 데이터로더가 H5에서 구성 — 추출 헤더와 동일해야 함)
- **lane 입력** = raw lane 이미지에 seg 3채널을 색으로 alpha-blend
  (ch0=red, ch1=green, ch2=blue; `visualize_labels.overlay_seg`와 동일 방식).
- **front 입력** = raw front 이미지에 car bbox 그림 (`det[4] > 0`일 때).
- 둘 다 (3, 224, 224), **BGR** 채널 순서 유지. ImageNet 정규화는 BGR 기준으로 적용하거나
  RGB로 변환 후 정규화하되 **추론과 동일하게** 맞출 것.

### 손실
```
loss = 1.0·MSE(steer) + 0.5·MSE(throttle) + 0.5·MSE(waypoint)   # E2ELoss 기본값
```
waypoint는 보조 task(추론 시 버림) — backbone을 멀티스텝 의도 쪽으로 regularize.

### Colab 흐름 (TODO 노트북 `training/train_e2e_colab.ipynb`)
1. `labels_cache.h5` 업로드(또는 Drive 마운트) → `Dataset`이 H5에서 오버레이 합성
2. `E2ENet` 학습 (train/val split, AdamW, early-stop)
3. best 체크포인트 저장 → 5단계 ONNX export로 연결

---

## 5단계 — ONNX export

```python
import torch
from model import E2ENet
m = E2ENet().eval()
m.load_state_dict(torch.load("e2e_best.pt", map_location="cpu"))
dummy_lane  = torch.randn(1, 3, 224, 224)
dummy_front = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    m, (dummy_lane, dummy_front), "e2e.onnx",
    input_names=["lane", "front"],
    output_names=["steer", "throttle", "waypoints"],
    opset_version=13, do_constant_folding=True)
```

추론(실주행)에서는 `waypoints` 출력은 사용하지 않는다(시각화 전용).

---

## 6단계 — Jetson TensorRT 변환

ONNX→engine은 **Jetson(Orin Nano)에서만** 빌드한다 (GPU 아키텍처가 박히므로
Colab/x86에서 만든 engine은 호환 안 됨):

```bash
# Jetson에서
/usr/src/tensorrt/bin/trtexec \
    --onnx=e2e.onnx --fp16 --saveEngine=e2e.engine
```

- YOLO26은 NMS-free라 그 자체 ONNX도 후처리 없이 깔끔하게 변환됨
  (Front bbox를 런타임에서 합성할지, 사전 합성된 입력만 받을지는 7단계 노드 설계에 따름).

---

## 7단계 — 실차 (rover_lane 추론 노드) [별도 구현 TODO]

추론 노드 흐름:
```
/lane_image/compressed  → SegFormer → seg 오버레이 lane (224×224, BGR)
/front_image/compressed → YOLO26    → bbox 오버레이 front (224×224, BGR)
  → e2e.engine (TensorRT) → steer, throttle
  → 역변환 후 /cmd_vel:
       linear.x  = -(0.15 + abs(steer) * 0.10)   # -0.15 ~ -0.25
       angular.z = steer * 0.8                     # -0.8 ~ +0.8
  → motor_bridge_node → UART
```

- 전처리(ROI 크롭/세그/bbox 오버레이)는 **학습 때와 픽셀 단위로 동일**해야 함 — 4단계
  합성 계약 그대로. lane은 `crop_lane_roi`(같은 `LANE_CROP_TOP`) 후 224 resize, 색공간
  (BGR)·정규화·resize 순서 일치 필수.
- waypoint 출력은 무시(또는 디버그 시각화용으로만).

---

## DDS / QoS / SSH / Colab 메모

- **DDS/QoS**: 단일 호스트(Jetson) 안에서만 ROS2를 돌리면
  `export ROS_LOCALHOST_ONLY=1`로 외부 노드·멀티캐스트를 차단해 지연/간섭을 줄인다.
  이미지 토픽은 대용량이라 필요시 QoS depth를 줄여 지연 누적을 막는다.
- **SSH**: 카메라/모터는 Jetson 로컬 하드웨어 → 모든 노드는 Jetson에서 실행, 노트북은
  SSH 포트포워딩 또는 같은 네트워크에서 브라우저 모니터(`:8080`)로 본다.
- **Colab**: GPU 런타임 + Roboflow API 다운로드(Phase 1). 학습 산출물(best.pt /
  segformer_lane / e2e_best.pt)은 브라우저로 받아 WSL/Jetson에 배치.

---

## 완료 기준

- [ ] 대량 rosbag 수집 (다양한 차선 + 정지차 회피 시퀀스)
- [ ] `extract_labels.py`로 `labels_cache.h5` 생성 + `visualize_labels.py` 눈 검증 통과
- [ ] E2E 학습 → `e2e.onnx` export  (학습 노트북 구현 포함)
- [ ] Jetson에서 `e2e.engine` 빌드 (fp16)
- [ ] rover_lane 추론 노드 → 실차에서 차선 주행 + 정지차 회피·추월 확인
