# Phase 2 — 대량 수집 & E2E 본 학습 → ONNX

> Phase 1에서 freeze한 SegFormer/YOLO를 써서 **라벨링 없이** 대량 데이터를
> 자동 라벨링하고, ResNet18×2 + ControlHead/WaypointHead를 E2E로 학습해
> `e2e.onnx`까지 만드는 단계.
>
> Jetson engine 빌드 / 실차 추론 노드 / 주행 모니터링 / 추가학습은
> **[PHASE3.md](PHASE3.md)** 로 분리했다.
>
> Phase 1 상세는 [PHASE1.md](PHASE1.md), 모델 구조는 [model.py](model.py) 참고.

---

## 전체 흐름

```
1. 대량 rosbag 수집 (라벨링 X)   → final_project/rover_data/<session>_<ts>/bag
2. 라벨 자동 추출                → labels_cache.h5  (SegFormer/YOLO freeze + cmd_vel)
3. 라벨 검증 (눈 확인)           → debug_samples/*.png
4. E2E 학습 (Kaggle/Colab)      → e2e_best.pt  (train_e2e.py, 의도 시각화/이어학습 지원)
5. ONNX export                  → e2e.onnx     (export_onnx.py)
─────────────────────────  여기까지 Phase 2  ─────────────────────────
6~7. Jetson engine 빌드 + rover_lane 추론 노드 + 주행 모니터링 + 추가학습 → PHASE3.md
```

Phase 2 수집 데이터는 **라벨링 0** — cmd_vel은 텔레옵에서, 차선 세그/차량 bbox는
freeze된 SegFormer/YOLO가 자동 생성한다.

---

## 0단계 — 라벨 검증

```bash
cd final_project/ros2_ws
source install/setup.bash
ros2 launch rover_recorder record.launch.py session_name:=phase2_preview overlay_viz:=true
```

`http://<젯슨-IP>:8080/`에서 raw `lane`/`front`와 함께 `lane_seg`/`front_det` 오버레이가
뜬다. 이 미리보기는 `extract_labels.py`의 ROI crop, 224 resize, SegFormer 색상 합성,
YOLO bbox 합성 계약을 그대로 재사용한다.

---

## 1단계 — 대량 rosbag 수집

Phase 1과 동일한 스택 사용:

```bash
cd ~/team/final_project/ros2_ws
colcon build --symlink-install
source install/setup.bash

# 터미널 A — 카메라 + 모터 + bag 레코더 http://192.168.0.123:8080/
cd ~/team/final_project/ros2_ws && source install/setup.bash
ros2 launch rover_recorder record.launch.py session_name:=phase2

# 터미널 B — 키보드 텔레옵 (별도 SSH TTY)
cd ~/team/final_project/ros2_ws && source install/setup.bash
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
cd data_pipeline
python3 extract_labels.py \
    --bag ../rover_data/phase2_20260610_164824/bag \
    --segformer_ckpt ../models/segformer_lane \
    --yolo_weights ../models/best.pt \
    --out ../labels_cache12.h5 \
    --device cuda
```

- Phase 1 산출물 배치: `final_project/models/best.pt` (YOLO26) +
  `final_project/models/segformer_lane/` (SegFormer 폴더, config.json +
  model.safetensors + preprocessor_config.json).
- `--segformer_ckpt` **필수** (없으면 에러). `--yolo_weights` 기본값 `models/best.pt`.
- `--skip_det` : YOLO 생략 (차량 없는 차선 전용 bag일 때).
- `--limit N` : 앞 N 프레임만 (빠른 디버그용, 0=전체).
- `--debug_dir DIR` : **줄 때만** 검증용 디버그 PNG(~100장) 저장. 생략하면 안 만든다
  (bag마다 같은 폴더에 쌓이지 않게 기본은 끔). 첫 검증 때만 한 번 주면 충분.
- bag 여러 개면 `--bag_root ... --out_dir labels_all`로 한 번에 추출한다. 각 bag은
  `<out_dir>/<세션명>.h5`로 저장되고, 모델은 한 번만 로드된다.

### labels_cache.h5 스키마

각 lane 프레임 t마다 가장 가까운 front/cmd_vel을 ±50 ms로 동기화해 1 샘플 생성:

> **두 카메라 정합 기준**: lane↔front 매칭은 **`header.stamp`(캡처 시각)** 로 한다.
> camera_node가 reader 스레드에서 프레임을 받은 직후 시각을 stamp에 찍으므로,
> JPEG 인코딩·DDS 송신·publish 순서로 누적되는 지연이 stamp에 실리지 않는다.
> cmd_vel/steer는 header가 없어 lane↔cmd 매칭·waypoint 적분만 rosbag write 시각
> 끼리 비교한다(두 시계를 섞지 않음). 옛 bag(stamp 미설정)은 write 시각으로 폴백.
> 두 CSI 카메라는 free-running이라 캡처 위상차(~한 프레임 이하, 측정 median ≈26 ms)
> 자체는 genlock 없이 못 없애지만, 위 매칭이 캡처 시각 기준이라 항상 최근접 프레임을
> 짝짓는다.

| 데이터셋 | shape | dtype | 내용 |
|---|---|---|---|
| `lane` | (224,224,3) | uint8 | lane 이미지, 상단 ROI 크롭 후 resize (**BGR**) |
| `front` | (224,224,3) | uint8 | raw front 이미지, resize (**BGR**) |
| `seg` | (3,224,224) | uint8 | SegFormer 마스크 {0,255}. ch0=좌실선/ch1=우실선/ch2=중앙점선 |
| `det` | (5,) | float32 | YOLO car bbox `[x,y,w,h,conf]` (front 픽셀). 차 없으면 0 |
| `waypoint` | (5,2) | float32 | cmd_vel 적분 미래 2.5초 궤적 (로봇 프레임, 미터) |
| `steer` | () | float32 | angular.z at t |
| `throttle` | () | float32 | linear.x at t |
| `timestamp_ns` | () | int64 | lane 프레임 **캡처 시각**(`header.stamp`) |

> **색공간 주의**: lane/front는 **BGR**로 저장된다(추출이 wire JPEG를 BGR로 디코드 후
> resize). 학습 데이터로더/오버레이 합성/추론 전처리가 모두 BGR로 일관되면 된다.
>
> **lane ROI 크롭**: lane 경로는 224 resize **직전에** `crop_lane_roi`로 상단
> `LANE_CROP_TOP`(extract_labels.py 상수, 현재 0.30) 비율을 잘라낸다. 0.0이면
> 크롭하지 않는다. 차선 카메라 상단이 도로 밖 배경이면 라벨링 **전에** 이 값을 정한다. 한 번 정하면
> 라벨링·추출·추론이 **모두 같은 값**을 써야 좌표계가 맞는다(라벨링 후 변경 금지).
> front/YOLO는 크롭하지 않는다.

---

## 3단계 — E2E 학습 (Kaggle/Colab)

> 구현됨: [training/dataset.py](training/dataset.py)(H5→오버레이 합성 Dataset),
> [training/train_e2e.py](training/train_e2e.py)(학습 루프),
> [training/export_onnx.py](training/export_onnx.py)(5단계 ONNX),
> [training/e2e-train_kaggle.ipynb](training/e2e-train_kaggle.ipynb)(Kaggle/Colab).

모델: [model.py](model.py) 의 `E2ENet`
- `LaneEncoder` / `FrontEncoder` (ResNet18×2, ImageNet pretrained) → 각 256-d
- concat(512) → `ControlHead`(steer, throttle Tanh) + `WaypointHead`(5×2 미터)

### 오버레이 합성 계약 (학습 데이터로더가 H5에서 구성 — 추출 헤더와 동일해야 함)
- **lane 입력** = raw lane 이미지에 seg 3채널을 색으로 alpha-blend
  (ch0=red, ch1=green, ch2=blue; `visualize_labels.overlay_seg`와 동일 방식).
- **front 입력** = raw front 이미지에 car bbox 그림 (`det[4] > 0`일 때).
- 둘 다 합성까지는 **BGR**. 텐서화 직전에 `to_input_tensor()`가 BGR→RGB 변환 후
  ImageNet mean/std로 정규화한다. 추론 노드도 같은 함수를 import해 픽셀 계약을 맞춘다.

### 손실
```
loss = 1.0·MSE(steer) + 0.5·MSE(throttle) + 0.5·MSE(waypoint)   # E2ELoss 기본값
```
waypoint는 보조 task로 학습한다. 실차 추론에서는 launch 파라미터로 `head` 조향 또는
`waypoint` 기반 조향을 고를 수 있고, 현재 검증 기본값은 `steer_source:=waypoint`,
`steer_mode:=pursuit`이다. throttle 출력은 진단용으로 유지하되 실제 속도는 검증된
`|steer|` coupling으로 만든다.

### 학습 전 H5 audit / legacy waypoint 부호

학습 전에 H5 분포와 waypoint 부호 설정을 확인한다:

```bash
cd final_project/training
python3 audit_h5.py --cache phase2_*.h5

# 옛 extract_labels 부호 버그로 만든 legacy H5는 Kaggle 노트북과 동일하게:
python3 audit_h5.py --cache phase2_*.h5 --wp_fix_sign
```

- **legacy H5**: `WP_FIX_SIGN=True` 로 두고 `train_e2e.py`에 `--wp_fix_sign`을 넘긴다.
  이 옵션은 `waypoint[:, 0]`(전방 x)만 반전한다.
- **수정 후 새 H5**: `WP_FIX_SIGN=False` 로 두고 변환 없이 학습한다.
- `zero-control ratio`, `det ratio`, `steer/throttle` 범위, effective waypoint 방향을 보고
  의도한 세션만 학습에 포함한다. 음수 raw waypoint x 자체는 legacy H5에서는 정상이다.

### Colab 흐름 (`training/e2e-train_kaggle.ipynb`)
1. `labels_cache.h5` 업로드(또는 Drive 마운트) → `Dataset`이 H5에서 오버레이 합성
2. `E2ENet` 학습 (train/val split, AdamW, early-stop)
3. `WP_FIX_SIGN`, `AVOID_OVERSAMPLE`, `STEER_SMOOTH`, `WAYPOINT_WEIGHT`,
   `RESUME`/`FINETUNE` 값을 데이터 목적에 맞게 설정
4. best 체크포인트 저장 → 5단계 ONNX export로 연결

---

## 4단계 — ONNX export

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

추론(실주행)에서는 `waypoints`를 디버그 오버레이에 그리고, 기본 설정에서는 waypoint 기반
pure-pursuit 조향에도 사용할 수 있다. `steer_source:=head`로 바꾸면 ControlHead steer를
직접 사용한다.

---

## 다음 → Phase 3 (배포 · 추론 · 모니터링 · 추가학습)

`e2e.onnx`가 나오면 여기서부터는 **[PHASE3.md](PHASE3.md)**:

- Jetson에서 `trtexec --fp16`로 `e2e.engine` 빌드 (engine은 Jetson에서만)
- `rover_lane` 추론 노드: 전처리(학습과 픽셀 단위 동일) → engine → cmd_vel 역변환
- 주행 중 의도 모니터링 (`viz.draw_intent`, :8080 오버레이)
- 새 bag으로 `--resume` 추가학습 루프

---

## DDS / QoS / SSH / Colab 메모

- **DDS/QoS**: 단일 호스트(Jetson) 안에서만 ROS2를 돌리면
  `export ROS_LOCALHOST_ONLY=1`로 외부 노드·멀티캐스트를 차단해 지연/간섭을 줄인다.
  이미지 토픽은 대용량이라 **실시간 소비자**(monitor_node·overlay_viz_node)는
  `BEST_EFFORT + KEEP_LAST depth=1` QoS를 쓴다 — 밀린 옛 프레임을 버리고 최신만
  처리해 지연 누적을 끊는다(트레이드오프: 프레임 드롭). 단 **학습 bag을 저장하는
  bag_recorder 는 RELIABLE + 넉넉한 depth 유지** — 데이터 완결성이 필요해 프레임을
  버리면 안 된다(빠진 프레임은 extract_labels 동기화·waypoint 적분을 망친다).
  camera_node 송신은 `RELIABLE + depth=1`(송신측 옛 프레임 미보관) — RELIABLE이라
  recorder(RELIABLE sub)와 매칭되고, BEST_EFFORT 소비자 sub과도 매칭된다.
  (큐 적체 지연만 QoS로 끊는 것이고, 추론 처리시간 자체 지연은 TensorRT fp16의 몫.)
- **카메라 timestamp**: camera_node는 `header.stamp`에 publish 시각이 아니라
  **캡처 직후 시각**을 찍는다(reader 스레드가 프레임 받은 순간). 두 카메라가
  free-running이라 위상이 어긋나는데, 캡처 시각을 실어야 extract_labels가
  lane↔front를 캡처 기준으로 정확히 매칭한다(인코딩/송신 순서 지연 비반영).
- **SSH**: 카메라/모터는 Jetson 로컬 하드웨어 → 모든 노드는 Jetson에서 실행, 노트북은
  SSH 포트포워딩 또는 같은 네트워크에서 브라우저 모니터(`:8080`)로 본다.
- **Colab**: GPU 런타임 + CVAT export zip 업로드(Phase 1). 학습 산출물(best.pt /
  segformer_lane / e2e_best.pt)은 브라우저로 받아 WSL/Jetson에 배치.

---

## 완료 기준

- [ ] 대량 rosbag 수집 (다양한 차선 + 정지차 회피 시퀀스)
- [ ] `extract_labels.py`로 `labels_cache.h5` 생성 + `visualize_labels.py` 눈 검증 통과
- [x] E2E 학습 코드 구현 (dataset/train/export + Colab 노트북 + 의도 시각화 + 이어학습)
- [ ] 실제 데이터로 E2E 학습 → `e2e.onnx` export

> 이후(engine 빌드 · 실차 추론 · 주행 모니터링 · 추가학습)는 [PHASE3.md](PHASE3.md) 완료 기준 참고.
