# AUE4040 — Autonomous Rover (Personal Research)

closed-world 회전교차로 자율주행 로버 프로젝트. NVIDIA Jetson Orin Nano + ROS 2 Humble + TensorRT 10 기반.

좌/우 두 미션 중 하나가 런타임에 latch되고, 로버는 **E2E BC 모델**로 차선을 추적하면서 표지판·신호·차량·사람은 **YOLO + 손코딩 FSM**으로 처리한다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Repo layout](#2-repo-layout)
3. [E2E BC 모델 (차선추적)](#3-e2e-bc-모델-차선추적)
4. [YOLO 모델 (객체인식)](#4-yolo-모델-객체인식)
5. [통합 주행 (FSM + ROS 노드)](#5-통합-주행-fsm--ros-노드)
6. [Build & Run](#6-build--run)
7. [캘리브레이션](#7-캘리브레이션)
8. [참고 자료](#8-참고-자료)

---

## 1. 프로젝트 개요

### 무엇을 하는 프로젝트인가
closed-world 회전교차로 트랙에서 로버가:
1. **차선 추적** — E2E BC (Behavior Cloning) 모델 3개 (`common` / `left` / `right`)이 단일 rectified 이미지에서 (steer, speed)를 직접 출력
2. **객체 인식** — YOLOv8n 7-class fine-tune (`car`, `green`, `left`, `person`, `red`, `right`, `stop`)
3. **상태 결정** — 손코딩 FSM이 `COMMON / SLOW / WAITING / STOPPED / TURNING / ARRIVED` 전이
4. **모터 제어** — FSM이 게이트한 (steer, throttle)을 UART JSON으로 base controller에 전송

### 핵심 설계 결정
- **closed-world 가정**: 단일 차량, 고정 표지판 → 스테레오 disparity 없이 bbox 높이만으로 거리 계산 (`d = K / bbox_h_px`)
- **단일 이미지 진입점**: 듀얼 CSI는 rectify에만 사용하고 다운스트림은 좌측 한 장만 소비 (`/image_rectified`)
- **TensorRT 10 직접 호출**: ultralytics·pycuda 의존성 없이 torch CUDA 버퍼만 사용 → Orin Nano 30Hz 예산 확보
- **미션 latch + grace window**: 표지판은 1회만 latch되고, 출발 후 N초 경과 시점에 자동 분기 (분기 시점에 표지판이 다시 보일 필요 없음)

### 용어 정리
- **BC (Behavior Cloning)**: 사람이 텔레옵 주행 중 누른 키 + 그때의 이미지를 학습해 같은 이미지에서 같은 액션을 내도록 만드는 모방학습
- **E2E BC**: end-to-end BC. 차선검출 같은 중간 단계 없이 이미지 → 액션을 한 번에 학습
- 이 프로젝트에서 *lane 모델* = *BC 모델* = *E2E 모델* 모두 동일 (`rover_lane` 패키지)

---

## 2. Repo layout

```
AUE4040/
├── main/                       ← 자율주행 본 코드
│   ├── README.md               ← main/ 내부 빌드/실행 상세
│   ├── PROJECT_PLAN.md         ← 설계 결정·진행상황 (체크박스)
│   ├── ros2_ws/                ← colcon 워크스페이스
│   │   ├── src/                ← 7개 패키지 (아래 §5 표 참조)
│   │   └── models/             ← TRT engine + metadata.json (gitignored)
│   ├── infer_local.py          ← best.pt로 ROS 없이 로컬 추론
│   ├── train_yolo_colab.ipynb  ← Colab YOLO 파인튜닝
│   ├── best.pt                 ← 학습된 YOLO 가중치 (gitignored)
│   ├── inference/              ← 로컬 추론 입력
│   ├── inference_out/          ← 로컬 추론 결과 (bbox 그려진 이미지)
│   └── labeling/               ← 데이터셋 라벨링 보조 스크립트
├── calibration/                ← IMX219 듀얼 CSI 캘리브레이션 (jetcam 기반)
├── control/                    ← UART 베이스 컨트롤러 참조 구현 (HYU-ECL3003 포팅 원본)
├── tutorials/                  ← Linux/SSH/quantization/ROS 2 입문
└── README.md                   ← 이 파일
```

---

## 3. E2E BC 모델 (차선추적)

### 3.1 모델 구조
- **백본**: ResNet18 (ImageNet pretrained, 마지막 FC 제거)
- **입력**: 224×224 BGR→RGB (ImageNet 정규화) + step 스칼라 (정규화 진행도 [0,1])
- **step embedding**: 1-D scalar → 32-D MLP로 임베딩 (단순 concat하면 FC가 무시함)
- **출력**: 6-class action logits (up / down / left / right / straight / space)
- **postprocess**: argmax → `ACTIONS[idx]`로 (steer, speed) 룩업

### 3.2 왜 회귀가 아닌 action classification?
- 사람 텔레옵 키 입력이 이미 6개 이산 액션
- 회귀로 풀면 평균값 액션이 자주 나와 "약하게 휘기"가 됨 (jerky and short turns)
- 6-class classification + 키별 (steer, speed) 평균 lookup이 더 안정적

### 3.3 왜 step 입력?
- 회전구간이 trajectory의 어디쯤인지 모델에 알려줌
- 같은 이미지라도 시작/중간/끝에서 다른 액션을 낼 수 있음 (예: 회전 시작 직후엔 살짝, 끝나갈 땐 풀기)
- step은 추론 중 자체 카운터로 증가, 모델 스왑 시 0으로 리셋

### 3.4 학습 흐름
```
teleop 주행 (record_and_label.ipynb)
  → annotation.txt: filename xpos ypos segment steer_tel speed_tel last_key step
    │
    ▼
preprocess.py
  - key → action_idx (up=0, down=1, left=2, right=3, straight=4, space=5)
  - segment별 분리 (common / left / right)
  - 세션 단위 train/test split (행 단위로 나누면 인접 프레임 leak)
  - down/space (텔레옵 보정 키) 기본 제외
    │
    ▼
train_e2e.py
  - segment별로 ActionCNN 학습
  - class-balanced sampling, weighted CE loss
  - metadata.json에 step_max 기록 (추론 정규화에 필요)
    │
    ▼
export_to_onnx.py → trtexec (Jetson) → e2e_{common,left,right}.engine
```

### 3.5 런타임 흐름
```
/image_rectified (bgr8)
  → lane_node
  → preprocess (224×224, ImageNet norm)
  → step counter ++ (현재 active 모델의 카운터)
  → TensorRT 추론 (~7.7ms Python e2e, ~0.9ms pure GPU)
  → argmax(logits) → ACTIONS[idx] = (steer, speed)
  → /bc_cmd (geometry_msgs/Twist)
```

**모델 스왑**: `/active_model` 토픽(`"common"` / `"left"` / `"right"`)으로 전환. 스왑 시 `set_active()`가 새 모델의 step을 0으로 리셋 (각 segment 학습 시 0부터 시작했기 때문에 의미를 맞춤).

### 3.6 액션 매핑 (현재값)
키별 `(steer_tel, speed_tel)` 평균:
```
0 UP        ( 0.000, -0.143)
1 DOWN      ( 0.000,  0.000)   학습 제외
2 LEFT      (-0.799, -0.263)
3 RIGHT     (+0.794, -0.263)
4 STRAIGHT  ( 0.000, -0.152)
5 SPACE     ( 0.000,  0.000)   학습 제외
```
부호 컨벤션: **negative speed = forward** (텔레옵 원본).

---

## 4. YOLO 모델 (객체인식)

### 4.1 클래스
7개 — `best.pt` 학습 인덱스 순:
```
0 car      1 green   2 left     3 person   4 red    5 right   6 stop
```
- `yellow`는 클래스 자체가 없음 → 감지 안 됨, 평시 주행 유지

### 4.2 학습 흐름
```
Roboflow 데이터셋 (7-class, train/valid/test)
  - Auto-Orient: ON
  - Resize: OFF (ultralytics가 letterbox 처리하므로 stretch resize는 왜곡 발생)
    │
    ▼
Colab에서 fine-tune (train_yolo_colab.ipynb)
  yolo train model=yolov8n.pt data=data.yaml imgsz=320 epochs=N
           mosaic=0.5 mixup=0 (hsv 튜닝)
    │
    ▼
best.pt (drive에 저장)
    │
    ▼
yolo export model=best.pt format=onnx imgsz=320 opset=12
    │
    ▼ (Jetson에서)
trtexec --onnx=best.onnx --saveEngine=yolov8n.engine --fp16
```

### 4.3 왜 imgsz=320?
- Orin Nano 추론 예산 (stock 640 대비 ~4배 빠름)
- 표지판/차량 모두 충분히 큰 객체 — 작은 객체 detection 필요 없음

### 4.4 왜 ultralytics가 아닌 TensorRT 직접 호출?
- ultralytics import만 ~수백ms (런타임에 부담)
- pycuda 의존성 제거 → torch CUDA 버퍼만 사용
- 자체 letterbox + numpy NMS 구현 (`yolo_inference.py`)

### 4.5 런타임 흐름
```
/image_rectified (bgr8)
  → yolo_node
  → detect_every_n=2 데시메이션 (표지판 상태는 빠르게 안 바뀜)
  → letterbox 320×320 (aspect 유지 + 114 회색 패딩)
  → BGR→RGB, /255, CHW transpose
  → TensorRT 추론 (5~8ms 예상)
  → conf 임계 + per-class numpy NMS
  → un-letterbox → 원본 좌표계 bbox
  → /detections (DetectionArray: class_name, score, xyxy)
```

### 4.6 알려진 이슈와 해소
**문제**: NMS는 클래스 간 중복을 거르지 않아서 같은 표지판에 left/right 박스가 함께 잡힘.
**해소**: decision_node에서 프레임당 score 큰 쪽만 채택해 stabilizer 카운터에 반영. → 안정화 임계 도달 시 score가 우세했던 쪽으로 mission latch.

### 4.7 거리 추정
스테레오 disparity 미사용. 단일 차량 가정으로:
```
d = K / bbox_h_px        K = bbox_h_px × d (px·m)
```
운영 좌표계(rectified+ROI crop)에서 한 번 측정 → `vehicle_dist_K`에 박음. 현재값 `K=156.2` (bbox_h=390.5px @ 0.4m).

---

## 5. 통합 주행 (FSM + ROS 노드)

### 5.1 노드 토폴로지
```
[CSI L/R 카메라]
       ↓
rover_stereo (rectify + ROI crop)         ← 단일 이미지 진입점
       ↓ /image_rectified
   ┌───┴────────────────┐
   ▼                    ▼
rover_perception   rover_lane
(YOLO TRT)         (BC TRT, model_manager)
   ↓                    ↓
/detections         /bc_cmd
   └─────────┬──────────┘
             ▼
       rover_decision     ← Stabilizer + FSM + 거리 게이팅
             ↓
   /cmd_vel, /active_model, /fsm_state
             ↓
       rover_control      ← FSM SAFE 게이트 + 좌/우 모터 믹싱 + UART
             ↓
       rover MCU → motors
```

### 5.2 ros2_ws 패키지 표

| 패키지 | 역할 | 주 토픽 |
|---|---|---|
| `rover_msgs` | 커스텀 메시지 | `Detection`, `DetectionArray`, `FSMState` |
| `rover_bringup` | launch + `config/params.yaml` (단일 진실원) | — |
| `rover_stereo` | 듀얼 CSI 캡처 + rectify+ROI crop | pub `/image_rectified` |
| `rover_perception` | YOLO TRT 추론 | pub `/detections` |
| `rover_lane` | BC TRT 추론 + 모델 스왑 | pub `/bc_cmd` |
| `rover_decision` | 안정화 + FSM + 미션 latch + 거리 게이팅 | pub `/cmd_vel`, `/active_model`, `/fsm_state` |
| `rover_control` | FSM 게이트 + 모터 믹싱 + UART | sub `/cmd_vel`, `/fsm_state` |
| `rover_recorder` | 데이터 수집 노드 + teleop 스크립트 | — |
| `rover_training` | preprocess + train_e2e + ONNX export (ROS 외부) | — |

### 5.3 FSM 상태
| 상태 | 의미 | throttle |
|---|---|---|
| `COMMON` | 평시 주행 (common BC 모델) | 모델 출력 |
| `SLOW` | person 감지 → 감속 | `-0.13` 고정 (평시 ~`-0.18` 대비 약 0.05 감속) |
| `WAITING` | 전방 차량 가까움 → 정지 | 0 |
| `STOPPED` | stop/red/turn-trigger로 정지 | 0 |
| `TURNING` | 분기 진행 (mission BC 모델) | 모델 출력 |
| `ARRIVED` | 최종 정지 (lane_lost) | 0 |

`STOPPED.entered_by`로 진입 원인 추적 (`"stop"` / `"red"` / `"turn"`) → 해제 조건 분기.

### 5.4 라벨별 처리 로직

| 라벨 | 안정화(연속 N프레임) | 추가 조건 | 진입 상태 | 동작 | 해제 |
|------|---|---|---|---|---|
| `stop` | 4 | – | `STOPPED(stop)` | 정지 | 2초 타이머 |
| `red` | 4 | – | `STOPPED(red)` | 정지 | green stable OR red 끊김 |
| `green` | 4 | – | (해제 신호) | – | – |
| `left` | 5 | – | mission latch (left) | – | latch 후 무시 |
| `right` | 5 | – | mission latch (right) | – | latch 후 무시 |
| `person` | 4 | – | `SLOW` | -0.13 | 3초 타이머 |
| `car` | 4 | bbox_h≥390px | `WAITING` | 정지 | 거리 멀어지면 prev_state |
| (분기) | – | grace window 경과 | `STOPPED(turn)→TURNING` | 정지 → 모델 스왑 | `lane_lost` → `ARRIVED` |

**우선순위**: `stop` > `red` > `car`(close) > `person` > 분기 트리거 > 일반 주행.

### 5.5 미션 분기 메커니즘 (핵심 흐름)
```
1. 출발 → COMMON 상태, common BC 모델로 차선 추적
2. left/right 표지판이 5프레임 연속 안정 감지
   → self.mission = "left" or "right" latch
   → 이후 turn-sign 감지는 무시 (한 번만 결정)
3. 출발 후 common_grace_s 경과
   → 자동 STOPPED(entered_by="turn") 진입, 2초 정지
4. stop 타이머 만료 → TURNING 상태로 전이
   → active_model() = mission 반환
   → /active_model 발행 → lane_node가 모델 스왑 (step 카운터 리셋)
5. mission 모델로 회전 구간 주행
6. lane_lost 감지 → ARRIVED (최종 정지)
```

**핵심 trick**:
- 분기 시점에 표지판이 다시 보일 필요 없음 (`self.mission`이 latch 시 기억)
- grace window가 출발 직전 표지판을 잘못 trigger하는 것을 방지
- `allowed_missions=["right"]` 설정으로 left 모델이 없을 때 stray 감지로 인한 오latch 차단

### 5.6 Stabilizer (노이즈 필터)
- 클래스별 연속 감지 카운터 유지
- 한 프레임이라도 사라지면 0으로 리셋
- 임계 N프레임 도달 시 `stable=True`
- → 단일 프레임 오인식 무시, 시각적으로 의미 있는 감지만 FSM에 전달

---

## 6. Build & Run

### 6.1 모델 산출물 위치 (gitignored)
```
main/ros2_ws/models/
├── yolov8n.engine            ← YOLO TRT (best.pt → ONNX → engine, Jetson에서)
├── e2e_common.engine         ← BC 공통 구간
├── e2e_common.metadata.json  ← step_max 등
├── e2e_right.engine          ← BC 우회전 구간 (또는 e2e_left.engine)
└── e2e_right.metadata.json
```

### 6.2 Jetson에서 빌드 + 실행
```bash
cd ~/Personal_Research/AUE4040/main/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch rover_bringup autonomous.launch.py
```

미션은 launch 인자가 아니라 **런타임 latch** — `params.yaml`의 `allowed_missions`로만 제한.

### 6.3 TRT engine 변환 (Jetson 전용)
WSL/노트북에서 만든 engine은 GPU 아키텍처가 달라 Jetson에서 사용 불가.
```bash
# YOLO
yolo export model=best.pt format=onnx imgsz=320 opset=12
trtexec --onnx=best.onnx \
        --saveEngine=$HOME/Personal_Research/AUE4040/main/ros2_ws/models/yolov8n.engine \
        --fp16
# (trtexec 없으면 /usr/src/tensorrt/bin/trtexec)
```

### 6.4 로컬 추론 (개발용, WSL/노트북)
```bash
conda activate py310
cd ~/Personal_Research/AUE4040/main
python infer_local.py                    # main/inference/ 전체
python infer_local.py /path/to/img.jpg   # 단일 이미지
python infer_local.py 0                  # 웹캠
python infer_local.py --conf 0.15        # 임계 조정
```
결과는 `main/inference_out/run/`에 bbox 그려진 이미지로 저장.

---

## 7. 캘리브레이션

### 7.1 스테레오 캘리브 freeze
[`main/ros2_ws/src/rover_stereo/config/stereo_calib.yaml`](main/ros2_ws/src/rover_stereo/config/stereo_calib.yaml)은 카메라 마운트 확정 직후 한 번 생성한 뒤 학기 동안 재생성 금지. 이 파일이 바뀌면 그 이전 BC 학습 데이터가 무효화됨. 카메라가 흔들리면 *재캘리브가 아니라 마운트 물리 복원*으로 대응.

### 7.2 차량 거리 K 캘리브
거리 측정은 closed-world 단일 차량 가정 → `d = K / bbox_h_px`. 운영 좌표계(rectified+ROI crop)에서 측정해야 정확.

**방법 A — `/image_rectified` 프레임에서 직접 측정**:
```bash
ros2 launch rover_bringup debug.launch.py
# 다른 터미널에서 차량을 정지 임계 위치에 두고:
python3 main/ros2_ws/src/rover_perception/scripts/calibrate_vehicle_distance.py \
        --distance 0.4 --write
```
`--write`가 `params.yaml`의 `vehicle_dist_K`를 자동 갱신.

**방법 B — 사진 1장으로 측정** (현재 적용된 방식):
임계 위치에서 찍은 사진에서 YOLO가 그린 bbox 높이 `h`를 읽고 `K = h × safe_dist_m`로 계산. 현재값: `K = 390.5 × 0.4 = 156.2`.

---

## 8. 프로젝트 요지 (발표용 요약)

### E2E 주행 AI
Jetson 자체 카메라 드라이버와 주행 제어 명령 동기화를 위해 Python 환경을 사용. `record_and_label` ipynb 하나로 데이터 수집과 주행 명령을 동시에 수행해 GT(ground truth) 데이터 수집을 용이하게 함. 학습은 Colab에서 git clone 후 ResNet18 파인튜닝. 상위 주행 명령(키보드 버튼)을 라벨로 사용해 학습 난이도를 낮춤. 학습된 `.pt`는 Jetson에서 TensorRT `.engine`으로 변환해 사용.

회전교차로까지 가는 `common` 모델, 회전 표지판에 따라 분기하는 `right` / `left` 모델을 **별도로 학습** — 유사한 경로 데이터가 누적되어 서로 다른 task 간 액션이 평균화되는 현상을 사전 방지.

### 표지판 인지
YOLOv8n 사용. Robust한 데이터 확보를 위해 rover 카메라 촬영 이미지와 핸드폰 촬영 이미지를 모두 수집하고 Roboflow의 augmentation을 활용. GPU 가속을 위해 Colab에서 드라이브 연동 후 파인튜닝. 결과 `.pt`를 Jetson으로 옮겨 TensorRT `.engine`으로 변환해 사용.

### 실주행
여러 노드와 두 모델 실행을 용이하게 하기 위해 ROS 2 사용. 카메라는 기존 Jetson 드라이버로 띄운 스트림을 ROS 2 토픽으로 변환해 publish. 주행에 필요한 모든 노드를 순차적으로 동시에 띄우는 launch 파일 구성.

**안전 처리**: `try/except KeyboardInterrupt`로 Ctrl+C 시 모터에 throttle 0 명령을 명시적으로 보내 종료 — 일반적인 Ctrl+C 종료 시 마지막 cmd_vel이 유지되어 로버가 계속 전진하려 하는 문제를 사전 해결 (`control_node.stop_motors()`).

---

## 9. 참고 자료

- [`main/PROJECT_PLAN.md`](main/PROJECT_PLAN.md) — 설계 결정 + 체크박스 진행상황
- [`main/README.md`](main/README.md) — main/ 내부 빌드/실행/학습 상세
- [`calibration/README.md`](calibration/README.md) — 스테레오 캘리브 도구 사용법
- [`tutorials/`](tutorials/) — Linux / SSH / quantization / ROS 2 입문

### 외부 의존성 (HYU-ECL3003 재사용)
- `rover_control/motor_driver.py` ← `control/base_ctrl.py`
- `rover_stereo/` jetcam wrapper ← `~/team/calibration/camera.py` 경로 의존 (Jetson)
- `rover_stereo/calib/capture_stereo.py` ← HYU stereo_depth_tutorial 포팅
- `rover_perception/yolo_inference.py` ← `rover_lane/center_inference.py` TRT 패턴 차용
