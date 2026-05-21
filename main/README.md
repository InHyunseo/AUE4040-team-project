# Rover Autonomous Driving — Roundabout (closed-world)

회전교차로 자율주행 로버 프로젝트. NVIDIA Jetson Orin Nano + ROS 2 Humble 기반.

학기 프로젝트 차원에서 closed-world (정해진 트랙) 환경의 자율주행 로버를 만든다. 좌/우 두 미션 중 하나가 시작 시 선택되고, 로버는 도로 중심점을 회귀하는 CNN으로 차선을 추적하면서 표지판·신호·전방차량은 YOLO + hand-coded FSM으로 처리한다. 자세한 설계 결정과 진행 상황은 [PROJECT_PLAN.md](PROJECT_PLAN.md) 참조.

## 무엇을 하는 레포인가

이 레포는 **로버 한 대를 자율주행시키기 위한 모든 코드**를 담는다. 크게 세 축:

1. **데이터 수집** — Jetson에 연결된 듀얼 CSI 카메라로 영상을 찍고, 키보드 텔레옵으로 로버를 직접 운전하면서 (이미지, 시연한 조향값) 쌍을 저장한다. 이후 노트북에서 각 이미지에 도로 중심점을 수동 클릭으로 라벨링한다. YOLO용으로는 표지판·신호·차량 클래스를 별도로 수집·라벨링한다.
2. **학습** — 라벨된 데이터로 (a) **lane**: ResNet18 기반 회귀 CNN을 공통 / 좌회전 / 우회전 세 구간별로 학습, (b) **YOLO**: 7-class fine-tune (`traffic_light_{red,green,yellow}`, `stop_sign`, `vehicle`, `turn_{left,right}_sign`).
3. **추론·통합** — 학습된 모델을 ONNX → TensorRT engine으로 변환해 Jetson에 올리고, ROS 2 노드들이 카메라 → YOLO + lane → FSM → 모터까지 한 파이프라인으로 묶어 자율주행한다. 추론은 모두 TensorRT 10 Python API를 직접 호출 (ultralytics / pycuda 의존성 없음) — Orin Nano에서 동시 실행해도 30 Hz 예산 안에 들어가도록 설계.

> **2026-05-21 현재 상태:** 인프라·lane 파이프라인·YOLO 추론·FSM·모터 wire-up 모두 합성 데이터/스톡 모델로 end-to-end 검증 완료. 실제 트랙·표지판·차량이 마련되는 대로 데이터 수집 → 학습 → 배포로 바로 진행할 수 있는 상태.

## Layout

```
main/
├── PROJECT_PLAN.md             # 프로젝트 계획서 (체크박스 + 설명)
├── ros2_ws/                    # ROS 2 워크스페이스
│   └── src/
│       ├── rover_bringup/      # launch + config (params.yaml 등)
│       ├── rover_msgs/         # 커스텀 메시지 (RoadCenter, Detection, FSMState)
│       ├── rover_stereo/       # 듀얼 CSI rectify-only (단일 이미지 진입점, disparity 안 씀)
│       ├── rover_perception/   # YOLO 검출 노드 (TRT 10 직접 호출) + bbox 거리 캘리브
│       ├── rover_lane/         # 도로 중심점 회귀 노드 (ex. rover_pilotnet)
│       ├── rover_decision/     # FSM + 안전 + 미션
│       ├── rover_control/      # 모터 (UART JSON) 노드
│       ├── rover_recorder/     # 데이터 수집 (notebook 기반)
│       └── rover_training/     # 학습 스크립트 (ROS 외부)
└── README.md                   # 이 파일
```

## 동작 흐름 (런타임)

```
[CSI 듀얼 카메라]
       │
       ▼
[rover_stereo]  rectify only (frozen LUT)  →  /image_rectified
       │
       ├──────────────────────────────┐
       ▼                              ▼
[rover_perception]              [rover_lane]
   YOLOv8n TRT 320 FP16          ResNet18 TRT 224 FP16
   7-class (detect_every_n=2)    active_model: common/left/right
   /detections                   /road_center  (x, y, model_tag)
       │                              │
       └──────────────┬───────────────┘
                      ▼
               [rover_decision]
        Stabilizer + FSM + bbox→거리 + mission 교차검증
                      │
       ┌──────────────┼────────────────┐
       ▼              ▼                ▼
    /cmd_vel    /active_model     /fsm_state
                                       │
       ▼                               │
[rover_control] ◄──────────────────────┘
   FSM SAFE 상태 throttle gate + (steer,throttle)→(L,R) + UART JSON
       │
       ▼
   rover MCU → 모터
```

> 주의: `/vehicle_distance` 토픽은 없다. 차량 거리는 `decision_node`가 `Detection.vehicle`의 bbox 높이에서 직접 계산 (`d = K / bbox_h`).

자세한 데이터 흐름과 결정 이유는 PROJECT_PLAN.md §3 / §4-A 참조.

## Build / Run

### 빌드

`~/.bashrc`에 정의된 `ros_start` 함수를 쓰는 게 가장 간단:

```bash
ros_start    # cd ~/team/main/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

수동:

```bash
cd ~/team/main/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 데이터 수집 (현재 권장 경로: 노트북)

```bash
# Part A: 운전하면서 이미지 + 텔레옵 로그 저장
# Part B: 저장된 이미지마다 도로 중심점을 마우스로 클릭해 라벨링
jupyter notebook ros2_ws/src/rover_recorder/notebooks/record_and_label.ipynb
```

ROS 2 노드 기반 recorder도 있지만 현재는 노트북 쪽이 안정적.

### 학습 파이프라인 (현재는 합성 데이터로만 검증됨)

```bash
cd ros2_ws/src/rover_training/scripts

# 1) 원본 세션을 segment별로 분리
python3 preprocess.py --raw ~/data/raw --out ~/data/processed

# 2) 각 segment마다 학습 (CenterDataset이 CWD에서 이미지를 찾음 — 주의)
cd ~/data/processed/common/images
python3 ~/team/main/ros2_ws/src/rover_training/scripts/train_center.py \
    --segment common --data ~/data/processed/common --epochs 20 \
    --out ~/models/center_common.pth

# 3) ONNX → TensorRT engine
python3 export_to_onnx.py --ckpt ~/models/center_common.pth --out ~/models/center_common.onnx
bash export_to_trt.sh ~/models/center_common.onnx ~/models/center_common.engine fp16
```

좌/우 모델도 같은 방식으로 학습. 최종 3개 `.engine`이 `rover_lane`이 로드할 산출물.

### YOLO 학습 / 배포 (데이터 확보 후)

```bash
# 1) Ultralytics로 YOLOv8n 7-class fine-tune (320 입력, FP16 export)
yolo train model=yolov8n.pt data=rover.yaml imgsz=320 epochs=100
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=320

# 2) ONNX → TRT engine
bash ros2_ws/src/rover_training/scripts/export_to_trt.sh best.onnx \
    ros2_ws/src/rover_perception/models/yolov8n.engine fp16
```

### 차량 거리 K 캘리브레이션 (한 번)

```bash
python3 ros2_ws/src/rover_perception/scripts/calibrate_vehicle_distance.py \
    --distance 0.5 --write
# 차량을 카메라로부터 0.5 m 앞에 놓고, OpenCV 창에서 bbox 코너 두 번 클릭
# K = bbox_h * d 가 params.yaml의 vehicle_dist_K에 저장됨
```

### 자율주행 (통합 후)

```bash
ros2 launch rover_bringup autonomous.launch.py mission:=left
```

`mission:=left|right`로 어느 분기 모델을 쓸지 결정.

## 모델 / 추론 스택

**Lane (도로 중심점 회귀):**
- 백본: ResNet18 (ImageNet pretrained, 마지막 FC를 512→2로 교체)
- 입력: 224×224 BGR → RGB, ImageNet 정규화
- 출력: `(x, y) ∈ [-1, +1]²` — 도로 중심점의 정규화 이미지 좌표
- 모델 3개: `common` / `left` / `right` — `/active_model` 토픽으로 스위치
- 런타임: TensorRT 10 Python API + torch CUDA 버퍼
- 측정 지연: ~0.9 ms (pure GPU, trtexec), ~7.7 ms (Python end-to-end)
- 소비처: `rover_decision`이 `x`를 `steering = clip(k·x, -1, +1)`로 결정적 변환

**Perception (YOLOv8n):**
- 클래스 7개: `traffic_light_red`, `traffic_light_green`, `traffic_light_yellow`, `stop_sign`, `vehicle`, `turn_left_sign`, `turn_right_sign`
- 입력: letterboxed 320×320 BGR→RGB / 255
- 런타임: TensorRT 10 직접 호출 (no ultralytics, no pycuda) — `yolo_inference.py`에 letterbox + numpy NMS만 직접 구현
- 데시메이션: `detect_every_n=2` (표지판/신호 상태는 빠르게 변하지 않음)
- 측정 지연: stock COCO 640으로 27.9 ms — 본 모델 320 7-class에선 5~8 ms 예상
- 거리: `vehicle` bbox 높이 → `d = K / bbox_h_px` (스테레오 disparity 안 씀)

자세한 모델 설계와 라벨링 규약은 PROJECT_PLAN.md §4 / §4-A 참조.

## Reused from HYU-ECL3003

이 프로젝트는 `~/HYU-ECL3003/`의 기존 자산을 적극 재사용한다:

- `rover_control/motor_driver.py` ← `rover/base_ctrl.py`
- `rover_recorder/jetcam/` ← `rover/jetcam/`
- `rover_stereo/calib/capture_stereo.py` ← `stereo_depth_tutorial/.../capture-stereo.py`
- `rover_training/scripts/train_center.py` ← `rover/train_road_center_model.ipynb`
- `rover_perception/yolo_inference.py` ← `center_inference.py` 패턴 차용 (TRT 10 + torch CUDA); HYU `week07/YOLOv8/yolov8n.pytorch.onnx`는 스모크 테스트용 engine 빌드에 사용
- `HYU-ECL3003/rover/cnn/center_dataset.py` ← 학습 시 직접 import

세부 매핑은 PROJECT_PLAN.md §8 참조.

## Calibration is frozen

`rover_stereo/config/stereo_calib.yaml`은 카메라 마운트가 확정된 직후 **한 번 생성한 뒤 학기 동안 재생성 금지**. 이 파일이 바뀌면 그 이전에 모은 BC 학습 데이터는 모두 무효가 된다. 카메라가 흔들리면 *재캘리브가 아니라 마운트 물리적 복원*으로 대응할 것. 자세한 이유는 PROJECT_PLAN.md §10 참조.

## 알려진 이슈

- `train_center.py`는 `CenterDataset`이 bare 파일명을 PIL.Image.open에 넘기는 구조라 CWD가 이미지 디렉토리여야 동작한다. 위 예시처럼 `cd <data>/images` 후 실행하면 된다. 향후 chdir 자동화 예정.
- 모터 명령은 통신되지만 움직이지 않으면 배터리 저전압을 먼저 의심할 것.

## 진행 상황

[PROJECT_PLAN.md §0](PROJECT_PLAN.md#0-진행-상황-한눈에-보기)의 체크박스 참조.
