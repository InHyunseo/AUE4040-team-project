# 회전교차로 자율주행 로버 — 프로젝트 계획서

NVIDIA Jetson Orin Nano + ROS 2 Humble 기반 closed-world 자율주행 로버.
좌/우 두 미션 중 하나를 표지판 trigger로 선택하고, 도로 중심점 회귀 CNN으로 차선을 따라간다. 안전(정지/대기)은 규칙 기반 FSM이 담당한다.

**마지막 업데이트:** 2026-05-21
**현재 단계:** 전체 ROS 파이프라인(perception + lane + FSM + control) 코드 검증 완료. 실차 데이터·모델 수집 대기 중.

---

## 0. 진행 상황 한눈에 보기

- [x] **Phase 0 — 인프라** : 패키지 구조, msgs, bringup, base_ctrl 연결
- [x] **Phase 1 — 하드웨어 검증** : 모터/UART 통신, 듀얼 CSI 카메라
- [x] **Phase 2 — 데이터 수집 도구** : 노트북 기반 텔레옵 + 라벨링 (record_and_label.ipynb)
- [x] **Phase 3 — 학습/추론 파이프라인 (lane)** : preprocess, train, ONNX, TRT engine, ROS 추론 노드 (스모크 테스트 통과)
- [x] **Phase 3.5 — 경량 perception + FSM 통합** : YOLO TRT 직접 호출(ultralytics 제거), `rover_msgs` 빌드, decision_node ↔ control_node 풀 wire-up, 스모크 테스트 통과
- [x] **Phase 4 — Stereo 캘리브레이션 동결** : 2026-05-19에 완료 (commit `c74c7fd`, RMS≈1.84). `rover_stereo/config/stereo_calib.yaml` git-committed, `team/calibration/calib/` 노트북·LUT 보존
- [ ] **Phase 5 — 차량 bbox 거리 K 캘리브레이션** : `scripts/calibrate_vehicle_distance.py` 한 번 실행
- [ ] **Phase 6 — 실차 데이터 수집** : lane 미션당 50~100회, YOLO 클래스별 ~수백장
- [ ] **Phase 7 — 3개 center CNN 학습** : common / left / right
- [ ] **Phase 8 — YOLO 7-class fine-tune + TRT 변환 (320×320 FP16)**
- [ ] **Phase 9 — 통합 자율주행** : `autonomous.launch.py mission:=left|right`
- [ ] **Phase 10 — 실차 디버깅 및 튜닝**
- [ ] **Phase 11 — 데모 준비**

> **현재 차단 요인 (blocker):** 실제 도로/표지판/차량 부재. Phase 5(차량 K) ~ 6(데이터 수집) 진행 전엔 학습할 데이터가 없다. 인프라·툴체인·노드 코드·stereo 캘리브는 모두 준비됨.

---

## 1. 시나리오

```
[출발점] → [공통 구간] → [회전교차로 분기] ─┬→ [좌 경로] → [도착 A]
                                          └→ [우 경로] → [도착 B]
```

- 미션은 launch 인자로 시작 시 결정 (`mission:=left|right`)
- Localization·GT map 사용 안 함 (closed-world에서 line tracking + 표지판 trigger로 충분)
- 인식 클래스 (YOLOv8n, **7-class**): `traffic_light_red`, `traffic_light_green`, `traffic_light_yellow`, `stop_sign`, `vehicle`, `turn_left_sign`, `turn_right_sign`
  - `turn_left_sign` / `turn_right_sign`은 회전교차로 중앙에만 존재하며 lane 모델 스위치 trigger
  - `vehicle`은 단 한 대(고정된 외형). 거리는 **bbox 높이**로 계산 — stereo disparity 사용 안 함
  - `stop_sign`은 3초 정지 후 진행 (`stop_wait_seconds: 3.0`)

## 2. 설계 원칙

- **안전 = hand-coded FSM** (정지 표지판, 신호등, 전방 차량). 학습 모델이 안전을 결정하지 않는다.
- **주행 = Behavior Cloning** (도로 중심점 회귀). 인간 시연을 외운다.
- **미션 = 규칙** (좌/우 결정 + 어느 모델을 쓸지 선택). 학습 모델이 분기를 결정하지 않는다.
- **이미지 진입점은 한 곳**: 모든 노드는 `stereo_node`가 발행하는 `/image_rectified`만 본다. raw 카메라 토픽 직접 구독 금지.

## 3. 데이터 흐름

```
   /camera/left/image_raw  ┐
                           ├─→ [stereo_node]  rectify only (frozen LUT)
   /camera/right/image_raw ┘     │            ※ 우측 카메라는 rectify 후 폐기
                                 ▼
                          /image_rectified   (left, ROI-cropped, bgr8)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
      [yolo_node]                              [lane_node]
        TRT 320×320 FP16                       TRT 224×224 FP16
        7-class (decimate ×2)                  active_model: common/left/right
        /detections (DetectionArray)           /road_center (x, y, model_tag)
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                         [decision_node]
                  Stabilizer(N-frame) + FSM + bbox→거리
                  + 미션 vs turn-sign 교차검증
                                 │
              ┌──────────────────┼──────────────────────────┐
              ▼                  ▼                          ▼
         /cmd_vel           /active_model              /fsm_state
        (Twist)             (String)                   (FSMState)
              │                                              │
              ▼                                              │
        [control_node] ◄────────────────────────────────────┘
        throttle gate (STOPPED/WAITING/ARRIVED → 0)
        (steer, throttle) → (L, R) → UART JSON
                                 │
                                 ▼
                        UART JSON  {"T":1,"L":L,"R":R}\n
                                 │
                                 ▼
                          rover MCU → 모터 PWM
```

> **명명 주의:** 도로 중심점 회귀 노드는 `rover_lane` 패키지의 `lane_node`. 과거 `rover_pilotnet` 이름은 2026-05-21에 폐기됨 (원본 NVIDIA PilotNet과 혼동되어).

> **중요:** `/vehicle_distance` 토픽은 더 이상 존재하지 않는다. 거리는 `decision_node`가 `Detection.vehicle`의 bbox 높이에서 직접 계산 (`d = K / bbox_h`, `K`는 캘리브 한 번). stereo disparity는 런타임에 돌지 않는다.

## 4. BC 모델: 입력·출력·라벨

- [x] 모델 팩토리 ([rover_training/models/center_cnn.py](ros2_ws/src/rover_training/models/center_cnn.py))
- [x] 데이터셋 ([HYU-ECL3003/rover/cnn/center_dataset.py](../../HYU-ECL3003/rover/cnn/center_dataset.py) 재사용)
- [x] 학습 스크립트 ([rover_training/scripts/train_center.py](ros2_ws/src/rover_training/scripts/train_center.py))
- [x] ONNX → TRT engine 빌드 ([scripts/export_to_onnx.py](ros2_ws/src/rover_training/scripts/export_to_onnx.py), [scripts/export_to_trt.sh](ros2_ws/src/rover_training/scripts/export_to_trt.sh))
- [x] 추론 노드 ([rover_lane/lane_node.py](ros2_ws/src/rover_lane/rover_lane/lane_node.py))
- [ ] 실차 데이터로 실제 학습된 가중치 확보

| 항목 | 값 |
|---|---|
| 입력 | rectified-left, 224×224 RGB, ImageNet 정규화 |
| 모델 | ResNet18 (ImageNet pretrained) + FC(512→2) |
| 출력 | `(x, y)` ∈ [-1, +1]² — 도로 중심점의 정규화된 이미지 좌표 |
| 손실 | MSE |
| 학습 데이터 | 인간이 시연하며 클릭(또는 자동 추출)한 픽셀 좌표 |
| 라벨 파일 | `annotation.txt` 한 줄: `filename xpos ypos segment steer_tel speed_tel` |
| 라벨 정규화 | `CenterDataset`이 `x = 2*(xpos/width - 0.5)`, `y = 2*(ypos/height - 0.5)`로 변환 |
| 추론 출력 사용처 | `decision_node`가 `steering = clip(k * x, -1, +1)` 결정적 변환 |
| 추론 런타임 | TensorRT 10 Python API + torch CUDA buffers (pycuda 의존성 없음) |
| 측정된 추론 지연 | ~0.9 ms pure GPU (trtexec), ~7.7 ms end-to-end (preprocess + H2D + D2H) |

**라벨링 정책**:
- `xpos, ypos`는 픽셀 정수 (CenterDataset 호환).
- `steer_tel, speed_tel`은 텔레옵 시점의 정규화 값 ([-1, +1]). 학습 라벨이 아님. 분석/이상치 탐지용.
- `segment ∈ {common, left, right, pause}` — 키보드 라벨로 실시간 기록. `pause`는 학습 제외.
- Part B 수동 클릭 라벨링: [rover_recorder/notebooks/record_and_label.ipynb](ros2_ws/src/rover_recorder/notebooks/record_and_label.ipynb).

BC는 픽셀 공간에서 닫혀 있어야 하므로, 학습/추론 모두 **같은 `/image_rectified` 스트림** (같은 rectify LUT, 같은 ROI)을 사용한다.

## 4-A. Perception: YOLO 7-class + bbox 거리

- [x] 경량 TRT 10 직접 호출 (no ultralytics, no pycuda) — [yolo_inference.py](ros2_ws/src/rover_perception/rover_perception/yolo_inference.py)
- [x] `rover_msgs/DetectionArray` publish — [yolo_node.py](ros2_ws/src/rover_perception/rover_perception/yolo_node.py)
- [x] 프레임 데시메이션 (`detect_every_n=2`)
- [x] bbox 거리 캘리브 헬퍼 — [scripts/calibrate_vehicle_distance.py](ros2_ws/src/rover_perception/scripts/calibrate_vehicle_distance.py)
- [ ] 7-class 데이터 라벨링
- [ ] YOLOv8n 320×320 fine-tune + ONNX → TRT engine

| 항목 | 값 |
|---|---|
| 모델 | YOLOv8n (Ultralytics 학습 → ONNX → TRT engine FP16) |
| 입력 | letterboxed 320×320 BGR→RGB, /255 |
| 출력 | `(1, 4+nc, N)` — cxcywh + class scores, anchor-free |
| 클래스 (nc=7) | `traffic_light_red`, `traffic_light_green`, `traffic_light_yellow`, `stop_sign`, `vehicle`, `turn_left_sign`, `turn_right_sign` |
| Post-process | numpy NMS (`_nms_numpy`), letterbox 역변환 |
| 런타임 | TensorRT 10 Python + torch CUDA buffers — `center_inference.py`와 동일 패턴 |
| 추론율 | 카메라 30 Hz, YOLO는 `detect_every_n=2` (~15 Hz) |
| 측정 (stock COCO 640) | 27.9 ms/inf — 본 모델 320 7-class 예상 5~8 ms |

**거리 계산 (decision_node)**:
```
d = K / bbox_h_px
K = bbox_h_at_known_d * known_d   # 한 번 캘리브, params.yaml에 저장
```
- 차량 외형이 단 하나로 고정된 closed-world라서 모노큘러 bbox로 충분.
- safe_dist_m (기본 0.4 m) 미만이면 WAITING.
- 캘리브 절차: `python3 scripts/calibrate_vehicle_distance.py --distance 0.5 --write`

**왜 stereo depth를 안 쓰는가**:
- Orin Nano에 YOLO + lane CNN + rectify + motor I/O를 동시에 돌릴 때 SGBM disparity는 가장 무거운 CPU 작업이었다. 닫힌 환경에서 단일 고정 차량이면 bbox 단조성으로 충분.
- stereo 캘리브 자체는 rectify(영상 정렬)용으로 여전히 필요. disparity만 끔.

## 5. 모터 명령 변환 (control_node)

- [x] BaseController 래퍼 (HYU-ECL3003 base_ctrl.py 재사용)
- [x] 실차 모터 동작 확인 (2026-05 배터리 교체 후)
- [x] `steer_speed_to_lr` 믹싱 + curvature decel 통합 + FSM 연동 (`/fsm_state` 구독, SAFE 상태 throttle=0)

`/cmd_vel`은 추상 명령이고, 실제 모터 값으로의 변환은 `rover_control/control_node.py`에서 일어난다.

1. **Throttle 계산** — FSM 상태가 `STOPPED/WAITING/ARRIVED`면 0, 아니면:
   ```
   throttle = target_speed * (1 - curvature_decel_factor * |steering|)
   ```
2. **(steering, throttle) → (L, R)** — HYU-ECL3003 `ctrl_with_keyboard.py`의 믹싱 그대로:
   ```
   L = throttle * (1 - steering)    # 음수면 0으로 clamp
   R = throttle * (1 + steering)
   ```
3. **부호 반전** — `BaseController`는 *전진 시 음수* 컨벤션이라 `L, R = -L, -R` (config의 `motor.invert_drive: true`).
4. **UART JSON 송신** — `BaseController.base_speed_ctrl(L, R)`이 `{"T":1,"L":L,"R":R}\n`을 `/dev/ttyUSB0`로 write. 로버 펌웨어가 `T=1`을 모터 속도 명령으로 해석.

이 변환 코드는 `rover_control/control_node.py:steer_speed_to_lr` + `rover_control/motor_driver.py:BaseController` 두 곳에 있다.

## 6. FSM 상태 + 안전 우선순위

- [x] 상태 enum + 전이 테이블 구현 ([fsm.py](ros2_ws/src/rover_decision/rover_decision/fsm.py))
- [x] safety stabilizer (N-frame 누적, [safety.py](ros2_ws/src/rover_decision/rover_decision/safety.py))
- [x] bbox 기반 vehicle gating (`vehicle_close(bbox_h, K, safe_dist)`)
- [x] **turn_sign → model_tag 스위칭** (mission 인자와 교차검증, 어긋나면 warn + ignore)

```
COMMON   : 공통 구간 (model = common)
TURNING  : 회전교차로 통과 (model = mission tag: left or right)
WAITING  : 전방 차량 가까움 (throttle=0)
STOPPED  : 정지 표지판/적신호 (throttle=0, stop_sign이면 3초 후 resume)
ARRIVED  : 도착
```

매 프레임 평가 순서: stop_sign / red_light → vehicle_close → turn_sign_stable → lane_lost.
모든 trigger는 N프레임 stabilization 후 발동 (false positive 방지).

**Vehicle 거리**: stereo 사용 안 함. `Detection.vehicle`의 bbox 높이로 직접 계산.
```
d = K / bbox_h_px       # K = bbox_h_px * d (px·m), scripts/calibrate_vehicle_distance.py로 한 번 측정
```
`safe_dist_m` (기본 0.4m) 미만이면 WAITING.

**Turn-sign 교차검증**: launch 인자 `mission=left`인데 `turn_right_sign`만 stable이면 → `turn_sign_stable=False` + WARN 로그. 실수로 반대 모델 스위치되는 것을 차단.

## 7. 패키지 구조

- [x] rover_msgs (Detection, DetectionArray, RoadCenter, FSMState — 빌드 완료)
- [x] rover_bringup (launch + params.yaml, 7-class names + `detect_every_n` 추가)
- [x] rover_stereo (rectify-only — SGBM/disparity 코드 제거됨)
- [x] **rover_perception** (TRT 10 직접 호출, ultralytics 제거, `DetectionArray` publish)
- [x] **rover_lane** (前 rover_pilotnet — 2026-05-21 rename) ← 추론 노드 구현 완료
- [x] **rover_decision** (FSM + Stabilizer + bbox 거리 + turn-sign 교차검증 wire-up 완료)
- [x] **rover_control** (BaseController 래퍼 + `/fsm_state` SAFE 게이팅)
- [x] rover_recorder (노트북 기반 — ROS 노드는 stub)
- [x] rover_training (preprocess, train, export)

```
ros2_ws/src/
├── rover_bringup/      launch + params.yaml + missions.yaml + hardware.yaml
├── rover_msgs/         Detection, DetectionArray, RoadCenter, FSMState
├── rover_stereo/       듀얼 CSI rectify + disparity (단일 이미지 진입점)
│   └── config/stereo_calib.yaml   ← FROZEN, git-committed
├── rover_perception/   YOLO TRT 10 직접 호출 (no ultralytics) + bbox 거리 calib 스크립트
├── rover_lane/         Center-regression CNN (3개 모델 스위칭)  ※ ex rover_pilotnet
├── rover_decision/     FSM + safety stabilizer + steering 변환
├── rover_control/      BaseController UART JSON 래퍼
├── rover_recorder/     키보드 텔레옵 + /image_rectified 저장 (현재는 노트북 기반)
└── rover_training/     학습 스크립트 (ROS 외부, models/center_cnn.py + scripts/train_center.py)
```

## 8. HYU-ECL3003 재사용

| 자산 | → 사용처 |
|---|---|
| `rover/base_ctrl.py` | `rover_control/motor_driver.py` |
| `rover/ctrl_with_keyboard.py` (steer/speed→L,R 믹싱) | `rover_control/control_node.py:steer_speed_to_lr` |
| `rover/jetcam/` | 카메라 캡처 |
| `rover/cnn/center_dataset.py` | `rover_training/scripts/train_center.py` |
| `rover/train_road_center_model.ipynb` | `train_center.py` 베이스 |
| `rover/road_following_model.pth` | sanity check (학습 전 추론 파이프라인 검증) |
| `week07/YOLOv8/yolov8n.pytorch.engine` | `rover_perception/models/` 초기 배치 |
| `week07/YOLOv8/demo_livecam_local.py` (draw_boxes, YOLO load) | `rover_perception/yolo_inference.py` |
| `stereo_depth_tutorial/.../capture-stereo.py` | `rover_stereo/calib/capture_stereo.py` |
| `rover/camera_live_dual_failover.py` | 듀얼 CSI 초기화 패턴 |

## 9. 개발 순서

- [x] 1. **rover_msgs 정의 + 빌드** — 의존성 출발점.
- [x] 2. **rover_bringup config (`params.yaml`, `hardware.yaml`)** — 임계값·UART 코드 중앙화.
- [x] 3. **rover_control 노드** — `BaseController` 래퍼. 첫 동작 milestone.
- [x] 4. **듀얼 CSI 카메라 발행** — `/camera/left/image_raw`, `/camera/right/image_raw` (노트북 환경에서 검증).
- [x] 5. **카메라 마운트 최종 고정 + stereo 캘리브레이션 동결** — 2026-05-19 완료. `team/calibration/calib/{K_l,K_r,dist_coeff_l,dist_coeff_r}.npy` + `rectify_map_imx219_160deg_720p.yaml` + commit된 `rover_stereo/config/stereo_calib.yaml` (RMS≈1.84). **이후 재캘리 금지.**
- [x] 6. **rover_stereo 노드** — rectify-only → `/image_rectified` (disparity/`/vehicle_distance` 제거됨).
- [x] 7. **rover_recorder** — 키보드 텔레옵 + 이미지 저장 + segment 라벨링 (notebook 형태).
- [ ] 8. **수동 주행 한 사이클 닫기** — 한 미션 시연 → 저장 → 재생 검증.
- [ ] 9. **YOLO fine-tune + TensorRT 변환** — 200~500장 라벨링 후 학습.
- [ ] 10. **본격 데이터 수집** — 미션당 50~100회, 진입 각도·속도·복귀 시연 다양화.
- [ ] 11. **3개 center CNN 학습** — common/left/right 분리. 노트북에서 정성 평가 (cross-eval).
- [x] 12. **자율주행 노드 통합 (코드)** — decision_node + lane_node + yolo_node + control_node 모두 wire-up 완료. 실차 검증은 단계 13에서.
- [ ] 13. **실차 디버깅** — smoothing, segment 데이터 보강, 임계값 튜닝.
- [ ] 14. **데모 준비** — latency 측정, rosbag 로그, 데모 영상.

## 10. 캘리브레이션 동결 원칙

`rover_stereo/config/stereo_calib.yaml`은 단계 5에서 **한 번 생성한 뒤 학기 동안 절대 재생성 금지**. 카메라가 흔들리면 *재캘리브가 아니라 마운트 물리적 복원*으로 대응. 위반 시 BC silent degradation (모델은 안 죽고 차만 비뚤게 감) → 디버깅 어려움.

## 11. 주요 결정

| 결정 | 선택 | 사유 |
|---|---|---|
| Localization / GT map | 사용 안 함 | 누적 오차 위험 |
| 의사결정 | 규칙 기반 FSM | 안전 critical은 hand-coded |
| 주행 학습 | Center-regression CNN | 해석 가능, hflip 무료, HYU 검증 |
| 학습 모델 수 | 3개 (common/left/right) | 라벨 충돌 회피, 디버깅 단위 명확 |
| 카메라 | 듀얼 CSI + calibration 동결 | 단일 이미지 진입점, stereo 거리 |
| BC 입력 | rectified-left (3, 224, 224) | stereo가 어차피 있으니 통일이 깔끔 |
| Vehicle 거리 | **bbox-height만** (`d = K / bbox_h`) | 차량 외형 고정 + closed-world. SGBM 런타임 비용 회피 |
| YOLO 런타임 | TRT 10 + torch CUDA (no ultralytics) | ultralytics wrapper의 PyTorch 오버헤드 제거 (5×↑ 빠름) |
| YOLO 입력 크기 | 320×320 FP16 (fine-tuned) | 640 대비 4× 적은 연산. 7-class closed-world면 충분 |
| YOLO 프레임 데시메이션 | `detect_every_n: 2` | 표지판/신호 상태는 빠르게 바뀌지 않음. lane은 매 프레임 그대로 |
| Throttle | 규칙 + 곡률 감속 | 안전·튜닝 우월 |
| 텔레옵 | 키보드 (`pynput`) → 노트북 widget | HYU 검증, SSH 환경에서도 동작 |
| 모터 명령 | UART JSON `{T:1, L, R}` | `BaseController` 그대로 |
| 추론 런타임 | TRT 10 Python + torch CUDA | pycuda 빌드 회피, deps 최소화 |
| 라벨링 방법 | 수동 클릭 (Part B 노트북) | 자동화는 후순위. 클릭이 가장 정확 |

## 12. 주요 함정

- **이미지/명령 timestamp**: `ApproximateTimeSynchronizer` slop ≤ 30ms.
- **Segment 라벨링 엄격성**: 회전 한복판이 common에 섞이면 직선에서 핸들 꺾음.
- **카메라 노출 manual 고정**: auto면 같은 장면이 다르게 들어옴.
- **steering 분포 불균형**: 직진 데이터가 압도적이면 모델이 항상 직진. histogram 확인 후 oversample.
- **모델 스위칭 jump**: 데이터 segment 경계 1~2초 겹침 + EMA 스무딩 α≈0.7.
- **YOLO false positive**: N프레임 안정화 + bbox 최소 크기.
- **TensorRT 콜백 블로킹**: multi-threaded executor.
- **`/dev/ttyUSB*` 권한**: `sudo usermod -aG dialout $USER` + 재로그인.
- **`nvargus-daemon`**: CSI 카메라 dead 시 `sudo systemctl restart nvargus-daemon`.
- **`BaseController` 부호**: 전진이 음수 (`invert_drive: true`).
- **캘리브 동결 원칙 위반**: §10 참조.
- **`CenterDataset` 파일 경로 (알려진 이슈)**: `PIL.Image.open(filename)`이 bare 파일명을 받기 때문에 `train_center.py`는 CWD가 `<data>/images`여야 동작. 해결 옵션: train_center.py에서 chdir, 또는 preprocess.py가 절대경로 기록. 우선순위 낮음.
- **배터리 저전압**: 모터 명령은 통신되지만 실제 움직임이 약하거나 없을 수 있음. 디버그 1번 항목으로 항상 확인.

## 13. 인프라 검증 기록 (스모크 테스트)

**2026-05-21 lane 파이프라인** — 합성 데이터 8장으로 end-to-end 검증:

| 단계 | 결과 |
|---|---|
| preprocess.py | 6-col annotation → 3-col 분리 출력 (common/left/right) |
| train_center.py (1 epoch, 3 frames) | loss=0.1162, .pth 저장 |
| export_to_onnx.py | 44 MB ONNX 생성 |
| export_to_trt.sh fp16 | 22 MB engine, trtexec ~0.9 ms/inf |
| CenterInference.infer | 정상 (x, y) 반환, end-to-end 7.7 ms/inf |

**2026-05-21 perception + FSM wire-up** — HYU 스톡 YOLOv8n으로 통합 검증:

| 단계 | 결과 |
|---|---|
| `rover_msgs` colcon build | 4개 메시지 import OK |
| `colcon build --symlink-install` | 8개 패키지 모두 clean (16 초) |
| YOLO TRT engine 재빌드 (FP16, 640) | 9.4 MB engine 생성 |
| `YoloInference.infer` (stock COCO 640) | input/output 추론 일치, 27.9 ms/inf |
| FSM 전이 단위 테스트 | COMMON → TURNING (`turn_sign_stable=True`) → STOPPED (`stop_sign_stable=True`) |
| `Stabilizer` N=3 | frame 2부터 stable=True ✓ |
| `vehicle_close` (K=180, safe=0.4) | bbox 600→close, 50→far ✓ |

**예상 본 모델 지연 (7-class, 320×320 FP16):** 5–8 ms/inf (4× 적은 픽셀 + 단순 head). 카메라 30 Hz 예산 33 ms 내 lane(7.7) + yolo(~6) + rectify 합쳐도 여유.

실차 도로 + 데이터만 확보되면 즉시 학습/배포 가능한 상태.

## 14. Out of Scope

MoE routing, SNN, ISO 26262 / SOTIF, open-world 일반화, sim-to-real, DAgger, RL fine-tuning. 본진 연구 트랙이지 이 프로젝트 안 다룸.

## 15. 참고

- ResNet18: He et al., "Deep Residual Learning for Image Recognition", 2015.
- YOLOv8: https://docs.ultralytics.com/
- ROS 2 Humble: https://docs.ros.org/en/humble/
- TensorRT 10 Python API: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/
- HYU-ECL3003 (재사용 베이스): `~/HYU-ECL3003/`
