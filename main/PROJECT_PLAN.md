# 회전교차로 자율주행 로버 — 프로젝트 계획서

NVIDIA Jetson Orin Nano + ROS 2 Humble 기반 closed-world 자율주행 로버.
좌/우 두 미션 중 하나를 표지판 trigger로 선택하고, 도로 중심점 회귀 CNN으로 차선을 따라간다. 안전(정지/대기)은 규칙 기반 FSM이 담당한다.

---

## 1. 시나리오

```
[출발점] → [공통 구간] → [회전교차로 분기] ─┬→ [좌 경로] → [도착 A]
                                          └→ [우 경로] → [도착 B]
```

- 미션은 launch 인자로 시작 시 결정 (`mission:=left|right`)
- Localization·GT map 사용 안 함 (closed-world에서 line tracking + 표지판 trigger로 충분)
- 인식 클래스 (YOLOv8n): `stop_sign`, `roundabout_sign`, `traffic_light_red`, `traffic_light_green`, `vehicle`

## 2. 설계 원칙

- **안전 = hand-coded FSM** (정지 표지판, 신호등, 전방 차량). 학습 모델이 안전을 결정하지 않는다.
- **주행 = Behavior Cloning** (도로 중심점 회귀). 인간 시연을 외운다.
- **미션 = 규칙** (좌/우 결정 + 어느 모델을 쓸지 선택). PilotNet이 분기를 결정하지 않는다.
- **이미지 진입점은 한 곳**: 모든 노드는 `stereo_node`가 발행하는 `/image_rectified`만 본다. raw 카메라 토픽 직접 구독 금지.

## 3. 데이터 흐름

```
   /camera/left/image_raw  ┐
                           ├─→ [stereo_node]  rectify (frozen LUT) + disparity
   /camera/right/image_raw ┘     │
                                 ├─→ /image_rectified   (left, ROI-cropped)
                                 └─→ /vehicle_distance  (meters; inf if invalid)
                                          │
              ┌───────────────────────────┴──────────┐
              ▼                                      ▼
      [yolo_node]                            [pilotnet_node]
        /detections                            /road_center  (x, y, model_tag)
              │                                      │
              └──────────────────┬───────────────────┘
                                 ▼
                         [decision_node]
                  FSM + 안전 layer + 모델 스위칭 + (x,y)→steering
                                 │
                                 ▼
                             /cmd_vel  (geometry_msgs/Twist)
                                 │  linear.x  = steering placeholder
                                 │  angular.z = steering ∈ [-1, +1]
                                 ▼
                         [control_node]
                  (steering, FSM state) → throttle → (L, R)
                                 │
                                 ▼
                        UART JSON  {"T":1,"L":L,"R":R}\n
                                 │
                                 ▼
                          rover MCU → 모터 PWM
```

## 4. BC 모델: 입력·출력·라벨

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

**라벨링 정책**:
- `xpos, ypos`는 픽셀 정수 (CenterDataset 호환).
- `steer_tel, speed_tel`은 텔레옵 시점의 정규화 값 ([-1, +1]). 학습 라벨이 아님. 분석/이상치 탐지용.
- `segment ∈ {common, left, right, pause}` — 키보드 라벨로 실시간 기록. `pause`는 학습 제외.

BC는 픽셀 공간에서 닫혀 있어야 하므로, 학습/추론 모두 **같은 `/image_rectified` 스트림** (같은 rectify LUT, 같은 ROI)을 사용한다.

## 5. 모터 명령 변환 (control_node)

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

```
COMMON   : 공통 구간 (model = common)
TURNING  : 회전교차로 통과 (model = left or right)
WAITING  : 전방 차량 가까움 (throttle=0)
STOPPED  : 정지 표지판/적신호 (throttle=0)
ARRIVED  : 도착
```

매 프레임 평가 순서: stop_sign / red_light → vehicle_close → roundabout_trigger → lane_lost.
모든 trigger는 N프레임 stabilization 후 발동 (false positive 방지).

**Vehicle 거리** = stereo disparity 기반 (1차) → 신뢰도 낮으면 bbox-height fallback (`dist ≈ K / bbox_h`).

## 7. 패키지 구조

```
ros2_ws/src/
├── rover_bringup/      launch + params.yaml + missions.yaml + hardware.yaml
├── rover_msgs/         Detection, DetectionArray, RoadCenter, FSMState
├── rover_stereo/       듀얼 CSI rectify + disparity (단일 이미지 진입점)
│   └── config/stereo_calib.yaml   ← FROZEN, git-committed
├── rover_perception/   YOLO (Ultralytics + TRT engine)
├── rover_pilotnet/     Center-regression CNN (3개 모델 스위칭)
├── rover_decision/     FSM + safety stabilizer + steering 변환
├── rover_control/      BaseController UART JSON 래퍼
├── rover_recorder/     키보드 텔레옵 + /image_rectified 저장
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

1. **rover_msgs 정의 + 빌드** — 의존성 출발점.
2. **rover_bringup config (`params.yaml`, `hardware.yaml`)** — 임계값·UART 코드 중앙화.
3. **rover_control 노드** — `BaseController` 래퍼. 첫 동작 milestone.
4. **듀얼 CSI 카메라 발행** — `/camera/left/image_raw`, `/camera/right/image_raw`.
5. **카메라 마운트 최종 고정 + stereo 캘리브레이션 동결** — `rover_stereo/calib/`로 30~50쌍 캡처 → `stereo_calib.yaml` git commit. **이 단계 통과 전 데이터 수집 금지.**
6. **rover_stereo 노드** — rectify pair → `/image_rectified` + `/vehicle_distance`.
7. **rover_recorder** — 키보드 텔레옵 + `/image_rectified` 저장 + segment 라벨링.
8. **수동 주행 한 사이클 닫기** — 한 미션 시연 → 저장 → 재생 검증.
9. **YOLO fine-tune + TensorRT 변환** — 200~500장 라벨링 후 학습.
10. **본격 데이터 수집** — 미션당 50~100회, 진입 각도·속도·복귀 시연 다양화.
11. **3개 center CNN 학습** — common/left/right 분리. 노트북에서 정성 평가 (cross-eval).
12. **자율주행 통합** — decision_node + pilotnet_node + autonomous.launch.py.
13. **실차 디버깅** — smoothing, segment 데이터 보강, 임계값 튜닝.
14. **데모 준비** — latency 측정, rosbag 로그, 데모 영상.

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
| Vehicle 거리 | stereo 1차 + bbox-height fallback | 메트릭 단위 + closed-world 보험 |
| Throttle | 규칙 + 곡률 감속 | 안전·튜닝 우월 |
| 텔레옵 | 키보드 (`pynput`) | HYU 검증, 조이스틱 불필요 |
| 모터 명령 | UART JSON `{T:1, L, R}` | `BaseController` 그대로 |

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

## 13. Out of Scope

MoE routing, SNN, ISO 26262 / SOTIF, open-world 일반화, sim-to-real, DAgger, RL fine-tuning. 본진 연구 트랙이지 이 프로젝트 안 다룸.

## 14. 참고

- PilotNet: Bojarski et al., "End to End Learning for Self-Driving Cars", NVIDIA 2016.
- YOLOv8: https://docs.ultralytics.com/
- ROS 2 Humble: https://docs.ros.org/en/humble/
- HYU-ECL3003 (재사용 베이스): `~/HYU-ECL3003/`
