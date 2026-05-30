# 자율주행 AI — 차선 주행 + 정지 차량 추월

> AUE4040 자동차임베디드AI — 멀티태스크 E2E 주행 모델

---

## 한 줄 요약

**"전처리된 BEV(차선) + Front(차량 bbox) 이미지를 보고 차선을 따라 주행하며, 정지한 차량을 회피·추월한다"**

---

## 태스크

단순화된 차선 도로에서:

- 차선을 따라 주행 (직선 / 코너 / S자)
- 전방에 **정지한 차량**이 있으면 회피·추월
- 추월 결정은 모델이 학습 (명시적 차선 변경 개념 없음 — 어디를 주행해도 무관)

---

## 입력

```
BEV 카메라
  → SegFormer로 차선 세그 (좌실선 / 우실선 / 중앙점선 3클래스)
  → 색상 오버레이된 BEV 이미지
  → ResNet18-A (BEVEncoder)

Front 카메라
  → YOLO로 차량 감지 (단일 클래스: car)
  → bbox 오버레이된 Front 이미지
  → ResNet18-B (FrontEncoder)
```

- 차량 bbox 위치 + 크기를 보고 거리감을 학습 → 회피·추월 타이밍 결정.
- SegFormer와 YOLO는 별도 fine-tuning 후 freeze (이미지 전처리 전용).
- ResNet18 두 개가 전처리된 이미지를 보고 제어값을 학습.

---

## 출력 (멀티태스크)

```
ControlHead   → steer, throttle      (메인, Tanh [-1,1])
WaypointHead  → waypoints (5점, 0.5s) (보조, 미터)
```

### steer / throttle (메인)

rover_control에서 실제 값으로 역변환:

```
linear.x  = -(0.15 + abs(steer) * 0.10)   # -0.15 ~ -0.25
angular.z = steer * 0.8                     # -0.8 ~ +0.8
```

### waypoint (보조)

- 출력. GT는 cmd_vel을 forward-Euler로 적분한 미래 0.5초 궤적 (로봇 프레임).
- 멀티스텝 의도를 backbone에 학습시켜 메인 task(steer/throttle)를 regularize.
- 추론 시 사용 안 함 (디버깅·시연 때만 BEV에 그려 의도 시각화).

loss: `1.0 * steer + 0.5 * throttle + 0.5 * waypoint` (모두 MSE).

---

## 2단계 학습

```
Phase 1 — 소량 라벨 데이터 (한 번만)
  소량 라벨 → SegFormer fine-tune (차선 세그)  → freeze
  소량 라벨 → YOLO fine-tune     (car 단일클래스) → freeze

Phase 2 — 대량 rosbag 데이터 (라벨링 X)
  rosbag (대량)
    → extract_labels.py
        SegFormer(freeze) → BEV 차선 오버레이
        YOLO(freeze)      → Front 차량 bbox 오버레이
        cmd_vel           → steer, throttle, waypoint (자동 생성)
    → labels_cache.h5
    → train: ResNet18×2 + ControlHead + WaypointHead
```

- Phase 1 라벨링은 한 번만 (클래스당 200장+). 실제 트랙 환경 사진으로.
- Phase 2 본 학습 데이터는 라벨링 0 — cmd_vel은 텔레옵에서, 세그/bbox는 freeze 모델이 자동 생성.

Phase 1 상세는 [PHASE1.md](PHASE1.md) 참고.

---

## 텔레옵 (데이터 수집)

**1D steering level + throttle coupling**.

```
turn_level: -5 ~ +5  (a/d 키로 1단계씩 조절)
직진:  linear.x = -0.15, angular.z = 0.0
회전:  linear.x = -0.25까지 자동 증가, angular.z = ±0.8
```

회전 시 throttle이 자동으로 높아져 차동 모터 토크 부족을 해결. 이 coupling이 학습 데이터에 반영됨.

| level | linear.x | angular.z |
|---:|---:|---:|
| 0 | -0.15 | 0.00 |
| ±1 | -0.17 | ±0.16 |
| ±2 | -0.19 | ±0.32 |
| ±3 | -0.21 | ±0.48 |
| ±4 | -0.23 | ±0.64 |
| ±5 | -0.25 | ±0.80 |

smoothing(approach 보간)으로 실제 cmd_vel은 연속적으로 변함.

---

## 파이프라인

```
1. SegFormer fine-tuning (좌실선/우실선/중앙점선) → freeze   [training/train_segformer_colab.ipynb]
2. YOLO fine-tuning (car 단일 클래스) → freeze              [training/train_yolo_colab.ipynb]
3. BEV 캘리브레이션
     ros2 launch rover_calib bev_capture.launch.py → calib 이미지 한 장
     python data_pipeline/bev_calibration.py --image ... → calib.json (M, bev_size, ppm)
4. 데이터 수집 (rosbag)
     ros2 launch rover_recorder record.launch.py + ros2 run rover_teleop teleop_node
     직선 → 코너 → 복귀, 정지 차량 접근 → 회피 → 복귀 시퀀스 반복 수집
5. 라벨 추출
     python data_pipeline/extract_labels.py --bag ... --calib calib.json \
         --segformer_ckpt ... --yolo_weights main/best.pt → labels_cache.h5
6. E2E 학습 (Colab)
     ResNet18×2 + ControlHead + WaypointHead
7. ONNX export → Jetson에서 trtexec --fp16 (engine은 Jetson에서만 빌드)
8. rover_lane 노드 교체 → 실차 테스트
```

ROS2 노드 사용법은 [ros2_ws/README.md](ros2_ws/README.md) 참고.
