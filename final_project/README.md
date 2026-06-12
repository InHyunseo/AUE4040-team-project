# 자율주행 AI — 차선 주행 + 정지 차량 추월

> AUE4040 자동차임베디드AI — 멀티태스크 E2E 주행 모델

---

## 한 줄 요약

**"전처리된 Lane(차선 세그) + Front(차량 bbox) 이미지를 보고 차선을 따라 주행하며, 정지한 차량을 회피·추월한다"**

---

## 태스크

단순화된 차선 도로에서:

- 차선을 따라 주행 (직선 / 코너 / S자)
- 전방에 **정지한 차량**이 있으면 회피·추월
- 추월 결정은 모델이 학습 (명시적 차선 변경 개념 없음 — 어디를 주행해도 무관)

---

## 입력

```
Lane 카메라 (sensor 0)
  → SegFormer로 차선 세그 (좌실선 / 우실선 / 중앙점선 3클래스)
  → 색상 오버레이된 raw lane 이미지
  → ResNet18-A (LaneEncoder)

Front 카메라 (sensor 1)
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
WaypointHead  → waypoints (5점, 2.5s) (보조, 미터)
```

### steer / throttle (메인)

추론 노드(`rover_lane`)에서 실제 cmd_vel로 역변환 (throttle 출력은 안 씀 — 속도는
`|steer|`에 커플링, teleop과 동일):

```
linear.x  = -(0.20 + abs(steer) * 0.05)   # -0.20 ~ -0.25  (음수 = 전진)
angular.z = steer * 1.2                     # -1.2 ~ +1.2
```

### waypoint (보조)

- 출력. GT는 cmd_vel을 forward-Euler로 적분한 미래 **2.5초** 궤적 (로봇 프레임, 5점).
- 멀티스텝 의도를 backbone에 학습시켜 메인 task(steer/throttle)를 regularize.
- 추론 시 사용 안 함 (디버깅·시연 때만 lane 이미지에 그려 의도 시각화).
- horizon이 짧으면(예: 0.5초, 저속이라 ~15cm) 점이 뭉쳐 trivial → aux 효과 미미.
  2.5초(~55cm)로 벌려 직진/코너가 구분되는 의미있는 궤적을 학습시킨다.
  ⚠️ 점 개수(WP_N)나 horizon을 바꾸면 기존 체크포인트 `--resume` 정합성에 영향
  (개수 변경 시 shape 불일치로 불가, horizon만 변경 시 head 재적응 1회 발생).

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
        SegFormer(freeze) → raw lane 이미지 차선 오버레이
        YOLO(freeze)      → Front 차량 bbox 오버레이
        cmd_vel           → steer, throttle, waypoint (자동 생성)
    → labels_cache.h5
    → train: ResNet18×2 + ControlHead + WaypointHead
```

- Phase 1 라벨링은 한 번만 (클래스당 200장+). 실제 트랙 환경 사진으로.
- Phase 2 본 학습 데이터는 라벨링 0 — cmd_vel은 텔레옵에서, 세그/bbox는 freeze 모델이 자동 생성.

Phase 1 상세는 [PHASE1.md](PHASE1.md), Phase 2(수집·학습)는 [PHASE2.md](PHASE2.md),
Phase 3(실차 추론·모니터링·추가학습)는 [PHASE3.md](PHASE3.md) 참고.

---

## 텔레옵 (데이터 수집)

**1D steering level + throttle coupling**.

```
turn_level: -2 ~ +2  (a/d 키로 1단계씩 조절, 좌/우 두 번이면 최대 회전)
직진:  linear.x = -0.20, angular.z = 0.0
회전:  linear.x = -0.25까지 자동 증가, angular.z = ±1.2
```

회전 시 throttle이 자동으로 높아져 차동 모터 토크 부족을 해결. 이 coupling이 학습 데이터에 반영됨.

| level | linear.x | angular.z |
|---:|---:|---:|
| 0 | -0.20 | 0.00 |
| ±1 | -0.225 | ±0.96 |
| ±2 | -0.25 | ±1.20 |

(`BASE_V=0.20, TURN_V=0.25, MAX_OMEGA=1.2, TURN_FRAC=(0, 0.8, 1.0)` — teleop_node.py)

smoothing(approach 보간)으로 실제 cmd_vel은 연속적으로 변함.

---

## 실차 추론 (rover_lane)

학습된 `e2e.onnx`를 Jetson에서 TensorRT engine으로 빌드하고, `rover_lane` 단일 노드가
실차를 주행시킨다. 명령은 [PHASE3.md](PHASE3.md), 구조 설명은 아래.

**단일 통합 노드** — `e2e_infer_node`가 raw 카메라 토픽을 구독해 SegFormer + YOLO +
E2E(TensorRT)를 **한 프로세스에서** 돌리고 `/cmd_vel`을 publish한다. 노드를 나눠 SegFormer
출력을 JPEG 토픽으로 넘기면 매 제어 주기에 인코딩/디코딩 + 노드 간 홉이 더해지므로, 저지연
제어를 위해 한 프로세스에 묶어 텐서를 메모리에 흘린다.

```
/lane_image  ─→ crop ROI → resize224 → SegFormer → composite_lane ─┐
/front_image ─→ resize224 → YOLO → composite_front ────────────────┤→ e2e.engine
                                                                    │   → steer
                          linear.x = -(0.20+|steer|*0.05) ←─────────┘   (throttle/wp 미사용)
                          angular.z = steer * 1.2  →  /cmd_vel → motor_bridge → UART
```

- **전처리는 학습과 픽셀 단위 동일.** 추론 노드는 `extract_labels`(`crop_lane_roi`,
  `SegFormerLaneSeg`, `YoloCarDet`)와 `dataset`(`composite_lane/front`, `to_input_tensor`)을
  직접 import — 재구현 금지. 색공간/정규화/crop 순서가 한 픽셀이라도 어긋나면 오작동.
- **안전장치(watchdog) + smoothing.** 독립 타이머(20Hz)가 추론이 준 *목표* steer로 발행
  명령을 매 틱 `approach()` 저역통과(`smooth_alpha`=0.35, teleop과 동일)로 당기며 재발행한다.
  추론은 raw 출력을 바로 publish하지 않고 목표만 저장 → 학습 cmd_vel(전부 teleop smoothing
  거친 값)과 분포가 맞고, 프레임마다 steer가 튀어 모터가 "뚝뚝" 끊기는 현상이 사라진다.
  추론이 `cmd_timeout_s`(0.4초)보다 오래 멈추면(스트림 끊김/행/로딩) 자동 정지(모델 로딩 중에도 정지).
- **제어 주기 = min(카메라 fps, SegFormer+YOLO+engine 속도).** SegFormer가 PyTorch로 도는 한
  병목일 수 있으니 `ros2 topic hz /cmd_vel`로 실측, 낮으면 SegFormer도 TRT로 변환.
- **QoS.** 이미지 sub는 `BEST_EFFORT + KEEP_LAST depth=1` — 밀린 옛 프레임 버리고 최신만 처리.

---

## 파이프라인

```
1. SegFormer fine-tuning (좌실선/우실선/중앙점선) → freeze   [training/train_segformer_colab.ipynb]
2. YOLO fine-tuning (car 단일 클래스) → freeze              [training/train_yolo_colab.ipynb]
3. 데이터 수집 (rosbag)
     ros2 launch rover_recorder record.launch.py + ros2 run rover_teleop teleop_node
     직선 → 코너 → 복귀, 정지 차량 접근 → 회피 → 복귀 시퀀스 반복 수집
4. 라벨 추출
     python data_pipeline/extract_labels.py --bag ... \
         --segformer_ckpt models/segformer_lane --yolo_weights models/best.pt → labels_cache.h5
5. E2E 학습 (Colab) → ONNX export
     ResNet18×2 + ControlHead + WaypointHead  [training/train_e2e_colab.ipynb → e2e.onnx]
─────────────────────────  여기까지 Phase 2 (PHASE2.md)  ─────────────────────────
6. Jetson에서 trtexec --fp16 → e2e.engine (engine은 Jetson에서만 빌드)
7. rover_lane 추론 노드 → 실차 주행 + 주행 모니터링 + 추가학습   [Phase 3 (PHASE3.md)]
```

단계 상세: 수집~학습은 [PHASE2.md](PHASE2.md), 배포~추론~추가학습은 [PHASE3.md](PHASE3.md).
ROS2 노드 사용법은 [ros2_ws/README.md](ros2_ws/README.md) 참고.
