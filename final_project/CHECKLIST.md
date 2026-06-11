# Troubleshooting — 다른 팀 해결 사례 기반

> 차선 주행 + 정지 차량 추월 태스크 기준. 다른 팀들이 실제로 겪고 해결한 문제 정리

---

## 객체인식 / 라벨링 (YOLO car 단일 클래스)

### 객체 인식 성능이 낮다
**해결**:
- 클래스당 라벨 최소 200장 이상 확보 필수
- 정지 차량을 다양한 각도/거리/위치에서 촬영
- 부족하면 특정 거리/각도에서 아예 못 잡거나 오인식 빈번

---

### Augmentation 시 flip 사용 금지
**이유**:
- 차선을 좌우 flip하면 좌/우 실선 레이블이 뒤바뀜 (SegFormer 학습 데이터)
- 방향 정보가 있는 데이터는 flip augmentation 금지

**사용 가능한 augmentation**:
- 밝기/대비 조정
- 색조(Hue) 약간 변환
- 가우시안 노이즈 추가
- 랜덤 크롭 (방향 유지되는 범위 내)

---

### 라벨 export 때 resize/왜곡되면 bbox가 틀어진다
**해결**:
- export 단계에서 **stretch resize 금지** (원본 비율 유지). CVAT는 원본 그대로 내보냄.
- ultralytics가 학습 시 letterbox로 자체 처리하므로, 미리 stretch resize하면 bbox 왜곡 발생

---

## 차선 세그멘테이션 (SegFormer)

### 차선 종류를 구별해서 세그해야 한다
**SegFormer로 세그 후 raw lane 이미지에 오버레이하는 방식**

세그 대상 (3클래스):
```
좌측 실선   — 좌측 트랙 경계
우측 실선   — 우측 트랙 경계
중앙 점선   — 차선 구분선 (centering 참고용)
```

좌/우 실선을 구분하는 이유: 좌우 경계 곡률이 코너링 판단에 유용. 단 **차선 변경 개념은 없으므로 어디를 주행해도 무관** — 점선은 좌우 구분 없이 중앙 1채널로 충분.

> Phase 1에서 소량 라벨로 fine-tune 후 freeze. 라벨 추출(extract_labels.py)과 실차 추론에서 동일 모델 사용.

---

## 거리 / 추월

### 전방 정지 차량까지 거리 / 추월 타이밍
**해결**:
- 별도 거리 임계값(FSM) 없음. 모델이 YOLO bbox **위치 + 크기**(det 라벨)를 보고 거리감을 직접 학습.
- bbox가 클수록(가까울수록) 회피·추월 동작을 하도록 cmd_vel GT와 매칭되어 학습됨.
- waypoint 보조 head가 "옆으로 비켰다 복귀하는" 추월 궤적 의도를 함께 학습.

---

## 모터 / 제어

### 회전 시 토크 부족으로 로버가 멈춘다
**원인**:
차동 모터 구조라 회전 시 토크가 분산되고, 차량 무게까지 더해져 throttle -0.15로는 회전 중 멈추는 현상 발생.

**해결**:
회전 시 throttle을 -0.25로 높여야 안정적인 코너링 가능. 직선 복귀 시 -0.15로 낮추고 steer 0으로 복귀.

```python
# throttle-steer coupling (텔레옵 + 모델 출력 역변환 공통)
base_v   = 0.15
turn_v   = 0.25
max_omega = 0.8

a = abs(turn)  # turn: -1.0 ~ 1.0
linear_x  = -(base_v + a * (turn_v - base_v))
angular_z = turn * max_omega
```

| steer (turn) | linear.x | angular.z |
|---:|---:|---:|
| 0.0 | -0.15 | 0.00 |
| ±0.5 | -0.20 | ±0.40 |
| ±1.0 | -0.25 | ±0.80 |

---

### 회전 구간 학습 데이터가 부족하다
**원인**:
로버가 빠르게 회전하므로 회전 구간에서 녹화되는 프레임 수가 직선 구간보다 적음.

**해결**:
- 직선 → 코너링 → 직선 복귀 시퀀스를 의도적으로 반복 수집
- 코너 진입/탈출 직전 구간도 충분히 포함 (전환 타이밍 학습)

---

### 정지 차량 추월 데이터가 부족하다
**원인**:
추월은 짧은 시퀀스라 전체 주행 중 비율이 낮아지기 쉬움.

**해결**:
- 정지 차량 접근 → 회피 → 복귀 시퀀스를 의도적으로 반복 수집
- 차량 위치(좌/우/중앙), 거리(멀리서/가까이서 회피 시작)를 다양하게
- 추월 직전 감속 구간도 포함 (감속→회피 타이밍 학습)

---

## 텔레옵 (데이터 수집용)

### 기존 버튼식 텔레옵이 불편하고 데이터 품질이 낮다
**해결**:
1D steering level + throttle coupling 방식 (rover_teleop teleop_node).

```
기본 전진:  linear.x = -0.15, angular.z = 0.0
좌우 5단계 조향 (turn_level: -5 ~ +5, a/d 키)
조향 강도에 비례해서 throttle도 자동으로 -0.25까지 증가
space = 정지
```

```python
turn = turn_level / 5.0          # -1.0 ~ 1.0
a    = abs(turn)
linear_x  = -(0.15 + a * 0.10)  # -0.15 ~ -0.25
angular_z = turn * 0.8           # -0.8 ~ +0.8

# smoothing (부드러운 cmd_vel 발행)
linear_x  = approach(current_v, linear_x,  alpha)
angular_z = approach(current_w, angular_z, alpha)
```

**장점**:
- 회전 시 토크 부족 구조적으로 해결 (throttle 자동 증가)
- 단계식이라 학습 라벨이 안정적
- smoothing으로 실제 cmd_vel은 연속적으로 변함
- SSH 환경: pynput(X 의존) 대신 termios cbreak 사용 → 디스플레이 없이 키 입력 가능

---

## 데이터 수집

### 이미지 수천 장을 로컬에 저장하기 어렵다
**해결**:
- rosbag으로 수집 → db3 파일 하나에 모든 토픽 저장 (이미지 파일 별도 생성 안 됨)
- compressed image 토픽 사용으로 용량 절감
- Colab에서 `rosbags` 라이브러리로 직접 파싱 가능 (ROS2 설치 불필요)

```bash
ros2 bag record /bev_image/compressed /front_image/compressed /cmd_vel /steer_level
```

---

## 시스템 / ROS2

### 카메라/추론/제어 노드가 서로 블로킹된다
**해결**:
- ROS2 `MultiThreadedExecutor` + `ReentrantCallbackGroup` 사용
- 스레드 분리: 이미지 수신 / 모델 추론 / 제어 명령 발행

```python
executor = MultiThreadedExecutor(num_threads=3)
```

---

## 배포

### Colab에서 만든 TRT engine이 Jetson에서 동작 안 한다
**원인**: 노트북/WSL에서 만든 engine은 GPU 아키텍처가 달라 Jetson에서 사용 불가

**해결**:
- ONNX export는 어디서든 가능
- trtexec는 반드시 **Jetson에서 직접** 실행

```bash
# Jetson에서 실행
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

---

### 실차 추론 제어율(`ros2 topic hz /cmd_vel`)이 낮다
**증상**: 제어 주기가 카메라 fps(15Hz)보다 한참 낮음. 차가 끊기거나 코너 반응 지연.

**원인**: E2E 는 TRT 인데 **SegFormer/YOLO 는 아직 PyTorch** → 이 둘이 병목.
제어율 = `min(카메라 fps, SegFormer+YOLO+engine 처리 속도)`.

**판단 (먼저 측정 — 낮다고 다 문제 아님)**:

- 13~15Hz: 정상.
- 8~12Hz: 저속 주행(0.3m/s)이면 보통 OK. 실주행이 멀쩡하면 그냥 둔다(조기 최적화 금지).
- 5Hz 이하 또는 주행이 끊김/코너 이탈: 아래 해결.
- 안전은 watchdog 이 지킴 — 너무 느려 `cmd_timeout_s`(0.4s) 초과 시 자동 정지(폭주 X).

**해결 (효과 큰 순서)**:

1. **SegFormer 를 TensorRT 로 변환** (가장 효과 큼, fp16 2~4배). 단 E2E 처럼 공짜 아님:
   onnx export(Transformer 라 op 호환 확인 필요) + 전처리(`SegformerImageProcessor`)를
   numpy 로 재현 + 추론 노드에 SegFormer 용 TRTEngine 추가. 반나절 작업 + fp16 세그 품질 검증.
2. **YOLO 격프레임 실행** — 차가 자주 안 나오니 2~3프레임에 1번만, 나머지는 직전 bbox 재사용.
3. 입력 해상도 ↓(224→160) 또는 더 작은 백본 — 재학습 필요, 최후 수단.
