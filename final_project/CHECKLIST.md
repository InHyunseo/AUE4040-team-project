# Troubleshooting — 다른 팀 해결 사례 기반

> 다른 팀들이 실제로 겪고 해결한 문제들 정리

---

## 객체인식 / 라벨링

### 신호등(red/green) 인식이 RGB로 잘 안 된다
**해결**:
- RGB 대신 HSV 사용 — 조명 변화에 훨씬 강건함
- 다른 팀들 대부분 HSV로 신호 인식 잘 됐다고 함
- HSV 범위는 실제 촬영 환경에서 직접 측정해서 설정

---

### left를 right로 오인식하는 경향이 있다
**원인**: 모델이 right 클래스로 편향되어 left를 right로 잘못 분류

**해결**:
- 5프레임 슬라이딩 윈도우 다수결로 단일 프레임 오인식 보정
- 3프레임 이상 일치해야 latch → 순간적인 편향 오인식 무시

```python
from collections import deque
window = deque(maxlen=5)

def update_mission(detections):
    left_det  = next((d for d in detections if d.class_name == 'left'), None)
    right_det = next((d for d in detections if d.class_name == 'right'), None)

    if left_det and right_det:
        vote = 'left' if left_det.score > right_det.score else 'right'
    elif left_det:
        vote = 'left'
    elif right_det:
        vote = 'right'
    else:
        vote = None

    window.append(vote)

    if window.count('left') >= 3:
        return 'left'
    if window.count('right') >= 3:
        return 'right'
    return None
```

> 근본적인 해결은 left 클래스 학습 데이터 보강 (200장 이상, 다양한 각도/거리)

---

### 객체 인식 성능이 낮다
**해결**:
- 클래스당 라벨 최소 200장 이상 확보 필수
- 부족하면 특정 클래스를 아예 못 잡거나 오인식 빈번

---

### Augmentation 시 flip 사용 금지
**이유**:
- 좌회전/우회전 표지판을 좌우 flip하면 반대 방향 표지판이 됨
- 차선도 flip하면 좌/우 레이블이 뒤바뀜
- 방향 정보가 있는 데이터는 flip augmentation 절대 금지

**사용 가능한 augmentation**:
- 밝기/대비 조정
- 색조(Hue) 약간 변환
- 가우시안 노이즈 추가
- 랜덤 크롭 (방향 유지되는 범위 내)

---

### Roboflow에서 Resize ON으로 내보냈더니 bbox가 틀어진다
**해결**:
- Auto-Orient: ON, **Resize: OFF** 설정
- ultralytics가 letterbox로 자체 처리하므로 stretch resize하면 bbox 왜곡 발생

---

## 차선 세그멘테이션

### 차선 종류를 구별해서 세그해야 한다
**SegFormer로 세그 후 BEV 이미지에 오버레이하는 방식**

세그 대상:
```
좌측 실선   — 차선 이탈 판단
우측 실선   — 차선 이탈 판단
중앙 점선   — 회전교차로 내부 위치 판단용
```

**중앙 점선의 역할**:
회전교차로 내부에서는 점선이 곡선 형태로 복잡하게 나타나고, 출구 구간에서는 패턴이 바뀐다. 점선 세그를 오버레이 이미지에 포함시키면 모델이 이미지만 보고 "아직 회전교차로 안인지, 나가야 하는지"를 스스로 학습할 수 있다. FSM 규칙 없이 E2E로 해결.

```
회전교차로 진입:  점선 곡선 패턴 시작
회전 중:         점선 지속
출구 근처:       점선 패턴 변화 → 모델이 나가야 할 타이밍 학습
```

> 별도 FSM 규칙 추가 없이 라벨링만으로 해결 — E2E 학습 입력에 포함시키는 것으로 충분

---

## 거리 추정

### 전방 차량까지 거리 추정
**해결**:
- y축 bbox 높이로 거리 역산 (단일 카메라, 단일 차량 환경 가정)
- 차량 충돌 회피 전용 — bbox 높이가 임계값 이상이면 WAITING 상태 진입

```python
d = K / bbox_h_px
# K는 실제 환경에서 캘리브: K = bbox_h_px × 실제거리(m)
# bbox_h가 클수록 차량이 가까움
```

---

## 모터 / 제어

### 회전 시 토크 부족으로 로버가 멈춘다
**원인**:
차동 모터 구조라 회전 시 한쪽 바퀴는 빨라지고 반대쪽은 느려짐. 이 과정에서 토크가 분산되고, 차량 무게까지 더해져 throttle -0.15로는 회전 중 멈추는 현상 발생.

**특성**:
```
직선 주행:  throttle -0.15, steer 0.0  → 정상
90도 회전:  steer ±0.8 → 토크 부족 → 멈춤
```

**해결**:
회전 시 throttle을 -0.1 추가해서 -0.25로 높여야 안정적인 90도 코너링 가능.
직선 복귀 시 throttle 다시 -0.15로 낮추고 steer 0으로 복귀.

```python
# throttle-steer coupling
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
- 전체 주행 중 코너 구간 비율이 충분히 확보될 때까지 반복
- 코너 진입/탈출 직전 구간도 충분히 포함 (전환 타이밍 학습)

---

## 텔레옵 (데이터 수집용)

### 기존 버튼식 텔레옵이 불편하고 데이터 품질이 낮다
**해결**:
1D steering level + throttle coupling 방식으로 교체.

```
기본 전진:  linear.x = -0.15, angular.z = 0.0
좌우 5단계 조향 (turn_level: -5 ~ +5)
조향 강도에 비례해서 throttle도 자동으로 -0.25까지 증가
Space = 정지
```

```python
# turn_level: -5 ~ +5 (버튼으로 1단계씩 조절)
turn = turn_level / 5.0          # -1.0 ~ 1.0
a    = abs(turn)
linear_x  = -(0.15 + a * 0.10)  # -0.15 ~ -0.25
angular_z = turn * 0.8           # -0.8 ~ +0.8

# smoothing (부드러운 cmd_vel 발행)
linear_x  = approach(current_v, linear_x,  delta_v)
angular_z = approach(current_w, angular_z, delta_w)
```

| level | linear.x | angular.z |
|---:|---:|---:|
| 0 | -0.15 | 0.00 |
| ±1 | -0.17 | ±0.16 |
| ±2 | -0.19 | ±0.32 |
| ±3 | -0.21 | ±0.48 |
| ±4 | -0.23 | ±0.64 |
| ±5 | -0.25 | ±0.80 |

**장점**:
- 회전 시 토크 부족 구조적으로 해결 (throttle 자동 증가)
- 단계식이라 학습 라벨이 안정적
- smoothing으로 실제 cmd_vel은 연속적으로 변함
- 코너 진입/탈출 조작이 재현 가능

---

## 데이터 수집

### 이미지 수천 장을 로컬에 저장하기 어렵다
**해결**:
- rosbag으로 수집 → db3 파일 하나에 모든 토픽 저장 (이미지 파일 별도 생성 안 됨)
- compressed image 토픽 사용으로 용량 약 1/3 절감
- Colab에서 `rosbags` 라이브러리로 직접 파싱 가능 (ROS2 설치 불필요)

```bash
ros2 bag record /bev_image/compressed /front_image/compressed /cmd_vel /fsm_state
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