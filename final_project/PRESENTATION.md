# 발표자료 — 자율주행 AI (차선 주행 + 정지 차량 추월)

> 자유주제 E2E 멀티태스크 주행 모델 · AUE4040 자동차임베디드AI
> 총 7슬라이드 · 슬라이드당 ≈1분 · (결과 도출 완료 전제)

---

## 슬라이드 1 — 주제 & 목적

**주제**: 단순화된 차선 도로에서 **차선을 따라 자율주행하며, 전방의 정지 차량을 스스로 회피·추월**하는 End-to-End 멀티태스크 모델 (자유주제)

**한 줄 요약**: "전처리된 Lane(차선 세그) + Front(차량 bbox) 이미지를 보고, 차선을 따라가며 정지 차량을 피해 추월한다"

### 왜 이 주제 / 목적a
- 단순 차선추종을 넘어 **"정지 차량 추월"이라는 판단(decision)** 을 얹어 난이도·의미를 키움.
- 핵심 철학: 추월을 명시적 FSM(거리 임계값·차선변경 규칙)으로 짜지 않고 **모델이 데이터로 학습**.
  "차선 변경" 개념 자체를 없애 트랙 어디를 주행해도 무관 → 추월이 특별한 상태가 아니라 주행의 일부.
- 목적 4가지(= 이후 구현 슬라이드):
  1. 쓸 만한 수동 제어·모니터링 인프라
  2. 라벨 효율적 2단계 학습 파이프라인
  3. 지각-제어 분리 멀티태스크 네트워크
  4. 딜레이 없는 임베디드 추론

### 시스템 한눈에 (다이어그램)
```
[Lane cam sensor0] → SegFormer(freeze) → 차선세그 오버레이 ┐
                                                           ├→ ResNet18×2 → ControlHead → steer/throttle → cmd_vel → UART
[Front cam sensor1] → YOLO(freeze) → 차량 bbox 오버레이    ┘                 └ WaypointHead(보조)
```

---

## 슬라이드 2 — 구현 (1) 수동 컨트롤 로직 & 로컬 웹 모니터링

### 개선된 수동 컨트롤 — 1D steering level + throttle coupling
> `ros2_ws/src/rover_teleop/rover_teleop/teleop_node.py`

- 기존 버튼식: 조작 불편 + 라벨 품질 낮음 + **차동 모터라 회전 시 토크 분산 → 코너에서 멈춤**.
- `a`/`d`로 -2~+2 단계 조향, **조향 강도에 비례해 throttle 자동 증가** (코너 토크 부족 구조적 해결).

  | level | linear.x | angular.z |
  |---:|---:|---:|
  | 0  | -0.15 | 0.00  |
  | ±1 | -0.20 | ±0.40 |
  | ±2 | -0.25 | ±0.80 |

- `approach()` smoothing(α=0.35, 20Hz)으로 cmd_vel이 연속적으로 변함 → 라벨 품질↑.
  이 coupling이 곧 학습 분포가 되고, 모델 출력 역변환도 동일 공식 사용.

### 딜레이 없는 카메라 로컬 웹 모니터링 (SSH 환경)
> `ros2_ws/src/rover_camera/rover_camera/monitor_node.py`

- 헤드리스 Jetson SSH: pynput(X 의존) 대신 **termios cbreak**로 디스플레이 없이 키 입력.
- 카메라 JPEG를 **재인코딩 없이** 브라우저 MJPEG로 스트림(`http://<host>:8080/`) → Jupyter 위젯보다 표시 지연 낮음.
- **제어 경로(teleop→cmd_vel→UART)와 완전히 별개 노드** → 모니터링이 주행 명령 타이밍에 영향 0.
  `Condition` var로 새 프레임에만 깨움(busy-spin 없음).

---

## 슬라이드 3 — 구현 (2) 라벨 데이터 수집 & 학습 (멀티헤드 전처리)

### 멀티헤드 구조: 두 카메라 = 두 태스크 분리
```
Lane cam(sensor0)  → SegFormer → 차선 세그 (좌실선/우실선/중앙점선 3클래스)
Front cam(sensor1) → YOLO      → 차량 객체인식 (단일클래스 car)
```
- 좌/우 실선 구분 이유: 경계 곡률이 **코너링 판단**에 유용. 점선은 중앙 1채널이면 충분(차선변경 개념 없음).
- 두 전처리기는 **별도 fine-tune 후 freeze** → 이미지 전처리 전용, 뒤단 학습과 분리돼 안정적.

### Phase 1 — 소량 라벨로 한 번만
> `PHASE1.md`

- rosbag → jpg 추출 → CVAT 라벨링 → Colab fine-tune.
- CVAT task 2개 분리: Lane=**polyline**(선만, 노트북이 띠로 자동 rasterize) / Front=**bbox**.
- 양: car bbox 200장+, 차선 100~300장.
- 시행착오: **flip aug 금지**(좌/우 실선 뒤바뀜), **export stretch resize 금지**(bbox 왜곡) → 밝기/대비/노이즈만.
- 산출물: SegFormer 체크포인트 + YOLO `best.pt` → freeze 확정.

---

## 슬라이드 4 — 구현 (2-b) 라벨링 없는 대량 자동 라벨 추출 ⭐

> **본 학습 데이터는 수작업 라벨 0장** — 라벨 효율의 핵심
> `data_pipeline/extract_labels.py`: rosbag 한 개 → `labels_cache.h5`

```
rosbag (대량 텔레옵 주행)
 ├ SegFormer(freeze) → 차선 세그 마스크 (자동)
 ├ YOLO(freeze)      → 차량 bbox        (자동)
 └ cmd_vel           → steer/throttle/waypoint (자동 적분 생성)
```

- lane 프레임마다 가장 가까운 front/cmd_vel을 **±50ms 시간 동기화**해 1샘플 생성.
- H5에 raw 이미지 + seg/det를 **따로** 저장 → 오버레이 합성은 dataloader에서 → 재추출 없이 합성 튜닝 가능.
- 색공간(BGR)·ROI 크롭(`LANE_CROP_TOP=0.30`)을 라벨/추출/추론 전부 동일하게 고정(좌표계 정합).
- **눈 검증**(`data_pipeline/visualize_labels.py`)으로 세그/bbox 품질 확인 후 학습.

> ※ 슬라이드 3·4는 "구현 (2)"의 앞/뒤 흐름. 발표 분량상 한 장으로 합쳐도 됨(그러면 총 6장).

---

## 슬라이드 5 — 구현 (3) 주행 데이터 학습 & 멀티태스크 네트워크 ⭐

### 네트워크 구조
> `model.py` `E2ENet`

```
오버레이 Lane 이미지  → LaneEncoder (ResNet18-A) → 256d ┐
오버레이 Front 이미지 → FrontEncoder(ResNet18-B) → 256d ┘ concat(512)
                                     ├→ ControlHead  → steer, throttle (Tanh, 메인)
                                     └→ WaypointHead → waypoints 5×2 (보조)
```

### 정확히 뭘 학습하나 (앞 슬라이드와 연결)
- SegFormer/YOLO는 freeze(지각). **새로 학습하는 건 ResNet18×2 + 두 헤드**(판단+제어).
  ResNet은 ImageNet pretrained 재사용 후 전체 fine-tune.
- 입력은 raw가 아니라 **전처리된 오버레이 이미지**:
  Lane=seg 3채널 색 blend(좌빨강/우초록/중앙파랑), Front=car bbox.
- **별도 거리 센서/임계값 없음** — 모델이 bbox **위치+크기**로 거리감을 암묵 학습 → 회피·추월 타이밍.

### 멀티태스크 loss & 알아서 주행하는 원리
> `E2ELoss`

```
loss = 1.0·MSE(steer) + 0.5·MSE(throttle) + 0.5·MSE(waypoint)
```

- 메인 = steer/throttle(실제 제어). 보조 = waypoint(cmd_vel 적분한 미래 0.5초 궤적, 추론 시 버림).
- 두 헤드가 **같은 512-d feature 공유** → 보조 task가 backbone을 "옆으로 비켰다 복귀하는 추월 의도" 쪽으로 **regularize** → 메인 일반화↑.
- 명시적 추월 규칙 없이 텔레옵 분포를 모방 → **자율 주행·추월이 창발**.

---

## 슬라이드 6 — 구현 (4) 코드 & 임베디드 최적화

### ROS2 시스템 / 동시성
- 노드 5개(camera/monitor/teleop/motor_bridge/bag_recorder) 단일 책임 분리.
- `MultiThreadedExecutor` + `ReentrantCallbackGroup` — 수신/추론/제어 스레드 분리(블로킹 방지).
- 카메라 hot-path: jetcam **native BGR 그대로 JPEG 인코딩** → BGR↔RGB 왕복 제거.
  리더 스레드 분리, capture_fps 2배로 버퍼 신선.
- DDS/QoS: `ROS_LOCALHOST_ONLY=1`로 외부 멀티캐스트 차단(지연↓). 이미지 토픽은
  **실시간 소비자(모니터·오버레이·추론)만 BEST_EFFORT+depth1**로 옛 프레임 버려 지연 누적 차단
  (트레이드오프 프레임 드롭). **학습 bag recorder는 RELIABLE 유지**(완결성). 처리시간 지연은 TensorRT fp16로 별도 해결.

### 데이터·코딩 기법
- rosbag으로 compressed 토픽을 db3 하나에 저장(용량↓), Colab에선 `rosbags`로 ROS2 없이 파싱.
- H5 gzip+chunk+resizable로 메모리 폭발 없이 대량 스트리밍 저장.
- `/record_enable` 엣지로 bag 녹화 자동 시작/종료 + lane 프레임 없으면 자동 종료(쓰레기 방지).

### 임베디드 배포 최적화
- PyTorch → ONNX → Jetson `trtexec --fp16` TensorRT engine.
  **engine은 Jetson에서만 빌드**(아키텍처 박힘). YOLO26은 NMS-free라 후처리 없이 변환.

---

## 슬라이드 7 — 결론: 시연 영상 & 시행착오

### 주행 시연 (영상 재생)
- 직선 추종 → 코너링 → 정지 차량 접근 → **회피·추월** → 차선 복귀까지 한 사이클.
- waypoint 시각화 오버레이를 같이 띄워 "모델의 의도"를 보여줌.
- **결과**: 다양한 차량 위치(좌/우/중앙)·거리에서 안정적으로 추월·복귀 성공,
  명시적 규칙 없이 데이터만으로 판단이 창발함을 확인. 임베디드 추론도 딜레이 없이 실시간 동작.

### 시행착오 정리
> `CHECKLIST.md`

| 문제 | 해결 |
|---|---|
| 회전 시 토크 부족으로 정지 | throttle-steer **coupling** (회전 시 -0.25 자동 증가) |
| 회전·추월 데이터 부족 | 직선→코너→복귀 / 접근→회피→복귀 **반복 수집** |
| 객체 인식 저조 | 클래스당 200장+, 다양한 거리/각도 |
| 차선 좌우 뒤바뀜 | **flip aug 금지** |
| bbox 틀어짐 | export **stretch resize 금지** |
| SSH 키 입력 / 모니터링 지연 | **termios cbreak** / **재인코딩 없는 MJPEG**(제어와 분리) |
| Jetson engine 미동작 | **trtexec는 Jetson에서만** |
| (작업방식) 변경마다 전체 재작성 | **단계(Phase)별 분리 + 계약 고정**으로 부분만 교체 |

### 작업방식 회고 — 왜 단계(Phase)로 쪼갰나
- 중간 프로젝트 때는 **처음부터 agentic하게 전 과정을 한 번에 다 생성**했더니, 실전에서 문제·변경점이 생길 때마다(센서 좌표, 라벨 품질, 코너 토크 등) **매번 전체를 새로 짜야 했다.**
- 그래서 작업방식을 바꿔, 파이프라인을 **Phase 1(지각 학습) / Phase 2(수집·E2E 학습) / Phase 3(실차 추론·모니터링·추가학습)** 로 분리하고, 단계 사이를 **고정 계약**(H5 스키마, 합성·정규화·ROI 크롭, cmd_vel 역변환 공식)으로 묶었다.
- 효과: 한 단계에서 문제가 나도 **그 단계만 교체**(예: SegFormer만 재학습, 데이터로더 합성만 튜닝, 추론 노드만 수정) → 나머지는 계약이 보장돼 그대로 재사용. 변경 비용이 전체→부분으로 줄었다.

**마무리 한 줄**: 지각(SegFormer/YOLO)–제어(ResNet+멀티태스크) 분리와 라벨 효율적 2단계 파이프라인으로, 규칙 없이도 차선 주행 + 정지 차량 추월을 학습·실차 검증 완료. 단계별 계약 고정으로 실전 변경에도 부분 교체만으로 대응.
