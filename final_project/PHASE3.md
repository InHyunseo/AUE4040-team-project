# Phase 3 — 실차 배포 · 추론 · 주행 모니터링 · 추가학습

> Phase 2에서 만든 `e2e.onnx`를 Jetson에서 TensorRT engine으로 바꾸고,
> `rover_lane` 추론 노드로 실차를 굴린다. 주행 의도를 :8080으로 모니터링하고,
> 새로 모은 bag으로 추가학습(`--resume`)한다.
>
> 학습 파이프라인은 [PHASE2.md](PHASE2.md), 모델 구조는 [model.py](model.py),
> 합성·정규화 계약은 [training/dataset.py](training/dataset.py),
> 의도 시각화는 [training/viz.py](training/viz.py) 참고.

---

## 전체 흐름

```
(Phase 2 산출물) e2e.onnx
   │
1. Jetson TensorRT 변환        → e2e.engine  (trtexec --fp16, Jetson에서만)
2. rover_lane 추론 노드         → /cmd_vel    (전처리 → engine → 역변환)
3. 주행 모니터링                → :8080       (예측 의도 오버레이, 켜고/끔)
4. 추가학습 루프                → e2e_best.pt (새 bag 수집 → extract → --resume)
```

전처리(ROI 크롭 / SegFormer seg 오버레이 / YOLO bbox 오버레이 / BGR→RGB→ImageNet)는
**학습 때와 픽셀 단위로 동일**해야 한다. 이 계약의 단일 소스는
`training/dataset.py`(`composite_lane`/`composite_front`/`to_input_tensor`)와
`data_pipeline/extract_labels.py`(`crop_lane_roi`, `LANE_CROP_TOP`, `SegFormerLaneSeg`,
`YoloCarDet`)다. **추론 노드는 이 함수들을 import해서 그대로 쓴다 — 재구현 금지.**

> 참고: 이미 `rover_camera/overlay_viz_node.py`가 동일 전처리 계약으로
> `/lane_seg`·`/front_det` 오버레이를 publish한다. 추론 노드 전처리는 이 노드와
> 같은 패턴(project root 탐색 → `extract_labels`/`dataset` import)을 따른다.

---

## 1단계 — Jetson TensorRT 변환

ONNX→engine은 **Jetson(Orin Nano)에서만** 빌드한다 (GPU 아키텍처가 박히므로
Colab/x86에서 만든 engine은 호환 안 됨):

```bash
# Jetson에서
/usr/src/tensorrt/bin/trtexec \
    --onnx=e2e.onnx --fp16 --saveEngine=e2e.engine
```

- 입력 2개(`lane`, `front`, 각 1×3×224×224), 출력 3개(`steer`, `throttle`, `waypoints`).
- `waypoints`는 추론 주행에선 쓰지 않는다(모니터링 전용). engine에는 남아 있어도 무방.
- 변환 정합성은 Phase 2의 `export_onnx.py --check`에서 이미 확인(PyTorch↔onnxruntime).

---

## 2단계 — rover_lane 추론 노드 [골격 — 구현 TODO]

> 패키지는 `ros2_ws/src/rover_lane/`에 만들 예정(**아직 없음** — 실차 추론 단계에서 생성).
> 기존 `rover_camera`/`rover_recorder` 패키지 구조(package.xml/setup.py)를 복제하고,
> SegFormer/YOLO 전처리·합성은 `extract_labels`/`dataset`에서 import, 엔진 추론과
> cmd_vel 역변환만 노드에서 새로 구현한다.

### 데이터 흐름

```
/lane_image/compressed  ─┐
                         ├─ 전처리(학습과 동일) ─→ lane_t, front_t (1×3×224×224)
/front_image/compressed ─┘        │
                                  ▼
                         e2e.engine (TensorRT) ─→ steer, throttle  (+waypoints, 모니터링용)
                                  │
                                  ▼   역변환 (PHASE1/README 계약과 일치)
                         /cmd_vel:
                            linear.x  = -(0.15 + abs(steer) * 0.10)   # -0.15 ~ -0.25
                            angular.z = steer * 0.8                     # -0.8 ~ +0.8
                                  │
                                  ▼
                         motor_bridge_node → UART
```

### 전처리 계약 (재사용)

추론 노드는 lane 경로에서 반드시:
1. `crop_lane_roi(lane)` (같은 `LANE_CROP_TOP`) → `cv2.resize(.., LANE_SIZE)`
2. `SegFormerLaneSeg`(freeze) → seg (3,224,224)
3. `dataset.composite_lane(lane, seg)` → BGR 오버레이
4. `dataset.to_input_tensor(...)` → RGB ImageNet 정규화 텐서

front 경로: `cv2.resize(.., FRONT_SIZE)` → `YoloCarDet`(freeze) → `dataset.composite_front`
→ `dataset.to_input_tensor`. (front는 crop 안 함.)

> 색공간(BGR→RGB)·정규화·resize·crop 순서가 학습과 한 픽셀이라도 어긋나면
> 엉뚱하게 주행한다. 위 함수를 직접 호출하는 것이 그걸 보장하는 유일한 방법.

### 엔진 추론

- TensorRT engine 로드 + 바인딩(`lane`,`front` 입력 / `steer`,`throttle`,`waypoints` 출력).
- engine이 아직 없으면(개발 단계) **PyTorch fallback**으로 `model.E2ENet` + `e2e_best.pt`를
  써서 노드 구조만 검증 가능하게 한다(`--backend torch|trt`).

### QoS / DDS (Phase 2 메모와 동일)

- 단일 Jetson 호스트면 `export ROS_LOCALHOST_ONLY=1`.
- 추론 노드는 **실시간 소비자** → 이미지 sub에 `BEST_EFFORT + KEEP_LAST depth=1`
  (monitor_node·overlay_viz_node와 동일 QoS). 밀린 옛 프레임을 버리고 최신만 처리해
  지연 누적을 끊는다 — 과거 프레임 보고 조향하면 위험하므로 최신성이 정답.
  트레이드오프는 프레임 드롭이지만 제어엔 무해(완결성 불필요). 큐 적체 지연만 끊는
  것이고, SegFormer+YOLO+engine **처리시간 자체 지연은 TensorRT fp16의 몫**.
- 카메라 `header.stamp`는 **캡처 시각**(Phase 2 메모 참조) — 두 카메라 정합 기준.

---

## 3단계 — 주행 모니터링 (의도 오버레이) [켜고/끔]

추론 노드가 매 프레임 예측 `waypoints`(+steer/throttle)를 lane 오버레이 이미지에
그려 JPEG로 publish → 브라우저(:8080)에서 실시간 확인. **파라미터로 끈다**
(`monitor:=false`면 추론만, 시각화 토픽 publish 안 함 → 지연 최소).

- 그리기는 `training/viz.draw_intent(lane_bgr, waypoints, color)`를 그대로 쓴다
  (좌표 변환이 `visualize_labels.draw_waypoints`와 단일 소스로 일치).
- publish 토픽 예: `/lane_intent/compressed` (overlay_viz의 `/lane_seg`와 같은 결).
- launch 인자: `ros2 launch rover_lane drive.launch.py monitor:=true engine:=.../e2e.engine`

> 학습 쪽 모니터링(예측 vs GT 패널)은 Phase 2 `train_e2e.py --viz_dir`. 추론 쪽은
> GT가 없으니 예측 의도만 그린다. 둘 다 `viz.draw_intent`를 공유.

---

## 4단계 — 추가학습 (incremental)

실주행에서 약한 시나리오(특정 코너, 특정 차량 위치)가 보이면 그 상황 위주로 더 수집
→ 추가학습:

```bash
# 1) 부족한 시나리오 위주로 새 bag 수집 (Phase 2 1단계와 동일 스택)
# 2) 새 bag → labels_cache 추출 (Phase 2 2~3단계)
python3 data_pipeline/extract_labels.py --bag .../phase3_<ts>/bag \
    --segformer_ckpt models/segformer_lane --yolo_weights models/best.pt \
    --out labels_phase3.h5 --debug_dir debug_samples

# 3) 기존 best에서 이어학습 (가중치 + optimizer state 복원)
python3 training/train_e2e.py \
    --cache labels_cache.h5 --cache labels_phase3.h5 \
    --resume models/e2e_best.pt \
    --viz_dir debug_samples/train_viz --viz_every 5 \
    --out models/e2e_best.pt

# 4) 다시 ONNX → Jetson engine 재빌드 → 노드 교체
python3 training/export_onnx.py --ckpt models/e2e_best.pt --out models/e2e.onnx --check
```

- `--cache`를 여러 개 주면 기존+신규 bag을 합쳐 학습한다(분포 망각 완화).
- `--resume`는 처음부터 안 돌리고 fine-tune. 데이터가 작으면 합쳐서 재학습이 더 안정적
  (Phase 2 train_e2e docstring 참고).

---

## 완료 기준

- [ ] Jetson에서 `e2e.engine` 빌드 (fp16)
- [ ] `rover_lane` 노드: 전처리(학습과 픽셀 동일) → engine → cmd_vel 역변환 동작
- [ ] 실차에서 차선 주행 (직선/코너/S자)
- [ ] 실차에서 정지 차량 회피·추월
- [ ] 주행 의도 모니터링(:8080) 켜고/끔 동작
- [ ] 추가학습 루프 1회(부족 시나리오 수집 → --resume → engine 재빌드) 검증
