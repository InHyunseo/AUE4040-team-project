# Phase 3 — 실차 배포 · 추론 · 주행 · 추가학습

> `e2e.onnx` → Jetson TensorRT engine → `rover_lane` 추론 노드로 실차 주행.
> 새 bag으로 추가학습(`--resume`). 노드 구조 설명은 [README](README.md#실차-추론-rover_lane).

---

## 모델 파일 준비

[Kaggle 노트북](https://www.kaggle.com/code/hyunseoin/e2e-train) Output 탭에서 `e2e.onnx` 를
받아 `final_project/models/e2e.onnx` 에 둔다.

---

## TL;DR — 명령만

```bash
# ── 0. Jetson: ONNX → TensorRT engine (Jetson에서만, 최초 1회) ───────────────
#      (먼저 위 "모델 파일 준비"대로 e2e.onnx 를 models/ 에 받아둘 것)
cd ~/team/final_project
/usr/src/tensorrt/bin/trtexec --onnx=models/e2e.onnx --fp16 --saveEngine=models/e2e.engine

# ── 1. 빌드 ─────────────────────────────────────────────────────────────────
cd ~/team/final_project/ros2_ws
colcon build --symlink-install
source install/setup.bash

# ── 2. 자율주행 (한 방에: 카메라 + 모터 + 추론) ──────────────────────────────
cd ~/team/final_project/ros2_ws
source install/setup.bash
ros2 launch rover_lane drive.launch.py monitor:=false publish_overlay:=false
# 최종 주행: 브라우저 모니터와 디버그 JPEG 오버레이를 꺼서 자원 절약

# 디버그 주행/드라이런: raw + lane_intent + front_det 를 :8080 에 표시
ros2 launch rover_lane drive.launch.py dry_run:=true monitor:=true publish_overlay:=true

# ── 3. 점검 ─────────────────────────────────────────────────────────────────
cd ~/team/final_project/ros2_ws
source install/setup.bash
ros2 topic hz /cmd_vel        # 실제 제어 주기 (SegFormer 병목 확인)
ros2 topic echo /cmd_vel      # 명령 값 확인
```


### 추가학습 (실주행에서 약한 시나리오 보강)

추출은 SegFormer/YOLO 가 있는 **로버(Jetson)에서**, 학습은 **Kaggle 노트북에서** 한다.
h5 를 로버에서 만들어 Kaggle 로 옮기고, 학습은 노트북 설정만 바꿔 돌린다.

1. **[로버] 부족 시나리오 위주로 새 bag 수집** (record.launch + teleop, Phase 2와 동일)

   ```bash
   ros2 launch rover_recorder record.launch.py session_name:=phase3_corner
   ```

2. **[로버] bag → labels 일괄 추출** — `--bag_root` 에 bag 들이 든 폴더(record.launch
   기본 저장 위치 = `rover_data/`)를 주면 안의 모든 bag 을 순회해 각각
   `<out_dir>/<세션명>.h5` 로 만든다(모델 1회만 로드). 기존 17개 + 새 코너 bag 이
   모두 `rover_data/` 에 있으면 한 명령으로 전부 2.5초 정의로 재라벨링된다.

   ```bash
   cd ~/team/final_project

   # 먼저 잡히는 bag 확인 (기존+신규 다 나오는지 — 경로 틀리면 일부 누락)
   find rover_data -name metadata.yaml | sort

   # 이전 추출 결과가 섞이지 않게 비우고 새로 생성
   rm -rf labels_all

   python3 data_pipeline/extract_labels.py --bag_root rover_data \
       --segformer_ckpt models/segformer_lane --yolo_weights models/best.pt \
       --out_dir labels_all
   # → labels_all/<세션명>.h5 들이 생성됨
   ```

   > bag 하나만 추출하려면 `--bag rover_data/<session>/bag --out labels_all/<session>.h5`.

   학습 전에 H5 계약을 확인한다. 옛 부호 버그 시절 H5는 Kaggle 노트북에서
   `WP_FIX_SIGN=True` 로 두고, 새 `extract_labels.py`로 재추출한 H5는 `False` 로 둔다.

   ```bash
   cd ~/team/final_project/training
   python3 audit_h5.py --cache ../labels_all/*.h5
   # legacy H5 확인용:
   python3 audit_h5.py --cache ../labels_all/*.h5 --wp_fix_sign
   ```

3. **[로버→로컬→Kaggle] h5 폴더 통째로 옮기기** — 여러 h5 가 든 폴더를 한 번에 압축해
   로컬로 내린 뒤, 풀어서 **Kaggle Dataset 으로 업로드**.

   ```bash
   # [로버] labels_all/ 폴더 전체를 tar 로 묶기
   cd ~/team/final_project && tar czf labels_all.tar.gz labels_all

   # [로컬] 로버에서 받아 풀기
   scp <rover>:~/team/final_project/labels_all.tar.gz .
   tar xzf labels_all.tar.gz
   # → labels_all/ 안의 *.h5 들을 Kaggle Dataset 에 업로드 (드래그&드롭)
   ```

4. **[Kaggle] `e2e-train` 노트북에서 이어학습** — 명령을 치는 게 아니라 설정 셀만 바꾼다:
   - 노트북이 `/kaggle/input/**/*.h5` 를 자동 수집(`CACHES`)하므로, Dataset 에 h5 만 추가하면
     기존+신규가 전부 학습에 들어간다(개수 신경 안 써도 됨, 분포 망각 완화).
   - 이어학습 켜기: `RESUME = ""` → `RESUME = CKPT` (또는 이전 output 의 `e2e_best.pt` 경로).
   - 노트북 실행 → 재학습 → `e2e.onnx` / `e2e_best.pt` output 생성.

5. **[→ Jetson] 새 `e2e.onnx` 를 받아 0단계(engine 재빌드)로** 돌아간다.

---

## 메모

- **engine은 Jetson(Orin)에서만 빌드** — GPU 아키텍처가 박혀 x86/Colab engine은 호환 안 됨.
  입력 `lane`,`front`(1×3×224×224), 출력 `steer`,`throttle`,`waypoints`. 노드는 바인딩을
  이름으로 매칭해 I/O 순서 변경에 안전.
- **전처리·노드 구조·안전장치(watchdog)·QoS 설명** → [README 실차 추론 섹션](README.md#실차-추론-rover_lane).
- **제어율이 낮으면** 먼저 `monitor:=false publish_overlay:=false` 로 디버그 JPEG 비용을 끈다.
  그래도 낮으면 YOLO 격프레임/직전 bbox 재사용을 검토하고, 마지막으로 SegFormer TensorRT
  변환을 진행한다. (overlay_viz의 3Hz는 디버그용 인위적 캡이지 SegFormer 성능 한계가 아니다.)
- **추가학습 시 `--cache` 여러 개** = 기존+신규 합쳐 학습(분포 망각 완화). 데이터가 작으면
  합쳐서 재학습이 `--resume`보다 안정적일 수 있음.
- ⚠️ **waypoint 정의를 바꾸면** `--resume` 정합성 영향 — 점 개수(WP_N) 변경은 shape 불일치로
  불가, horizon만 변경은 첫 회 head 재적응 발생. (현재 2.5초/5점.)

---

## 완료 기준

- [ ] Jetson `e2e.engine` 빌드 (fp16)
- [ ] `rover_lane` 노드: 전처리(학습과 픽셀 동일) → engine → cmd_vel 동작
- [ ] 실차 차선 주행 (직선 / 코너 / S자)
- [ ] 실차 정지 차량 회피·추월
- [ ] `ros2 topic hz /cmd_vel` 제어율 확인 (13~15Hz 목표)
- [ ] 최종 주행은 `monitor:=false publish_overlay:=false` 로 확인
- [ ] (제어율 낮고 주행 끊길 때만) overlay off → YOLO 격프레임 → SegFormer TensorRT 순서로 최적화
- [ ] 추가학습 루프 1회 (수집 → 재추출 → scratch 학습 → engine 재빌드) 검증
