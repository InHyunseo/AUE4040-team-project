# Phase 1 — 라벨용 데이터 수집 & SegFormer/YOLO 학습

> SegFormer(차선 세그)와 YOLO(car 감지)를 fine-tune해 freeze하는 단계.
> 이후 Phase 2(대량 본 학습)에서 재사용.

---

## 전체 흐름

```
1. BEV 캘리브레이션        → calib.json
2. rosbag 주행 영상 수집    → ~/rover_data/<session>/bag
3. bag → jpg 추출          → roboflow_input/{bev,front}/*.jpg
4. Roboflow 라벨링 (수작업) → BEV 세그 + Front bbox
5. 학습                    → SegFormer 체크포인트 + YOLO best.pt
```

---

## 1단계 — BEV 캘리브레이션

BEV jpg를 warp하려면 `calib.json`이 필요. (이미 있으면 건너뜀)

```bash
# 카메라 켜고 체커보드 한 컷 캡처
ros2 launch rover_calib bev_capture.launch.py
#  → 체커보드를 BEV 카메라에 놓고, 이 터미널에서 'c' 키
#  → final_project/calib/bev_capture_<ts>.jpg 저장 후 자동 종료

# calib.json 생성 (rows/cols = 체커보드 내부 코너 수, square_m = 한 칸 실측 m)
cd final_project/data_pipeline
python3 bev_calibration.py --image ../calib/bev_capture_<ts>.jpg \
    --rows 6 --cols 9 --square_m 0.025
#  → final_project/calib/calib.json  (M, bev_size, pixels_per_meter)
```

Front만 먼저 모을 거면 calib 없이 3단계에서 `--target front` 가능.

---

## 2단계 — rosbag 주행 영상 수집

```bash
# 터미널 A — 카메라 + 모터 + bag 레코더
ros2 launch rover_recorder record.launch.py session_name:=phase1

# 터미널 B — 키보드 텔레옵 (별도 TTY)
cd final_project/ros2_ws && source install/setup.bash
ros2 run rover_teleop teleop_node
```

**텔레옵 키**: `a`/`d` 조향(-5~+5), `space` 정지, `g` 주행 on/off, `r` 녹화 on/off, `q` 종료.
`r`로 녹화 켜면 `~/rover_data/phase1_<ts>/bag/`에 저장.

다양하게 수집:
- 차선: 직선 / 코너 / S자 — 좌우 실선과 중앙 점선이 다양한 각도로
- 차량: 정지 차량을 좌 / 우 / 중앙, 가까이 / 멀리 다양한 위치에 놓고 주행

---

## 3단계 — bag → jpg 추출

```bash
cd final_project/data_pipeline
python3 extract_for_labeling.py \
    --bag ~/rover_data/phase1_<ts>/bag \
    --calib ../calib/calib.json \
    --out ../roboflow_input \
    --stride 15
#  → roboflow_input/bev/<bag>_<idx>.jpg    (warp된 BEV, SegFormer용)
#  → roboflow_input/front/<bag>_<idx>.jpg  (224x224 Front, YOLO용)
```

- `--stride 15` : 15fps에서 1초당 1장.
- `--target {bev,front,both}` : 기본 both. Front만 먼저면 `--target front` (calib 불필요).
- bag 여러 개면 각각 실행 (파일명에 bag 이름 prefix).

BEV는 `extract_labels.py`가 SegFormer 돌리는 것과 동일한 warp 좌표계로 추출됨.

---

## 4단계 — Roboflow 라벨링 (수작업)

### Front jpg → bbox
- 클래스: **car** 단일 클래스
- 클래스당 **200장 이상**
- 정지 차량을 다양한 위치/거리에서

### BEV jpg → polygon 세그
- 클래스 3개: **좌실선 / 우실선 / 중앙점선**
- 100~300장

### 공통 주의
- **Resize: OFF**, Auto-Orient: ON
- **flip augmentation 금지** (좌/우 실선 레이블 뒤바뀜)
- 사용 가능한 aug: 밝기/대비, 색조 약간, 가우시안 노이즈, 방향 유지 랜덤 크롭

---

## 5단계 — 학습

두 노트북 모두 Roboflow **Export → Show download code** 의 `api_key / workspace / project / version` 을 `CONFIG`에 채우고 실행.

### YOLO (car 단일 클래스)
[training/train_yolo_colab.ipynb](training/train_yolo_colab.ipynb)

1. Roboflow에서 car bbox export (YOLOv8 포맷) → download code 확인
2. 노트북 `CONFIG`에 `RF_*` 값 채우고 실행
3. → `best.pt` 브라우저로 다운로드 → WSL에 둠

→ Phase 2: `extract_labels.py --yolo_weights <best.pt>`

### SegFormer (차선 세그)
[training/train_segformer_colab.ipynb](training/train_segformer_colab.ipynb)

1. Roboflow에서 polygon 세그 라벨링 → **PNG Mask Semantic** 포맷 export
   - 클래스 순서 고정: 배경 0 / 좌실선 1 / 우실선 2 / 중앙점선 3
     (extract_labels.py `SegFormerLaneSeg`의 id2label과 일치해야 함)
2. 노트북 `CONFIG`에 `RF_*` 값 채우고 실행
3. → `segformer_lane.zip` 다운로드 → 압축 해제한 폴더가 체크포인트

→ Phase 2: `extract_labels.py --segformer_ckpt <segformer_lane 폴더>`

---

## 완료 기준

- [ ] `calib.json` 존재
- [ ] Front car bbox 200장+ 라벨링 → YOLO `best.pt`
- [ ] BEV 차선 세그 100장+ 라벨링 → SegFormer 체크포인트
- [ ] 두 모델 freeze 확정

이후 Phase 2: 대량 rosbag 수집 → `extract_labels.py`로 `labels_cache.h5` 자동 생성 → E2E 학습.
자세한 건 [README.md](README.md) 참고.
