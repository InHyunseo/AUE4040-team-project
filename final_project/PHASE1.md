# Phase 1 — 라벨용 데이터 수집 & SegFormer/YOLO 학습

> SegFormer(차선 세그)와 YOLO(car 감지)를 fine-tune해 freeze하는 단계.
> 이후 Phase 2(대량 본 학습)에서 재사용.

---

## 전체 흐름

```
1. rosbag 주행 영상 수집    → final_project/rover_data/<session>_<ts>/bag
2. bag → jpg 추출          → roboflow_input/{lane,front}/*.jpg
3. CVAT 라벨링 (수작업)     → Lane 세그 + Front bbox
4. 학습                    → SegFormer 체크포인트 + YOLO best.pt
```

> 차선 세그는 raw lane 이미지(sensor 0) 위에서 바로 동작한다.

---

## 1단계 — rosbag 주행 영상 수집

```bash
cd ~/team/final_project/ros2_ws
colcon build --symlink-install
source install/setup.bash


# 터미널 A — 카메라 + 모터 + bag 레코더 http://192.168.0.123:8080/
cd ~/team/final_project/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch rover_recorder record.launch.py session_name:=phase1


# 터미널 B — 키보드 텔레옵 (별도 TTY)
cd ~/team/final_project/ros2_ws && source install/setup.bash
ros2 run rover_teleop teleop_node
```

**텔레옵 키**: `a`/`d` 조향(-2~+2, 좌/우 두 번이면 최대 회전), `space` 정지, `g` 주행 on/off, `r` 녹화 on/off, `q` 종료.
`r`로 녹화 켜면 `final_project/rover_data/phase1_<ts>/bag/`에 저장.

다양하게 수집:
- 차선: 직선 / 코너 / S자 — 좌우 실선과 중앙 점선이 다양한 각도로
- 차량: 정지 차량을 좌 / 우 / 중앙, 가까이 / 멀리 다양한 위치에 놓고 주행

---

## 2단계 — bag → jpg 추출

```bash
cd final_project/data_pipeline
python3 extract_for_labeling.py \
    --bag ../rover_data/phase1_20260604_164551/bag \
    --out ../roboflow_input \
    --stride 15
#  → roboflow_input/lane/<bag>_<idx>.jpg   (224x224 raw lane, SegFormer용)
#  → roboflow_input/front/<bag>_<idx>.jpg  (224x224 Front, YOLO용)
```

- `--stride 15` : 15fps에서 1초당 1장. 저장 장수 ≈ (녹화 프레임 수 ÷ stride).
  예) 3분 주행 bag → lane/front 각 ~180장. 목표 장수는 bag 길이·개수로 맞추면 됨.
- `--target {lane,front,both}` : 기본 both. Front만 먼저면 `--target front`.
- bag 여러 개면 각각 실행 (파일명에 bag 이름 prefix).

Lane jpg는 `extract_labels.py`가 SegFormer 돌리는 것과 동일한 raw 이미지(resize만)로 추출됨.

---

## 3단계 — CVAT 라벨링 (수작업)

> CVAT task **2개**를 따로 만든다: Lane = polyline(차선), Front = bbox(rectangle).
> 같은 task에 섞지 말 것.

### Front jpg → bbox
- 클래스: **car** 단일 클래스 (rectangle)
- 클래스당 **200장 이상**
- 정지 차량을 다양한 위치/거리에서
- Export: **YOLO 1.1** (또는 Ultralytics YOLO)

### Lane jpg → polyline 차선
- 클래스 3개: **left-solid / right-solid / center-dashed** (**polyline**)
  (라벨 이름은 노트북 `CVAT_LABEL2ID`와 맞추면 됨)
- 차선을 따라 선만 그으면 됨 (면 칠하기 X). 노트북이 `LANE_THICKNESS` px 띠로 자동
  rasterize → SegFormer 학습 마스크 생성.
- 점선(center-dashed)도 끊김 없이 한 줄로 쭉 긋고 두께만 주면 됨.
- 100~300장
- Export: **CVAT for images 1.1** (`annotations.xml` + 원본 이미지)

### 공통 주의
- **resize 하지 말 것** (추출 jpg가 이미 모델 입력 좌표계). CVAT는 원본 그대로 라벨.
- **flip augmentation 금지** (좌/우 실선 레이블 뒤바뀜) — aug는 학습 노트북에서만.

---

## 4단계 — 학습 (Colab)

두 노트북 모두 **CVAT export zip을 업로드 셀에서 올리고** 실행.

### YOLO (car 단일 클래스)
[training/train_yolo_colab.ipynb](training/train_yolo_colab.ipynb)

1. CVAT에서 car bbox → **YOLO 1.1** export → zip 다운로드
2. 노트북 업로드 셀에서 zip 업로드 → `CONFIG` 하이퍼파라미터만 확인 후 실행
3. → `best.pt` 브라우저로 다운로드 → WSL에 둠

→ Phase 2: `extract_labels.py --yolo_weights <best.pt>`

### SegFormer (차선 세그)
[training/train_segformer_colab.ipynb](training/train_segformer_colab.ipynb)

1. CVAT에서 polyline 차선 → **CVAT for images 1.1** export → zip 다운로드
   - 클래스 순서 고정: 배경 0 / 좌실선 1 / 우실선 2 / 중앙점선 3
     (extract_labels.py `SegFormerLaneSeg`의 id2label과 일치해야 함)
   - 노트북 `CVAT_LABEL2ID`를 CVAT 라벨 이름에, `LANE_THICKNESS`를 원하는 띠 두께에 맞춤
     (polyline → 마스크 rasterize는 노트북이 자동)
2. 노트북 업로드 셀에서 zip 업로드 후 실행
3. → `segformer_lane.zip` 다운로드 → 압축 해제한 폴더가 체크포인트

→ Phase 2: `extract_labels.py --segformer_ckpt <segformer_lane 폴더>`

---

## 완료 기준

- [x] Front car bbox 200장+ 라벨링 → YOLO `best.pt`
- [x] Lane 차선 세그 100장+ 라벨링 → SegFormer 체크포인트
- [x] 두 모델 freeze 확정

이후 Phase 2: 대량 rosbag 수집 → `extract_labels.py`로 `labels_cache.h5` 자동 생성 → E2E 학습.
자세한 건 [README.md](README.md) 참고.
