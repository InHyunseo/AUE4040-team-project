# 내일 할 일 (2026-05-22 예정)

> 작성일: 2026-05-21
> 대상: 같이 작업하는 팀원
> 한 줄 요약: **내일은 하드웨어 데이`. 도로 없이 못 하는 것만 함.**

지금까지 책상에서 할 수 있는 건 거의 다 했다. ROS 노드 7개 모두 wire-up 끝, stereo 캘리브 완료(2026-05-19, commit `c74c7fd`, RMS≈1.84), lane / YOLO 추론 파이프라인 스모크 테스트 통과. 남은 건 **실제 도로 + 실제 차량 + 실제 표지판이 있어야만** 할 수 있는 것들이다.

## 0. 시작 전 체크 (5분)

- [ ] Jetson 부팅, 배터리 충분한지 확인 (지난번처럼 저전압이면 모터가 통신은 되는데 안 움직임)
- [ ] CSI 카메라 두 개 다 인식되는지: `ls /dev/video*` → 두 개 보여야 함. 안 보이면 `sudo systemctl restart nvargus-daemon`
- [ ] `/dev/ttyUSB0` 보이는지 (모터 UART): `ls /dev/ttyUSB*`
- [ ] 워크스페이스 빌드 & source:
  ```bash
  ros_start
  # = cd ~/team/main/ros2_ws && colcon build --symlink-install && source install/setup.bash
  ```

---

## 1. 런치 그래프 스모크 테스트 (30분, 실내, 도로 X)

먼저 노드 5개가 다 뜨고 토픽이 흐르는지만 확인. 모델 파일(`.engine`) 없어도 됨 — 노드는 warn만 띄우고 살아있을 것이다.

```bash
# 터미널 A
ros2 launch rover_bringup autonomous.launch.py mission:=left

# 터미널 B
ros2 topic list
ros2 topic hz /image_rectified /road_center /detections /fsm_state /cmd_vel
```

**기대 결과:**
- `/image_rectified`이 카메라 fps(~30 Hz)로 흐름
- `/road_center`, `/detections`, `/fsm_state`, `/cmd_vel` 모두 보임 (값은 의미 없음 — 모델 없음)
- 노드 log에 "models not loaded" / "YOLO engine not loaded" warn은 정상. **error로 죽으면 안 됨.**

**여기서 막히면 도로 가기 전에 디버그.** 도로에서 디버그하면 시간 다 날린다.

---

## 2. 차량 bbox 거리 K 캘리브레이션 (5분, 실차 필요)

차량(회전교차로에서 움직일 그 차)이 실제로 있어야 하는 작업.

```bash
# 차량을 카메라 정면 0.5 m 앞에 정확히 세움
python3 ros2_ws/src/rover_perception/scripts/calibrate_vehicle_distance.py \
    --distance 0.5 --write

# OpenCV 창에서: 차량 bbox의 좌상단 → 우하단 두 번 클릭
# K = bbox_h_px * 0.5 가 자동으로 params.yaml의 vehicle_dist_K에 들어감
```

**왜 이게 필요한가**: decision_node가 `d = K / bbox_h_px`로 차량까지 거리를 계산. K 한 번만 캘리브하면 closed-world 내내 사용 가능. stereo disparity는 안 쓴다 (런타임 비용 줄이려고).

**실차가 없으면**: 비슷한 크기의 대체 물체로 일단 K 측정 → 실차 도착하면 다시 측정. 어차피 한 줄 갱신.

---

## 3. Lane 데이터 수집 — 첫 세션 (1~2시간, 도로 필요)

```bash
jupyter notebook ros2_ws/src/rover_recorder/notebooks/record_and_label.ipynb
```

### Part A — 운전하면서 저장
- 화살표 키: 조향·속도
- `1` / `2` / `3` : segment 라벨 = `common` / `left` / `right` 로 실시간 전환
- `p` : `pause` (학습에서 제외)
- `r` : 녹화 토글
- `esc` : 종료

**첫날 목표:**
- 미션 **하나만** (예: `left`) end-to-end 시연 3~5바퀴
- 200~500 프레임이면 충분. 다 모으겠다는 욕심 X
- 세그먼트 전환을 칼같이: 회전 시작 직전에 `2`(left) 눌러야지, common 구간에 회전이 섞이면 학습 망함

### Part B — 라벨링
- Part A 저장이 끝나면 같은 노트북 Part B 셀 실행
- 이미지마다 마우스로 도로 중심점 클릭
- 200장이면 30~45분 작업
- `xpos == -1`인 줄은 preprocess가 자동으로 건너뜀 — 헷갈리거나 잘못 찍은 건 그냥 패스

---

## 4. 첫 lane 모델 학습 (15분)

합성 데이터로 이미 검증된 파이프라인. 실데이터에 그대로 돌리면 됨.

```bash
# 세션 → segment별로 분리
cd ~/team/main/ros2_ws/src/rover_training/scripts
python3 preprocess.py --raw ~/rover_data --out ~/data/processed

# CenterDataset이 CWD에서 이미지를 찾는 버그가 아직 있음 → cd 필수
cd ~/data/processed/common/images
python3 ~/team/main/ros2_ws/src/rover_training/scripts/train_center.py \
    --segment common --data ~/data/processed/common --epochs 20 \
    --out ~/models/center_common.pth

# ONNX → TensorRT engine
cd ~/team/main/ros2_ws/src/rover_training/scripts
python3 export_to_onnx.py --ckpt ~/models/center_common.pth \
    --out ~/models/center_common.onnx
bash export_to_trt.sh ~/models/center_common.onnx \
    ~/team/main/ros2_ws/src/rover_lane/models/center_common.engine fp16
```

**확인:** `ls -la ~/team/main/ros2_ws/src/rover_lane/models/` 에 `.engine` 파일 있어야 함.

좌/우 segment는 데이터가 모이면 같은 방식. 첫날은 `common` 하나만으로 충분.

---

## 5. Lane만 실차 테스트 (30분, YOLO 없이)

가장 중요한 단계. YOLO 없이 lane CNN 출력 → steering 만으로 차선 추적이 되는지 확인.

```bash
ros2 launch rover_bringup autonomous.launch.py mission:=left
```

- YOLO engine 없어서 warn 뜸 → 정상. decision_node는 표지판 없이도 `/road_center` 받아 `/cmd_vel` 발행.
- FSM 상태가 계속 `COMMON`에 머무름 (turn_sign 못 보니까) → 정상.
- 직선 구간에서 똑바로 따라가면 성공.

**잘 안 되면 디버깅 순서:**
1. `ros2 topic echo /road_center` → x 값이 도로 위치에 따라 변하나? (도로 왼쪽 보면 x<0, 오른쪽 보면 x>0)
2. `ros2 topic echo /cmd_vel` → angular.z가 x 따라 변하나?
3. 모터가 그 방향으로 도나? — 안 돌면 `params.yaml`의 `motor.invert_drive` 의심
4. 차가 비뚤어진다 → steering_gain_k 튜닝 (`params.yaml`, 기본 1.2)

---

## 내일 **하지 말 것**

- **YOLO 학습 X**: 클래스당 200장씩 라벨링부터 해야 함, 멀티데이 작업. lane이 먼저.
- **Stereo 재캘리 X**: 2026-05-19에 동결됨. 카메라 흔들리면 *물리적으로 복원*, 재캘리 X. `stereo_calib.yaml` 건드리면 그 전 데이터 다 무효.
- **FSM 임계값 튜닝 X**: `stable_frames_sign`, `safe_dist_m`, `steering_gain_k` 같은 건 차가 실제로 굴러갈 때 튜닝하는 거. 미리 손대지 말것.

---

## 알려진 함정

- **모터가 통신은 되는데 안 움직임** → 99% 배터리 저전압. 충전 / 교체 먼저.
- **CSI 카메라 죽음** → `sudo systemctl restart nvargus-daemon`
- **`/dev/ttyUSB0` permission denied** → `sudo usermod -aG dialout $USER` 후 **재로그인**
- **`CenterDataset` FileNotFoundError** → `cd <data>/images` 안 했기 때문. 알려진 이슈, 우선순위 낮음.
- **`ros2 topic hz` 결과 안 뜸** → 새 터미널에서 `source install/setup.bash` 안 했음.

---

## 내일 끝나면 어디까지 와있어야 하나

**현실적 목표:**
- 직선 구간에서 `common` lane 모델로 차선 따라가는 것 한 번이라도 성공
- 차량 K 캘리브레이션 값이 `params.yaml`에 들어가 있음
- 200장 이상 라벨된 lane 데이터셋 확보

**욕심 부리지 않는 게 핵심.** 첫날에 회전교차로까지 가려고 하면 다 망한다. baseline 하나 잡고, 다음날 left/right segment, 그 다음날 YOLO 라벨링 시작이 정상 페이스.

---

## 참고

- 전체 설계: [PROJECT_PLAN.md](PROJECT_PLAN.md)
- 빌드 & 실행: [README.md](README.md)
- 학습 파이프라인 세부: `ros2_ws/src/rover_training/`
- 노드별 wire-up 상태: PROJECT_PLAN §7 패키지 구조
