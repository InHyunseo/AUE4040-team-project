# ros2_ws — data collection stack

데이터 수집 전용 ROS2 워크스페이스.

## 패키지

| 패키지 | 노드 | 역할 |
|---|---|---|
| `rover_camera`   | `camera_node`        | jetcam 두 대 → `/lane_image/compressed`(sensor0, 차선세그), `/front_image/compressed`(sensor1, 객체인식) |
| `rover_camera`   | `overlay_viz_node`   | frozen SegFormer/YOLO → `/lane_seg/compressed`, `/front_det/compressed` 실시간 오버레이 |
| `rover_camera`   | `monitor_node`       | 카메라 JPEG를 재인코딩 없이 브라우저 MJPEG로 스트림 (`http://<host>:8080/`) |
| `rover_teleop`   | `teleop_node`        | 키보드(cbreak) 1D steering level → `/cmd_vel`, `/steer_level`, `/record_enable` |
| `rover_recorder` | `motor_bridge_node`  | `/cmd_vel` → UART (데이터 수집 전용) |
| `rover_recorder` | `bag_recorder_node`  | `/record_enable` 토글 시 `ros2 bag record` 자동 시작/종료. lane 프레임 없으면 종료 |

> 두 카메라 모두 raw로 녹화하고, 차선 세그 헤드는 raw lane 이미지 위에서 동작한다.

## 토픽 contract

`final_project/data_pipeline/extract_labels.py`가 기대하는 그대로:

```
/lane_image/compressed   sensor_msgs/CompressedImage  # sensor 0, 차선 세그 헤드
/front_image/compressed  sensor_msgs/CompressedImage  # sensor 1, 객체인식 헤드
/lane_seg/compressed     sensor_msgs/CompressedImage  # optional: lane + SegFormer overlay
/front_det/compressed    sensor_msgs/CompressedImage  # optional: front + YOLO bbox overlay
/cmd_vel                 geometry_msgs/Twist          # linear.x throttle, angular.z steering
/steer_level             std_msgs/Int8                # -2..+2 raw teleop
/record_enable           std_msgs/Bool                # bag on/off toggle
```

## 빌드

```bash
cd /home/ircv16/team/final_project/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 사용

### 주행 + 녹화 세션

**노트북 하나로 launch + 미리보기**:
```bash
jupyter notebook ../data_pipeline/launch_and_record.ipynb
```
셀 순서대로 실행하면 카메라/모터/bag 레코더 + 모니터가 켜짐.

**미리보기 = 브라우저 모니터** (노트북 위젯 아님). launch가 `rover_monitor`를 같이 띄움.
카메라 JPEG를 **재인코딩 없이** MJPEG로 흘려서 노트북 위젯보다 표시 지연이 낮고, 제어 경로
(teleop→`/cmd_vel`→motor_bridge→UART)와 완전히 별개 노드라 주행 명령 타이밍에 영향이 없음.
- 젯슨 로컬: `http://localhost:8080/`
- 같은 네트워크 노트북: `http://<젯슨-IP>:8080/` (기본 `monitor_host:=0.0.0.0`)

끄려면 `monitor:=false`. 로컬만 열려면 `monitor_host:=127.0.0.1`.

**모델 결과까지 실시간 확인**하려면 Phase 1 산출물
`final_project/models/segformer_lane/`, `final_project/models/best.pt`를 둔 뒤:

```bash
ros2 launch rover_recorder record.launch.py session_name:=phase2_preview overlay_viz:=true
```

브라우저에는 `lane`, `front`, `lane_seg`, `front_det` 네 화면이 뜬다. 오버레이 노드는
학습/추출과 같은 계약으로 lane 상단 ROI crop → 224 resize → SegFormer 색상 합성,
front 224 resize → YOLO bbox 합성을 수행한다. 추론이 무거우면 `viz_fps:=1.0`처럼 낮춘다.

Jetson에서 `ModuleNotFoundError: transformers` 또는 NumPy 2.x 경고가 뜨면:

```bash
python3 -m pip install --user transformers ultralytics
python3 -m pip uninstall -y opencv-python opencv-contrib-python
python3 -m pip install --user "numpy<2.0"
python3 -c "import numpy, torch, transformers, ultralytics; print(numpy.__version__, torch.__version__, transformers.__version__, ultralytics.__version__)"
```

`opencv-python not installed` 경고는 무시 가능하다. Jetson 카메라는 apt/JetPack OpenCV를 써야 한다.

**별도 SSH 터미널 1개** — 키보드 텔레옵:
```bash
cd /home/ircv16/team/final_project/ros2_ws
source install/setup.bash
ros2 run rover_teleop teleop_node
```

순수 CLI:
```bash
# 터미널 1
ros2 launch rover_recorder record.launch.py session_name:=loop_test
# (브라우저에서 http://<host>:8080/ 열면 두 카메라 라이브)
# 터미널 2
ros2 run rover_teleop teleop_node
```

**키 매핑** (teleop_node 터미널에서):
- `a` / `d` : turn_level −1 / +1 (−2..+2, 두 번이면 최대 회전)
- `space` : 정지 (level=0, drive off)
- `g` : drive on/off 토글 (UART 송신)
- `r` : 녹화 on/off 토글 (bag 시작/종료)
- `q` 또는 `ESC` : 종료

녹화 결과: `final_project/rover_data/<session>_<ts>/bag/` → 그대로 `extract_labels.py --bag`에 입력.
