# ros2_ws — data collection stack

데이터 수집 전용 ROS2 워크스페이스. `main/ros2_ws`(자율주행 본체)와 별도로,
final_project 전용 노드만 모았다.

## 패키지

| 패키지 | 노드 | 역할 |
|---|---|---|
| `rover_camera`   | `camera_node`        | jetcam 두 대 → `/bev_image/compressed`, `/front_image/compressed` |
| `rover_teleop`   | `teleop_node`        | 키보드(cbreak) 1D steering level → `/cmd_vel`, `/steer_level`, `/record_enable` |
| `rover_recorder` | `motor_bridge_node`  | `/cmd_vel` → UART (데이터 수집 전용) |
| `rover_recorder` | `bag_recorder_node`  | `/record_enable` 토글 시 `ros2 bag record` 자동 시작/종료. BEV 프레임 없으면 종료 |
| `rover_calib`    | `bev_capture_node`   | `/bev_image/compressed` 한 프레임 캡처 후 종료 (BEV 캘리브용) |

## 토픽 contract

`final_project/data_pipeline/extract_labels.py`가 기대하는 그대로:

```
/bev_image/compressed    sensor_msgs/CompressedImage
/front_image/compressed  sensor_msgs/CompressedImage
/cmd_vel                 geometry_msgs/Twist          # linear.x throttle, angular.z steering
/steer_level             std_msgs/Int8                # -5..+5 raw teleop
/record_enable           std_msgs/Bool                # bag on/off toggle
```

## 빌드

```bash
cd /home/hyunseo/Personal_Research/AUE4040/final_project/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 사용

### A. BEV 캘리브 이미지 캡처 (한 번만)

```bash
ros2 launch rover_calib bev_capture.launch.py
# 카메라 떠 있는 상태에서 체커보드 BEV 카메라에 놓고
# 터미널에서 'c' 키 → final_project/calib/bev_capture_<ts>.jpg 저장 후 자동 종료
```

이어서:
```bash
cd ../data_pipeline
python bev_calibration.py --image ../calib/bev_capture_<ts>.jpg \
    --rows 6 --cols 9 --square_m 0.025
```

### B. 주행 + 녹화 세션

**노트북 하나로 launch + 미리보기**:
```bash
jupyter notebook ../data_pipeline/launch_and_record.ipynb
```
셀 순서대로 실행하면 카메라/모터/bag 레코더가 켜지고 두 카메라 화면이 보임.

**별도 SSH 터미널 1개** — 키보드 텔레옵:
```bash
cd /home/hyunseo/Personal_Research/AUE4040/final_project/ros2_ws
source install/setup.bash
ros2 run rover_teleop teleop_node
```

순수 CLI:
```bash
# 터미널 1
ros2 launch rover_recorder record.launch.py session_name:=loop_test
# 터미널 2
ros2 run rover_teleop teleop_node
```

**키 매핑** (teleop_node 터미널에서):
- `a` / `d` : turn_level −1 / +1 (−5..+5)
- `space` : 정지 (level=0, drive off)
- `g` : drive on/off 토글 (UART 송신)
- `r` : 녹화 on/off 토글 (bag 시작/종료)
- `q` 또는 `ESC` : 종료

녹화 결과: `~/rover_data/<session>_<ts>/bag/` → 그대로 `extract_labels.py --bag`에 입력.
