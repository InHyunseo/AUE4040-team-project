# Rover Autonomous Driving — Roundabout (closed-world)

회전교차로 자율주행 로버 프로젝트. NVIDIA Jetson Orin Nano + ROS 2 Humble.

자세한 설계는 [PROJECT_PLAN.md](PROJECT_PLAN.md) 참조.

## Layout

```
main/
├── PROJECT_PLAN.md             # 프로젝트 계획서 (v2)
├── ros2_ws/                    # ROS 2 워크스페이스
│   └── src/
│       ├── rover_bringup/      # launch + config
│       ├── rover_msgs/         # 커스텀 메시지
│       ├── rover_stereo/       # 듀얼 CSI rectify + disparity (단일 이미지 진입점)
│       ├── rover_perception/   # YOLO 검출 노드
│       ├── rover_pilotnet/     # 도로 중심점 회귀 노드
│       ├── rover_decision/     # FSM + 안전 + 미션
│       ├── rover_control/      # 모터 (UART JSON) 노드
│       ├── rover_recorder/     # 데이터 수집
│       └── rover_training/     # 학습 스크립트 (ROS 외부)
```

## Build

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Quickstart

```bash
# 데이터 수집
ros2 launch rover_bringup recording.launch.py

# 자율주행
ros2 launch rover_bringup autonomous.launch.py mission:=left
```

## Reused from HYU-ECL3003

- `rover_control/motor_driver.py` ← `rover/base_ctrl.py`
- `rover_recorder/jetcam/` ← `rover/jetcam/`
- `rover_stereo/calib/capture_stereo.py` ← `stereo_depth_tutorial/.../capture-stereo.py`
- `rover_training/scripts/train_center.py` ← `rover/train_road_center_model.ipynb`
- `rover_perception/yolo_inference.py` ← `week07/YOLOv8/demo_livecam_local.py`

세부 매핑은 PROJECT_PLAN.md §6.1 참조.

## Calibration is frozen

`rover_stereo/config/stereo_calib.yaml`은 Phase 0.5에서 **한 번 생성한 뒤 학기 동안 재생성 금지**입니다. 이 파일이 바뀌면 모든 BC 학습 데이터를 다시 수집해야 합니다. 자세한 이유는 PROJECT_PLAN.md §9.6 참조.
