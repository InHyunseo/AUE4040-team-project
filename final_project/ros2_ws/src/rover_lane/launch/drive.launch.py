"""Autonomous-driving bringup (Phase 3).

Camera + motor bridge + E2E inference. The inference counterpart of
rover_recorder/record.launch.py: same camera + motor_bridge, but the bag
recorder is dropped and the E2E inference node replaces manual teleop.

Usage:
  # camera + motor + autonomous inference
  ros2 launch rover_lane drive.launch.py

  # final run: drop the browser monitor to free resources
  ros2 launch rover_lane drive.launch.py monitor:=false

  # dry run the motor bridge (no UART), watch http://<host>:8080/
  ros2 launch rover_lane drive.launch.py dry_run:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    fps = LaunchConfiguration("fps")
    dry_run = LaunchConfiguration("dry_run")
    monitor = LaunchConfiguration("monitor")
    monitor_host = LaunchConfiguration("monitor_host")
    monitor_port = LaunchConfiguration("monitor_port")
    engine_path = LaunchConfiguration("engine_path")
    segformer_ckpt = LaunchConfiguration("segformer_ckpt")
    yolo_weights = LaunchConfiguration("yolo_weights")
    device = LaunchConfiguration("device")
    max_rate_hz = LaunchConfiguration("max_rate_hz")
    watchdog_hz = LaunchConfiguration("watchdog_hz")
    cmd_timeout_s = LaunchConfiguration("cmd_timeout_s")
    smooth_alpha = LaunchConfiguration("smooth_alpha")
    steer_source = LaunchConfiguration("steer_source")
    steer_mode = LaunchConfiguration("steer_mode")
    lookahead_idx = LaunchConfiguration("lookahead_idx")
    idx_lo = LaunchConfiguration("idx_lo")
    idx_hi = LaunchConfiguration("idx_hi")
    pursuit_gain = LaunchConfiguration("pursuit_gain")

    return LaunchDescription([
        DeclareLaunchArgument("fps", default_value="15"),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="motor_bridge dry run (no UART)"),
        DeclareLaunchArgument("monitor", default_value="true",
                              description="launch the browser MJPEG monitor"),
        DeclareLaunchArgument("monitor_host", default_value="0.0.0.0",
                              description="monitor bind host (127.0.0.1 = local only)"),
        DeclareLaunchArgument("monitor_port", default_value="8080"),
        DeclareLaunchArgument("engine_path", default_value="",
                              description="TensorRT e2e.engine path (empty = "
                                          "<project>/models/e2e.engine)"),
        DeclareLaunchArgument("segformer_ckpt", default_value="",
                              description="SegFormer ckpt dir (empty = default)"),
        DeclareLaunchArgument("yolo_weights", default_value="",
                              description="YOLO best.pt path (empty = default)"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("max_rate_hz", default_value="30.0",
                              description="inference rate cap (keep above camera "
                                          "rate; effectively infer-every-frame)"),
        DeclareLaunchArgument("watchdog_hz", default_value="20.0",
                              description="steady republish + deadman rate"),
        DeclareLaunchArgument("cmd_timeout_s", default_value="0.4",
                              description="stop if no inference within this window"),
        DeclareLaunchArgument("smooth_alpha", default_value="0.35",
                              description="steer low-pass per watchdog tick "
                                          "(matches teleop SMOOTH_ALPHA; 0=off, "
                                          "lower=smoother/laggier, higher=snappier)"),
        # 조향 소스/모드 (회피 비교용). head=ControlHead steer 직접,
        # waypoint=waypoint 추종(steer_mode: pursuit|heading|max_y|mean).
        DeclareLaunchArgument("steer_source", default_value="waypoint",
                              description="head | waypoint"),
        DeclareLaunchArgument("steer_mode", default_value="pursuit",
                              description="pursuit | heading | max_y | mean (waypoint 일 때)"),
        DeclareLaunchArgument("lookahead_idx", default_value="3",
                              description="pursuit/heading 단일 점 인덱스(0~4)"),
        DeclareLaunchArgument("idx_lo", default_value="2",
                              description="max_y/mean 구간 시작 인덱스"),
        DeclareLaunchArgument("idx_hi", default_value="4",
                              description="max_y/mean 구간 끝 인덱스"),
        DeclareLaunchArgument("pursuit_gain", default_value="0.25",
                              description="곡률/각도→정규화 조향 게인 (heading류는 0.6~0.8 부터)"),

        # Camera (same as record.launch).
        Node(package="rover_camera", executable="camera_node",
             name="rover_camera", output="screen",
             parameters=[{"fps": fps}]),

        # Browser monitor: raw lane/front + E2E 디버그 오버레이(seg+예측 의도, bbox).
        # e2e_infer_node 가 publish_overlay=True 일 때 /lane_intent · /front_det 를 낸다.
        # raw 와 오버레이를 나란히 봐 seg/bbox 품질과 모델 의도를 한 화면에서 확인.
        Node(package="rover_camera", executable="monitor_node",
             name="rover_monitor", output="screen",
             condition=IfCondition(monitor),
             parameters=[{
                 "host": monitor_host,
                 "port": monitor_port,
                 "streams": [
                     "lane:/lane_image/compressed",
                     "front:/front_image/compressed",
                     "lane_intent:/lane_intent/compressed",
                     "front_det:/front_det/compressed",
                 ],
             }]),

        # Motor bridge (same as record.launch) — drives /cmd_vel over UART.
        Node(package="rover_recorder", executable="motor_bridge_node",
             name="motor_bridge", output="screen",
             parameters=[{"dry_run": dry_run}]),

        # E2E inference -> /cmd_vel.
        Node(package="rover_lane", executable="e2e_infer_node",
             name="rover_e2e_infer", output="screen",
             parameters=[{
                 "engine_path": engine_path,
                 "segformer_ckpt": segformer_ckpt,
                 "yolo_weights": yolo_weights,
                 "device": device,
                 "max_rate_hz": max_rate_hz,
                 "watchdog_hz": watchdog_hz,
                 "cmd_timeout_s": cmd_timeout_s,
                 "smooth_alpha": smooth_alpha,
                 "steer_source": steer_source,
                 "steer_mode": steer_mode,
                 "lookahead_idx": lookahead_idx,
                 "idx_lo": idx_lo,
                 "idx_hi": idx_hi,
                 "pursuit_gain": pursuit_gain,
             }]),
    ])
