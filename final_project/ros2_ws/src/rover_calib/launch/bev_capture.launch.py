"""BEV calibration capture bringup.

Starts camera_node + bev_capture_node. Press 'c' in the terminal that
launched this to save one frame to final_project/calib/, then it exits.

Usage:
  ros2 launch rover_calib bev_capture.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    out_dir = LaunchConfiguration("out_dir")
    fps = LaunchConfiguration("fps")

    return LaunchDescription([
        DeclareLaunchArgument(
            "out_dir",
            default_value="/home/hyunseo/Personal_Research/AUE4040/final_project/calib",
        ),
        DeclareLaunchArgument("fps", default_value="10"),

        Node(package="rover_camera", executable="camera_node",
             name="rover_camera", output="screen",
             parameters=[{"fps": fps}]),

        # bev_capture_node owns the TTY (cbreak) — output=screen so we see prompts.
        Node(package="rover_calib", executable="bev_capture_node",
             name="bev_capture", output="screen",
             parameters=[{"out_dir": out_dir}]),
    ])
