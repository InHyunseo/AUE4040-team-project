"""Full autonomous stack: camera + perception + lane + decision + control.

Launch order: stereo first; the rest start only after stereo logs
"cameras up" (both CSI sensors opened and rectify maps ready). The stereo
node takes ~10-15 s to bring both cameras up, so launching everything in
parallel means the BC/yolo nodes spin uselessly until frames arrive.
Mission (left/right) is latched at runtime by decision_node from the first
stable turn-sign detection — no launch-time argument needed.

All nodes load rover_bringup/config/params.yaml so the YAML is the single
source of truth for thresholds, durations, and the YOLO class_names list.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch_ros.actions import Node


def generate_launch_description():
    params = str(Path(get_package_share_directory("rover_bringup"))
                 / "config" / "params.yaml")

    stereo = Node(
        package="rover_stereo", executable="stereo_node",
        name="stereo", output="screen",
        parameters=[params],
    )

    downstream = [
        Node(package="rover_perception", executable="yolo_node",
             name="yolo", output="screen", parameters=[params]),
        Node(package="rover_lane", executable="lane_node",
             name="lane", output="screen", parameters=[params]),
        Node(package="rover_decision", executable="decision_node",
             name="decision", output="screen", parameters=[params]),
        Node(package="rover_control", executable="control_node",
             name="control", output="screen", parameters=[params]),
    ]

    started = {"done": False}

    def on_stereo_io(event):
        # rclpy logging goes to stderr by default — register both streams.
        if started["done"]:
            return None
        text = event.text.decode(errors="replace")
        if "cameras up" not in text:
            return None
        started["done"] = True
        return downstream

    return LaunchDescription([
        stereo,
        RegisterEventHandler(OnProcessIO(
            target_action=stereo,
            on_stdout=on_stereo_io,
            on_stderr=on_stereo_io,
        )),
    ])
