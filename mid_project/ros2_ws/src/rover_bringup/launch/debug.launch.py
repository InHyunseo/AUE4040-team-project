"""Debug launch: camera + perception only (visualization-friendly)."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="rover_stereo", executable="stereo_node",
             name="stereo", output="screen"),
        Node(package="rover_perception", executable="yolo_node",
             name="yolo", output="screen"),
    ])
