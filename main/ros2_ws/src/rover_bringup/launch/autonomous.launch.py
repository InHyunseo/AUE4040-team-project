"""Full autonomous stack: camera + perception + lane + decision + control.

Mission (left/right) is latched at runtime by decision_node from the first
stable turn-sign detection — no launch-time argument needed.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="rover_stereo", executable="stereo_node",
             name="stereo", output="screen"),
        Node(package="rover_perception", executable="yolo_node",
             name="yolo", output="screen"),
        Node(package="rover_lane", executable="lane_node",
             name="lane", output="screen"),
        Node(package="rover_decision", executable="decision_node",
             name="decision", output="screen"),
        Node(package="rover_control", executable="control_node",
             name="control", output="screen"),
    ])
