"""Launch file for data collection: camera + teleop + recorder."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("session_name", default_value="session"),
        Node(
            package="rover_stereo",
            executable="stereo_node",
            name="stereo",
            output="screen",
        ),
        Node(
            package="rover_recorder",
            executable="recorder_node",
            name="recorder",
            output="screen",
            parameters=[{"session_name": LaunchConfiguration("session_name")}],
        ),
        Node(
            package="rover_control",
            executable="control_node",
            name="control",
            output="screen",
        ),
    ])
