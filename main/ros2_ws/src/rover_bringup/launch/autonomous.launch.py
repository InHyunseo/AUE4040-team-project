"""Full autonomous stack: camera + perception + pilotnet + decision + control."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mission", default_value="left"),
        Node(package="rover_stereo", executable="stereo_node",
             name="stereo", output="screen"),
        Node(package="rover_perception", executable="yolo_node",
             name="yolo", output="screen"),
        Node(package="rover_pilotnet", executable="pilotnet_node",
             name="pilotnet", output="screen"),
        Node(package="rover_decision", executable="decision_node",
             name="decision", output="screen",
             parameters=[{"mission": LaunchConfiguration("mission")}]),
        Node(package="rover_control", executable="control_node",
             name="control", output="screen"),
    ])
