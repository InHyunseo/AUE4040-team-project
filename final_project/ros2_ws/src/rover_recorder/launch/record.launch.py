"""Data-collection bringup.

Camera + motor bridge + bag recorder.  Teleop is intentionally NOT launched
here — run it in a separate SSH terminal so cbreak owns its own TTY:

  ros2 run rover_teleop teleop_node

Usage:
  ros2 launch rover_recorder record.launch.py session_name:=loop_test
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    session = LaunchConfiguration("session_name")
    out_root = LaunchConfiguration("out_root")
    fps = LaunchConfiguration("fps")
    dry_run = LaunchConfiguration("dry_run")
    monitor = LaunchConfiguration("monitor")
    monitor_host = LaunchConfiguration("monitor_host")
    monitor_port = LaunchConfiguration("monitor_port")
    overlay_viz = LaunchConfiguration("overlay_viz")
    viz_fps = LaunchConfiguration("viz_fps")
    viz_device = LaunchConfiguration("viz_device")
    raw_monitor = PythonExpression([
        "'", monitor, "'.lower() in ('true','1','yes') and '",
        overlay_viz, "'.lower() not in ('true','1','yes')",
    ])
    overlay_monitor = PythonExpression([
        "'", monitor, "'.lower() in ('true','1','yes') and '",
        overlay_viz, "'.lower() in ('true','1','yes')",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("session_name", default_value="session"),
        DeclareLaunchArgument("out_root",
                              default_value="/home/ircv16/team/final_project/rover_data"),
        DeclareLaunchArgument("fps", default_value="15"),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="motor_bridge dry run (no UART)"),
        DeclareLaunchArgument("monitor", default_value="true",
                              description="launch the browser MJPEG monitor"),
        DeclareLaunchArgument("monitor_host", default_value="0.0.0.0",
                              description="monitor bind host (127.0.0.1 = local only)"),
        DeclareLaunchArgument("monitor_port", default_value="8080"),
        DeclareLaunchArgument("overlay_viz", default_value="false",
                              description="run SegFormer/YOLO overlay preview topics"),
        DeclareLaunchArgument("viz_fps", default_value="3.0",
                              description="overlay preview inference rate"),
        DeclareLaunchArgument("viz_device", default_value="cuda",
                              description="device for SegFormer/YOLO preview"),

        Node(package="rover_camera", executable="camera_node",
             name="rover_camera", output="screen",
             parameters=[{"fps": fps}]),

        Node(package="rover_camera", executable="monitor_node",
             name="rover_monitor", output="screen",
             condition=IfCondition(raw_monitor),
             parameters=[{
                 "host": monitor_host,
                 "port": monitor_port,
                 "streams": [
                     "lane:/lane_image/compressed",
                     "front:/front_image/compressed",
                 ],
             }]),

        Node(package="rover_camera", executable="monitor_node",
             name="rover_monitor", output="screen",
             condition=IfCondition(overlay_monitor),
             parameters=[{
                 "host": monitor_host,
                 "port": monitor_port,
                 "streams": [
                     "lane:/lane_image/compressed",
                     "front:/front_image/compressed",
                     "lane_seg:/lane_seg/compressed",
                     "front_det:/front_det/compressed",
                 ],
             }]),

        Node(package="rover_camera", executable="overlay_viz_node",
             name="rover_overlay_viz", output="screen",
             condition=IfCondition(overlay_viz),
             parameters=[{"viz_fps": viz_fps, "device": viz_device}]),

        Node(package="rover_recorder", executable="motor_bridge_node",
             name="motor_bridge", output="screen",
             parameters=[{"dry_run": dry_run}]),

        Node(package="rover_recorder", executable="bag_recorder_node",
             name="bag_recorder", output="screen",
             parameters=[{"session_name": session, "out_root": out_root}]),
    ])
