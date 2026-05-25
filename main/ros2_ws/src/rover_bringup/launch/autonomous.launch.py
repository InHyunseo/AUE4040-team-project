"""Full autonomous stack: camera + perception + lane + decision + control.

Launch order: stereo first; the rest start only after stereo logs
"cameras up" (both CSI sensors opened and rectify maps ready). The stereo
node takes ~10-15 s to bring both cameras up, so launching everything in
parallel means the BC/yolo nodes spin uselessly until frames arrive.
Mission (left/right) is latched at runtime by decision_node from the first
stable turn-sign detection — no launch-time argument needed.
"""
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch_ros.actions import Node


def generate_launch_description():
    stereo = Node(
        package="rover_stereo", executable="stereo_node",
        name="stereo", output="screen",
    )

    downstream = [
        Node(package="rover_perception", executable="yolo_node",
             name="yolo", output="screen"),
        Node(package="rover_lane", executable="lane_node",
             name="lane", output="screen"),
        Node(package="rover_decision", executable="decision_node",
             name="decision", output="screen"),
        Node(package="rover_control", executable="control_node",
             name="control", output="screen"),
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
