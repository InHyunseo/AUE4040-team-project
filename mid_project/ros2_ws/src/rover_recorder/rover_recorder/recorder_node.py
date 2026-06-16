"""
Teleop + recorder.

Reuses HYU-ECL3003 pieces:
  - jetcam.csi_camera.CSICamera (drop the directory into this package)
  - update_vehicle_motion mixing from rover/ctrl_with_keyboard.py
  - publishes /camera/{left,right}/image_raw and /cmd_vel just like the
    autonomous stack — rover_stereo turns them into /image_rectified

On disk we save ONLY the rectified-left stream (the source of truth for
BC training). The raw pair is dropped after stereo_node consumes it; this
keeps training data and inference inputs from drifting apart.

Per-frame on disk:
  <session_dir>/images/<timestamp>.jpg   # rectified-left, ROI-cropped
  <session_dir>/annotation.txt           # filename steer speed segment
"""
import csv
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from rover_recorder.segment_labeler import SegmentLabeler


class RecorderNode(Node):
    def __init__(self):
        super().__init__("rover_recorder")
        self.declare_parameter("session_name", "session")
        self.declare_parameter("out_root", str(Path.home() / "rover_data"))

        out_root = Path(self.get_parameter("out_root").value)
        name = self.get_parameter("session_name").value
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = out_root / f"{name}_{ts}"
        (self.session_dir / "images").mkdir(parents=True, exist_ok=True)
        self.ann_path = self.session_dir / "annotation.txt"

        self.labeler = SegmentLabeler()
        self.steering = 0.0
        self.speed = 0.0

        # TODO:
        #   1. Spin up two jetcam.CSICamera instances; publish
        #      /camera/left/image_raw and /camera/right/image_raw so that
        #      rover_stereo produces /image_rectified.
        #   2. Subscribe to /image_rectified — save THAT to disk (not raw).
        #   3. pynput.keyboard.Listener for steering/speed and segment label,
        #      port from HYU-ECL3003/rover/ctrl_with_keyboard.py.
        #   4. Publish /cmd_vel so control_node drives during recording.
        #   5. Append (filename, steer, speed, label) to annotation.txt.
        self.get_logger().info(f"recording into {self.session_dir}")


def main():
    rclpy.init()
    rclpy.spin(RecorderNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
