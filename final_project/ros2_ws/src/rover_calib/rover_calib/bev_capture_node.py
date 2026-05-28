"""One-shot BEV calibration image grabber.

Subscribes /bev_image/compressed (publish it via `rover_camera camera_node`),
holds the latest frame, and writes it to disk on key press in the controlling
TTY:

  c        : capture (save to <out_dir>/bev_capture_<ts>.jpg) and EXIT
  q / ESC  : quit without saving

BEV calibration only needs ONE good checkerboard image, so this exits after
a successful capture. Feed the resulting jpg straight into:

  python data_pipeline/bev_calibration.py --image calib/bev_capture_<ts>.jpg \
      --rows 6 --cols 9 --square_m 0.025

If no /bev_image/compressed frames arrive within `require_frames_within_s`
the node logs an error and exits non-zero — same loud-fail policy as the
bag recorder.
"""
from __future__ import annotations

import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class BevCaptureNode(Node):
    def __init__(self) -> None:
        super().__init__("bev_capture")
        self.declare_parameter("out_dir",
                               "/home/hyunseo/Personal_Research/AUE4040/final_project/calib")
        self.declare_parameter("require_frames_within_s", 5.0)

        self.out_dir = Path(self.get_parameter("out_dir").value)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.deadline_s = float(self.get_parameter("require_frames_within_s").value)

        self._latest_jpg: bytes | None = None
        self._started_at = time.time()

        self.create_subscription(CompressedImage, "/bev_image/compressed",
                                 self._on_bev, 10)

        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY; run in a real terminal.")
            raise SystemExit(2)
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"waiting on /bev_image/compressed.  press 'c' to capture, 'q'/ESC to quit. "
            f"out_dir={self.out_dir}"
        )

    def _on_bev(self, msg: CompressedImage) -> None:
        self._latest_jpg = bytes(msg.data)

    def _tick(self) -> None:
        # 1. fail loudly if no frames arrive
        if self._latest_jpg is None and (time.time() - self._started_at) > self.deadline_s:
            self.get_logger().error(
                f"no /bev_image/compressed in {self.deadline_s:.1f}s. "
                "Is rover_camera camera_node running?"
            )
            rclpy.shutdown()
            return

        # 2. drain keys
        while select.select([sys.stdin], [], [], 0)[0]:
            c = sys.stdin.read(1)
            if not c:
                break
            if c in ("q", "\x1b"):
                self.get_logger().info("quit without saving")
                rclpy.shutdown()
                return
            if c == "c":
                self._save_and_exit()
                return

    def _save_and_exit(self) -> None:
        if self._latest_jpg is None:
            self.get_logger().warn("no frame yet, ignoring capture")
            return
        # Decode and re-encode to ensure on-disk is a valid jpg even if the
        # sender used a non-jpeg format param.
        arr = np.frombuffer(self._latest_jpg, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().error("imdecode failed on /bev_image/compressed payload")
            rclpy.shutdown()
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"bev_capture_{ts}.jpg"
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.get_logger().info(f"saved {path}")
        self.get_logger().info(
            f"next: python data_pipeline/bev_calibration.py --image {path} "
            "--rows 6 --cols 9 --square_m 0.025"
        )
        rclpy.shutdown()

    def destroy_node(self) -> bool:
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = BevCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
