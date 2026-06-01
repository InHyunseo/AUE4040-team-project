"""Start/stop `ros2 bag record` on /record_enable edges.

Topic schema is locked by final_project/data_pipeline/extract_labels.py:
  /lane_image/compressed    sensor_msgs/CompressedImage
  /front_image/compressed   sensor_msgs/CompressedImage
  /cmd_vel                  geometry_msgs/Twist
Plus side-channel we want available later:
  /steer_level              std_msgs/Int8     (raw teleop input)

The bag goes to <out_root>/<session>_<ts>/bag/  so it lines up with the path
extract_labels.py expects (`--bag <session_dir>/bag`).

The recorder subscribes to /lane_image/compressed for a side-effect: if no
camera frames arrive within `require_frames_within` seconds after start, it
logs a loud error AND stops the bag — exactly the "no images = fail loud"
behavior you wanted. No silent jpg fallback.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


DEFAULT_TOPICS = [
    "/lane_image/compressed",
    "/front_image/compressed",
    "/cmd_vel",
    "/steer_level",
]


class BagRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("bag_recorder")
        self.declare_parameter("session_name", "session")
        self.declare_parameter("out_root", str(Path.home() / "rover_data"))
        self.declare_parameter("topics", DEFAULT_TOPICS)
        self.declare_parameter("require_frames_within_s", 3.0)

        self.session = self.get_parameter("session_name").value
        self.out_root = Path(self.get_parameter("out_root").value)
        self.topics = list(self.get_parameter("topics").value)
        self.frame_deadline_s = float(self.get_parameter("require_frames_within_s").value)

        if shutil.which("ros2") is None:
            self.get_logger().error("`ros2` not on PATH. source the workspace first.")
            raise SystemExit(2)

        self._proc: subprocess.Popen | None = None
        self._session_dir: Path | None = None
        self._bag_started_at: float | None = None
        self._got_lane_after_start = False

        self.create_subscription(Bool, "/record_enable", self._on_toggle, 10)
        self.create_subscription(CompressedImage, "/lane_image/compressed",
                                 self._on_lane, 10)
        self.create_timer(1.0, self._watchdog)
        self.get_logger().info(
            f"waiting on /record_enable.  topics={self.topics}  out_root={self.out_root}"
        )

    def _on_lane(self, _msg: CompressedImage) -> None:
        if self._proc is not None:
            self._got_lane_after_start = True

    def _on_toggle(self, msg: Bool) -> None:
        if msg.data and self._proc is None:
            self._start()
        elif (not msg.data) and self._proc is not None:
            self._stop()

    def _start(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = self.out_root / f"{self.session}_{ts}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        bag_path = self._session_dir / "bag"

        cmd = ["ros2", "bag", "record", "-o", str(bag_path), *self.topics]
        self.get_logger().info(f"START bag: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        self._bag_started_at = time.time()
        self._got_lane_after_start = False

    def _stop(self) -> None:
        p = self._proc
        if p is None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            p.wait(timeout=5)
        except Exception:
            try: p.kill()
            except Exception: pass
        self.get_logger().info(f"STOP bag -> {self._session_dir}")
        self._proc = None
        self._bag_started_at = None
        self._session_dir = None
        self._got_lane_after_start = False

    def _watchdog(self) -> None:
        if self._proc is None or self._bag_started_at is None:
            return
        if self._proc.poll() is not None:
            self.get_logger().error(
                f"ros2 bag record died (returncode={self._proc.returncode}). stopping."
            )
            self._proc = None
            self._bag_started_at = None
            return
        elapsed = time.time() - self._bag_started_at
        if (not self._got_lane_after_start) and elapsed > self.frame_deadline_s:
            self.get_logger().error(
                f"NO /lane_image/compressed frames seen in {elapsed:.1f}s after bag start. "
                "Is rover_camera running?  Stopping bag to avoid junk data."
            )
            self._stop()

    def destroy_node(self) -> bool:
        self._stop()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = BagRecorderNode()
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
