"""1D steering-level teleop. termios cbreak reads keys from the controlling
TTY, so it works over SSH (no X). Terminal must be a foreground TTY — do NOT pipe.

  turn_level ∈ {-2..+2}, throttle coupled to |turn_level|, smoothed via approach():
    level=0:  linear.x=-0.15, angular.z=0
    level=±2: linear.x=-0.25, angular.z=±1.2   (좌/우 두 번이면 최대 회전)

Keys:
  a / d         : turn_level -1 / +1
  space         : level=0, drive off (hard stop)
  g             : toggle drive on/off
  r             : toggle /record_enable (recorder listens to this)
  q  or  ESC    : quit

Publishes:
  /cmd_vel        geometry_msgs/Twist        (smoothed)
  /steer_level    std_msgs/Int8              (raw current level)
  /record_enable  std_msgs/Bool              (toggle for recorder_node)
"""
from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int8


BASE_V       = 0.15
TURN_V       = 0.25
MAX_OMEGA    = 1.2
LEVELS       = 2   # 좌/우 두 번 누르면 최대 회전 (turn_level ∈ -2..+2)
SMOOTH_ALPHA = 0.35
TICK_HZ      = 20.0

# Per-|level| steering fraction (0..1). lv1 sits close to lv2 so it's a slightly
# softer version of the full turn rather than a much weaker one.
#   index 0 = level 0, 1 = level 1, 2 = level 2(max)
TURN_FRAC    = (0.0, 0.8, 1.0)


def approach(cur: float, target: float, alpha: float) -> float:
    return cur + (target - cur) * alpha


def drain_stdin() -> str:
    chars: list[str] = []
    while select.select([sys.stdin], [], [], 0)[0]:
        c = sys.stdin.read(1)
        if not c:
            break
        chars.append(c)
    return "".join(chars)


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("rover_teleop")
        self.pub_cmd     = self.create_publisher(Twist,  "/cmd_vel",       10)
        self.pub_level   = self.create_publisher(Int8,   "/steer_level",   10)
        self.pub_rec     = self.create_publisher(Bool,   "/record_enable", 10)

        self.turn_level  = 0
        self.driving     = False
        self.recording   = False

        self.cur_lin = 0.0
        self.cur_ang = 0.0

        if not sys.stdin.isatty():
            self.get_logger().error(
                "stdin is not a TTY. Run this directly in an SSH terminal, "
                "not piped or backgrounded."
            )
            raise SystemExit(2)
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

        # Publish initial state so subscribers latch a known value.
        self._publish_record()

        self.timer = self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(
            "teleop ready. keys: a/d=level, space=stop, g=drive, r=record, "
            "q/ESC=quit"
        )

    def _publish_record(self) -> None:
        m = Bool(); m.data = self.recording; self.pub_rec.publish(m)

    def _target(self) -> tuple[float, float]:
        if not self.driving:
            return 0.0, 0.0
        sign = 1.0 if self.turn_level >= 0 else -1.0
        a = TURN_FRAC[abs(self.turn_level)]   # steering fraction for this level
        lin = -(BASE_V + a * (TURN_V - BASE_V))
        ang = sign * a * MAX_OMEGA
        return lin, ang

    def _handle_keys(self) -> bool:
        """Return True if quit was requested."""
        for c in drain_stdin():
            if c in ("q", "\x1b"):
                return True
            elif c == "a":
                self.turn_level = max(-LEVELS, self.turn_level - 1)
            elif c == "d":
                self.turn_level = min( LEVELS, self.turn_level + 1)
            elif c == " ":
                self.turn_level = 0
                self.driving = False
            elif c == "g":
                self.driving = not self.driving
                self.get_logger().info(f"drive = {self.driving}")
            elif c == "r":
                self.recording = not self.recording
                self._publish_record()
                self.get_logger().info(f"recording = {self.recording}")
        return False

    def _tick(self) -> None:
        if self._handle_keys():
            rclpy.shutdown()
            return

        tgt_lin, tgt_ang = self._target()
        self.cur_lin = approach(self.cur_lin, tgt_lin, SMOOTH_ALPHA)
        self.cur_ang = approach(self.cur_ang, tgt_ang, SMOOTH_ALPHA)

        cmd = Twist()
        cmd.linear.x  = float(self.cur_lin)
        cmd.angular.z = float(self.cur_ang)
        self.pub_cmd.publish(cmd)

        lvl = Int8(); lvl.data = int(self.turn_level); self.pub_level.publish(lvl)

    def destroy_node(self) -> bool:
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        # Send a zero cmd_vel on the way out so control_node stops the motor.
        try:
            stop = Twist(); self.pub_cmd.publish(stop)
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = TeleopNode()
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
