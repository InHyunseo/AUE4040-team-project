"""Subscribe /cmd_vel and drive the rover motor over UART.

No FSM gating (vs. main/.../rover_control). Data-collection only —
teleop_node owns the safety semantics (space = hard stop, drive toggle).

cmd_vel convention (matches teleop_node + extract_labels.py):
  linear.x  : throttle, negative = forward on this rover wiring
              (record_and_label.ipynb sends state['speed'] straight to
               base_speed_ctrl with no inversion; we match that)
  angular.z : steering rad/s-ish, ±MAX_OMEGA

Mixing is the same diff-drive formula main/rover_control uses, so the
trained model's outputs are compatible with the autonomous stack.
"""
from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

sys.path.insert(0, "/home/ircv16/team")
try:
    from control.base_ctrl import BaseController
except Exception as e:  # pragma: no cover — import-time error reported at runtime
    BaseController = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


MAX_OMEGA = 0.8


def mix(linear_x: float, angular_z: float, max_speed: float,
        invert_drive: bool = False) -> tuple[float, float]:
    """Same shape as main/rover_control.control_node.steer_speed_to_lr but
    parameterised on angular_z directly (teleop publishes rad/s-ish).

    teleop convention: forward = negative linear.x.  We turn that into forward
    drive (L,R same sign as forward) here. `invert_drive` flips the final L/R
    sign for rovers whose firmware treats positive L/R as reverse — set it so
    that pressing drive moves the rover *forward*. Steering (a/d) stays correct
    either way because the flip is applied to both wheels equally."""
    turn = max(-1.0, min(1.0, angular_z / max(MAX_OMEGA, 1e-6)))
    base = abs(linear_x)
    fwd = -1.0 if linear_x < 0 else (1.0 if linear_x > 0 else 0.0)  # teleop: <0 = forward
    L = fwd * base * max(0.0, 1.0 - turn)
    R = fwd * base * max(0.0, 1.0 + turn)
    if invert_drive:
        L, R = -L, -R
    L = max(-max_speed, min(max_speed, L))
    R = max(-max_speed, min(max_speed, R))
    return L, R


class MotorBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("motor_bridge")
        self.declare_parameter("uart_dev", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("max_speed", 0.5)
        self.declare_parameter("dry_run", False)
        # Flip final L/R sign if your rover drives backward when it should go
        # forward. This rover's firmware matches the default (invert=False).
        self.declare_parameter("invert_drive", False)

        self.dry = bool(self.get_parameter("dry_run").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.invert_drive = bool(self.get_parameter("invert_drive").value)

        if self.dry:
            self.base = None
            self.get_logger().warn("dry_run=True — no UART writes")
        elif BaseController is None:
            self.get_logger().error(f"BaseController import failed: {_IMPORT_ERR}")
            raise SystemExit(2)
        else:
            uart = self.get_parameter("uart_dev").value
            baud = self.get_parameter("baudrate").value
            self.base = BaseController(uart, baud)
            self.get_logger().info(f"motor ready on {uart}@{baud}")

        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self._dbg = 0

    def _on_cmd(self, msg: Twist) -> None:
        L, R = mix(float(msg.linear.x), float(msg.angular.z), self.max_speed,
                   self.invert_drive)
        if self.base is not None:
            try:
                self.base.base_speed_ctrl(L, R)
            except Exception as e:
                self.get_logger().error(f"base_speed_ctrl: {e}")
        self._dbg += 1
        if self._dbg % 20 == 0:
            self.get_logger().info(
                f"cmd v={msg.linear.x:+.3f} w={msg.angular.z:+.3f} -> L={L:+.3f} R={R:+.3f}"
            )

    def stop(self) -> None:
        if self.base is None:
            return
        try:
            self.base.base_speed_ctrl(0.0, 0.0)
        except Exception:
            pass


def main() -> None:
    rclpy.init()
    node = MotorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
