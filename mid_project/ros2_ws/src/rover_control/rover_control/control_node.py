"""
Motor control node.

Subscribes:  /cmd_vel (geometry_msgs/Twist) — linear.x = throttle in [-1,+1],
                                              angular.z = steering in [-1,+1]
             /fsm_state (rover_msgs/FSMState) — gates throttle to 0 in STOPPED/WAITING/ARRIVED
Publishes:   none (writes to motor over UART)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rover_msgs.msg import FSMState
from rover_control.motor_driver import BaseController


SAFE_STATES = {"STOPPED", "WAITING", "ARRIVED"}


def steer_speed_to_lr(steering: float, speed: float, max_steer: float, max_speed: float):
    """Mirror of HYU-ECL3003/rover/ctrl_with_keyboard.py update_vehicle_motion."""
    steer = max(min(steering, max_steer), -max_steer)
    base = abs(speed)
    left_ratio = max(0.0, 1.0 - steer)
    right_ratio = max(0.0, 1.0 + steer)
    L = max(min(base * left_ratio, max_speed), -max_speed)
    R = max(min(base * right_ratio, max_speed), -max_speed)
    if speed < 0:
        L, R = -L, -R
    return L, R


class ControlNode(Node):
    def __init__(self):
        super().__init__("rover_control")
        self.declare_parameter("uart_dev", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        # record_and_label.ipynb sends state["speed"] (negative = forward) to
        # base_speed_ctrl directly with no inversion, and that's the known-good
        # path. Matching that here: negative L,R = forward at the firmware.
        self.declare_parameter("invert_drive", False)
        self.declare_parameter("max_steer", 0.8)
        self.declare_parameter("max_speed", 0.5)
        # SLOW state throttle (telop sign convention: negative = forward; matches record_and_label).
        self.declare_parameter("slow_speed", -0.05)

        uart = self.get_parameter("uart_dev").value
        baud = self.get_parameter("baudrate").value
        try:
            self.base = BaseController(uart, baud)
        except NotImplementedError as e:
            self.get_logger().warn(f"motor_driver not yet ported: {e}")
            self.base = None

        self.fsm_state = "COMMON"
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(FSMState, "/fsm_state", self.on_fsm, 10)

    def on_fsm(self, msg: FSMState) -> None:
        self.fsm_state = msg.state

    def on_cmd_vel(self, msg: Twist) -> None:
        steering = float(msg.angular.z)
        if self.fsm_state in SAFE_STATES:
            throttle = 0.0
        elif self.fsm_state == "SLOW":
            throttle = float(self.get_parameter("slow_speed").value)
        else:
            throttle = float(msg.linear.x)
        L, R = steer_speed_to_lr(
            steering,
            throttle,
            self.get_parameter("max_steer").value,
            self.get_parameter("max_speed").value,
        )
        if self.get_parameter("invert_drive").value:
            L, R = -L, -R
        if self.base is not None:
            self.base.base_speed_ctrl(L, R)
            # Throttled debug so we can see what's actually going to the motor.
            self._dbg_i = getattr(self, "_dbg_i", 0) + 1
            if self._dbg_i % 10 == 0:
                self.get_logger().info(
                    f"state={self.fsm_state} cmd_v={msg.linear.x:.3f} "
                    f"cmd_w={msg.angular.z:.3f} -> L={L:.3f} R={R:.3f}"
                )
        else:
            self.get_logger().info(f"[dry] L={L:.3f} R={R:.3f}")

    def stop_motors(self) -> None:
        """Send a hard zero to the motor controller. Safe to call repeatedly."""
        if self.base is None:
            self.get_logger().info("[dry] stop_motors: L=0 R=0")
            return
        try:
            self.base.base_speed_ctrl(0.0, 0.0)
        except Exception as e:
            self.get_logger().error(f"stop_motors failed: {e}")


def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C / shutdown: explicitly zero the motors so the last cmd_vel
        # doesn't keep the rover rolling after the node exits.
        node.stop_motors()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
