"""
Motor control node.

Subscribes:  /cmd_vel (geometry_msgs/Twist) — linear.x = throttle, angular.z = steering
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


def compute_throttle(steering: float, target_speed: float, decel: float) -> float:
    return target_speed * (1.0 - decel * abs(steering))


class ControlNode(Node):
    def __init__(self):
        super().__init__("rover_control")
        self.declare_parameter("uart_dev", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("invert_drive", True)
        self.declare_parameter("target_speed", 0.35)
        self.declare_parameter("curvature_decel_factor", 0.6)
        self.declare_parameter("max_steer", 0.8)
        self.declare_parameter("max_speed", 0.5)

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
        else:
            throttle = compute_throttle(
                steering,
                self.get_parameter("target_speed").value,
                self.get_parameter("curvature_decel_factor").value,
            )
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
        else:
            self.get_logger().info(f"[dry] L={L:.3f} R={R:.3f}")


def main():
    rclpy.init()
    rclpy.spin(ControlNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
