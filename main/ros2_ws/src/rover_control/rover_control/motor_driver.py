"""
Ported from HYU-ECL3003/rover/base_ctrl.py:BaseController.

Drops the file-relative YAML load; all config comes through the constructor.
Only the surface area that ControlNode uses is kept (serial write of motor
commands via a background queue). Other helpers from the upstream class
(lights, gimbal, OLED, lidar, sensors) were intentionally not ported.
"""
import json
import queue
import threading

import serial


class BaseController:
    def __init__(self, uart_dev: str, baudrate: int, cmd_motion: int = 1):
        self.cmd_motion = cmd_motion
        self.ser = serial.Serial(uart_dev, baudrate, timeout=1)
        self.command_queue: "queue.Queue[dict]" = queue.Queue()
        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.command_thread.start()

    def _process_commands(self) -> None:
        while True:
            data = self.command_queue.get()
            try:
                self.ser.write((json.dumps(data) + "\n").encode("utf-8"))
            except Exception as e:
                print(f"[motor_driver] write error: {e}")

    def send_command(self, data: dict) -> None:
        self.command_queue.put(data)

    def base_speed_ctrl(self, input_left: float, input_right: float) -> None:
        self.send_command({"T": self.cmd_motion, "L": input_left, "R": input_right})

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass
