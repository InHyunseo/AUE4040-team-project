"""
Thin re-export of HYU-ECL3003/rover/base_ctrl.py BaseController.

When porting, copy base_ctrl.py contents here and replace the YAML loading
block with values supplied by ROS parameters (uart_dev, baudrate, cmd codes).
For now this stub raises on import so the missing port is obvious.
"""

class BaseController:
    def __init__(self, uart_dev: str, baudrate: int):
        raise NotImplementedError(
            "Port HYU-ECL3003/rover/base_ctrl.py:BaseController here. "
            "Drop the file-relative YAML load; take config via constructor args."
        )

    def base_speed_ctrl(self, L: float, R: float) -> None:  # pragma: no cover
        ...
