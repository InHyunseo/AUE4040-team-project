"""Project-wide runtime contracts.

Keep values here aligned with the verified vehicle behavior. These constants
are intentionally small and boring: they exist to prevent topic/control drift
between teleop, inference, recorder, launch files, and documentation.
"""
from __future__ import annotations

from typing import Final


# ROS topics
LANE_IMAGE_TOPIC: Final = "/lane_image/compressed"
FRONT_IMAGE_TOPIC: Final = "/front_image/compressed"
CMD_VEL_TOPIC: Final = "/cmd_vel"
STEER_LEVEL_TOPIC: Final = "/steer_level"
RECORD_ENABLE_TOPIC: Final = "/record_enable"
LANE_SEG_TOPIC: Final = "/lane_seg/compressed"
FRONT_DET_TOPIC: Final = "/front_det/compressed"
LANE_INTENT_TOPIC: Final = "/lane_intent/compressed"

# Verified teleop/control contract.
BASE_V: Final = 0.20
TURN_V: Final = 0.25
MAX_OMEGA: Final = 1.2
LEVELS: Final = 2
TURN_FRAC: Final = (0.0, 0.8, 1.0)
SMOOTH_ALPHA: Final = 0.35
TICK_HZ: Final = 20.0

# E2E TensorRT/ONNX contract.
E2E_LANE_INPUT: Final = "lane"
E2E_FRONT_INPUT: Final = "front"
E2E_STEER_OUTPUT: Final = "steer"
E2E_THROTTLE_OUTPUT: Final = "throttle"
E2E_WAYPOINT_OUTPUT: Final = "waypoints"
E2E_INPUT_SHAPE: Final = (1, 3, 224, 224)
