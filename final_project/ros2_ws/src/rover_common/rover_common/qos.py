"""Shared QoS profiles.

These two profiles were redeclared verbatim across camera/monitor/overlay_viz/
e2e_infer. Centralizing them keeps the pub/sub matching rules (and the reasoning
behind them) in one place.

  IMAGE_PUB_QOS  : camera image publishers. RELIABLE + KEEP_LAST depth=1.
    RELIABLE so the bag_recorder (RELIABLE sub, needs completeness) matches;
    depth=1 so the sender doesn't hold stale frames. RELIABLE pub still matches
    BEST_EFFORT consumer subs (monitor/overlay/inference).

  SENSOR_QOS     : real-time image consumers (monitor/overlay/inference).
    BEST_EFFORT + KEEP_LAST depth=1 — drop backed-up frames, act on the latest
    only, so per-cycle latency never accumulates. NOT for bag_recorder.
"""
from __future__ import annotations

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

IMAGE_PUB_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
