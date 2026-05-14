"""
Per-frame safety detector stabilization. Maintains rolling counters and
decides which detections are "stable" enough to drive an FSM transition.
"""
from collections import deque
from typing import Dict


def vehicle_close(bbox_h_px: float, K_px_m: float, safe_dist_m: float) -> bool:
    """Bbox-height fallback distance check (used when stereo is unavailable).

    Closed-world single-vehicle assumption: bbox height is monotonic in distance,
    so one constant K (= fx * H_real) is enough. Calibrate K once with the
    actual vehicle at 2-3 known distances.
    """
    if bbox_h_px <= 0.0:
        return False
    dist = K_px_m / bbox_h_px
    return dist < safe_dist_m


class Stabilizer:
    def __init__(self, thresholds: Dict[str, int]):
        # class_name -> N frames required
        self.thresholds = thresholds
        self.streaks: Dict[str, int] = {k: 0 for k in thresholds}

    def update(self, present: Dict[str, bool]) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for cls, threshold in self.thresholds.items():
            if present.get(cls, False):
                self.streaks[cls] += 1
            else:
                self.streaks[cls] = 0
            out[cls] = self.streaks[cls] >= threshold
        return out
