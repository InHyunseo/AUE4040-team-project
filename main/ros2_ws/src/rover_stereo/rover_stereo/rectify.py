"""
Stereo rectification helpers. The calibration is FROZEN for the project — we
load YAML + cached remap LUTs once, then `cv2.remap` every frame.

Calibration provenance:
  - Captured by calib/capture_stereo.py (ported from
    HYU-ECL3003/stereo_depth_tutorial/jetson-stereo-depth/calib/capture-stereo.py)
  - Computed by calib/stereo_calibrate.py
  - Output: config/stereo_calib.yaml (committed to git, never overwritten
    after Phase 3 data collection starts)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


@dataclass
class StereoCalib:
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: Tuple[int, int]  # (W, H)
    # Filled in after stereoRectify:
    R1: np.ndarray = None
    R2: np.ndarray = None
    P1: np.ndarray = None
    P2: np.ndarray = None
    Q: np.ndarray = None
    roi1: Tuple[int, int, int, int] = None
    roi2: Tuple[int, int, int, int] = None

    @classmethod
    def load(cls, path: Path) -> "StereoCalib":
        data = yaml.safe_load(Path(path).read_text())
        arr = lambda k: np.array(data[k], dtype=np.float64)
        return cls(
            K1=arr("K1"), D1=arr("D1"),
            K2=arr("K2"), D2=arr("D2"),
            R=arr("R"), T=arr("T"),
            image_size=tuple(data["image_size"]),
        )


class StereoRectifier:
    def __init__(self, calib: StereoCalib, alpha: float = 0.0):
        # alpha=0 -> tight crop (no black borders), alpha=1 -> keep all pixels.
        self.calib = calib
        W, H = calib.image_size
        (calib.R1, calib.R2, calib.P1, calib.P2, calib.Q,
         calib.roi1, calib.roi2) = cv2.stereoRectify(
            calib.K1, calib.D1, calib.K2, calib.D2, (W, H),
            calib.R, calib.T, alpha=alpha,
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            calib.K1, calib.D1, calib.R1, calib.P1, (W, H), cv2.CV_16SC2,
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            calib.K2, calib.D2, calib.R2, calib.P2, (W, H), cv2.CV_16SC2,
        )
        # Conservative shared ROI: intersection of left/right valid regions.
        x1, y1, w1, h1 = calib.roi1
        x2, y2, w2, h2 = calib.roi2
        xa, ya = max(x1, x2), max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        self.shared_roi = (xa, ya, xb - xa, yb - ya)

    def rectify_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        lr = cv2.remap(left_bgr, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rr = cv2.remap(right_bgr, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return lr, rr

    def crop_to_roi(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.shared_roi
        return img[y:y + h, x:x + w]


def disparity_to_distance(disparity_px: float, fx: float, baseline_m: float) -> float:
    if disparity_px <= 0.0:
        return float("inf")
    return fx * baseline_m / disparity_px
