"""Shared image (de)serialization for CompressedImage topics.

camera/overlay_viz/e2e_infer each had a near-identical `_encode`. One helper
keeps the JPEG settings and message layout consistent across publishers.
"""
from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

DEFAULT_JPEG_QUALITY = 80


def encode_bgr(bgr: np.ndarray, header, quality: int = DEFAULT_JPEG_QUALITY) -> CompressedImage:
    """BGR uint8 image → CompressedImage(jpeg) carrying `header` verbatim.

    Callers pass a std_msgs/Header (copy the source image's header to keep the
    capture timestamp, or build one from a clock stamp).
    """
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    out = CompressedImage()
    out.header = header
    out.format = "jpeg"
    out.data = jpg.tobytes()
    return out
