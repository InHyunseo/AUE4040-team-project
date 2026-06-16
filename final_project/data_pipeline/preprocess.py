"""Shared E2E overlay-compositing + tensor contract.

These functions define the exact pixels the E2E model sees. They MUST be
identical across:
  - training  (training/dataset.py builds inputs from labels_cache.h5)
  - inference (rover_lane/e2e_infer_node.py builds inputs from live frames)
  - preview   (rover_camera/overlay_viz_node.py for the browser monitor)

If color/normalization/crop order differs by a single pixel between training and
inference the model misbehaves, so all three import from here — never reimplement.

Color space: H5 lane/front are BGR uint8; compositing stays in BGR, then
to_input_tensor converts BGR→RGB and applies ImageNet mean/std (ResNet18
ImageNet pretrained is RGB-based).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

# Works both as a package module (data_pipeline.preprocess, ROS nodes) and as a
# top-level module (preprocess, with data_pipeline/ on sys.path — training).
try:
    from .extract_labels import SEG_N_CLASSES
except ImportError:  # pragma: no cover
    from extract_labels import SEG_N_CLASSES

# seg channel → BGR color (ch0=left-solid red, ch1=right-solid green,
# ch2=center-dashed blue). Matches extract_labels.save_debug / visualize_labels.
SEG_COLORS_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
SEG_ALPHA = 0.6   # color weight (base 0.4)

BBOX_COLOR_BGR = (0, 255, 0)
BBOX_THICK = 2

# ImageNet normalization (RGB).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def composite_lane(lane_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """raw lane(BGR uint8) + seg 3 channels alpha-blended by color. Returns BGR uint8."""
    out = lane_bgr.copy()
    for c in range(SEG_N_CLASSES):
        m = seg[c] > 0
        if not m.any():
            continue
        color = np.array(SEG_COLORS_BGR[c], dtype=np.float32)
        out[m] = (out[m] * (1.0 - SEG_ALPHA) + color * SEG_ALPHA).astype(np.uint8)
    return out


def composite_front(front_bgr: np.ndarray, det: np.ndarray) -> np.ndarray:
    """raw front(BGR uint8) + car bbox rectangle. det=[x,y,w,h,conf]; conf<=0 → unchanged."""
    out = front_bgr.copy()
    if det[4] > 0:
        x, y, w, h, _ = det
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                      BBOX_COLOR_BGR, BBOX_THICK)
    return out


def to_input_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Composited BGR uint8 (H,W,3) → RGB ImageNet-normalized tensor (3,H,W) float32.

    Inference must use this exact transform (BGR→RGB→ImageNet)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())
