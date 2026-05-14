"""
Pure inference for the road-center regression CNN.

Preprocessing mirrors HYU-ECL3003/rover/cnn/center_dataset.py TRAIN_TRANSFORMS:
  Resize 224x224, ToTensor, Normalize(ImageNet mean/std).
"""
from typing import Tuple

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img, (2, 0, 1))[None, ...]


class CenterInference:
    """Wraps a TensorRT or torch model that outputs (x, y) in [-1, +1]."""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.runtime = None  # filled in port step
        # TODO: load TRT engine here, or torch.load fallback.

    def infer(self, image_bgr: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError("Port the runtime load + execute here.")
