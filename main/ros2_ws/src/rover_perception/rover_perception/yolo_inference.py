"""
Pure YOLO inference helpers — no ROS dependency.

Port plan:
  - Reuse the load / draw_boxes pattern from
    HYU-ECL3003/week07/YOLOv8/demo_livecam_local.py.
  - For TensorRT engine inference, use `ultralytics.YOLO("path.engine", task="detect")`.
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Det:
    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class YoloInference:
    def __init__(self, engine_path: str, conf: float = 0.4, iou: float = 0.5):
        from ultralytics import YOLO  # lazy import; not needed for unit tests
        self.model = YOLO(engine_path, task="detect")
        self.names = self.model.names
        self.conf = conf
        self.iou = iou

    def infer(self, image_bgr: np.ndarray) -> List[Det]:
        results = self.model(image_bgr, conf=self.conf, iou=self.iou, verbose=False)
        out: List[Det] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cid = int(box.cls[0])
                out.append(Det(cid, self.names[cid], float(box.conf[0]),
                               x1, y1, x2, y2))
        return out
