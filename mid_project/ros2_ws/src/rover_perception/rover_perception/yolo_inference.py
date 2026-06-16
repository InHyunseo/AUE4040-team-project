"""
Pure YOLO inference helpers — no ROS dependency.

Runtime: TensorRT 10 Python API + torch CUDA buffers (matches the lightweight
pattern used by rover_lane/center_inference.py). No ultralytics, no pycuda.

Engine assumptions (standard YOLOv8 export):
  input  : (1, 3, H, W) float32, RGB, 0..1, letterboxed
  output : (1, 4+nc, N) float32   — N anchors, 4 bbox (cx, cy, w, h) + nc class scores
The input H, W and the number of classes are read from the engine itself, so
this helper works for both the stock COCO YOLOv8n at 640 and a fine-tuned
7-class model at 320 without changes.
"""
from dataclasses import dataclass
from typing import List, Sequence

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


def letterbox(image_bgr: np.ndarray, size: int) -> tuple:
    """Resize-with-pad to (size, size). Returns (padded, scale, pad_x, pad_y)."""
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    import cv2
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # YOLO default pad value
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Plain numpy NMS. boxes: (N,4) xyxy."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


class YoloInference:
    """Wraps a YOLOv8 TensorRT engine — minimal, lightweight."""

    def __init__(self, engine_path: str, conf: float = 0.4,
                 iou: float = 0.5, class_names: Sequence[str] = ()):
        import tensorrt as trt
        import torch

        self.engine_path = engine_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_names = list(class_names)
        self._torch = torch

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("engine missing input/output tensors")

        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        if len(in_shape) != 4 or in_shape[1] != 3:
            raise RuntimeError(f"unexpected input shape {in_shape}")
        self.input_size = int(in_shape[2])
        # YOLOv8 head: (1, 4+nc, N). 80 classes for stock COCO.
        self.nc = int(out_shape[1]) - 4
        self.n_anchors = int(out_shape[2])

        self.context.set_input_shape(self.input_name, in_shape)

        self.dev_in = torch.empty(in_shape, dtype=torch.float32, device="cuda")
        self.dev_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.input_name, self.dev_in.data_ptr())
        self.context.set_tensor_address(self.output_name, self.dev_out.data_ptr())
        self.stream = torch.cuda.Stream()

    def _preprocess(self, image_bgr: np.ndarray):
        canvas, scale, pad_x, pad_y = letterbox(image_bgr, self.input_size)
        # BGR -> RGB, HWC -> CHW, /255, contiguous.
        rgb = canvas[:, :, ::-1]
        x = rgb.astype(np.float32) / 255.0
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None, ...])
        return x, scale, pad_x, pad_y

    def infer(self, image_bgr: np.ndarray) -> List[Det]:
        torch = self._torch
        x, scale, pad_x, pad_y = self._preprocess(image_bgr)
        with torch.cuda.stream(self.stream):
            self.dev_in.copy_(torch.from_numpy(x), non_blocking=True)
            self.context.execute_async_v3(self.stream.cuda_stream)
            raw = self.dev_out.detach().cpu().numpy()
        self.stream.synchronize()

        # raw shape: (1, 4+nc, N) -> (N, 4+nc)
        pred = raw[0].T
        box_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4 + self.nc]
        cls_id = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)
        mask = cls_conf > self.conf
        if not mask.any():
            return []
        box_xywh = box_xywh[mask]
        cls_id = cls_id[mask]
        cls_conf = cls_conf[mask]

        # cxcywh -> xyxy (still in letterboxed coords)
        cx, cy, w, h = box_xywh[:, 0], box_xywh[:, 1], box_xywh[:, 2], box_xywh[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms_numpy(boxes, cls_conf, self.iou)
        if not keep:
            return []
        boxes = boxes[keep]
        cls_id = cls_id[keep]
        cls_conf = cls_conf[keep]

        # Un-letterbox back into the original image coords.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        out: List[Det] = []
        for i in range(len(keep)):
            cid = int(cls_id[i])
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            out.append(Det(
                class_id=cid, class_name=name, score=float(cls_conf[i]),
                x1=float(boxes[i, 0]), y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]), y2=float(boxes[i, 3]),
            ))
        return out


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
