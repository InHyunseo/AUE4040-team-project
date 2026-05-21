"""
Pure inference for the road-center regression CNN.

Preprocessing mirrors HYU-ECL3003/rover/cnn/center_dataset.py TRAIN_TRANSFORMS:
  Resize 224x224, ToTensor, Normalize(ImageNet mean/std).

Runtime: TensorRT 10 Python API. CUDA buffers and stream come from torch
(already a dependency for training); we just hand TRT the device pointers via
set_tensor_address and drive the stream with torch.cuda. The model is exported
from center_cnn.build_center_cnn() with input (1,3,224,224) f32 and output
(1,2) f32 in [-1, +1].
"""
from typing import Tuple

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SHAPE = (1, 3, 224, 224)
_OUTPUT_SHAPE = (1, 2)


def preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None, ...])


class CenterInference:
    """Wraps a TensorRT engine that outputs (x, y) in [-1, +1]."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self.engine_path = engine_path
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

        self.context.set_input_shape(self.input_name, _INPUT_SHAPE)

        self.dev_in = torch.empty(_INPUT_SHAPE, dtype=torch.float32, device="cuda")
        self.dev_out = torch.empty(_OUTPUT_SHAPE, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.input_name, self.dev_in.data_ptr())
        self.context.set_tensor_address(self.output_name, self.dev_out.data_ptr())
        self.stream = torch.cuda.Stream()

    def infer(self, image_bgr: np.ndarray) -> Tuple[float, float]:
        torch = self._torch
        x = preprocess(image_bgr)
        with torch.cuda.stream(self.stream):
            self.dev_in.copy_(torch.from_numpy(x), non_blocking=True)
            self.context.execute_async_v3(self.stream.cuda_stream)
            out = self.dev_out.detach().cpu()
        self.stream.synchronize()
        return float(out.flatten()[0]), float(out.flatten()[1])
