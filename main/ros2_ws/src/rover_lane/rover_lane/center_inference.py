"""
Pure inference for the action-classification BC CNN with step input.

Model inputs: image (1,3,224,224) + step (1,1) normalized to [0,1].
Model output: 6 logits (one per action class).
Postprocess: argmax -> (steer, speed) via ACTIONS lookup.

Preprocessing: Resize 224x224, ToTensor, Normalize(ImageNet mean/std).
Runtime: TensorRT 10 Python API on CUDA buffers via torch.

Step counter is internal: increments each infer() call from 0. Must match
training-time step semantics (frames from session start). reset_step() at
the start of each run.
"""
from typing import Tuple

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IMG_SHAPE = (1, 3, 224, 224)
_STEP_SHAPE = (1, 1)
_OUTPUT_SHAPE = (1, 6)
STEP_MAX = 1000.0   # must match action_dataset.STEP_MAX

# Action lookup: idx -> (steer, speed) in [-1, +1].
# Tune these values to match the rover's physical response.
ACTIONS = {
    0: (0.0,  -0.15),   # UP        forward
    1: (0.0,  +0.10),   # DOWN      backward
    2: (-0.8, -0.25),   # LEFT      sharp left + accel
    3: (+0.8, -0.25),   # RIGHT     sharp right + accel
    4: (0.0,  -0.10),   # STRAIGHT  straighten + gentle accel
    5: (0.0,   0.0),    # SPACE     stop
}


def preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None, ...])


class CenterInference:
    """Wraps a TensorRT engine that outputs 6-class action logits."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self.engine_path = engine_path
        self._torch = torch
        self.step = 0

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Identify the two inputs (image, step) and one output by shape.
        self.img_name = None
        self.step_name = None
        self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                # 4D = image, 2D = step scalar
                shape = tuple(self.engine.get_tensor_shape(name))
                if len(shape) == 4:
                    self.img_name = name
                else:
                    self.step_name = name
            else:
                self.out_name = name
        if not (self.img_name and self.step_name and self.out_name):
            raise RuntimeError("engine missing expected I/O tensors")

        self.context.set_input_shape(self.img_name, _IMG_SHAPE)
        self.context.set_input_shape(self.step_name, _STEP_SHAPE)

        self.dev_img = torch.empty(_IMG_SHAPE, dtype=torch.float32, device="cuda")
        self.dev_step = torch.empty(_STEP_SHAPE, dtype=torch.float32, device="cuda")
        self.dev_out = torch.empty(_OUTPUT_SHAPE, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.img_name, self.dev_img.data_ptr())
        self.context.set_tensor_address(self.step_name, self.dev_step.data_ptr())
        self.context.set_tensor_address(self.out_name, self.dev_out.data_ptr())
        self.stream = torch.cuda.Stream()

    def reset_step(self) -> None:
        self.step = 0

    def infer(self, image_bgr: np.ndarray) -> Tuple[float, float]:
        torch = self._torch
        x = preprocess(image_bgr)
        step_norm = np.array([[min(self.step / STEP_MAX, 1.0)]], dtype=np.float32)
        with torch.cuda.stream(self.stream):
            self.dev_img.copy_(torch.from_numpy(x), non_blocking=True)
            self.dev_step.copy_(torch.from_numpy(step_norm), non_blocking=True)
            self.context.execute_async_v3(self.stream.cuda_stream)
            out = self.dev_out.detach().cpu()
        self.stream.synchronize()
        self.step += 1
        action_idx = int(out.flatten().argmax().item())
        return ACTIONS[action_idx]
