"""Holds 3 CenterInference instances and selects active model by tag."""
from typing import Dict

from rover_pilotnet.center_inference import CenterInference


class ModelManager:
    def __init__(self, model_paths: Dict[str, str]):
        self.models: Dict[str, CenterInference] = {
            tag: CenterInference(path) for tag, path in model_paths.items()
        }
        self.active = "common"

    def set_active(self, tag: str) -> None:
        if tag not in self.models:
            raise KeyError(f"unknown model tag: {tag}")
        self.active = tag

    def infer(self, image_bgr):
        return self.models[self.active].infer(image_bgr)
