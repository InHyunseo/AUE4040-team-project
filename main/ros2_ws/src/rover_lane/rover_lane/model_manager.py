"""Holds up to 3 CenterInference instances and selects active model by tag.

Missing engine files are skipped (logged) — useful for partial testing with
just the common model.
"""
import os
from typing import Dict

from rover_lane.center_inference import CenterInference


class ModelManager:
    def __init__(self, model_paths: Dict[str, str]):
        self.models: Dict[str, CenterInference] = {}
        for tag, path in model_paths.items():
            if not os.path.exists(path):
                print(f"[ModelManager] skip {tag}: {path} not found")
                continue
            self.models[tag] = CenterInference(path)
        if not self.models:
            raise RuntimeError("no engines loaded")
        self.active = "common" if "common" in self.models else next(iter(self.models))

    def set_active(self, tag: str) -> None:
        if tag not in self.models:
            print(f"[ModelManager] {tag} not loaded; keeping {self.active}")
            return
        self.active = tag

    def infer(self, image_bgr):
        # Share step counter across models so a mid-run model switch keeps
        # step values aligned with training (step = frames since session start).
        active = self.models[self.active]
        for other_tag, other in self.models.items():
            if other_tag != self.active:
                other.step = active.step
        return active.infer(image_bgr)

    def reset_step(self):
        for m in self.models.values():
            m.reset_step()
