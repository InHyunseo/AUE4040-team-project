"""Holds up to 3 CenterInference instances and selects active model by tag.

Each engine path must have a sibling `<stem>.metadata.json` (written by
train_e2e.py) carrying that segment's step_max. Missing engine files are
skipped (logged); missing metadata is a hard error since step normalization
depends on it.

step policy: each segment was recorded as its own session starting at step 0
(common, sessionL, sessionR are independent recordings). On segment switch
we therefore reset the new model's step to 0 — keeping step semantics aligned
with training. The previous "share step across models" policy was wrong for
this dataset.
"""
import json
import os
from pathlib import Path
from typing import Dict

from rover_lane.center_inference import CenterInference


def _load_step_max(engine_path: str) -> float:
    meta = Path(engine_path).with_suffix(".metadata.json")
    if not meta.exists():
        # train_e2e saves models/e2e_<seg>.metadata.json; TRT engine may be
        # models/e2e_<seg>.engine — with_suffix replaces .engine so this works
        # for both .onnx and .engine. Fall back to <path>.metadata.json too.
        alt = Path(str(engine_path) + ".metadata.json")
        if alt.exists():
            meta = alt
        else:
            raise FileNotFoundError(
                f"metadata.json missing for engine {engine_path}; "
                f"expected at {meta} or {alt}"
            )
    with open(meta, "r") as f:
        return float(json.load(f)["step_max"])


class ModelManager:
    def __init__(self, model_paths: Dict[str, str]):
        self.models: Dict[str, CenterInference] = {}
        for tag, path in model_paths.items():
            if not os.path.exists(path):
                print(f"[ModelManager] skip {tag}: {path} not found")
                continue
            step_max = _load_step_max(path)
            self.models[tag] = CenterInference(path, step_max=step_max)
            print(f"[ModelManager] loaded {tag}: step_max={step_max}")
        if not self.models:
            raise RuntimeError("no engines loaded")
        self.active = "common" if "common" in self.models else next(iter(self.models))

    def set_active(self, tag: str) -> None:
        if tag not in self.models:
            print(f"[ModelManager] {tag} not loaded; keeping {self.active}")
            return
        if tag == self.active:
            return
        self.active = tag
        # Each segment was trained on its own session starting at step 0.
        # Reset the now-active model's step so inference-time step semantics
        # match training-time semantics.
        self.models[tag].reset_step()

    def infer(self, image_bgr):
        return self.models[self.active].infer(image_bgr)

    def reset_step(self):
        for m in self.models.values():
            m.reset_step()

    def current_step(self) -> int:
        return self.models[self.active].step

    def current_step_max(self) -> float:
        return self.models[self.active].step_max

    def step_done(self) -> bool:
        return self.models[self.active].step_done()
