"""
Action-classification BC dataset: (image, step) -> action class (0..5).

Reads `annotation.txt` lines: `filename action_idx step` (3 cols, written by
preprocess.py). Step is normalized to [0, 1] by dividing by step_max, which
is loaded from <root>/metadata.json (also written by preprocess.py).
"""
import json
import os

import PIL.Image
import torch
import torch.utils.data
import torchvision.transforms as transforms


NUM_ACTIONS = 6

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def normalize_step(step: float, step_max: float) -> float:
    return min(float(step) / float(step_max), 1.0)


def load_step_max(root_dir) -> float:
    meta_path = os.path.join(root_dir, "metadata.json")
    with open(meta_path, "r") as f:
        return float(json.load(f)["step_max"])


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=TRAIN_TRANSFORMS,
                 annotation_file="annotation.txt"):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.step_max = load_step_max(root_dir)
        with open(os.path.join(root_dir, annotation_file), "r") as f:
            self.data = [
                line.split()
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, action_str, step_str = self.data[idx]
        action = int(action_str)
        step = normalize_step(int(step_str), self.step_max)

        image = PIL.Image.open(os.path.join(self.root_dir, "images", filename))
        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            torch.tensor([step], dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
        )
