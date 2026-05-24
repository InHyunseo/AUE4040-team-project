"""
Action-classification BC dataset: (image, step) -> action class (0..5).

Reads `annotation.txt` lines: `filename action_idx step` (3 cols, written by
preprocess.py). Step is normalized to [0, 1] by dividing by STEP_MAX.

hflip: image flipped horizontally + left/right swap (2 <-> 3). Step unchanged.
"""
import os

import numpy as np
import PIL.Image
import torch
import torch.utils.data
import torchvision.transforms as transforms


NUM_ACTIONS = 6
HFLIP_SWAP = {2: 3, 3: 2}
# Frames per session ceiling for step normalization. Sessions longer than this
# saturate at 1.0 — set to a value safely above your longest expected run.
STEP_MAX = 1000.0

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


def normalize_step(step: float) -> float:
    return min(float(step) / STEP_MAX, 1.0)


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, random_hflip=True, transform=TRAIN_TRANSFORMS):
        super().__init__()
        self.root_dir = root_dir
        self.random_hflip = random_hflip
        self.transform = transform
        with open(os.path.join(root_dir, "annotation.txt"), "r") as f:
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
        step = normalize_step(int(step_str))

        image = PIL.Image.open(os.path.join(self.root_dir, "images", filename))
        if self.transform is not None:
            image = self.transform(image)

        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.flip(image, [-1])
            action = HFLIP_SWAP.get(action, action)

        return (
            image,
            torch.tensor([step], dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
        )
