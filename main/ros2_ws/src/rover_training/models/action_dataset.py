"""
Action-classification BC dataset: (image, step) -> action class (0..5).

Reads `annotation.txt` lines: `filename action_idx step` (3 cols, written by
preprocess.py). Step is normalized to [0, 1] by dividing by STEP_MAX.
"""
import os

import PIL.Image
import torch
import torch.utils.data
import torchvision.transforms as transforms


NUM_ACTIONS = 6
# Frames-per-session ceiling for step normalization. Set just above your longest
# real run so the normalized signal fills [0,1] instead of getting squished into
# the low end (which would let the FC head learn to ignore it).
STEP_MAX = 400.0

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
    def __init__(self, root_dir, transform=TRAIN_TRANSFORMS):
        super().__init__()
        self.root_dir = root_dir
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

        return (
            image,
            torch.tensor([step], dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
        )
