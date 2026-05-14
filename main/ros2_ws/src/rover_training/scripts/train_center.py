"""
Train one center-regression CNN per segment.

Adapted from HYU-ECL3003/rover/train_road_center_model.ipynb. Expects the
CenterDataset format (annotation.txt: filename xpos ypos) — see ../models/
and the preprocess.py output.

Usage:
  python train_center.py --segment common --data data/processed/common \
                         --epochs 20 --out models/center_common.pth
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make HYU-ECL3003 CenterDataset importable without copy.
HYU_ROVER = Path.home() / "HYU-ECL3003" / "rover"
if (HYU_ROVER / "cnn" / "center_dataset.py").exists():
    sys.path.insert(0, str(HYU_ROVER))
from cnn.center_dataset import CenterDataset, TRAIN_TRANSFORMS  # noqa: E402

# Local model factory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.center_cnn import build_center_cnn  # noqa: E402


def train(segment: str, data_dir: Path, epochs: int, out_path: Path, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CenterDataset(str(data_dir), transform=TRAIN_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = build_center_cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
        print(f"[{segment}] epoch {epoch+1}/{epochs} loss={total/len(dataset):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True, choices=["common", "left", "right"])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    train(args.segment, args.data, args.epochs, args.out)
