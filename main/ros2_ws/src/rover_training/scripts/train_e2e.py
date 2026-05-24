"""
Train one action-classification BC CNN per segment.

Usage:
  python train_e2e.py --segment common --data data/processed/common \
                      --epochs 20 --out models/e2e_common.pth
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action_dataset import ActionDataset, NUM_ACTIONS, TRAIN_TRANSFORMS
from models.center_cnn import build_center_cnn


def train(segment: str, data_dir: Path, epochs: int, out_path: Path, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ActionDataset(str(data_dir), transform=TRAIN_TRANSFORMS)

    # Inverse-frequency class weighting on the sampler.
    counts = Counter(int(r[1]) for r in dataset.data)
    cls_w = {c: 1.0 / max(n, 1) for c, n in counts.items()}
    weights = [cls_w[int(r[1])] for r in dataset.data]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    print(f"[{segment}] class counts: {dict(counts)} (balanced sampling)")

    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=2)
    model = build_center_cnn(num_classes=NUM_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_correct = 0
        n_seen = 0
        for imgs, steps, targets in loader:
            imgs = imgs.to(device)
            steps = steps.to(device)
            targets = targets.to(device)
            logits = model(imgs, steps)
            loss = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            n_correct += (logits.argmax(1) == targets).sum().item()
            n_seen += imgs.size(0)
        print(f"[{segment}] epoch {epoch+1}/{epochs} "
              f"loss={total_loss/n_seen:.4f} acc={n_correct/n_seen:.3f}")

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
