"""
Train one action-classification BC CNN per segment.

Usage:
  python train_e2e.py --segment common --data data/processed/common \
                      --epochs 20 --out models/e2e_common.pth
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
from models.action_dataset import ActionDataset, NUM_ACTIONS, TRAIN_TRANSFORMS, TEST_TRANSFORMS
from models.center_cnn import build_center_cnn

DEFAULT_DATA_ROOT = PKG_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PKG_ROOT / "models"


def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0
    with torch.no_grad():
        for imgs, steps, targets in loader:
            imgs = imgs.to(device)
            steps = steps.to(device)
            targets = targets.to(device)
            logits = model(imgs, steps)
            loss = loss_fn(logits, targets)
            total_loss += loss.item() * imgs.size(0)
            n_correct += (logits.argmax(1) == targets).sum().item()
            n_seen += imgs.size(0)
    if n_seen == 0:
        return None, None
    return total_loss / n_seen, n_correct / n_seen


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

    # Optional test set: held-out sessions (preprocess writes test_annotation.txt).
    test_loader = None
    test_ann = data_dir / "test_annotation.txt"
    if test_ann.exists() and test_ann.stat().st_size > 0:
        test_ds = ActionDataset(str(data_dir), transform=TEST_TRANSFORMS,
                                annotation_file="test_annotation.txt")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
        print(f"[{segment}] test set: {len(test_ds)} frames")
    else:
        print(f"[{segment}] no test set (test_annotation.txt missing/empty)")

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
        msg = (f"[{segment}] epoch {epoch+1}/{epochs} "
               f"train_loss={total_loss/n_seen:.4f} train_acc={n_correct/n_seen:.3f}")
        if test_loader is not None:
            test_loss, test_acc = _evaluate(model, test_loader, loss_fn, device)
            msg += f"  test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        print(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    # Copy metadata.json next to the checkpoint so inference (ONNX/TRT) can
    # resolve the matching step_max without depending on data/processed paths.
    src_meta = Path(data_dir) / "metadata.json"
    if src_meta.exists():
        dst_meta = out_path.with_suffix(".metadata.json")
        shutil.copy2(src_meta, dst_meta)
        print(f"saved -> {out_path}  (+ {dst_meta.name})")
    else:
        print(f"saved -> {out_path}  (warning: {src_meta} missing)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True, choices=["common", "left", "right"])
    ap.add_argument("--data", type=Path, default=None,
                    help=f"segment dir (default: {DEFAULT_DATA_ROOT}/<segment>)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output .pth (default: {DEFAULT_MODELS_DIR}/e2e_<segment>.pth)")
    args = ap.parse_args()

    data = args.data if args.data is not None else DEFAULT_DATA_ROOT / args.segment
    out = args.out if args.out is not None else DEFAULT_MODELS_DIR / f"e2e_{args.segment}.pth"
    train(args.segment, data, args.epochs, out)
