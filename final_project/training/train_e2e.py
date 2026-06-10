"""E2E 학습 루프 — labels_cache.h5 → E2ENet best 체크포인트.

  python train_e2e.py \
      --cache ../labels_cache.h5 \
      --out ../models/e2e_best.pt \
      --epochs 60 --batch 32 --lr 3e-4 --device cuda

여러 bag 을 합치려면 --cache 를 여러 번:
  python train_e2e.py --cache a.h5 --cache b.h5 ...

의도 시각화 켜기 (예측 vs GT waypoint/steer/throttle 패널 저장):
  python train_e2e.py --cache ... --viz_dir ../debug_samples/train_viz --viz_every 5

이어학습 (새 bag 추가 후 기존 best 에서 fine-tune, optimizer state 복원):
  python train_e2e.py --cache old.h5 --cache new.h5 --resume ../models/e2e_best.pt

흐름: make_splits 로 train/val 분할 → E2ENet(ImageNet pretrained backbone)
학습 → val total loss 기준 early-stop, best 만 저장. 체크포인트는 ONNX export 가
바로 load 할 수 있게 model state_dict 를 담고(optimizer/메타데이터는 별도 키),
--viz_dir 가 켜지면 매 --viz_every epoch + 새 best 마다 의도 패널을 저장한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# model.py 는 final_project/ 루트, dataset.py 는 이 파일과 같은 training/ 에 있다.
# cwd 와 무관하게 import 되도록 둘 다 sys.path 에 절대경로로 넣는다.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # final_project/ → model.py
sys.path.insert(0, str(_HERE))          # training/      → dataset.py
from model import E2ENet, E2ELoss  # noqa: E402

from dataset import (E2EDataset, make_splits,  # noqa: E402
                     IMAGENET_MEAN, IMAGENET_STD)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from viz import pred_vs_gt_panel  # noqa: E402


def _denorm_bgr(t):
    """정규화 텐서 (3,H,W) → 합성 BGR uint8 (viz 입력용, to_input_tensor 역변환)."""
    rgb = t.detach().cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def save_viz(model, dataset, viz_dir, epoch, device, n=6):
    """val 데이터셋 앞 n개에 대해 예측 vs GT 의도 패널을 저장 (켜졌을 때만)."""
    model.eval()
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(dataset))):
        lane_t, front_t, steer_gt, thr_gt, wp_gt = dataset[i]
        steer_p, thr_p, wp_p = model(lane_t.unsqueeze(0).to(device),
                                     front_t.unsqueeze(0).to(device))
        panel = pred_vs_gt_panel(
            _denorm_bgr(lane_t), _denorm_bgr(front_t),
            wp_p[0].cpu().numpy(), wp_gt.numpy(),
            float(steer_p[0]), float(thr_p[0]),
            float(steer_gt), float(thr_gt))
        cv2.imwrite(str(viz_dir / f"ep{epoch:03d}_s{i:02d}.png"), panel)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals = dict(total=0.0, steer=0.0, throttle=0.0, wp=0.0, n=0)
    torch.set_grad_enabled(train)
    for lane, front, steer_gt, thr_gt, wp_gt in loader:
        lane = lane.to(device, non_blocking=True)
        front = front.to(device, non_blocking=True)
        steer_gt = steer_gt.to(device, non_blocking=True)
        thr_gt = thr_gt.to(device, non_blocking=True)
        wp_gt = wp_gt.to(device, non_blocking=True)

        steer, thr, wp = model(lane, front)
        loss, sl, tl, wl = criterion(steer, thr, wp, steer_gt, thr_gt, wp_gt)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = lane.size(0)
        totals["total"] += loss.item() * bs
        totals["steer"] += sl.item() * bs
        totals["throttle"] += tl.item() * bs
        totals["wp"] += wl.item() * bs
        totals["n"] += bs
    torch.set_grad_enabled(True)
    n = max(totals["n"], 1)
    return {k: totals[k] / n for k in ("total", "steer", "throttle", "wp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True, type=Path,
                    help="labels_cache.h5 (여러 bag 이면 반복 지정)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "models" / "e2e_best.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10, help="early-stop epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--viz_dir", type=Path, default=None,
                    help="set to enable pred-vs-GT intent viz (saved each --viz_every)")
    ap.add_argument("--viz_every", type=int, default=5,
                    help="save viz every N epochs (and on each new best)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from checkpoint (model + optimizer state)")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[warn] cuda unavailable, falling back to cpu")
    torch.manual_seed(args.seed)

    cache_paths = [str(p) for p in args.cache]
    train_idx, val_idx = make_splits(cache_paths, val_frac=args.val_frac, seed=args.seed)
    print(f"samples: train={len(train_idx)}  val={len(val_idx)}  (caches={len(cache_paths)})")
    if len(train_idx) == 0:
        raise SystemExit("no training samples — check --cache path / extraction")

    train_ds = E2EDataset(cache_paths, indices=train_idx, augment=True, seed=args.seed)
    val_ds   = E2EDataset(cache_paths, indices=val_idx, augment=False, seed=args.seed)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    model = E2ENet().to(device)
    criterion = E2ELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 이어학습: 가중치 + optimizer state 복원 (없으면 ImageNet pretrained 부터).
    best_val = float("inf")
    if args.resume is not None:
        ck = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        if isinstance(ck, dict) and "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if isinstance(ck, dict) and "val_total" in ck:
            best_val = float(ck["val_total"])
        print(f"resumed from {args.resume} (epoch={ck.get('epoch')} val_total={best_val:.4f})")

    args_meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args_meta["cache"] = cache_paths

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"ep {epoch:3d}/{args.epochs}  "
              f"train {tr['total']:.4f} (s{tr['steer']:.4f} t{tr['throttle']:.4f} w{tr['wp']:.4f})  "
              f"val {va['total']:.4f} (s{va['steer']:.4f} t{va['throttle']:.4f} w{va['wp']:.4f})  "
              f"lr {lr_now:.2e}  {dt:.0f}s")

        improved = va["total"] < best_val - 1e-5
        if improved:
            best_val = va["total"]
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # --resume 용
                "epoch": epoch,
                "val_total": best_val,
                "val_steer": va["steer"],
                "val_throttle": va["throttle"],
                "val_wp": va["wp"],
                "args": args_meta,
            }, args.out)
            print(f"    ↳ new best val={best_val:.4f}  saved {args.out}")
        else:
            bad += 1

        # 의도 시각화: 켜졌을 때만, 주기적으로 + 새 best 마다.
        if args.viz_dir is not None and (improved or epoch % args.viz_every == 0):
            save_viz(model, val_ds, args.viz_dir, epoch, device)

        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (no val improvement for {args.patience})")
            break

    print(f"done. best val total={best_val:.4f}  -> {args.out}")
    print("next: export_onnx.py --ckpt", args.out)


if __name__ == "__main__":
    main()
