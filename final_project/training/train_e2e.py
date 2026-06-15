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

from dataset import (E2EDataset, make_splits, oversample_avoidance,  # noqa: E402
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
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5, help="early-stop epochs")
    # ReduceLROnPlateau: val_steer 가 --lr_patience epoch 정체하면 LR×lr_factor.
    # cosine(T_max) 은 early-stop 길이를 미리 못 맞춰 LR 이 거의 안 식는 문제가 있어
    # plateau 로 교체(정체 시 자동 감쇠). early-stop patience > lr_patience 라야
    # LR 을 낮춘 뒤 더 내려가는지 볼 여유가 생긴다(patience 5 면 lr_patience 2).
    ap.add_argument("--lr_patience", type=int, default=2, help="LR decay patience (epochs)")
    ap.add_argument("--lr_factor", type=float, default=0.5, help="LR decay factor on plateau")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="LR floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--viz_dir", type=Path, default=None,
                    help="set to enable pred-vs-GT intent viz (saved each --viz_every)")
    ap.add_argument("--viz_every", type=int, default=5,
                    help="save viz every N epochs (and on each new best)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from checkpoint (model + optimizer/scheduler state 이어가기)")
    # 미세조정: --resume 의 체크포인트에서 **가중치만** 불러오고 optimizer/scheduler/
    # best 는 새로 시작한다. 새 --lr(낮게)로 깨끗이 시작하고 best 도 새 데이터 기준으로
    # 다시 잡아 조기종료를 막는다. (--resume 의 "정확히 이어가기"와 구분.)
    ap.add_argument("--finetune", action="store_true",
                    help="load only model weights from --resume; fresh optimizer/scheduler/best")
    # loss 가중치 (E2ELoss 기본 1.0/0.5/0.5). waypoint 부호 오염 격리 테스트엔
    # --waypoint_weight 0 으로 보조 task 를 꺼서 steer 학습에 미치는 영향을 본다.
    ap.add_argument("--steer_weight", type=float, default=1.0)
    ap.add_argument("--throttle_weight", type=float, default=0.5)
    ap.add_argument("--waypoint_weight", type=float, default=0.5)
    # steer GT 시간 스무딩: H5(세션) 내부에서 ±k 프레임 이동평균(teleop raw 조향의
    # 순간 떨림 완화). 0=끔. 세션 경계는 넘지 않는다(dataset 이 보장).
    ap.add_argument("--steer_smooth", type=int, default=0,
                    help="moving-average half-window over steer GT (0=off)")
    # 옛 부호 H5 의 waypoint 를 즉석 보정(x 만 반전). 재추출 없이 wp 부호 버그를
    # 학습/시각화에서 고친다. 새 부호로 재추출한 H5 엔 끈다(이중 반전 방지).
    ap.add_argument("--wp_fix_sign", action="store_true",
                    help="flip waypoint x for legacy (pre-sign-fix) H5")
    # 좌우 flip aug (train 만). 이미지+seg채널교환+det+steer+wp 일관 반전, 50% 확률.
    # 데이터 2배 + 좌우 균형으로 steer 과적합을 완화한다.
    ap.add_argument("--hflip", action="store_true", help="50% horizontal flip aug (train)")
    # 회피 oversampling: 차 감지+측면이동 큰 train 프레임을 N배 복제(클래스 불균형
    # 완화). 재수집 없이 회피 학습 강화. val 엔 적용 안 함(원본 분포 평가).
    ap.add_argument("--avoid_oversample", type=int, default=1,
                    help="duplicate avoidance (det>0 + lateral wp) train frames N times")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[warn] cuda unavailable, falling back to cpu")
    torch.manual_seed(args.seed)

    cache_paths = [str(p) for p in args.cache]
    train_idx, val_idx = make_splits(cache_paths, val_frac=args.val_frac, seed=args.seed)
    if args.avoid_oversample > 1:
        before = len(train_idx)
        train_idx = oversample_avoidance(cache_paths, train_idx, factor=args.avoid_oversample)
        print(f"avoid oversample x{args.avoid_oversample}: train {before} -> {len(train_idx)}")
    print(f"samples: train={len(train_idx)}  val={len(val_idx)}  (caches={len(cache_paths)})")
    if len(train_idx) == 0:
        raise SystemExit("no training samples — check --cache path / extraction")

    train_ds = E2EDataset(cache_paths, indices=train_idx, augment=True,
                          seed=args.seed, steer_smooth=args.steer_smooth,
                          wp_fix_sign=args.wp_fix_sign, hflip=args.hflip)
    val_ds   = E2EDataset(cache_paths, indices=val_idx, augment=False,
                          seed=args.seed, steer_smooth=args.steer_smooth,
                          wp_fix_sign=args.wp_fix_sign, hflip=False)  # val 은 원본

    # 워커별 numpy 재시드: PyTorch 는 torch/random 만 워커 시드하고 np.random 은
    # 안 해줘서, 안 하면 모든 워커가 같은 flip/jitter 패턴을 내 다양성이 준다.
    def _seed_worker(worker_id):
        import torch as _t
        seed = (_t.initial_seed() + worker_id) % (2 ** 32)
        np.random.seed(seed)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True,
                              worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    model = E2ENet().to(device)
    criterion = E2ELoss(steer_weight=args.steer_weight,
                        throttle_weight=args.throttle_weight,
                        waypoint_weight=args.waypoint_weight)
    print(f"loss weights: steer={args.steer_weight} throttle={args.throttle_weight} "
          f"waypoint={args.waypoint_weight} | steer_smooth=±{args.steer_smooth}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # val_steer(=best 기준 신호) 가 정체하면 LR 을 깎는다. 빠른 초기 하강 후
    # 정체되는 이 문제 패턴에 cosine 보다 적합.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr)

    # 이어학습: 가중치 + optimizer state 복원 (없으면 ImageNet pretrained 부터).
    best_val = float("inf")
    if args.resume is not None:
        ck = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        if args.finetune:
            # 미세조정: 가중치만. optimizer/scheduler/best 는 새 --lr 로 깨끗이 시작.
            print(f"finetune from {args.resume} (weights only; fresh optimizer/scheduler, "
                  f"lr={args.lr})")
        else:
            # 정확히 이어가기: optimizer/scheduler/best 복원.
            if isinstance(ck, dict) and "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            # scheduler state 복원 (plateau 의 정체 카운터/현재 LR 이어가기).
            if isinstance(ck, dict) and "scheduler" in ck:
                scheduler.load_state_dict(ck["scheduler"])
            # best 기준은 val steer (실제 구동 신호). 옛 체크포인트는 val_steer 키가 있음.
            if isinstance(ck, dict) and "val_steer" in ck:
                best_val = float(ck["val_steer"])
            print(f"resumed from {args.resume} (epoch={ck.get('epoch')} best_steer={best_val:.4f})")

    args_meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args_meta["cache"] = cache_paths

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(va["steer"])   # plateau: best 기준과 동일 신호로 감쇠
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"ep {epoch:3d}/{args.epochs}  "
              f"train {tr['total']:.4f} (s{tr['steer']:.4f} t{tr['throttle']:.4f} w{tr['wp']:.4f})  "
              f"val {va['total']:.4f} (s{va['steer']:.4f} t{va['throttle']:.4f} w{va['wp']:.4f})  "
              f"lr {lr_now:.2e}  {dt:.0f}s")

        # best/early-stop 은 val *steer* 기준. steer 만 실제 구동에 쓰이므로(throttle 은
        # |steer| 로 재구성, waypoint 는 추론 미사용) total 로 고르면 잡음 큰 wp/throttle
        # 이 steer-최적 아닌 epoch 를 고른다.
        improved = va["steer"] < best_val - 1e-5
        if improved:
            best_val = va["steer"]
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),    # --resume 용
                "scheduler": scheduler.state_dict(),    # --resume 시 LR 복원
                "epoch": epoch,
                "val_total": va["total"],
                "val_steer": va["steer"],
                "val_throttle": va["throttle"],
                "val_wp": va["wp"],
                "args": args_meta,
            }, args.out)
            print(f"    ↳ new best val_steer={best_val:.4f}  saved {args.out}")
        else:
            bad += 1

        # 의도 시각화: 켜졌을 때만, 주기적으로 + 새 best 마다.
        if args.viz_dir is not None and (improved or epoch % args.viz_every == 0):
            save_viz(model, val_ds, args.viz_dir, epoch, device)

        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (no val improvement for {args.patience})")
            break

    print(f"done. best val_steer={best_val:.4f}  -> {args.out}")
    print("next: export_onnx.py --ckpt", args.out)


if __name__ == "__main__":
    main()
