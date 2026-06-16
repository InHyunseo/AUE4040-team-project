"""Lightweight labels_cache.h5 audit before E2E training.

This intentionally avoids torch so it can run in local, Jetson, or Kaggle
environments before choosing the training flags.

Examples:
  python3 audit_h5.py --cache labels_all/*.h5
  python3 audit_h5.py --cache legacy/*.h5 --wp_fix_sign
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED = {
    "lane": (224, 224, 3),
    "front": (224, 224, 3),
    "seg": (3, 224, 224),
    "det": (5,),
    "waypoint": (5, 2),
    "steer": (),
    "throttle": (),
    "timestamp_ns": (),
}


def _fmt(v: float) -> str:
    return f"{v:+.4f}"


def audit_one(path: Path, wp_fix_sign: bool) -> bool:
    ok = True
    with h5py.File(path, "r") as h5:
        print(f"\n{path}")
        missing = [k for k in REQUIRED if k not in h5]
        if missing:
            print(f"  ERROR missing datasets: {missing}")
            return False

        n = int(h5["lane"].shape[0])
        print(f"  samples: {n}")
        for key, tail in REQUIRED.items():
            shape = tuple(h5[key].shape[1:])
            if shape != tail:
                print(f"  ERROR {key} shape tail {shape} != {tail}")
                ok = False

        attrs = {
            k: h5.attrs[k].tolist() if hasattr(h5.attrs[k], "tolist") else h5.attrs[k]
            for k in h5.attrs
        }
        print(f"  attrs: {attrs}")

        steer = h5["steer"][:].astype(np.float32)
        throttle = h5["throttle"][:].astype(np.float32)
        det_conf = h5["det"][:, 4].astype(np.float32)
        wp = h5["waypoint"][:].astype(np.float32)
        wp_eff = wp.copy()
        if wp_fix_sign:
            wp_eff[:, :, 0] *= -1.0

        zero = (np.abs(steer) < 1e-6) & (np.abs(throttle) < 1e-6)
        det_mask = det_conf > 0
        end_raw = wp[:, -1, :]
        end_eff = wp_eff[:, -1, :]

        print(
            "  steer raw    min/max/mean/std: "
            f"{_fmt(float(steer.min()))} {_fmt(float(steer.max()))} "
            f"{_fmt(float(steer.mean()))} {float(steer.std()):.4f}"
        )
        print(
            "  throttle     min/max/mean/std: "
            f"{_fmt(float(throttle.min()))} {_fmt(float(throttle.max()))} "
            f"{_fmt(float(throttle.mean()))} {float(throttle.std()):.4f}"
        )
        print(
            "  zero-control ratio: "
            f"{float(zero.mean()):.3f}  det ratio: {float(det_mask.mean()):.3f}"
        )
        if det_mask.any():
            print(f"  det conf mean/max: {float(det_conf[det_mask].mean()):.3f} {float(det_conf.max()):.3f}")
        print(
            "  waypoint end raw x/y mean: "
            f"{_fmt(float(end_raw[:, 0].mean()))} {_fmt(float(end_raw[:, 1].mean()))}"
        )
        print(
            f"  waypoint end effective x/y mean (wp_fix_sign={wp_fix_sign}): "
            f"{_fmt(float(end_eff[:, 0].mean()))} {_fmt(float(end_eff[:, 1].mean()))}"
        )
        print(
            "  waypoint effective |y| mean/max: "
            f"{float(np.abs(end_eff[:, 1]).mean()):.4f} {float(np.abs(end_eff[:, 1]).max()):.4f}"
        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True, type=Path,
                    help="one or more labels_cache.h5 files")
    ap.add_argument("--wp_fix_sign", action="store_true",
                    help="show waypoint stats after legacy x-axis sign fix")
    args = ap.parse_args()

    all_ok = True
    for path in args.cache:
        all_ok = audit_one(path, args.wp_fix_sign) and all_ok
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
