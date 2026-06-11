"""Overlay seg / bbox / waypoints for a single sample from labels_cache.h5.

  python visualize_labels.py --cache ../labels_cache.h5 --idx 0 \
      --out ../debug_samples/viz_000.png

Does not import from main/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


SEG_COLORS = [(0, 0, 255),     # left-solid    - red
              (0, 255, 0),     # right-solid   - green
              (255, 0, 0)]     # center-dashed - blue

# Display-only scale for drawing metric waypoints (labels are in meters).
# WP_HORIZON_S 2.5s 면 끝점 ~0.55m. 기존 200 ppm 으로는 점이 224px 밖으로
# 나가므로 180 으로 낮춰 0.55m(≈99px)가 화면 안에 들어오게 한다.
# (디스플레이 전용 — 라벨/학습 좌표는 미터 단위 그대로라 영향 없음.)
DEBUG_PPM = 180.0


def overlay_seg(lane: np.ndarray, seg: np.ndarray) -> np.ndarray:
    out = lane.copy()
    for c in range(len(SEG_COLORS)):
        m = seg[c] > 0
        if not m.any():
            continue
        out[m] = (out[m] * 0.4 + np.array(SEG_COLORS[c]) * 0.6).astype(np.uint8)
    return out


def draw_waypoints(lane: np.ndarray, wps: np.ndarray, ppm: float) -> np.ndarray:
    H, W = lane.shape[:2]
    ox, oy = W // 2, H - H // 8
    pts = []
    for (x_m, y_m) in wps:
        u = int(ox - y_m * ppm)   # +y_m (left) -> -u
        v = int(oy - x_m * ppm)   # +x_m (forward) -> -v
        pts.append((u, v))
    out = lane.copy()
    cv2.circle(out, (ox, oy), 4, (255, 255, 255), 1)
    for i, (u, v) in enumerate(pts):
        cv2.circle(out, (u, v), 4, (255, 255, 255), -1)
        cv2.putText(out, str(i + 1), (u + 5, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(out, a, b, (255, 255, 255), 1)
    return out


def draw_bbox(front: np.ndarray, det: np.ndarray) -> np.ndarray:
    out = front.copy()
    x, y, w, h, conf = det
    if conf <= 0:
        cv2.putText(out, "no det", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        return out
    cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                  (0, 255, 0), 2)
    cv2.putText(out, f"car {conf:.2f}", (int(x), max(0, int(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "debug_samples" / "viz.png")
    args = ap.parse_args()

    with h5py.File(args.cache, "r") as h5:
        n = h5["lane"].shape[0]
        if not (0 <= args.idx < n):
            raise IndexError(f"idx {args.idx} out of range [0,{n})")
        lane = h5["lane"][args.idx]
        front = h5["front"][args.idx]
        seg = h5["seg"][args.idx]
        det = h5["det"][args.idx]
        wps = h5["waypoint"][args.idx]
        steer = float(h5["steer"][args.idx])
        thr = float(h5["throttle"][args.idx])

    lane_vis = draw_waypoints(overlay_seg(lane, seg), wps, DEBUG_PPM)
    front_vis = draw_bbox(front, det)

    # match heights
    if lane_vis.shape[0] != front_vis.shape[0]:
        h = max(lane_vis.shape[0], front_vis.shape[0])
        def pad(im):
            ph = h - im.shape[0]
            if ph <= 0:
                return im
            return np.vstack([im, np.zeros((ph, im.shape[1], 3), np.uint8)])
        lane_vis = pad(lane_vis); front_vis = pad(front_vis)

    panel = np.hstack([lane_vis, front_vis])
    cv2.putText(panel, f"idx={args.idx}  steer(w)={steer:.2f}  throttle(v)={thr:.2f}",
                (5, panel.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), panel)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
