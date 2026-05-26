"""BEV perspective calibration using a checkerboard.

Outputs calib.json with:
  M:                  3x3 perspective matrix (source image -> BEV plane)
  pixels_per_meter:   scalar (BEV plane uses isotropic scale)
  bev_size:           [W, H]  (BEV output resolution; default 224x224)
  source_image_size:  [W, H]
  method:             "checkerboard" or "manual"

Auto mode (default):
  python bev_calibration.py --image checker.jpg --rows 6 --cols 9 --square_m 0.025
Manual fallback:
  python bev_calibration.py --image checker.jpg --manual --width_m 0.30 --height_m 0.30

No dependency on main/. OpenCV + numpy only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def calibrate_auto(img: np.ndarray, rows: int, cols: int, square_m: float,
                   bev_w: int, bev_h: int, square_px: int = 24):
    """Detect checkerboard, fit perspective from 4 outer corners."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if not found:
        raise RuntimeError(
            f"findChessboardCorners failed for rows={rows} cols={cols}. "
            "Try --manual, or verify the inner-corner counts (rows/cols are INNER corners)."
        )

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    pts = corners.reshape(rows, cols, 2)

    # Outer 4 corners in image: TL, TR, BR, BL
    src = np.float32([
        pts[0, 0], pts[0, -1], pts[-1, -1], pts[-1, 0],
    ])

    # Place the checkerboard at the bottom-center of the BEV plane,
    # rows of squares going UP from near (bottom) to far (top).
    board_w_px = (cols - 1) * square_px
    board_h_px = (rows - 1) * square_px
    cx = bev_w // 2
    bottom_margin = bev_h // 8
    x0 = cx - board_w_px // 2
    x1 = cx + board_w_px // 2
    y1 = bev_h - bottom_margin           # near edge (bottom)
    y0 = y1 - board_h_px                 # far edge (top)
    dst = np.float32([
        [x0, y0],  # TL (far-left)
        [x1, y0],  # TR (far-right)
        [x1, y1],  # BR (near-right)
        [x0, y1],  # BL (near-left)
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    pixels_per_meter = float(square_px / square_m)
    return M, pixels_per_meter, src, dst


def calibrate_manual(img: np.ndarray, width_m: float, height_m: float,
                     bev_w: int, bev_h: int):
    """Click 4 points (TL, TR, BR, BL) on a known rectangle in the ground plane."""
    clicks: list[tuple[int, int]] = []
    label = ["TL (far-left)", "TR (far-right)", "BR (near-right)", "BL (near-left)"]
    disp = img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(disp, label[len(clicks) - 1], (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    win = "click 4 corners: TL, TR, BR, BL  (Enter=confirm, Esc=cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(clicks) == 4:
            break
        if k == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("manual calibration cancelled")
    cv2.destroyWindow(win)

    # Pick px-per-meter so the rectangle fits nicely in BEV
    ppm = min((bev_w * 0.6) / width_m, (bev_h * 0.6) / height_m)
    w_px = width_m * ppm
    h_px = height_m * ppm
    cx = bev_w // 2
    bottom_margin = bev_h // 8
    x0 = cx - w_px / 2
    x1 = cx + w_px / 2
    y1 = bev_h - bottom_margin
    y0 = y1 - h_px

    src = np.float32(clicks)
    dst = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, float(ppm), src, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "calib" / "calib.json")
    ap.add_argument("--bev_w", type=int, default=224)
    ap.add_argument("--bev_h", type=int, default=224)
    # auto mode
    ap.add_argument("--rows", type=int, default=6, help="inner-corner rows")
    ap.add_argument("--cols", type=int, default=9, help="inner-corner cols")
    ap.add_argument("--square_m", type=float, default=0.025)
    ap.add_argument("--square_px", type=int, default=24,
                    help="how many BEV pixels per checkerboard square (sets ppm with square_m)")
    # manual mode
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--width_m", type=float, default=0.30)
    ap.add_argument("--height_m", type=float, default=0.30)
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(args.image)
    H_src, W_src = img.shape[:2]

    method = "manual" if args.manual else "checkerboard"
    if args.manual:
        M, ppm, src, dst = calibrate_manual(
            img, args.width_m, args.height_m, args.bev_w, args.bev_h)
    else:
        try:
            M, ppm, src, dst = calibrate_auto(
                img, args.rows, args.cols, args.square_m,
                args.bev_w, args.bev_h, args.square_px)
        except RuntimeError as e:
            print(f"[auto failed] {e}\nfalling back to manual click")
            M, ppm, src, dst = calibrate_manual(
                img, args.width_m, args.height_m, args.bev_w, args.bev_h)
            method = "manual"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "M": M.tolist(),
        "pixels_per_meter": ppm,
        "bev_size": [args.bev_w, args.bev_h],
        "source_image_size": [W_src, H_src],
        "method": method,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    print(f"  pixels_per_meter = {ppm:.2f}")

    # Debug overlays
    dbg_dir = args.out.parent
    warped = cv2.warpPerspective(img, M, (args.bev_w, args.bev_h))
    cv2.imwrite(str(dbg_dir / "calib_warped.png"), warped)

    overlay = img.copy()
    for (x, y) in src.astype(int):
        cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
    cv2.polylines(overlay, [src.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(str(dbg_dir / "calib_source.png"), overlay)
    print(f"  debug: {dbg_dir/'calib_warped.png'}, {dbg_dir/'calib_source.png'}")


if __name__ == "__main__":
    main()
