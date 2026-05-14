"""
Run stereo calibration from captured checkerboard pairs.

Output: config/stereo_calib.yaml — commit to git and DO NOT regenerate
after Phase 3 data collection starts (see PROJECT_PLAN.md §9.5).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_corners(images_dir: Path, board: tuple, square_m: float):
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_m
    obj_points, img_points, paths = [], [], []
    for p in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, board, None)
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
            )
            obj_points.append(objp.copy())
            img_points.append(corners)
            paths.append(p.name)
    return obj_points, img_points, paths, gray.shape[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="calib_data dir with left/, right/ subfolders")
    ap.add_argument("--board", type=int, nargs=2, default=[9, 6],
                    help="inner corners per row, column")
    ap.add_argument("--square", type=float, default=0.025,
                    help="checkerboard square size in meters")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    objL, imgL, pathsL, size = find_corners(args.data / "left", tuple(args.board), args.square)
    objR, imgR, pathsR, _ = find_corners(args.data / "right", tuple(args.board), args.square)

    common = sorted(set(pathsL) & set(pathsR))
    if len(common) < 15:
        print(f"Need >= 15 valid pairs, got {len(common)}", file=sys.stderr)
        sys.exit(1)
    idxL = [pathsL.index(p) for p in common]
    idxR = [pathsR.index(p) for p in common]
    objp = [objL[i] for i in idxL]
    imgL = [imgL[i] for i in idxL]
    imgR = [imgR[i] for i in idxR]

    _, K1, D1, _, _ = cv2.calibrateCamera(objp, imgL, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(objp, imgR, size, None, None)
    err, K1, D1, K2, D2, R, T, *_ = cv2.stereoCalibrate(
        objp, imgL, imgR, K1, D1, K2, D2, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"stereo reprojection error: {err:.4f} px")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({
        "image_size": list(size),
        "K1": K1.tolist(), "D1": D1.tolist(),
        "K2": K2.tolist(), "D2": D2.tolist(),
        "R": R.tolist(), "T": T.tolist(),
        "reprojection_error_px": float(err),
        "frozen": True,
        "note": ("FROZEN before Phase 3 data collection. Do not regenerate "
                 "without re-collecting BC training data."),
    }, sort_keys=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
