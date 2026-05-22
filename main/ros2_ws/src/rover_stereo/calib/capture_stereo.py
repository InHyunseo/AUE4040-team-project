"""
Checkerboard capture for stereo calibration.

Adapted from HYU-ECL3003/stereo_depth_tutorial/jetson-stereo-depth/calib/
capture-stereo.py. Keep ~30-50 pairs covering different angles + positions.
Press SPACE to save a pair, q to quit.

Output: out_dir/left/*.png, out_dir/right/*.png
"""
import argparse
from pathlib import Path

import cv2
import sys

# Vendored jetcam under team/calibration/camera/jetcam (carries sync=false fix).
sys.path.insert(0, str(Path.home() / "team" / "calibration" / "camera"))
from jetcam.csi_camera import CSICamera  # type: ignore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("calib_data"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    (args.out / "left").mkdir(parents=True, exist_ok=True)
    (args.out / "right").mkdir(parents=True, exist_ok=True)

    cam0 = CSICamera(capture_device=0, capture_width=args.width,
                     capture_height=args.height, downsample=2, capture_fps=30)
    cam1 = CSICamera(capture_device=1, capture_width=args.width,
                     capture_height=args.height, downsample=2, capture_fps=30)

    idx = 0
    while True:
        l = cv2.cvtColor(cam0.read(), cv2.COLOR_RGB2BGR)
        r = cv2.cvtColor(cam1.read(), cv2.COLOR_RGB2BGR)
        cv2.imshow("stereo (SPACE=save, q=quit)", cv2.hconcat([l, r]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:  # SPACE
            cv2.imwrite(str(args.out / "left" / f"{idx:03d}.png"), l)
            cv2.imwrite(str(args.out / "right" / f"{idx:03d}.png"), r)
            print(f"saved {idx}")
            idx += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
