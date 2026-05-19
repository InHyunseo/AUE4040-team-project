"""
Interactive stereo capture for chessboard calibration.

Usage:
    cd ~/team/calibration
    python3 calib/capture_stereo.py

Controls:
    space  save a paired (left, right) PNG to calib/calib_images/{left,right}/NNN.png
    q      quit
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make `from camera import Camera` work no matter where this is launched from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from camera import Camera  # noqa: E402


OUT_LEFT = ROOT / "calib" / "calib_images" / "left"
OUT_RIGHT = ROOT / "calib" / "calib_images" / "right"


def next_index(folder: Path) -> int:
    existing = [int(p.stem) for p in folder.glob("*.png") if p.stem.isdigit()]
    return max(existing) + 1 if existing else 0


def main():
    OUT_LEFT.mkdir(parents=True, exist_ok=True)
    OUT_RIGHT.mkdir(parents=True, exist_ok=True)

    print("Opening cameras ...")
    cam_l = Camera(0)
    cam_r = Camera(1)

    idx = max(next_index(OUT_LEFT), next_index(OUT_RIGHT))
    print(f"Capturing pairs starting at index {idx:03d}.")
    print("space=save  q=quit")

    win = "Stereo capture (space=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame_l = cam_l.read()  # RGB
            frame_r = cam_r.read()  # RGB
            if frame_l is None or frame_r is None:
                time.sleep(0.01)
                continue

            # Convert to BGR for OpenCV display/IO
            frame_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
            frame_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()
            label = f"saved: {idx:03d}"
            cv2.putText(disp_l, "L  " + label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(disp_r, "R", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            combined = np.hstack([disp_l, disp_r])
            h = combined.shape[0]
            if h > 720:
                scale = 720 / h
                combined = cv2.resize(combined, (int(combined.shape[1] * scale), 720))
            cv2.imshow(win, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                p_l = OUT_LEFT / f"{idx:03d}.png"
                p_r = OUT_RIGHT / f"{idx:03d}.png"
                cv2.imwrite(str(p_l), frame_l)
                cv2.imwrite(str(p_r), frame_r)
                print(f"saved pair {idx:03d}  ->  {p_l}  &  {p_r}")
                idx += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cam_l.stop()
        cam_r.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
