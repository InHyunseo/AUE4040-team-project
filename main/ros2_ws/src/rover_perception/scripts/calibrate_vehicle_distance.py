"""
One-shot calibration for vehicle bbox-height distance.

Closed-world: a single fixed-size car. We model distance as
    d = K / bbox_h_px        with K = bbox_h_px * d  (px * m)

Procedure:
  1. Park the car at a known distance `d` in front of the camera.
  2. Run this script. It opens CSI camera 0, shows one frame.
  3. Click the top-left of the car's bbox, then the bottom-right.
  4. Script prints K and (with --write) updates rover_bringup params.yaml.

Usage:
    python3 calibrate_vehicle_distance.py --distance 0.5
    python3 calibrate_vehicle_distance.py --distance 0.5 --write
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

# Reuse the same jetcam wrapper the recorder notebook uses.
CALIB_ROOT = Path.home() / "team" / "calibration"
if str(CALIB_ROOT) not in sys.path:
    sys.path.insert(0, str(CALIB_ROOT))
from camera import Camera  # noqa: E402


PARAMS_YAML = (Path.home() / "team" / "main" / "ros2_ws" /
               "src" / "rover_bringup" / "config" / "params.yaml")


def grab_frame(sensor_id: int = 0):
    cam = Camera(sensor_id)
    # Discard a few warmup frames.
    frame = None
    for _ in range(5):
        frame = cam.read()
        time.sleep(0.05)
    cam.stop()
    if frame is None:
        raise RuntimeError("camera produced no frames")
    # jetcam wrapper returns RGB; convert to BGR for cv2.imshow consistency.
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def pick_bbox(frame_bgr) -> tuple:
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    win = "click TL, then BR of the car's bbox  (q=abort)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_click)
    while True:
        disp = frame_bgr.copy()
        for pt in clicks:
            cv2.circle(disp, pt, 4, (0, 255, 0), -1)
        if len(clicks) == 2:
            cv2.rectangle(disp, clicks[0], clicks[1], (0, 255, 0), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if len(clicks) >= 2:
            cv2.imshow(win, disp)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            return clicks[0], clicks[1]


def update_params_yaml(K: float) -> None:
    text = PARAMS_YAML.read_text()
    new_lines = []
    written = False
    for line in text.splitlines():
        if line.strip().startswith("vehicle_dist_K:"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}vehicle_dist_K: {K:.2f}")
            written = True
        else:
            new_lines.append(line)
    if not written:
        raise RuntimeError(f"vehicle_dist_K key not found in {PARAMS_YAML}")
    PARAMS_YAML.write_text("\n".join(new_lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", required=True, type=float,
                    help="known distance to the car in meters")
    ap.add_argument("--sensor-id", default=0, type=int)
    ap.add_argument("--write", action="store_true",
                    help="update vehicle_dist_K in rover_bringup params.yaml")
    args = ap.parse_args()

    frame = grab_frame(args.sensor_id)
    tl, br = pick_bbox(frame)
    bbox_h = abs(br[1] - tl[1])
    if bbox_h < 1:
        raise SystemExit("degenerate bbox height")
    K = bbox_h * args.distance
    print(f"bbox_h = {bbox_h} px")
    print(f"d      = {args.distance} m")
    print(f"K      = bbox_h * d = {K:.2f}  (px * m)")
    print(f"=> at safe_dist=0.4 m, threshold bbox_h = {K / 0.4:.1f} px")

    if args.write:
        update_params_yaml(K)
        print(f"wrote vehicle_dist_K = {K:.2f} to {PARAMS_YAML}")


if __name__ == "__main__":
    main()
