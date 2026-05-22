"""
Pseudo-label sampled frames with COCO-pretrained YOLOv8.

Auto-fills the classes COCO already knows about:
  COCO  9 (traffic light) -> red / yellow / green via HSV inside the bbox
  COCO 11 (stop sign)     -> stop_sign
  COCO  2 (car) + 7 (truck) -> roundabout_vehicle  (size-filtered)

The remaining four classes (person_sign / left_arrow_sign /
right_arrow_sign and missed instances of the above) get drawn by
hand in CVAT after import.

Output: writes <stem>.txt next to each <stem>.jpg in YOLO format
        (cls cx cy w h, normalized). Also writes classes.txt copy.

Usage:
  python3 pseudo_label.py \
      --src ~/rover_data/_to_label \
      --weights yolov8n.pt \
      --car-min-area 0.005   # bbox area / image area; tune in-situ
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASSES = [
    "traffic_light_red",
    "traffic_light_green",
    "traffic_light_yellow",
    "stop_sign",
    "vehicle",
    "turn_left_sign",
    "turn_right_sign",
    "person_sign",
]
CLS_ID = {n: i for i, n in enumerate(CLASSES)}

COCO_TRAFFIC_LIGHT = 9
COCO_STOP_SIGN = 11
COCO_CAR = 2
COCO_TRUCK = 7


def classify_light_color(bgr_crop: np.ndarray) -> str:
    """Return 'red' / 'yellow' / 'green' from a traffic-light bbox crop."""
    if bgr_crop.size == 0:
        return "red"
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (v > 120) & (s > 80)
    masks = {
        "red":    bright & (((h <= 10) | (h >= 170))),
        "yellow": bright & ((h >= 18) & (h <= 35)),
        "green":  bright & ((h >= 40) & (h <= 90)),
    }
    counts = {k: int(m.sum()) for k, m in masks.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 20 else "red"  # fallback: red (safer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="dir of .jpg frames (output of sample_frames.py)")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--car-min-area", type=float, default=0.005,
                    help="reject cars/trucks smaller than this fraction"
                         " of the image (filters distant background traffic)")
    ap.add_argument("--device", default=None,
                    help="'cpu', '0', etc. None = ultralytics default")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"src not a dir: {args.src}")

    classes_dst = args.src / "classes.txt"
    classes_dst.write_text("\n".join(CLASSES) + "\n")
    print(f"[info] wrote {classes_dst}")

    yolo = YOLO(args.weights)
    images = sorted(args.src.glob("*.jpg"))
    print(f"[info] {len(images)} images")

    counts = {c: 0 for c in CLASSES}
    n_empty = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        results = yolo.predict(
            source=str(img_path), conf=args.conf, verbose=False,
            device=args.device,
        )[0]

        lines = []
        for box, cls_id_t, _conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy().astype(int),
            results.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(W, int(x2)); y2 = min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            bw = x2 - x1
            bh = y2 - y1
            our_cls = None

            if cls_id_t == COCO_STOP_SIGN:
                our_cls = "stop_sign"
            elif cls_id_t == COCO_TRAFFIC_LIGHT:
                color = classify_light_color(img[y1:y2, x1:x2])
                our_cls = f"traffic_light_{color}"
            elif cls_id_t in (COCO_CAR, COCO_TRUCK):
                if (bw * bh) / (W * H) >= args.car_min_area:
                    our_cls = "vehicle"
            if our_cls is None:
                continue

            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            nw = bw / W
            nh = bh / H
            lines.append(f"{CLS_ID[our_cls]} {cx:.6f} {cy:.6f} "
                         f"{nw:.6f} {nh:.6f}")
            counts[our_cls] += 1

        txt_path = img_path.with_suffix(".txt")
        if lines:
            txt_path.write_text("\n".join(lines) + "\n")
        else:
            txt_path.write_text("")
            n_empty += 1

    print("\n[counts]")
    for c, n in counts.items():
        print(f"  {c:24s} {n}")
    print(f"  (empty frames: {n_empty} / {len(images)})")
    print(f"\n[done] labels written next to images in {args.src}")
    print("[next] zip and import to CVAT as 'YOLO 1.1' format.")


if __name__ == "__main__":
    main()
