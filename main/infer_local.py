"""
Standalone YOLO inference on local images/video — no ROS, no TRT.

Usage:
    python infer_local.py                          # main/inference/*.jpg
    python infer_local.py path/to/img_or_dir       # custom input
    python infer_local.py video.mp4 --conf 0.3
    python infer_local.py --source 0               # webcam (if /dev/video0 available)

Outputs annotated images/video under main/inference_out/.
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "best.pt"
DEFAULT_SOURCE = HERE / "inference"
DEFAULT_OUT = HERE / "inference_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE),
                    help="image / dir / video / '0' for webcam")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src = args.source
    if src.isdigit():
        src = int(src)

    model = YOLO(args.weights)
    print(f"loaded {args.weights} | nc={len(model.names)} | names={model.names}")

    results = model.predict(
        source=src,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.out,
        name="run",
        exist_ok=True,
        verbose=False,
    )

    # Per-image summary
    for r in results:
        path = getattr(r, "path", "?")
        names = r.names
        if r.boxes is None or len(r.boxes) == 0:
            print(f"  {os.path.basename(str(path))}: (no detections)")
            continue
        items = []
        for b in r.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            items.append(f"{names[cls]}={conf:.2f}")
        print(f"  {os.path.basename(str(path))}: {', '.join(items)}")

    print(f"\nannotated outputs → {args.out}/run/")


if __name__ == "__main__":
    main()
