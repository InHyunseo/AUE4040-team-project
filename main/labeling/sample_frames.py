"""
Sample frames from one or more recording sessions for labeling.

Picks every Nth frame from each session's left/ folder, drops
out-of-focus frames (Laplacian variance < threshold), and copies
the survivors into one flat output directory with session-prefixed
names so CVAT shows the source plainly.

Usage:
  python3 sample_frames.py \
      --src ~/rover_data/test1_20260520_161431 ~/rover_data/test2_* \
      --dst ~/rover_data/_to_label \
      --stride 5 \
      --blur-thresh 80

Defaults assume 10 Hz recording: stride 5 -> ~2 Hz labeled rate.
"""
import argparse
import shutil
from glob import glob
from pathlib import Path

import cv2


def laplacian_var(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="session dirs (each must contain images/ from"
                         " record_and_label.ipynb)")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--subdir", default="images",
                    help="frame subdir inside each session (default: images,"
                         " matching record_and_label.ipynb output)")
    ap.add_argument("--stride", type=int, default=5,
                    help="keep 1 of every N frames")
    ap.add_argument("--blur-thresh", type=float, default=80.0,
                    help="reject if Laplacian variance below this")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (default: copy)")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped_blur = 0
    skipped_stride = 0

    sessions = []
    for s in args.src:
        for p in sorted(glob(s)):
            sessions.append(Path(p))
    if not sessions:
        raise SystemExit(f"no sessions matched: {args.src}")

    for sess in sessions:
        sub = sess / args.subdir
        if not sub.is_dir():
            print(f"[skip] {sess}: no {args.subdir}/")
            continue
        frames = sorted(sub.glob("*.jpg"))
        sess_tag = sess.name
        for i, f in enumerate(frames):
            if i % args.stride != 0:
                skipped_stride += 1
                continue
            lv = laplacian_var(f)
            if lv < args.blur_thresh:
                dropped_blur += 1
                continue
            out = args.dst / f"{sess_tag}__{f.stem}.jpg"
            if args.move:
                shutil.move(str(f), out)
            else:
                shutil.copy2(f, out)
            kept += 1
        print(f"[{sess_tag}] frames={len(frames)} kept={kept}")

    print(f"\n[done] kept={kept}  dropped_blur={dropped_blur}  "
          f"skipped_stride={skipped_stride}")
    print(f"[done] output dir: {args.dst}")


if __name__ == "__main__":
    main()
