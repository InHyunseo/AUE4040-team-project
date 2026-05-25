"""
Split raw session data into common/left/right subsets by segment label.

Input:  data/raw/<session>/{images/, annotation.txt}
        annotation.txt rows: filename xpos ypos segment steer_tel speed_tel last_key step
        (8 cols, as written by record_and_label.ipynb)
Output: data/processed/<segment>/{images/, annotation.txt}
        annotation.txt rows: filename action_idx step   (ActionDataset format)

Action map: up=0, down=1, left=2, right=3, straight=4, space=5
Rows with last_key=="none" (no key pressed yet) are skipped.
Old-format rows (<8 cols) are also skipped.
`down` and `space` frames are teleop corrections (reverse / stop) — including
them as BC targets makes the policy emit reverse mid-course. Skipped by default;
pass --keep-correction to include them.
"""
import argparse
import shutil
from pathlib import Path


SEGMENTS = ("common", "left", "right")
ACTION_MAP = {"up": 0, "down": 1, "left": 2, "right": 3, "straight": 4, "space": 5}
CORRECTION_KEYS = {"down", "space"}

# rover_training/scripts/preprocess.py -> rover_training/
PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path("~/rover_data").expanduser()
DEFAULT_OUT = PKG_ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                    help=f"raw sessions root (default: {DEFAULT_RAW})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"processed segments root (default: {DEFAULT_OUT})")
    ap.add_argument("--keep-correction", action="store_true",
                    help="include down/space frames (teleop corrections)")
    args = ap.parse_args()

    out_files = {}
    for seg in SEGMENTS:
        (args.out / seg / "images").mkdir(parents=True, exist_ok=True)
        out_files[seg] = open(args.out / seg / "annotation.txt", "w")

    routed = {s: 0 for s in SEGMENTS}
    class_counts = {s: {k: 0 for k in ACTION_MAP} for s in SEGMENTS}
    skipped_no_key = 0
    skipped_malformed = 0
    skipped_correction = 0

    for session in sorted(args.raw.iterdir()):
        ann = session / "annotation.txt"
        if not ann.exists():
            continue
        for line in ann.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 8:
                skipped_malformed += 1
                continue
            fname, _xp, _yp, seg, _st, _sp, key, step = parts[:8]
            if seg not in SEGMENTS:
                skipped_malformed += 1
                continue
            if key not in ACTION_MAP:
                skipped_no_key += 1
                continue
            if key in CORRECTION_KEYS and not args.keep_correction:
                skipped_correction += 1
                continue
            src = session / "images" / Path(fname).name
            dst = args.out / seg / "images" / src.name
            if not src.exists():
                skipped_malformed += 1
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
            out_files[seg].write(f"{src.name} {ACTION_MAP[key]} {step}\n")
            routed[seg] += 1
            class_counts[seg][key] += 1

    for f in out_files.values():
        f.close()

    for seg in SEGMENTS:
        print(f"{seg}: {routed[seg]} frames  {class_counts[seg]}")
    if skipped_no_key:
        print(f"skipped (no key pressed yet): {skipped_no_key}")
    if skipped_correction:
        print(f"skipped (down/space corrections): {skipped_correction}")
    if skipped_malformed:
        print(f"skipped (malformed/missing): {skipped_malformed}")


if __name__ == "__main__":
    main()
