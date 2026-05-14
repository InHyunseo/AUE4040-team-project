"""
Split raw session data into common/left/right subsets by segment label.

Input:  data/raw/<session>/{images/, annotation.txt}
        annotation.txt rows: filename steer speed segment
Output: data/processed/<segment>/{images/, annotation.txt}
        annotation.txt rows: filename xpos ypos     (CenterDataset format)

Center labels are filled by the labeler notebook; this script only routes
files into the right buckets.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    for seg in ("common", "left", "right"):
        (args.out / seg / "images").mkdir(parents=True, exist_ok=True)

    for session in args.raw.iterdir():
        ann = session / "annotation.txt"
        if not ann.exists():
            continue
        for line in ann.read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            fname, _steer, _speed, seg = parts[0], parts[1], parts[2], parts[3]
            if seg not in {"common", "left", "right"}:
                continue
            src = session / "images" / Path(fname).name
            dst = args.out / seg / "images" / src.name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
