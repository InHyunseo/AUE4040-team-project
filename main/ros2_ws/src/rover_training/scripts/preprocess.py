"""
Split raw session data into common/left/right subsets by segment label.

Input:  data/raw/<session>/{images/, annotation.txt}
        annotation.txt rows: filename xpos ypos segment steer_tel speed_tel
        (6 cols, as written by record_and_label.ipynb)
Output: data/processed/<segment>/{images/, annotation.txt}
        annotation.txt rows: filename xpos ypos     (CenterDataset format)

Rows with xpos == "-1" (unlabeled) are skipped. Run Part B of the labeler
notebook first to fill in xpos/ypos.
"""
import argparse
import shutil
from pathlib import Path


SEGMENTS = ("common", "left", "right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out_files = {}
    for seg in SEGMENTS:
        (args.out / seg / "images").mkdir(parents=True, exist_ok=True)
        out_files[seg] = open(args.out / seg / "annotation.txt", "w")

    routed = {s: 0 for s in SEGMENTS}
    skipped_unlabeled = 0
    skipped_malformed = 0

    for session in sorted(args.raw.iterdir()):
        ann = session / "annotation.txt"
        if not ann.exists():
            continue
        for line in ann.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                skipped_malformed += 1
                continue
            fname, xpos, ypos, seg = parts[0], parts[1], parts[2], parts[3]
            if seg not in SEGMENTS:
                skipped_malformed += 1
                continue
            if xpos == "-1" or ypos == "-1":
                skipped_unlabeled += 1
                continue
            src = session / "images" / Path(fname).name
            dst = args.out / seg / "images" / src.name
            if not src.exists():
                skipped_malformed += 1
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
            out_files[seg].write(f"{src.name} {xpos} {ypos}\n")
            routed[seg] += 1

    for f in out_files.values():
        f.close()

    for seg in SEGMENTS:
        print(f"{seg}: {routed[seg]} frames")
    if skipped_unlabeled:
        print(f"skipped (unlabeled, xpos=-1): {skipped_unlabeled}")
    if skipped_malformed:
        print(f"skipped (malformed/missing): {skipped_malformed}")


if __name__ == "__main__":
    main()
