"""
Split raw session data into common/left/right subsets by segment label.

Input:  data/raw/<session>/{images/, annotation.txt}
        annotation.txt rows: filename xpos ypos segment steer_tel speed_tel last_key step
        (8 cols, as written by record_and_label.ipynb)
Output: data/processed/<segment>/{images/, annotation.txt, test_annotation.txt}
        annotation rows: filename action_idx step   (ActionDataset format)

Train/test split is per-session: within each segment, the last ~test_ratio of
the sessions (sorted by name) are held out as test. Random row-level split
would leak — adjacent frames in the same session are nearly identical, so
random test rows would have near-duplicates in train and overstate test acc.

Action map: up=0, down=1, left=2, right=3, straight=4, space=5
Rows with last_key=="none" (no key pressed yet) are skipped.
Old-format rows (<8 cols) are also skipped.
`down` and `space` frames are teleop corrections (reverse / stop) — including
them as BC targets makes the policy emit reverse mid-course. Skipped by default;
pass --keep-correction to include them.
"""
import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


SEGMENTS = ("common", "left", "right")
ACTION_MAP = {"up": 0, "down": 1, "left": 2, "right": 3, "straight": 4, "space": 5}
CORRECTION_KEYS = {"down", "space"}

# rover_training/scripts/preprocess.py -> rover_training/
PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path("~/rover_data").expanduser()
DEFAULT_OUT = PKG_ROOT / "data" / "processed"


def parse_session(ann_path: Path, keep_correction: bool):
    """Yield (fname, action_idx, step_int, segment) for valid rows in a session.

    Also returns counters for skipped rows so the caller can report totals.
    """
    rows = []
    skipped = {"no_key": 0, "malformed": 0, "correction": 0}
    for line in ann_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 8:
            skipped["malformed"] += 1
            continue
        fname, _xp, _yp, seg, _st, _sp, key, step = parts[:8]
        if seg not in SEGMENTS:
            skipped["malformed"] += 1
            continue
        if key not in ACTION_MAP:
            skipped["no_key"] += 1
            continue
        if key in CORRECTION_KEYS and not keep_correction:
            skipped["correction"] += 1
            continue
        try:
            step_i = int(step)
        except ValueError:
            step_i = 0
        rows.append((fname, ACTION_MAP[key], step_i, seg, key))
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                    help=f"raw sessions root (default: {DEFAULT_RAW})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"processed segments root (default: {DEFAULT_OUT})")
    ap.add_argument("--keep-correction", action="store_true",
                    help="include down/space frames (teleop corrections)")
    ap.add_argument("--test-ratio", type=float, default=0.2,
                    help="fraction of sessions per segment held out as test "
                         "(0 disables test split). Default 0.2")
    args = ap.parse_args()

    for seg in SEGMENTS:
        (args.out / seg / "images").mkdir(parents=True, exist_ok=True)

    # Pass 1: parse all sessions, group rows per (segment, session_name).
    # We need per-segment session lists before we can decide the test holdout.
    seg_sessions: dict[str, dict[str, list]] = {s: defaultdict(list) for s in SEGMENTS}
    src_for_fname: dict[tuple[str, str], Path] = {}   # (seg, fname) -> src image path
    skipped_total = {"no_key": 0, "malformed": 0, "correction": 0}

    for session in sorted(args.raw.iterdir()):
        ann = session / "annotation.txt"
        if not ann.exists():
            continue
        rows, sk = parse_session(ann, args.keep_correction)
        for k, v in sk.items():
            skipped_total[k] += v
        for fname, action_idx, step_i, seg, _key in rows:
            src = session / "images" / Path(fname).name
            if not src.exists():
                skipped_total["malformed"] += 1
                continue
            seg_sessions[seg][session.name].append((fname, action_idx, step_i, _key))
            src_for_fname[(seg, fname)] = src

    # Pass 2: per-segment, hold out the trailing fraction of sessions as test.
    routed = {s: {"train": 0, "test": 0} for s in SEGMENTS}
    class_counts = {s: {k: 0 for k in ACTION_MAP} for s in SEGMENTS}
    max_step = {s: 0 for s in SEGMENTS}
    test_sessions_log: dict[str, list[str]] = {}

    for seg in SEGMENTS:
        sessions = sorted(seg_sessions[seg].keys())
        if not sessions:
            test_sessions_log[seg] = []
            (args.out / seg / "annotation.txt").write_text("")
            (args.out / seg / "test_annotation.txt").write_text("")
            continue
        if args.test_ratio <= 0:
            n_test = 0
        else:
            n_test = max(1, int(round(len(sessions) * args.test_ratio))) \
                     if len(sessions) >= 2 else 0
        test_set = set(sessions[len(sessions) - n_test:]) if n_test > 0 else set()
        test_sessions_log[seg] = sorted(test_set)

        train_f = open(args.out / seg / "annotation.txt", "w")
        test_f = open(args.out / seg / "test_annotation.txt", "w")
        try:
            for sess in sessions:
                bucket = "test" if sess in test_set else "train"
                out_f = test_f if bucket == "test" else train_f
                for fname, action_idx, step_i, key in seg_sessions[seg][sess]:
                    src = src_for_fname[(seg, fname)]
                    dst = args.out / seg / "images" / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    out_f.write(f"{src.name} {action_idx} {step_i}\n")
                    routed[seg][bucket] += 1
                    # class counts + max_step are computed from TRAIN only,
                    # since they feed sampler weights and step_max normalization.
                    if bucket == "train":
                        class_counts[seg][key] += 1
                        if step_i > max_step[seg]:
                            max_step[seg] = step_i
        finally:
            train_f.close()
            test_f.close()

    for seg in SEGMENTS:
        # Round max(step) up to next 100 with 10% headroom so the normalized
        # signal can still reach ~0.9 on the longest real run without saturating.
        step_max = int(math.ceil(max(max_step[seg], 1) * 1.1 / 100.0) * 100)
        meta = {
            "segment": seg,
            "num_train_frames": routed[seg]["train"],
            "num_test_frames": routed[seg]["test"],
            "max_step": max_step[seg],
            "step_max": step_max,
            "class_counts": class_counts[seg],
            "test_sessions": test_sessions_log[seg],
        }
        (args.out / seg / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"{seg}: train={routed[seg]['train']} test={routed[seg]['test']} "
              f"max_step={max_step[seg]} step_max={step_max} "
              f"test_sessions={test_sessions_log[seg]} {class_counts[seg]}")
    if skipped_total["no_key"]:
        print(f"skipped (no key pressed yet): {skipped_total['no_key']}")
    if skipped_total["correction"]:
        print(f"skipped (down/space corrections): {skipped_total['correction']}")
    if skipped_total["malformed"]:
        print(f"skipped (malformed/missing): {skipped_total['malformed']}")


if __name__ == "__main__":
    main()
