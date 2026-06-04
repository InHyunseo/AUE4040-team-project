"""Phase-1 라벨링용 jpg 추출 — rosbag → jpg (Lane + Front).

rosbag에서 stride 간격으로 프레임을 뽑아 라벨링 툴(Roboflow)에 올릴 jpg로 저장한다.
Lane/Front 모두 (224, 224)로 resize — extract_labels.py가 SegFormer/YOLO를 돌리는
입력과 동일 해상도라 라벨링-학습-추론 좌표계가 일치한다.

사용:
  python extract_for_labeling.py --bag <rosbag_dir> --out ../roboflow_input --stride 15

  --stride N : N 프레임마다 1장 저장 (15Hz·stride 15 → 1초에 1장)
  --target   : lane / front / both (기본 both)

출력:
  <out>/lane/<bag>_<idx:06d>.jpg   → Roboflow polygon 세그 (좌실선/우실선/중앙점선)
  <out>/front/<bag>_<idx:06d>.jpg  → Roboflow bbox (car)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

# 토픽/디코드/사이즈/크롭은 extract_labels.py 것을 공유 (좌표계 일치)
from extract_labels import (FRONT_SIZE, FRONT_TOPIC, LANE_SIZE, LANE_TOPIC,
                            crop_lane_roi, decode_compressed)
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def load_images_only(bag_path: Path):
    """Lane/Front 이미지만 timestamp순으로 로드 (cmd_vel은 라벨링에 불필요)."""
    lane, front = [], []
    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        conns = {c.topic: c for c in reader.connections}
        missing = [t for t in (LANE_TOPIC, FRONT_TOPIC) if t not in conns]
        if missing:
            raise RuntimeError(f"bag missing topics: {missing}\nfound: {list(conns)}")
        want = [conns[t] for t in (LANE_TOPIC, FRONT_TOPIC)]
        for conn, ts, raw in reader.messages(connections=want):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == LANE_TOPIC:
                lane.append((ts, decode_compressed(msg)))
            elif conn.topic == FRONT_TOPIC:
                front.append((ts, decode_compressed(msg)))
    lane.sort(key=lambda x: x[0])
    front.sort(key=lambda x: x[0])
    return lane, front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "roboflow_input")
    ap.add_argument("--stride", type=int, default=15,
                    help="N 프레임마다 1장 (15Hz·15 → 1초당 1장)")
    ap.add_argument("--target", choices=["lane", "front", "both"], default="both")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    do_lane  = args.target in ("lane", "both")
    do_front = args.target in ("front", "both")

    bag_name = args.bag.name
    lane_msgs, front_msgs = load_images_only(args.bag)
    print(f"loaded lane={len(lane_msgs)} front={len(front_msgs)} from {bag_name}")

    n_lane = n_front = 0
    if do_lane:
        out_lane = args.out / "lane"
        out_lane.mkdir(parents=True, exist_ok=True)
        for idx in range(0, len(lane_msgs), args.stride):
            resized = cv2.resize(crop_lane_roi(lane_msgs[idx][1]), LANE_SIZE)
            cv2.imwrite(str(out_lane / f"{bag_name}_{idx:06d}.jpg"), resized,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            n_lane += 1

    if do_front:
        out_front = args.out / "front"
        out_front.mkdir(parents=True, exist_ok=True)
        for idx in range(0, len(front_msgs), args.stride):
            resized = cv2.resize(front_msgs[idx][1], FRONT_SIZE)
            cv2.imwrite(str(out_front / f"{bag_name}_{idx:06d}.jpg"), resized,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            n_front += 1

    print(f"saved lane={n_lane} front={n_front} -> {args.out}")
    if do_lane:
        print(f"  Lane  → Roboflow polygon seg (좌실선/우실선/중앙점선)")
    if do_front:
        print(f"  Front → Roboflow bbox (car)")


if __name__ == "__main__":
    main()
