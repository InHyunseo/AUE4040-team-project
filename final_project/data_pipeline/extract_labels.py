"""Generate labels_cache.h5 from a rosbag2 .db3.

Reads three topics:
  /bev_image/compressed     sensor_msgs/CompressedImage
  /front_image/compressed   sensor_msgs/CompressedImage
  /cmd_vel                  geometry_msgs/Twist (linear.x m/s, angular.z rad/s)

For each BEV frame t, syncs the nearest front frame and the nearest cmd_vel
sample (both within ±SYNC_TOL ns), then generates:

  bev        (224,224,3) uint8   warped via calib.M
  front      (224,224,3) uint8   resized
  seg        (4,224,224) uint8   HSV mask + zone-validated, on BEV plane
  det        (5,)        float32 [x,y,w,h,conf] on front (pixels, conf in [0,1])
  waypoint   (5,2)       float32 future (x,y) in meters, robot frame
  steer      ()          float32 angular.z at t
  throttle   ()          float32 linear.x  at t
  timestamp_ns int64

Runs WITHOUT ROS2 or camera hardware. Pure rosbags + opencv + numpy + h5py
+ (optional) HF transformers for GroundingDINO.

  python extract_labels.py --bag /path/to/rosbag2_dir --calib ../calib/calib.json \
      --out labels_cache.h5 --debug_dir ../debug_samples --device cuda

NOTE on main/ coupling: this script imports NOTHING from main/. Its only
contract with main/ is the rosbag topic schema above, which a future ROS2
recorder under main/ros2_ws/src/rover_recorder/ must publish. See
final_project/README.md "Pipeline integration with main/".
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

# rosbag2 reading (no ROS2 install required)
from rosbags.highlevel import AnyReader

# -------------------------------------------------------------------- constants

BEV_TOPIC   = "/bev_image/compressed"
FRONT_TOPIC = "/front_image/compressed"
CMD_TOPIC   = "/cmd_vel"

SYNC_TOL_NS = 50_000_000      # 50 ms
WP_HORIZON_S = 0.5
WP_N         = 5
WP_DT        = WP_HORIZON_S / WP_N

FRONT_SIZE = (224, 224)

# HSV thresholds — tune on a sample frame.
HSV_WHITE = ((0, 0, 200), (180, 40, 255))            # solid lanes
HSV_YELLOW = ((15, 80, 120), (35, 255, 255))         # dashed lanes

# Zone validation bands on BEV (top=far, bottom=near). Near/mid must have a
# mask blob; far is allowed to be empty.
ZONE_BANDS = [("far", 0.0, 0.33, False),
              ("mid", 0.33, 0.66, True),
              ("near", 0.66, 1.0, True)]

# -------------------------------------------------------------------- helpers


def decode_compressed(msg) -> np.ndarray:
    arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return img


@dataclass
class CmdSample:
    t_ns: int
    v: float        # linear.x m/s
    w: float        # angular.z rad/s


def load_bag(bag_path: Path):
    bev, front, cmd = [], [], []
    with AnyReader([bag_path]) as reader:
        conns = {c.topic: c for c in reader.connections}
        missing = [t for t in (BEV_TOPIC, FRONT_TOPIC, CMD_TOPIC) if t not in conns]
        if missing:
            raise RuntimeError(f"bag missing topics: {missing}\nfound: {list(conns)}")
        for conn, ts, raw in reader.messages(connections=list(conns.values())):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == BEV_TOPIC:
                bev.append((ts, decode_compressed(msg)))
            elif conn.topic == FRONT_TOPIC:
                front.append((ts, decode_compressed(msg)))
            elif conn.topic == CMD_TOPIC:
                cmd.append(CmdSample(ts, float(msg.linear.x), float(msg.angular.z)))
    bev.sort(key=lambda x: x[0])
    front.sort(key=lambda x: x[0])
    cmd.sort(key=lambda c: c.t_ns)
    return bev, front, cmd


def nearest(sorted_ts, t):
    """Index of nearest timestamp in sorted_ts to t."""
    i = bisect.bisect_left(sorted_ts, t)
    if i == 0:
        return 0
    if i == len(sorted_ts):
        return len(sorted_ts) - 1
    return i - 1 if (t - sorted_ts[i - 1]) <= (sorted_ts[i] - t) else i


# -------------------------------------------------------------------- seg


def hsv_lane_mask(bgr_bev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr_bev, cv2.COLOR_BGR2HSV)
    solid = cv2.inRange(hsv, np.array(HSV_WHITE[0]), np.array(HSV_WHITE[1]))
    dashed = cv2.inRange(hsv, np.array(HSV_YELLOW[0]), np.array(HSV_YELLOW[1]))
    k = np.ones((3, 3), np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k)
    dashed = cv2.morphologyEx(dashed, cv2.MORPH_OPEN, k)
    return solid, dashed


def split_lr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    left = mask.copy(); left[:, w // 2:] = 0
    right = mask.copy(); right[:, :w // 2] = 0
    return left, right


def zone_ok(mask: np.ndarray) -> bool:
    h = mask.shape[0]
    for _, y0f, y1f, required in ZONE_BANDS:
        if not required:
            continue
        band = mask[int(y0f * h):int(y1f * h)]
        if band.sum() == 0:
            return False
    return True


def build_seg(bev_bgr: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """Return (4, H, W) uint8 in {0,255}: solidL, solidR, dashedL, dashedR."""
    solid, dashed = hsv_lane_mask(bev_bgr)
    sL, sR = split_lr(solid)
    dL, dR = split_lr(dashed)
    out = np.stack([sL, sR, dL, dR], axis=0)
    if prev is not None:
        for c in range(4):
            if not zone_ok(out[c]):
                out[c] = prev[c]
    return out


# -------------------------------------------------------------------- waypoint


def waypoint_gt(cmds: list[CmdSample], cmd_ts: list[int], t0_ns: int) -> np.ndarray | None:
    """Integrate cmd_vel from t0 to t0+horizon, sample at WP_N evenly spaced offsets.

    Uses the LAST cmd_vel command active in each integration sub-step (ZOH).
    Returns (5,2) float32 in robot frame (x forward, y left) — or None if there
    aren't enough future cmd_vel samples to cover the horizon.
    """
    horizon_ns = int(WP_HORIZON_S * 1e9)
    end_ns = t0_ns + horizon_ns
    # Need at least one cmd at or after end_ns to confirm coverage.
    if cmd_ts[-1] < end_ns:
        return None

    # Substep integration at fine dt for accuracy, then sample at WP_DT marks.
    sub_dt = 0.02   # 20 ms
    n_sub = int(WP_HORIZON_S / sub_dt)
    x = y = th = 0.0
    samples = []
    sample_marks = {int(round((i + 1) * WP_DT / sub_dt)) for i in range(WP_N)}
    for k in range(n_sub):
        t_cur = t0_ns + int(k * sub_dt * 1e9)
        i = bisect.bisect_right(cmd_ts, t_cur) - 1
        if i < 0:
            i = 0
        c = cmds[i]
        # forward euler
        th += c.w * sub_dt
        x += c.v * math.cos(th) * sub_dt
        y += c.v * math.sin(th) * sub_dt
        if (k + 1) in sample_marks:
            samples.append((x, y))
    if len(samples) != WP_N:
        return None
    return np.asarray(samples, dtype=np.float32)


# -------------------------------------------------------------------- det (GDINO)


class GroundingDinoDet:
    """Wrap HF IDEA-Research/grounding-dino-tiny. Lazy import."""

    def __init__(self, device: str = "cuda", prompt: str = "a car. a toy car."):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        self.torch = torch
        self.device = device
        self.prompt = prompt
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @staticmethod
    def empty() -> np.ndarray:
        return np.zeros(5, dtype=np.float32)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.3, text_threshold=0.25,
            target_sizes=[pil.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) else None
        scores = results["scores"].cpu().numpy() if len(results["scores"]) else None
        if boxes is None or len(boxes) == 0:
            return self.empty()
        k = int(scores.argmax())
        x1, y1, x2, y2 = boxes[k]
        return np.array([x1, y1, x2 - x1, y2 - y1, float(scores[k])], dtype=np.float32)


# -------------------------------------------------------------------- debug viz


def save_debug(path: Path, bev: np.ndarray, front: np.ndarray,
               seg: np.ndarray, det: np.ndarray, wps: np.ndarray, ppm: float):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    bev_vis = bev.copy()
    for c in range(4):
        bev_vis[seg[c] > 0] = (bev_vis[seg[c] > 0] * 0.4 + np.array(colors[c]) * 0.6).astype(np.uint8)
    H, W = bev_vis.shape[:2]
    ox, oy = W // 2, H - H // 8
    for (x_m, y_m) in wps:
        u = int(ox - y_m * ppm)
        v = int(oy - x_m * ppm)
        cv2.circle(bev_vis, (u, v), 3, (255, 255, 255), -1)

    front_vis = front.copy()
    if det[4] > 0:
        x, y, w, h, conf = det
        cv2.rectangle(front_vis, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 255, 0), 2)
        cv2.putText(front_vis, f"{conf:.2f}", (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    pad = np.hstack([bev_vis, front_vis])
    cv2.imwrite(str(path), pad)


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "labels_cache.h5")
    ap.add_argument("--debug_dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "debug_samples")
    ap.add_argument("--limit", type=int, default=0, help="cap N frames (0=all)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip_det", action="store_true", help="skip GroundingDINO")
    args = ap.parse_args()

    calib = json.loads(args.calib.read_text())
    M = np.asarray(calib["M"], dtype=np.float64)
    bev_w, bev_h = calib["bev_size"]
    ppm = float(calib["pixels_per_meter"])

    print(f"loading bag {args.bag} ...")
    bev_msgs, front_msgs, cmds = load_bag(args.bag)
    print(f"  bev={len(bev_msgs)} front={len(front_msgs)} cmd_vel={len(cmds)}")
    if not bev_msgs or not front_msgs or not cmds:
        raise RuntimeError("empty bag for one or more topics")

    front_ts = [t for t, _ in front_msgs]
    cmd_ts = [c.t_ns for c in cmds]

    det = None if args.skip_det else GroundingDinoDet(device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    n_candidate = len(bev_msgs)
    if args.limit:
        n_candidate = min(n_candidate, args.limit)
    debug_stride = max(1, n_candidate // 100)

    with h5py.File(args.out, "w") as h5:
        def dset(name, shape, dtype):
            return h5.create_dataset(
                name, shape=(0,) + shape, maxshape=(None,) + shape,
                dtype=dtype, chunks=(1,) + shape, compression="gzip",
                compression_opts=4)

        d_bev   = dset("bev", (bev_h, bev_w, 3), "uint8")
        d_front = dset("front", (FRONT_SIZE[1], FRONT_SIZE[0], 3), "uint8")
        d_seg   = dset("seg", (4, bev_h, bev_w), "uint8")
        d_det   = dset("det", (5,), "float32")
        d_wp    = dset("waypoint", (WP_N, 2), "float32")
        d_steer = dset("steer", (), "float32")
        d_thr   = dset("throttle", (), "float32")
        d_ts    = dset("timestamp_ns", (), "int64")

        prev_seg = None
        kept = 0
        for i, (t_ns, bev_src) in enumerate(bev_msgs[:n_candidate]):
            j = nearest(front_ts, t_ns)
            if abs(front_ts[j] - t_ns) > SYNC_TOL_NS:
                continue
            ci = nearest(cmd_ts, t_ns)
            if abs(cmd_ts[ci] - t_ns) > SYNC_TOL_NS:
                continue

            wps = waypoint_gt(cmds, cmd_ts, t_ns)
            if wps is None:
                continue

            bev = cv2.warpPerspective(bev_src, M, (bev_w, bev_h))
            front = cv2.resize(front_msgs[j][1], FRONT_SIZE)
            seg = build_seg(bev, prev_seg)
            prev_seg = seg
            det_arr = det(front) if det is not None else GroundingDinoDet.empty()

            v = cmds[ci].v
            w = cmds[ci].w

            for d in (d_bev, d_front, d_seg, d_det, d_wp, d_steer, d_thr, d_ts):
                d.resize(kept + 1, axis=0)
            d_bev[kept] = bev
            d_front[kept] = front
            d_seg[kept] = seg
            d_det[kept] = det_arr
            d_wp[kept] = wps
            d_steer[kept] = w
            d_thr[kept] = v
            d_ts[kept] = t_ns

            if kept % debug_stride == 0:
                save_debug(args.debug_dir / f"frame_{kept:05d}.png",
                           bev, front, seg, det_arr, wps, ppm)

            kept += 1
            if kept % 50 == 0:
                print(f"  kept {kept}/{i+1}")

        h5.attrs["pixels_per_meter"] = ppm
        h5.attrs["bev_size"] = [bev_w, bev_h]
        h5.attrs["wp_horizon_s"] = WP_HORIZON_S
        h5.attrs["wp_n"] = WP_N
        h5.attrs["bag"] = str(args.bag)

    print(f"done. kept={kept} -> {args.out}")
    print(f"debug samples -> {args.debug_dir}")


if __name__ == "__main__":
    main()
