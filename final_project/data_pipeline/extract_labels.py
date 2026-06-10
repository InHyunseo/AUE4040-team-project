"""Generate labels_cache.h5 from a rosbag2 .db3 (차선 주행 + 정지 차량 추월).

Reads three topics:
  /lane_image/compressed    sensor_msgs/CompressedImage
  /front_image/compressed   sensor_msgs/CompressedImage
  /cmd_vel                  geometry_msgs/Twist (linear.x m/s, angular.z rad/s)

For each lane frame t, syncs the nearest front frame and the nearest cmd_vel
sample (both within ±SYNC_TOL ns), then generates:

  lane       (224,224,3) uint8   raw lane image, resized
  front      (224,224,3) uint8   resized
  seg        (3,224,224) uint8   SegFormer lane masks on the raw lane image
                                 ch0=left-solid, ch1=right-solid, ch2=center-dashed
  det        (5,)        float32 [x,y,w,h,conf] on front (pixels, conf in [0,1])
                                 YOLO single-class (car); zeros if no car
  waypoint   (5,2)       float32 future (x,y) in meters, robot frame
  steer      ()          float32 angular.z at t
  throttle   ()          float32 linear.x  at t
  timestamp_ns int64

Runs without ROS2 or camera hardware: rosbags + opencv + numpy + h5py
+ transformers (SegFormer) + ultralytics (YOLO).

  python extract_labels.py --bag /path/to/rosbag2_dir \
      --segformer_ckpt ../models/segformer_lane \
      --yolo_weights ../models/best.pt \
      --out labels_cache.h5 --debug_dir ../debug_samples --device cuda

SegFormer and YOLO are frozen (fine-tuned once in Phase 1). The H5 stores raw
lane/front + seg/det separately; the dataloader composites the overlays
(lane: 3 seg channels alpha-blended ch0=red/ch1=green/ch2=blue; front: car
bbox drawn) so compositing can be tuned without re-extracting.
"""

from __future__ import annotations

import argparse
import bisect
from enum import IntEnum
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# h5py and rosbags are only needed for offline H5 extraction (main()). They are
# imported lazily inside the functions that use them so this module can be
# imported for its frozen-model classes (SegFormerLaneSeg / YoloCarDet /
# crop_lane_roi) on a vehicle/ROS2 host that has no h5py/rosbags installed.

# -------------------------------------------------------------------- constants

LANE_TOPIC  = "/lane_image/compressed"
FRONT_TOPIC = "/front_image/compressed"
CMD_TOPIC   = "/cmd_vel"

SYNC_TOL_NS = 50_000_000      # 50 ms
WP_HORIZON_S = 0.5
WP_N         = 5
WP_DT        = WP_HORIZON_S / WP_N

LANE_SIZE  = (224, 224)       # lane image resized to this
FRONT_SIZE = (224, 224)

# Fraction of the lane image to crop off the TOP before resizing to LANE_SIZE.
# The lane camera's upper region is off-road background (sky/wall/far scene)
# with no lane in it; cropping it gives the lanes more vertical resolution and
# removes background distractors. 0.0 = no crop; current project contract is
# 0.30. Set this from a real bag frame BEFORE labeling (changing it after
# labeling shifts the coordinate frame and invalidates labels). Only the lane
# path uses this; front/YOLO is never cropped (cars appear anywhere in frame).
LANE_CROP_TOP = 0.30

# Display-only scale for drawing metric waypoints onto the debug image. Does
# NOT affect stored labels (waypoints are kept in meters, robot frame).
DEBUG_PPM = 200.0

SEG_N_CLASSES = 3  # 0=left-solid, 1=right-solid, 2=center-dashed (background excluded)

# -------------------------------------------------------------------- helpers


def decode_compressed(msg) -> np.ndarray:
    arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return img


def crop_lane_roi(lane: np.ndarray) -> np.ndarray:
    """Drop the top LANE_CROP_TOP fraction of the lane image (off-road
    background). Must run BEFORE the LANE_SIZE resize, identically in every lane
    path (labeling export, h5 extraction, on-vehicle inference) so the
    label/train/infer coordinate frames stay aligned. LANE_CROP_TOP=0.0 returns
    the image unchanged."""
    if LANE_CROP_TOP <= 0.0:
        return lane
    y0 = int(lane.shape[0] * LANE_CROP_TOP)
    return lane[y0:]


@dataclass
class CmdSample:
    t_ns: int
    v: float        # linear.x m/s
    w: float        # angular.z rad/s


def _header_stamp_ns(msg, fallback_ns: int) -> int:
    """CompressedImage 의 header.stamp(캡처 시각)를 ns 로. stamp 가 0(미설정,
    구버전 camera_node)이면 bag-write ts 로 폴백해 옛 bag 도 그대로 추출된다."""
    st = msg.header.stamp
    ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
    return ns if ns > 0 else fallback_ns


def load_bag(bag_path: Path):
    """lane/front 는 (capture_stamp_ns, write_ts_ns, image) 로 반환한다.

    capture_stamp_ns = header.stamp(camera_node 가 찍은 캡처 시각). lane↔front
    정합은 이 값으로 한다(인코딩/송신 순서 지연이 stamp 에 안 실리므로 정확).
    write_ts_ns = rosbag write 시각. cmd_vel/steer 는 header 가 없어 write 시각만
    있으므로, lane↔cmd 매칭과 waypoint 적분은 write 시각끼리 비교해 시계를 안 섞는다.
    """
    from rosbags.highlevel import AnyReader  # offline-only; not needed on vehicle
    lane, front, cmd = [], [], []
    with AnyReader([bag_path]) as reader:
        conns = {c.topic: c for c in reader.connections}
        missing = [t for t in (LANE_TOPIC, FRONT_TOPIC, CMD_TOPIC) if t not in conns]
        if missing:
            raise RuntimeError(f"bag missing topics: {missing}\nfound: {list(conns)}")
        for conn, ts, raw in reader.messages(connections=list(conns.values())):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == LANE_TOPIC:
                lane.append((_header_stamp_ns(msg, ts), ts, decode_compressed(msg)))
            elif conn.topic == FRONT_TOPIC:
                front.append((_header_stamp_ns(msg, ts), ts, decode_compressed(msg)))
            elif conn.topic == CMD_TOPIC:
                cmd.append(CmdSample(ts, float(msg.linear.x), float(msg.angular.z)))
    lane.sort(key=lambda x: x[0])   # 캡처 시각 기준 정렬
    front.sort(key=lambda x: x[0])
    cmd.sort(key=lambda c: c.t_ns)
    return lane, front, cmd


def nearest(sorted_ts, t):
    """Index of nearest timestamp in sorted_ts to t."""
    i = bisect.bisect_left(sorted_ts, t)
    if i == 0:
        return 0
    if i == len(sorted_ts):
        return len(sorted_ts) - 1
    return i - 1 if (t - sorted_ts[i - 1]) <= (sorted_ts[i] - t) else i


# -------------------------------------------------------------------- seg (SegFormer)


class SegFormerLaneSeg:
    """Frozen SegFormer lane segmenter.

    Runs on the raw lane image and returns (3, H, W) uint8 masks in {0,255}:
      ch0 = left-solid, ch1 = right-solid, ch2 = center-dashed.

    The checkpoint's id2label must have 4 entries:
      0=background, 1=left-solid, 2=right-solid, 3=center-dashed.
    The output drops the background channel.
    """

    N_CLASSES = SEG_N_CLASSES

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # Ubuntu/Jetson apt Pillow can be old enough to lack Image.Resampling,
        # while recent transformers expects it. Alias the old constants module
        # before importing transformers so SegFormer can run on the vehicle.
        try:
            from PIL import Image
            if not hasattr(Image, "Resampling"):
                class _Resampling(IntEnum):
                    NEAREST = Image.NEAREST
                    BOX = getattr(Image, "BOX", Image.NEAREST)
                    BILINEAR = Image.BILINEAR
                    HAMMING = getattr(Image, "HAMMING", Image.BILINEAR)
                    BICUBIC = Image.BICUBIC
                    LANCZOS = Image.LANCZOS

                Image.Resampling = _Resampling
        except ImportError:
            pass

        try:
            from transformers.models.segformer.image_processing_segformer import (
                SegformerImageProcessor,
            )
            from transformers.models.segformer.modeling_segformer import (
                SegformerForSemanticSegmentation,
            )
        except ImportError:
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerImageProcessor)
        import torch

        self.torch = torch
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint_path).to(device)
        self.model.eval()

    @staticmethod
    def empty(h: int, w: int) -> np.ndarray:
        return np.zeros((SEG_N_CLASSES, h, w), dtype=np.uint8)

    def __call__(self, lane_bgr: np.ndarray) -> np.ndarray:
        h, w = lane_bgr.shape[:2]
        rgb = cv2.cvtColor(lane_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits           # (1, C, h', w')
        # upsample to lane-image resolution, then argmax over classes
        logits = self.torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False)
        cls_map = logits.argmax(dim=1)[0].cpu().numpy()     # (h, w), values 0..C-1
        # background = 0; semantic classes 1..SEG_N_CLASSES map to channels 0..N-1
        out = np.zeros((SEG_N_CLASSES, h, w), dtype=np.uint8)
        for c in range(SEG_N_CLASSES):
            out[c][cls_map == (c + 1)] = 255
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


# -------------------------------------------------------------------- det (YOLO)


class YoloCarDet:
    """Frozen YOLO (ultralytics best.pt) single-class car detector.

    Default model is YOLO26 (NMS-free, end-to-end), so no IoU/NMS threshold is
    used; only the confidence filter applies. __call__ returns the highest-conf
    car bbox as (5,) [x,y,w,h,conf] in front-image pixels (x,y = top-left), or
    zeros(5) if no car. Keeps bbox position and size so the model learns
    position and apparent distance.
    """

    def __init__(self, weights_path: str, device: str = "cuda",
                 imgsz: int = 320, conf: float = 0.25,
                 car_class: str = "car"):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        # resolve the car class id. names is {id: name}. Single-class models
        # accept any detection as a car.
        names = self.model.names
        if len(names) == 1:
            self.car_id = next(iter(names))
        else:
            self.car_id = next((i for i, n in names.items() if n == car_class), None)
            if self.car_id is None:
                raise RuntimeError(
                    f"class '{car_class}' not in YOLO names {names}")

    @staticmethod
    def empty() -> np.ndarray:
        return np.zeros(5, dtype=np.float32)

    def __call__(self, front_bgr: np.ndarray) -> np.ndarray:
        res = self.model.predict(front_bgr, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return self.empty()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        mask = cls == self.car_id
        if not mask.any():
            return self.empty()
        idx = np.where(mask)[0]
        k = idx[int(confs[idx].argmax())]
        x1, y1, x2, y2 = xyxy[k]
        return np.array([x1, y1, x2 - x1, y2 - y1, float(confs[k])], dtype=np.float32)


# -------------------------------------------------------------------- debug viz


def save_debug(path: Path, lane: np.ndarray, front: np.ndarray,
               seg: np.ndarray, det: np.ndarray, wps: np.ndarray):
    # ch0=left-solid red, ch1=right-solid green, ch2=center-dashed blue
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    lane_vis = lane.copy()
    for c in range(SEG_N_CLASSES):
        m = seg[c] > 0
        lane_vis[m] = (lane_vis[m] * 0.4 + np.array(colors[c]) * 0.6).astype(np.uint8)
    # Waypoints are stored in meters (robot frame); draw at a fixed display
    # scale just to eyeball them. Not metrically meaningful without calib.
    H, W = lane_vis.shape[:2]
    ox, oy = W // 2, H - H // 8
    for (x_m, y_m) in wps:
        u = int(ox - y_m * DEBUG_PPM)
        v = int(oy - x_m * DEBUG_PPM)
        cv2.circle(lane_vis, (u, v), 3, (255, 255, 255), -1)

    front_vis = front.copy()
    if det[4] > 0:
        x, y, w, h, conf = det
        cv2.rectangle(front_vis, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 255, 0), 2)
        cv2.putText(front_vis, f"car {conf:.2f}", (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    pad = np.hstack([lane_vis, front_vis])
    cv2.imwrite(str(path), pad)


# -------------------------------------------------------------------- main


def main():
    import h5py  # offline-only; not needed when importing frozen-model helpers

    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--segformer_ckpt", type=Path, default=None,
                    help="fine-tuned SegFormer lane checkpoint dir (required)")
    ap.add_argument("--yolo_weights", type=Path,
                    default=Path(__file__).resolve().parents[1] / "models" / "best.pt",
                    help="ultralytics YOLO best.pt (single-class car)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "labels_cache.h5")
    ap.add_argument("--debug_dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "debug_samples")
    ap.add_argument("--limit", type=int, default=0, help="cap N frames (0=all)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip_det", action="store_true", help="skip YOLO car detection")
    args = ap.parse_args()

    if args.segformer_ckpt is None:
        raise SystemExit(
            "--segformer_ckpt is required. Fine-tune SegFormer (Phase 1) first, "
            "then pass the checkpoint dir.")

    lane_w, lane_h = LANE_SIZE

    print(f"loading bag {args.bag} ...")
    lane_msgs, front_msgs, cmds = load_bag(args.bag)
    print(f"  lane={len(lane_msgs)} front={len(front_msgs)} cmd_vel={len(cmds)}")
    if not lane_msgs or not front_msgs or not cmds:
        raise RuntimeError("empty bag for one or more topics")

    front_cap = [cap for cap, _, _ in front_msgs]   # front 캡처시각 (lane↔front 매칭용)
    cmd_ts = [c.t_ns for c in cmds]

    print(f"loading SegFormer {args.segformer_ckpt} ...")
    segmenter = SegFormerLaneSeg(str(args.segformer_ckpt), device=args.device)
    det = None if args.skip_det else YoloCarDet(str(args.yolo_weights), device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    n_candidate = len(lane_msgs)
    if args.limit:
        n_candidate = min(n_candidate, args.limit)
    debug_stride = max(1, n_candidate // 100)

    with h5py.File(args.out, "w") as h5:
        def dset(name, shape, dtype):
            return h5.create_dataset(
                name, shape=(0,) + shape, maxshape=(None,) + shape,
                dtype=dtype, chunks=(1,) + shape, compression="gzip",
                compression_opts=4)

        d_lane  = dset("lane", (lane_h, lane_w, 3), "uint8")
        d_front = dset("front", (FRONT_SIZE[1], FRONT_SIZE[0], 3), "uint8")
        d_seg   = dset("seg", (SEG_N_CLASSES, lane_h, lane_w), "uint8")
        d_det   = dset("det", (5,), "float32")
        d_wp    = dset("waypoint", (WP_N, 2), "float32")
        d_steer = dset("steer", (), "float32")
        d_thr   = dset("throttle", (), "float32")
        d_ts    = dset("timestamp_ns", (), "int64")

        kept = 0
        for i, (cap_ns, write_ns, lane_src) in enumerate(lane_msgs[:n_candidate]):
            # lane↔front 는 캡처 시각으로 매칭(두 카메라 정합 정확도의 핵심).
            j = nearest(front_cap, cap_ns)
            if abs(front_cap[j] - cap_ns) > SYNC_TOL_NS:
                continue
            # lane↔cmd / waypoint 는 같은 시계(rosbag write 시각)끼리 비교한다.
            ci = nearest(cmd_ts, write_ns)
            if abs(cmd_ts[ci] - write_ns) > SYNC_TOL_NS:
                continue

            wps = waypoint_gt(cmds, cmd_ts, write_ns)
            if wps is None:
                continue

            lane = cv2.resize(crop_lane_roi(lane_src), LANE_SIZE)
            front = cv2.resize(front_msgs[j][2], FRONT_SIZE)
            seg = segmenter(lane)
            det_arr = det(front) if det is not None else YoloCarDet.empty()

            v = cmds[ci].v
            w = cmds[ci].w

            for d in (d_lane, d_front, d_seg, d_det, d_wp, d_steer, d_thr, d_ts):
                d.resize(kept + 1, axis=0)
            d_lane[kept] = lane
            d_front[kept] = front
            d_seg[kept] = seg
            d_det[kept] = det_arr
            d_wp[kept] = wps
            d_steer[kept] = w
            d_thr[kept] = v
            d_ts[kept] = cap_ns

            if kept % debug_stride == 0:
                save_debug(args.debug_dir / f"frame_{kept:05d}.png",
                           lane, front, seg, det_arr, wps)

            kept += 1
            if kept % 50 == 0:
                print(f"  kept {kept}/{i+1}")

        h5.attrs["lane_size"] = [lane_w, lane_h]
        h5.attrs["wp_horizon_s"] = WP_HORIZON_S
        h5.attrs["wp_n"] = WP_N
        h5.attrs["seg_n_classes"] = SEG_N_CLASSES
        h5.attrs["bag"] = str(args.bag)

    print(f"done. kept={kept} -> {args.out}")
    print(f"debug samples -> {args.debug_dir}")


if __name__ == "__main__":
    main()
