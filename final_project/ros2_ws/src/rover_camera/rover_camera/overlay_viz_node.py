"""Publish live SegFormer/YOLO overlay streams for the browser monitor.

Subscribes to the raw camera JPEG topics, runs the frozen Phase-1 models with
the same preprocessing contract used by data_pipeline/extract_labels.py, then
publishes JPEG overlays:

  /lane_seg/compressed   lane image + SegFormer mask alpha blend
  /front_det/compressed  front image + YOLO car bbox
"""
from __future__ import annotations

import os
import sys
import threading
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def _find_project_root() -> Path | None:
    env = os.environ.get("AUE4040_FINAL_PROJECT_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "data_pipeline" / "extract_labels.py").exists():
            return root

    for parent in Path(__file__).resolve().parents:
        if (parent / "data_pipeline" / "extract_labels.py").exists():
            return parent
    return None


PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT is not None:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data_pipeline.extract_labels import (  # noqa: E402
        FRONT_SIZE,
        LANE_SIZE,
        SEG_N_CLASSES,
        SegFormerLaneSeg,
        YoloCarDet,
        crop_lane_roi,
    )
except Exception as exc:  # pragma: no cover - reported clearly at node startup
    _HELPER_IMPORT_ERROR = exc
else:
    _HELPER_IMPORT_ERROR = None


SEG_COLORS_BGR = (
    np.array((0, 0, 255), dtype=np.float32),  # left-solid: red
    np.array((0, 255, 0), dtype=np.float32),  # right-solid: green
    np.array((255, 0, 0), dtype=np.float32),  # center-dashed: blue
)


class OverlayVizNode(Node):
    def __init__(self) -> None:
        super().__init__("rover_overlay_viz")
        if _HELPER_IMPORT_ERROR is not None:
            raise RuntimeError(
                "failed to import data_pipeline.extract_labels helpers. "
                "Run with --symlink-install from final_project/ros2_ws or set "
                "AUE4040_FINAL_PROJECT_ROOT=/path/to/final_project"
            ) from _HELPER_IMPORT_ERROR

        default_seg = PROJECT_ROOT / "models" / "segformer_lane" if PROJECT_ROOT else ""
        default_yolo = PROJECT_ROOT / "models" / "best.pt" if PROJECT_ROOT else ""

        self.declare_parameter("lane_topic", "/lane_image/compressed")
        self.declare_parameter("front_topic", "/front_image/compressed")
        self.declare_parameter("lane_overlay_topic", "/lane_seg/compressed")
        self.declare_parameter("front_overlay_topic", "/front_det/compressed")
        self.declare_parameter("segformer_ckpt", str(default_seg))
        self.declare_parameter("yolo_weights", str(default_yolo))
        self.declare_parameter("device", "cuda")
        self.declare_parameter("enable_seg", True)
        self.declare_parameter("enable_det", True)
        self.declare_parameter("viz_fps", 3.0)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("seg_alpha", 0.6)
        self.declare_parameter("yolo_imgsz", 320)
        self.declare_parameter("yolo_conf", 0.25)

        self.jpeg_q = int(self.get_parameter("jpeg_quality").value)
        self.seg_alpha = min(1.0, max(0.0, float(self.get_parameter("seg_alpha").value)))
        self.enable_seg = bool(self.get_parameter("enable_seg").value)
        self.enable_det = bool(self.get_parameter("enable_det").value)
        self._check_runtime_deps()

        lane_topic = str(self.get_parameter("lane_topic").value)
        front_topic = str(self.get_parameter("front_topic").value)
        lane_overlay_topic = str(self.get_parameter("lane_overlay_topic").value)
        front_overlay_topic = str(self.get_parameter("front_overlay_topic").value)

        self.pub_lane = self.create_publisher(
            CompressedImage, lane_overlay_topic, 1
        )
        self.pub_front = self.create_publisher(
            CompressedImage, front_overlay_topic, 1
        )
        self.create_subscription(
            CompressedImage, lane_topic, self._on_lane, 1
        )
        self.create_subscription(
            CompressedImage, front_topic, self._on_front, 1
        )

        self._lane_msg: CompressedImage | None = None
        self._front_msg: CompressedImage | None = None
        self._lane_seq = 0
        self._front_seq = 0
        self._done_lane_seq = 0
        self._done_front_seq = 0
        self._tick_i = 0

        self.segmenter = None
        self.detector = None

        fps = max(0.1, float(self.get_parameter("viz_fps").value))
        self.timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            "overlay viz ready: "
            f"{lane_topic}->{lane_overlay_topic} seg={self.enable_seg}, "
            f"{front_topic}->{front_overlay_topic} det={self.enable_det}, "
            f"fps={fps:.1f}"
        )
        threading.Thread(target=self._load_models, daemon=True).start()

    def _check_runtime_deps(self) -> None:
        major = int(np.__version__.split(".", 1)[0])
        if major >= 2:
            self.enable_seg = False
            self.enable_det = False
            self.get_logger().error(
                "NumPy 2.x is installed, but Jetson PyTorch wheels are usually "
                "built against NumPy 1.x. Publishing raw preview fallbacks only. "
                "Fix on the Jetson with:\n"
                "  python3 -m pip uninstall -y opencv-python opencv-contrib-python\n"
                "  python3 -m pip install --user 'numpy<2.0'"
            )
            return
        missing = []
        if self.enable_seg:
            for name in ("torch", "transformers"):
                if find_spec(name) is None:
                    missing.append(name)
        if self.enable_det and find_spec("ultralytics") is None:
            missing.append("ultralytics")
        if missing:
            if "torch" in missing or "transformers" in missing:
                self.enable_seg = False
            if "ultralytics" in missing:
                self.enable_det = False
            self.get_logger().error(
                "missing overlay runtime package(s): "
                + ", ".join(sorted(set(missing)))
                + ". Publishing raw preview fallbacks for unavailable models.\n"
                "Install on the Jetson, then keep NumPy on 1.x:\n"
                "  python3 -m pip install --user transformers ultralytics\n"
                "  python3 -m pip install --user 'numpy<2.0'"
            )

    def _load_models(self) -> None:
        device = str(self.get_parameter("device").value)
        seg_ckpt = Path(str(self.get_parameter("segformer_ckpt").value)).expanduser()
        yolo_weights = Path(str(self.get_parameter("yolo_weights").value)).expanduser()

        if self.enable_seg:
            try:
                if not seg_ckpt.exists():
                    raise RuntimeError(f"SegFormer checkpoint not found: {seg_ckpt}")
                self.get_logger().info(f"loading SegFormer from {seg_ckpt} on {device}")
                self.segmenter = SegFormerLaneSeg(str(seg_ckpt), device=device)
            except Exception as exc:
                self.segmenter = None
                self.get_logger().error(
                    "SegFormer unavailable; /lane_seg/compressed will publish "
                    f"resized raw lane frames. reason: {exc}"
                )

        if self.enable_det:
            try:
                if not yolo_weights.exists():
                    raise RuntimeError(f"YOLO weights not found: {yolo_weights}")
                imgsz = int(self.get_parameter("yolo_imgsz").value)
                conf = float(self.get_parameter("yolo_conf").value)
                self.get_logger().info(f"loading YOLO from {yolo_weights} on {device}")
                self.detector = YoloCarDet(
                    str(yolo_weights), device=device, imgsz=imgsz, conf=conf
                )
            except Exception as exc:
                self.detector = None
                self.get_logger().error(
                    "YOLO unavailable; /front_det/compressed will publish "
                    f"resized raw front frames. reason: {exc}"
                )

    def _on_lane(self, msg: CompressedImage) -> None:
        self._lane_msg = msg
        self._lane_seq += 1

    def _on_front(self, msg: CompressedImage) -> None:
        self._front_msg = msg
        self._front_seq += 1

    def _tick(self) -> None:
        self._tick_i += 1
        if self.enable_seg and self._lane_msg and self._lane_seq != self._done_lane_seq:
            try:
                self.pub_lane.publish(self._process_lane(self._lane_msg))
                self._done_lane_seq = self._lane_seq
            except Exception as exc:
                self.get_logger().warn(f"lane overlay failed: {exc}")

        if self.enable_det and self._front_msg and self._front_seq != self._done_front_seq:
            try:
                self.pub_front.publish(self._process_front(self._front_msg))
                self._done_front_seq = self._front_seq
            except Exception as exc:
                self.get_logger().warn(f"front overlay failed: {exc}")

        if self._tick_i % 30 == 0:
            self.get_logger().info(
                f"overlay seq lane={self._done_lane_seq}/{self._lane_seq} "
                f"front={self._done_front_seq}/{self._front_seq}"
            )

    def _process_lane(self, msg: CompressedImage) -> CompressedImage:
        lane = cv2.resize(crop_lane_roi(_decode_jpeg(msg)), LANE_SIZE)
        if self.segmenter is None:
            _put_status(lane, "seg unavailable")
            return self._encode(lane, msg)
        seg = self.segmenter(lane)
        overlay = _overlay_seg(lane, seg, self.seg_alpha)
        return self._encode(overlay, msg)

    def _process_front(self, msg: CompressedImage) -> CompressedImage:
        front = cv2.resize(_decode_jpeg(msg), FRONT_SIZE)
        if self.detector is None:
            _put_status(front, "det unavailable")
            return self._encode(front, msg)
        det = self.detector(front)
        overlay = _overlay_det(front, det)
        return self._encode(overlay, msg)

    def _encode(self, bgr: np.ndarray, src: CompressedImage) -> CompressedImage:
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        out = CompressedImage()
        out.header = src.header
        out.format = "jpeg"
        out.data = jpg.tobytes()
        return out


def _decode_jpeg(msg: CompressedImage) -> np.ndarray:
    arr = np.frombuffer(msg.data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    return bgr


def _overlay_seg(lane: np.ndarray, seg: np.ndarray, alpha: float) -> np.ndarray:
    vis = lane.copy()
    base = 1.0 - alpha
    for c in range(SEG_N_CLASSES):
        mask = seg[c] > 0
        vis[mask] = (vis[mask].astype(np.float32) * base + SEG_COLORS_BGR[c] * alpha)
    return vis.astype(np.uint8)


def _overlay_det(front: np.ndarray, det: np.ndarray) -> np.ndarray:
    vis = front.copy()
    if det[4] > 0:
        x, y, w, h, conf = det
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"car {conf:.2f}",
            (int(x), max(12, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return vis


def _put_status(img: np.ndarray, text: str) -> None:
    cv2.putText(
        img,
        text,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )


def main() -> None:
    rclpy.init()
    node = OverlayVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
