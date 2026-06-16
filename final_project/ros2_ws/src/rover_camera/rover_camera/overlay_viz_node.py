"""Publish live SegFormer/YOLO overlay streams for the browser monitor.

Subscribes to the raw camera JPEG topics, runs the frozen Phase-1 models with
the same preprocessing contract used by data_pipeline/extract_labels.py, then
publishes JPEG overlays:

  /lane_seg/compressed   lane image + SegFormer mask alpha blend
  /front_det/compressed  front image + YOLO car bbox
"""
from __future__ import annotations

import threading
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from rover_common.constants import (
    FRONT_DET_TOPIC,
    FRONT_IMAGE_TOPIC,
    LANE_IMAGE_TOPIC,
    LANE_SEG_TOPIC,
)
from rover_common.image_io import encode_bgr
from rover_common.paths import find_final_project_root
from rover_common.qos import SENSOR_QOS

try:
    PROJECT_ROOT = find_final_project_root(__file__)
    from data_pipeline.extract_labels import (  # noqa: E402
        FRONT_SIZE,
        LANE_SIZE,
        SegFormerLaneSeg,
        YoloCarDet,
        crop_lane_roi,
    )
    # Same overlay-compositing contract the dataset/inference path uses, so the
    # browser preview matches training inputs pixel-for-pixel.
    from data_pipeline.preprocess import composite_lane, composite_front  # noqa: E402
    _HELPER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - reported clearly at node startup
    PROJECT_ROOT = None
    _HELPER_IMPORT_ERROR = exc


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

        self.declare_parameter("lane_topic", LANE_IMAGE_TOPIC)
        self.declare_parameter("front_topic", FRONT_IMAGE_TOPIC)
        self.declare_parameter("lane_overlay_topic", LANE_SEG_TOPIC)
        self.declare_parameter("front_overlay_topic", FRONT_DET_TOPIC)
        self.declare_parameter("segformer_ckpt", str(default_seg))
        self.declare_parameter("yolo_weights", str(default_yolo))
        self.declare_parameter("device", "cuda")
        self.declare_parameter("enable_seg", True)
        self.declare_parameter("enable_det", True)
        self.declare_parameter("viz_fps", 3.0)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("yolo_imgsz", 320)
        self.declare_parameter("yolo_conf", 0.25)

        self.jpeg_q = int(self.get_parameter("jpeg_quality").value)
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
            CompressedImage, lane_topic, self._on_lane, SENSOR_QOS
        )
        self.create_subscription(
            CompressedImage, front_topic, self._on_front, SENSOR_QOS
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
        self._seg_status = "seg loading"
        self._det_status = "det loading"

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
            self._seg_status = "seg numpy 2.x"
            self._det_status = "det numpy 2.x"
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
                self._seg_status = "seg missing deps"
            if "ultralytics" in missing:
                self.enable_det = False
                self._det_status = "det missing deps"
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
                self._seg_status = "seg ready"
            except Exception as exc:
                self.segmenter = None
                self._seg_status = _short_status("seg", exc)
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
                self._det_status = "det ready"
            except Exception as exc:
                self.detector = None
                self._det_status = _short_status("det", exc)
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
            _put_status(lane, self._seg_status)
            return encode_bgr(lane, msg.header, self.jpeg_q)
        seg = self.segmenter(lane)
        # Shared compositing → preview matches the training/inference lane input
        # pixel-for-pixel (fixed SEG_ALPHA, not a monitor-only knob).
        overlay = composite_lane(lane, seg)
        return encode_bgr(overlay, msg.header, self.jpeg_q)

    def _process_front(self, msg: CompressedImage) -> CompressedImage:
        front = cv2.resize(_decode_jpeg(msg), FRONT_SIZE)
        if self.detector is None:
            _put_status(front, self._det_status)
            return encode_bgr(front, msg.header, self.jpeg_q)
        det = self.detector(front)
        # composite_front draws the same bbox the model sees; add a conf label on
        # top for the human monitor (debug-only, not part of the model input).
        overlay = composite_front(front, det)
        _label_det(overlay, det)
        return encode_bgr(overlay, msg.header, self.jpeg_q)


def _decode_jpeg(msg: CompressedImage) -> np.ndarray:
    arr = np.frombuffer(msg.data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    return bgr


def _label_det(vis: np.ndarray, det: np.ndarray) -> None:
    """Debug-only conf label above the bbox (the box itself is drawn by
    composite_front, identical to the model input)."""
    if det[4] > 0:
        x, y, _, _, conf = det
        cv2.putText(
            vis,
            f"car {conf:.2f}",
            (int(x), max(12, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


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


def _short_status(prefix: str, exc: Exception) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    if len(msg) > 36:
        msg = msg[:33] + "..."
    return f"{prefix} {msg}"


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
