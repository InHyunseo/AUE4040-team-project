"""E2E autonomous-driving inference node (single integrated node).

Phase-3 inference counterpart of the manual teleop pipeline. Subscribes to the
two raw camera JPEG topics, runs the full perception+control stack in one node
and publishes /cmd_vel so motor_bridge_node drives the rover:

  /lane_image/compressed  ─┐
                           ├─> [crop+resize] ─> SegFormer ─> composite ─┐
  /front_image/compressed ─┘   [resize]      ─> YOLO      ─> composite ─┤
                                                                        ▼
                                          E2ENet (TensorRT fp16 engine)
                                                  steer, throttle, wp
                                                        │ (waypoints unused)
                                                        ▼
                              linear.x = -(0.20 + |steer|*0.05)   (neg = forward)
                              angular.z = steer * 1.2
                                                        ▼
                                                    /cmd_vel  (geometry_msgs/Twist)

Why single node (not a chain of nodes): the perception->control loop must be
low-latency. Routing SegFormer/YOLO outputs through JPEG-encoded topics into a
separate E2E node adds encode/decode + inter-node hops to every control cycle.
Running all three models in one process keeps tensors in memory.

Preprocessing MUST match data_pipeline/extract_labels.py + training/dataset.py
pixel-for-pixel, or the model sees a different distribution than it trained on:
  lane : decode -> crop_lane_roi(top 30%) -> resize 224 -> SegFormer -> composite
  front: decode -> resize 224 -> YOLO -> composite
  both : composite (BGR uint8) -> to_input_tensor (RGB, /255, ImageNet norm)

The E2E model is loaded as a TensorRT engine (built on Jetson with
trtexec --onnx=e2e.onnx --fp16 --saveEngine=e2e.engine). SegFormer/YOLO reuse
the frozen Phase-1 helpers from extract_labels.py (same as overlay_viz_node).

Run (separate SSH terminal, like teleop):
  ros2 run rover_lane e2e_infer_node

Or via launch (camera + motor_bridge + this):
  ros2 launch rover_lane drive.launch.py
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist


# Camera publishes RELIABLE/KEEP_LAST depth=1; subscribe BEST_EFFORT depth=1 so
# we always act on the freshest frame and never queue stale ones (control wants
# latest, not complete). depth=1 matches the "no stale frames" intent.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def _find_project_root() -> Path | None:
    """Locate final_project/ so data_pipeline + training imports resolve.

    Same contract as overlay_viz_node: env override, else walk up from here.
    """
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
    sys.path.insert(0, str(PROJECT_ROOT))                 # data_pipeline.*, model
    sys.path.insert(0, str(PROJECT_ROOT / "training"))    # dataset.*

# Frozen Phase-1 perception helpers + preprocessing contract. Imported lazily-ish
# (at module load) but failures are surfaced clearly at node construction so a
# missing dep doesn't crash with an opaque traceback.
try:
    from data_pipeline.extract_labels import (  # noqa: E402
        LANE_SIZE,
        FRONT_SIZE,
        SegFormerLaneSeg,
        YoloCarDet,
        crop_lane_roi,
        decode_compressed,
    )
    from dataset import composite_lane, composite_front, to_input_tensor  # noqa: E402
    _HELPER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - reported at startup
    _HELPER_IMPORT_ERROR = exc


# --------------------------------------------------------------- TensorRT engine


class TRTEngine:
    """Minimal TensorRT fp16 engine runner for the E2E model.

    Loads e2e.engine (built by `trtexec --onnx=e2e.onnx --fp16`), runs a single
    forward with two image inputs (lane, front) and returns the named outputs.
    Inputs/outputs are matched by binding name so engine I/O order changes don't
    silently break the mapping. CUDA context is created on the calling thread.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (creates + manages CUDA context)

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Discover binding names/roles. The exporter names inputs lane/front and
        # outputs steer/throttle/waypoints (export_onnx.py dynamic_axes keys).
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _binding_shape(self, name):
        idx = self.engine.get_binding_index(name)
        return tuple(self.context.get_binding_shape(idx))

    def infer(self, lane_chw: np.ndarray, front_chw: np.ndarray) -> dict:
        """lane/front (3,224,224) float32 -> {output_name: np.ndarray}.

        Adds the batch dim, copies H2D, runs, copies D2H. Single-sample only.
        """
        cuda = self.cuda
        feeds = {
            self.input_names[0]: lane_chw[None].astype(np.float32),
            self.input_names[1]: front_chw[None].astype(np.float32),
        }
        # If the exporter happens to name inputs explicitly, honor the names.
        if "lane" in self.input_names and "front" in self.input_names:
            feeds = {
                "lane": lane_chw[None].astype(np.float32),
                "front": front_chw[None].astype(np.float32),
            }

        bindings = [0] * self.engine.num_bindings
        device_buffers = []
        host_outputs = {}

        for name, arr in feeds.items():
            arr = np.ascontiguousarray(arr)
            idx = self.engine.get_binding_index(name)
            self.context.set_binding_shape(idx, arr.shape)
            dptr = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(dptr, arr)
            bindings[idx] = int(dptr)
            device_buffers.append(dptr)

        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            host = np.empty(shape, dtype=np.float32)
            dptr = cuda.mem_alloc(host.nbytes)
            bindings[idx] = int(dptr)
            device_buffers.append(dptr)
            host_outputs[name] = (host, dptr)

        self.context.execute_v2(bindings)

        out = {}
        for name, (host, dptr) in host_outputs.items():
            cuda.memcpy_dtoh(host, dptr)
            out[name] = host
        return out


# ----------------------------------------------------------------- the ROS node


class E2EInferNode(Node):
    def __init__(self) -> None:
        super().__init__("rover_e2e_infer")
        if _HELPER_IMPORT_ERROR is not None:
            raise RuntimeError(
                "failed to import perception/preprocessing helpers. Run with "
                "--symlink-install from final_project/ros2_ws or set "
                "AUE4040_FINAL_PROJECT_ROOT=/path/to/final_project"
            ) from _HELPER_IMPORT_ERROR

        default_seg = str(PROJECT_ROOT / "models" / "segformer_lane") if PROJECT_ROOT else ""
        default_yolo = str(PROJECT_ROOT / "models" / "best.pt") if PROJECT_ROOT else ""
        default_engine = str(PROJECT_ROOT / "models" / "e2e.engine") if PROJECT_ROOT else ""

        self.declare_parameter("lane_topic", "/lane_image/compressed")
        self.declare_parameter("front_topic", "/front_image/compressed")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("segformer_ckpt", default_seg)
        self.declare_parameter("yolo_weights", default_yolo)
        self.declare_parameter("engine_path", default_engine)
        self.declare_parameter("device", "cuda")
        # Safety: hold the rover still until both models are loaded AND we have a
        # fresh pair of frames. If a camera stream stalls longer than this many
        # seconds, publish a stop instead of acting on a stale frame.
        self.declare_parameter("stale_timeout_s", 0.5)
        # Optional cap on inference rate. The camera (~15 Hz) is the natural
        # limit; keep this comfortably ABOVE the camera rate so the cap never
        # beats against frame timing (a cap == camera rate drops ~half the
        # frames to jitter). 30 Hz = effectively "infer every frame".
        self.declare_parameter("max_rate_hz", 30.0)
        # Watchdog: an independent timer republishes the last command at this
        # rate (teleop ran 20 Hz) so motor_bridge always has a fresh command,
        # AND issues a hard stop if no inference has completed within
        # cmd_timeout_s. This closes the gap where the lane stream dies entirely
        # (no callback fires) and the rover would otherwise keep its last command.
        self.declare_parameter("watchdog_hz", 20.0)
        self.declare_parameter("cmd_timeout_s", 0.4)

        self.lane_topic = self.get_parameter("lane_topic").value
        self.front_topic = self.get_parameter("front_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        # Empty string (e.g. passed by a launch file's default) falls back to the
        # project-relative default so callers can omit paths.
        self.seg_ckpt = self.get_parameter("segformer_ckpt").value or default_seg
        self.yolo_weights = self.get_parameter("yolo_weights").value or default_yolo
        self.engine_path = self.get_parameter("engine_path").value or default_engine
        self.device = self.get_parameter("device").value
        self.stale_timeout_s = float(self.get_parameter("stale_timeout_s").value)
        self.min_period = 1.0 / max(1e-3, float(self.get_parameter("max_rate_hz").value))
        self.cmd_timeout_s = float(self.get_parameter("cmd_timeout_s").value)
        watchdog_hz = max(1e-3, float(self.get_parameter("watchdog_hz").value))

        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 10)

        # Latest raw frames (decoded BGR) + their arrival time. Lane drives the
        # control loop; front is consumed opportunistically (latest available).
        self._lock = threading.Lock()
        self._front_bgr = None
        self._front_t = 0.0
        self._last_pub_t = 0.0
        # Last command produced by inference + when it was produced. The watchdog
        # timer republishes this (and stops if it goes stale). Guarded by _lock.
        self._last_cmd = Twist()
        self._last_cmd_t = 0.0

        self.create_subscription(
            CompressedImage, self.front_topic, self._on_front, SENSOR_QOS)
        self.create_subscription(
            CompressedImage, self.lane_topic, self._on_lane, SENSOR_QOS)

        # Independent watchdog/republish timer (see cmd_timeout_s / watchdog_hz).
        self.create_timer(1.0 / watchdog_hz, self._watchdog)

        # Models load on a background thread so the node spins (and keeps the
        # rover stopped) while heavy weights initialize. Until ready, _on_lane
        # publishes a stop.
        self.ready = False
        self.segmenter = None
        self.detector = None
        self.engine = None
        threading.Thread(target=self._load_models, daemon=True).start()

        self.get_logger().info(
            f"e2e_infer: {self.lane_topic}+{self.front_topic} -> {self.cmd_topic} | "
            f"engine={self.engine_path} device={self.device} | loading models...")

    # ---- model loading (background) ----

    def _load_models(self) -> None:
        try:
            self.segmenter = SegFormerLaneSeg(self.seg_ckpt, device=self.device)
            self.detector = YoloCarDet(self.yolo_weights, device=self.device)
            self.engine = TRTEngine(self.engine_path)
            self.ready = True
            self.get_logger().info(
                f"models ready (engine inputs={self.engine.input_names}, "
                f"outputs={self.engine.output_names}). driving.")
        except Exception as exc:
            self.get_logger().error(f"model load failed, staying stopped: {exc!r}")

    # ---- subscriptions ----

    def _on_front(self, msg: CompressedImage) -> None:
        try:
            bgr = decode_compressed(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"front decode failed: {exc!r}")
            return
        with self._lock:
            self._front_bgr = bgr
            self._front_t = self._now()

    def _on_lane(self, msg: CompressedImage) -> None:
        now = self._now()
        # Rate cap: skip frames that arrive faster than the engine can serve.
        # (Default cap is well above camera rate, so this is normally a no-op.)
        if now - self._last_pub_t < self.min_period:
            return

        if not self.ready:
            # Don't stamp _last_cmd_t — the watchdog keeps issuing stops while
            # models load (last cmd stays stale on purpose).
            return

        with self._lock:
            front_bgr = self._front_bgr
            front_t = self._front_t
        # Need a recent front frame to mirror training inputs; if the front
        # stream is stale, skip this cycle. We do NOT refresh _last_cmd, so the
        # watchdog will time out and stop if this persists.
        if front_bgr is None or (now - front_t) > self.stale_timeout_s:
            self.get_logger().warn("front frame missing/stale -> skip",
                                   throttle_duration_sec=1.0)
            return

        try:
            lane_bgr = decode_compressed(msg.data)
            steer, throttle = self._infer(lane_bgr, front_bgr)
        except Exception as exc:
            self.get_logger().error(f"inference failed -> skip: {exc!r}",
                                    throttle_duration_sec=1.0)
            return

        # Publish immediately for responsiveness AND store as the latest command
        # so the watchdog can republish it at a steady rate until the next frame.
        self._publish_cmd(steer, throttle)
        self._last_pub_t = now

    # ---- core ----

    def _infer(self, lane_bgr: np.ndarray, front_bgr: np.ndarray):
        """raw BGR pair -> (steer, throttle) floats in [-1, 1].

        Mirrors extract_labels.py preprocessing exactly:
          lane : crop top 30% -> resize 224 -> SegFormer -> composite_lane
          front: resize 224 -> YOLO -> composite_front
        then to_input_tensor (shared with training).
        """
        # lane path
        lane_c = crop_lane_roi(lane_bgr)
        lane_c = cv2.resize(lane_c, LANE_SIZE)
        seg = self.segmenter(lane_c)                 # (3,224,224) uint8 {0,255}
        lane_comp = composite_lane(lane_c, seg)

        # front path
        front_r = cv2.resize(front_bgr, FRONT_SIZE)
        det = self.detector(front_r)                 # (5,) [x,y,w,h,conf]
        front_comp = composite_front(front_r, det)

        # to tensors (numpy CHW float32) -> engine
        lane_t = to_input_tensor(lane_comp).cpu().numpy()
        front_t = to_input_tensor(front_comp).cpu().numpy()
        out = self.engine.infer(lane_t, front_t)

        steer, throttle = self._read_control(out)
        return float(steer), float(throttle)

    @staticmethod
    def _read_control(out: dict):
        """Pull steer/throttle from engine outputs by name, with fallbacks.

        Exporter names outputs steer/throttle/waypoints. If names differ, fall
        back to the two smallest (scalar) outputs as steer, throttle in order.
        """
        if "steer" in out and "throttle" in out:
            return np.asarray(out["steer"]).ravel()[0], np.asarray(out["throttle"]).ravel()[0]
        scalars = sorted(
            (k for k in out if np.asarray(out[k]).size <= 1),
            key=lambda k: np.asarray(out[k]).size,
        )
        if len(scalars) >= 2:
            return np.asarray(out[scalars[0]]).ravel()[0], np.asarray(out[scalars[1]]).ravel()[0]
        raise RuntimeError(f"cannot locate steer/throttle in engine outputs: {list(out)}")

    def _publish_cmd(self, steer: float, throttle: float) -> None:
        """Map model steer -> Twist using the training control contract.

        From model.py ControlHead docstring (throttle output is intentionally
        NOT used for cmd_vel; speed is coupled to |steer|, matching teleop):
          linear.x  = -(0.20 + |steer| * 0.05)   # negative = forward on this rover
          angular.z = steer * 1.2
        Publishes now and records it as the latest command for the watchdog.
        """
        steer = max(-1.0, min(1.0, steer))
        cmd = Twist()
        cmd.linear.x = -(0.20 + abs(steer) * 0.05)
        cmd.angular.z = steer * 1.2
        self.pub_cmd.publish(cmd)
        with self._lock:
            self._last_cmd = cmd
            self._last_cmd_t = self._now()

    def _publish_stop(self) -> None:
        self.pub_cmd.publish(Twist())  # all-zero = motor_bridge stops

    def _watchdog(self) -> None:
        """Steady-rate republish + deadman stop (independent of frame callbacks).

        - If the last inference command is fresh (< cmd_timeout_s), republish it
          so motor_bridge sees a steady ~watchdog_hz stream even between frames.
        - If it's stale (lane stream died, inference hung, models still loading),
          publish a hard stop. This is the only path that catches a fully dead
          lane topic, since _on_lane wouldn't fire at all in that case.

        Note: the node runs on the default single-threaded executor, so this
        timer cannot preempt an in-flight inference — it fires in the gaps
        between frame callbacks. That's fine: _on_lane already publishes on every
        successful inference, so the watchdog's job is steady republish between
        frames + the deadman stop when callbacks stop firing entirely (where a
        single thread is not a limitation, because nothing else is running then).
        """
        now = self._now()
        with self._lock:
            cmd = self._last_cmd
            age = now - self._last_cmd_t
        if not self.ready or age > self.cmd_timeout_s:
            self._publish_stop()
        else:
            self.pub_cmd.publish(cmd)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self) -> bool:
        # Best-effort stop on shutdown so the rover doesn't keep its last command.
        try:
            self._publish_stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = E2EInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
