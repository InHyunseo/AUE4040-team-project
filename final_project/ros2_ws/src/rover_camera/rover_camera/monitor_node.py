"""Browser-based MJPEG monitor for the rover camera streams.

Forwards camera_node's JPEG bytes as-is over an MJPEG
(multipart/x-mixed-replace) stream — the browser decodes natively, so there's
no re-encode and the monitor never touches the control path.

Serves on http://<host>:<port>/ :
  /            HTML page with all configured streams side by side
  /stream/<k>  multipart/x-mixed-replace MJPEG for stream key <k>

Streams come from the `streams` parameter: a list of "key:topic" entries, each
a sensor_msgs/CompressedImage (jpeg). Default: lane + front. Add another
"key:topic" entry to expose more streams — no code change needed.

Default binds 0.0.0.0 (open from a laptop on the same network). Set
host:=127.0.0.1 to keep it local only.
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from rover_common.constants import FRONT_IMAGE_TOPIC, LANE_IMAGE_TOPIC
from rover_common.qos import SENSOR_QOS

BOUNDARY = "rovermonitorframe"


class FrameStore:
    """Holds the latest JPEG bytes per stream key, with a condition var so
    the HTTP threads block until a genuinely new frame arrives (no busy spin)."""

    def __init__(self, keys: list[str]) -> None:
        self._cv = threading.Condition()
        self._data: dict[str, bytes] = {k: b"" for k in keys}
        self._seq: dict[str, int] = {k: 0 for k in keys}
        self._last_t: dict[str, float] = {k: 0.0 for k in keys}

    def set(self, key: str, jpg: bytes) -> None:
        with self._cv:
            self._data[key] = jpg
            self._seq[key] += 1
            self._last_t[key] = time.time()
            self._cv.notify_all()

    def wait_for(self, key: str, last_seq: int, timeout: float) -> tuple[bytes, int]:
        """Block until seq[key] != last_seq (new frame) or timeout. Returns
        (jpeg_bytes, new_seq). On timeout returns the current frame unchanged."""
        with self._cv:
            if self._seq[key] == last_seq:
                self._cv.wait_for(lambda: self._seq[key] != last_seq, timeout=timeout)
            return self._data[key], self._seq[key]

    def ages(self) -> dict[str, float]:
        now = time.time()
        with self._cv:
            return {k: (now - t if t else float("inf")) for k, t in self._last_t.items()}


def _make_handler(store: FrameStore, keys: list[str]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # silence per-request stderr spam
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._serve_index()
            elif self.path.startswith("/stream/"):
                key = self.path[len("/stream/"):]
                if key in keys:
                    self._serve_stream(key)
                else:
                    self.send_error(404, "unknown stream")
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            cells = "".join(
                f'<figure><figcaption>{k}</figcaption>'
                f'<img src="/stream/{k}" alt="{k}"></figure>'
                for k in keys
            )
            html = (
                "<!doctype html><html><head><meta charset=utf-8>"
                "<title>rover monitor</title><style>"
                "body{background:#111;color:#ddd;font-family:sans-serif;margin:0;padding:12px}"
                "main{display:flex;flex-wrap:wrap;gap:12px}"
                "figure{margin:0}figcaption{margin-bottom:4px;font-size:13px}"
                "img{max-width:48vw;border:1px solid #444;background:#000}"
                "</style></head><body><main>" + cells + "</main></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def _serve_stream(self, key: str) -> None:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            )
            self.end_headers()
            seq = 0
            try:
                while True:
                    jpg, seq = store.wait_for(key, seq, timeout=2.0)
                    if not jpg:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(
                        f"--{BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (browser tab) closed — normal

    return Handler


class MonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("rover_monitor")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        # "key:topic" pairs. Add lane-seg / detection overlays here when available.
        self.declare_parameter(
            "streams",
            [f"lane:{LANE_IMAGE_TOPIC}", f"front:{FRONT_IMAGE_TOPIC}"],
        )

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        spec = list(self.get_parameter("streams").value)

        self._keys: list[str] = []
        self._topic_to_key: dict[str, str] = {}
        for entry in spec:
            key, _, topic = entry.partition(":")
            if not key or not topic:
                self.get_logger().warn(f"ignoring bad stream spec: {entry!r}")
                continue
            self._keys.append(key)
            self._topic_to_key[topic] = key
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, k=key: self._on_image(msg, k), SENSOR_QOS,
            )

        self.store = FrameStore(self._keys)

        handler = _make_handler(self.store, self._keys)
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

        shown = host if host != "0.0.0.0" else "<this-host-ip>"
        self.get_logger().info(
            f"monitor on http://{shown}:{port}/  streams={self._keys}"
        )
        self._log_timer = self.create_timer(5.0, self._log_ages)

    def _on_image(self, msg: CompressedImage, key: str) -> None:
        # Forward the JPEG bytes as-is. No decode, no re-encode.
        self.store.set(key, bytes(msg.data))

    def _log_ages(self) -> None:
        ages = self.store.ages()
        parts = " ".join(f"{k}={a:.2f}s" for k, a in ages.items())
        self.get_logger().info(f"frame age {parts}")

    def destroy_node(self) -> bool:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = MonitorNode()
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
