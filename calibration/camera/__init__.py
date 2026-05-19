"""
Minimal Camera wrapper around jetcam.CSICamera.

Provides the small surface that the rest of the calibration code expects:
    Camera(sensor_id) -> object with .read(), .read_gray(), .stop(),
                         .running(), .wait_ready()

jetcam is sourced from /home/ircv16/HYU-ECL3003/rover/jetcam (same pattern
used by team/main/ros2_ws/src/rover_stereo/calib/capture_stereo.py). To
remove that dependency later, vendor the jetcam/ folder into this repo.
"""

import atexit
import os
import sys
import time

import cv2
import numpy as np

_JETCAM_PARENT = "/home/ircv16/HYU-ECL3003/rover"
if _JETCAM_PARENT not in sys.path:
    sys.path.insert(0, _JETCAM_PARENT)

from jetcam.csi_camera import CSICamera  # noqa: E402


CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
DOWNSAMPLE = 1


class Camera:
    def __init__(
        self,
        sensor_id,
        capture_width=CAPTURE_WIDTH,
        capture_height=CAPTURE_HEIGHT,
        capture_fps=CAPTURE_FPS,
        downsample=DOWNSAMPLE,
    ):
        self._sensor_id = sensor_id
        self._cam = CSICamera(
            capture_device=sensor_id,
            capture_width=capture_width,
            capture_height=capture_height,
            capture_fps=capture_fps,
            downsample=downsample,
        )

        t0 = time.time()
        first = None
        while time.time() - t0 < 3.0:
            first = self._cam.read()
            if first is not None:
                break
            time.sleep(0.05)
        if first is None:
            raise RuntimeError(
                f"CSI camera sensor-id={sensor_id} produced no frames "
                f"(try: sudo systemctl restart nvargus-daemon)"
            )

        atexit.register(self.stop)

    def read(self):
        """Return latest frame as RGB uint8 ndarray (H, W, 3), or None.

        jetcam's GStreamer pipeline ends in `format=BGR`, so the raw frame
        from cv2.VideoCapture is BGR — we convert here to RGB. Callers that
        pipe into cv2.imshow / cv2.imwrite must convert back to BGR.
        """
        frame = self._cam.read()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_gray(self):
        """Return latest frame as GRAY8 uint8 ndarray (H, W), or None."""
        frame = self._cam.read()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def running(self):
        return self._cam is not None and self._cam.cap.isOpened()

    def wait_ready(self):
        while not self.running():
            time.sleep(0.1)

    def stop(self):
        try:
            if self._cam is not None:
                self._cam.cap.release()
                self._cam = None
        except Exception:
            pass
