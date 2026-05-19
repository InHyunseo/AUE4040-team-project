"""
Pure OpenCV stereo depth pipeline. No VPI required.

Useful as a sanity check the moment notebook 02 produces the rectify YAML,
before installing VPI 3.
"""

import sys
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from camera import Camera  # noqa: E402


CAPTURE_SIZE = (1280, 720)
RESCALE_SIZE = (480, 270)
CALIB_YAML = ROOT / "calib" / "rectify_map_imx219_160deg_720p.yaml"

# StereoSGBM params — tune as needed.
MIN_DISP = 0
NUM_DISP = 96        # must be divisible by 16
BLOCK_SIZE = 7

# Shadow-suppression: CLAHE on the luminance channel before matching.
USE_CLAHE = True
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_color(rgb):
    """Return RGB uint8 with CLAHE applied to L of LAB (shadow-flattened)."""
    if not USE_CLAHE:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def get_calibration():
    if not CALIB_YAML.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIB_YAML}")
    fs = cv2.FileStorage(str(CALIB_YAML), cv2.FILE_STORAGE_READ)
    map_l = (fs.getNode("map_l_x").mat(), fs.getNode("map_l_y").mat())
    map_r = (fs.getNode("map_r_x").mat(), fs.getNode("map_r_y").mat())
    fs.release()
    return map_l, map_r


class CameraThread(Thread):
    def __init__(self, sensor_id):
        super().__init__(daemon=True)
        self._camera = Camera(sensor_id)
        self._should_run = True
        self._image = self._camera.read()  # BGR
        self.start()

    def run(self):
        while self._should_run:
            try:
                img = self._camera.read()  # BGR
            except RuntimeError:
                break  # camera was released during shutdown
            if img is not None:
                self._image = img

    @property
    def image(self):
        return self._image

    def stop(self):
        self._should_run = False
        self._camera.stop()


def main():
    print("Reading camera calibration...")
    map_l, map_r = get_calibration()

    cam_l = CameraThread(0)
    cam_r = CameraThread(1)

    matcher = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE ** 2,
        P2=32 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    t0 = time.perf_counter()
    i = 0
    try:
        while True:
            arr_l = cam_l.image
            arr_r = cam_r.image
            if arr_l is None or arr_r is None:
                time.sleep(0.005)
                continue

            rect_l = cv2.remap(arr_l, *map_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(arr_r, *map_r, cv2.INTER_LINEAR)

            rect_l = cv2.resize(rect_l, RESCALE_SIZE)
            rect_r = cv2.resize(rect_r, RESCALE_SIZE)

            rect_l = preprocess_color(rect_l)
            rect_r = preprocess_color(rect_r)

            disp = matcher.compute(rect_l, rect_r).astype(np.float32) / 16.0
            disp_vis = np.clip(disp / NUM_DISP * 255, 0, 255).astype(np.uint8)
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

            cv2.imshow("Disparity (OpenCV SGBM)", disp_vis)
            cv2.imshow("Left rectified", cv2.cvtColor(rect_l, cv2.COLOR_RGB2BGR))

            i += 1
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    finally:
        dt = time.perf_counter() - t0
        if i:
            print(f"~{i / dt:.1f} FPS over {i} frames")
        cam_l.stop()
        cam_r.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
