"""
VPI-accelerated stereo depth pipeline.

Requires NVIDIA VPI 3 (JetPack 6.x):
    sudo apt install libnvvpi3 vpi3-dev python3.10-vpi3

Reads rectification maps produced by calib/02_calibrate_create_rectification_map.ipynb
(rectify_map_imx219_160deg_720p.yaml with map_l_x/map_l_y/map_r_x/map_r_y, CV_32FC1).
"""

import os
import sys
import time
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from camera import Camera  # noqa: E402


MAX_DISP = 128
WINDOW_SIZE = 10
CAPTURE_SIZE = (1280, 720)       # (W, H) — must match the calibration capture res
RESCALE_SIZE = (480, 270)        # (W, H) for disparity output
CALIB_YAML = ROOT / "calib" / "rectify_map_imx219_160deg_720p.yaml"

# Shadow-suppression: CLAHE on the gray frames before VPI stereodisp.
USE_CLAHE = True
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def get_calibration():
    if not CALIB_YAML.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIB_YAML}\n"
            "Run the capture script and notebooks 01-02 first."
        )
    fs = cv2.FileStorage(str(CALIB_YAML), cv2.FILE_STORAGE_READ)
    map_l = (fs.getNode("map_l_x").mat(), fs.getNode("map_l_y").mat())
    map_r = (fs.getNode("map_r_x").mat(), fs.getNode("map_r_y").mat())
    fs.release()
    return map_l, map_r


def make_vpi_warpmap(cv_maps):
    """Convert OpenCV (map_x, map_y) float32 maps -> vpi.WarpMap."""
    import vpi
    map_x, map_y = cv_maps
    H, W = map_x.shape
    warp = vpi.WarpMap(vpi.WarpGrid((W, H)))
    arr_warp = np.asarray(warp)
    arr_warp[:H, :W, 0] = map_x
    arr_warp[:H, :W, 1] = map_y
    return warp


class CameraThread(Thread):
    def __init__(self, sensor_id):
        super().__init__(daemon=True)
        self._camera = Camera(sensor_id)
        self._should_run = True
        self._image = self._camera.read_gray()
        self.start()

    def run(self):
        while self._should_run:
            try:
                img = self._camera.read_gray()
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


class Depth(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        print("Reading camera calibration...")
        self._map_l, self._map_r = get_calibration()
        self._cam_l = CameraThread(0)
        self._cam_r = CameraThread(1)
        self._should_run = True
        self._dq = queue.deque(maxlen=3)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.start()
        while len(self._dq) < 1:
            time.sleep(0.1)

    def stop(self):
        self._should_run = False
        self._cam_l.stop()
        self._cam_r.stop()

    def disparity(self):
        while len(self._dq) == 0:
            time.sleep(0.01)
        return self._dq.pop()

    def _enqueue(self, vpi_image):
        arr = vpi_image.cpu().copy()
        self._dq.append(arr)

    def run(self):
        import vpi
        warp_l = make_vpi_warpmap(self._map_l)
        warp_r = make_vpi_warpmap(self._map_r)
        i = 0
        while self._should_run:
            i += 1
            with vpi.Backend.CUDA:
                arr_l = self._cam_l.image
                arr_r = self._cam_r.image
                if arr_l is None or arr_r is None:
                    time.sleep(0.005)
                    continue

                if USE_CLAHE:
                    arr_l = CLAHE.apply(arr_l)
                    arr_r = CLAHE.apply(arr_r)

                vpi_l = vpi.asimage(arr_l).remap(warp_l).rescale(
                    RESCALE_SIZE, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO
                )
                vpi_r = vpi.asimage(arr_r).remap(warp_r).rescale(
                    RESCALE_SIZE, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO
                )

                vpi_l_16 = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16 = vpi_r.convert(vpi.Format.U16, scale=1)

                disp_16 = vpi.stereodisp(
                    vpi_l_16, vpi_r_16,
                    out_confmap=None,
                    backend=vpi.Backend.CUDA,
                    window=WINDOW_SIZE,
                    maxdisp=MAX_DISP,
                )
                disp_8 = disp_16.convert(
                    vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)
                )
                self._executor.submit(self._enqueue, disp_8)

            if i % 10 == 0:
                vpi.clear_cache()


def main():
    depth = Depth()
    t0 = time.perf_counter()
    i = 0
    try:
        while True:
            disp = depth.disparity()
            disp_rgb = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
            cv2.imshow("Disparity", disp_rgb)
            left = depth._cam_l.image
            if left is not None:
                cv2.imshow("Left (rectified input)", cv2.resize(left, RESCALE_SIZE))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            i += 1
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    finally:
        depth.stop()
        cv2.destroyAllWindows()
        dt = time.perf_counter() - t0
        if i:
            print(f"~{i/dt:.1f} FPS over {i} frames")


if __name__ == "__main__":
    main()
