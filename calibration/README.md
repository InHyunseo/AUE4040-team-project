# Stereo CSI Calibration & Depth (team)

Standalone stereo calibration + depth pipeline for the two IMX219 CSI
cameras on this Jetson (L4T R36.5 / JetPack 6.2). Adapted from
`~/calibration` but rewritten to use `jetcam.CSICamera` (the same
backend used by `~/HYU-ECL3003/rover/camera_live_dual.py`) instead of
the broken PyGObject GStreamer bindings.

## Assumptions

- **Chessboard**: 9 × 7 inner corners, **20 mm** square size. Override
  in the notebooks if your target differs.
- **Capture resolution**: 1280 × 720 @ 30 fps.
- **Stereo baseline / focal**: not used here; configure in your
  downstream depth-to-distance code.
- `jetcam` is imported from `/home/ircv16/HYU-ECL3003/rover/jetcam`. To
  vendor it in, copy that folder into this directory and remove the
  `sys.path.insert(...)` line in `camera/__init__.py`.

## Layout

```
calibration/
├── camera/__init__.py                 # jetcam-backed Camera
├── calib/
│   ├── capture_stereo.py              # spacebar = save pair, q = quit
│   ├── calib_images/{left,right}/     # capture output
│   ├── 01_intrinsics_lens_dist.ipynb  # per-camera K, dist
│   ├── 02_calibrate_create_rectification_map.ipynb
│   ├── 03_remap.ipynb                 # sanity-check rectification
│   └── rectify_map_imx219_160deg_720p.yaml  # produced by 02
└── depth_pipeline_python/
    ├── depth_opencv.py                # no VPI needed
    └── depth_vpi.py                   # requires VPI 3
```

## Workflow

```bash
cd ~/team/calibration
```

1. **Capture** ~20–30 chessboard pairs across the working volume:
   ```bash
   python3 calib/capture_stereo.py
   # space = save, q = quit
   ```
2. **Per-camera intrinsics** — open and run all cells in
   `calib/01_intrinsics_lens_dist.ipynb`. Produces `K_l.npy`, `K_r.npy`,
   `dist_coeff_l.npy`, `dist_coeff_r.npy` in `calib/`.
3. **Stereo rectification** — run all cells in
   `calib/02_calibrate_create_rectification_map.ipynb`. Produces
   `calib/rectify_map_imx219_160deg_720p.yaml` with float32 maps
   (`map_l_x`, `map_l_y`, `map_r_x`, `map_r_y`).
4. **Visual check** — run `calib/03_remap.ipynb`; rectified rows should
   align across left/right.
5. **Live depth**:
   ```bash
   python3 depth_pipeline_python/depth_opencv.py   # CPU SGBM, works immediately
   ```
   For the VPI path:
   ```bash
   sudo apt update
   sudo apt install libnvvpi3 vpi3-dev python3.10-vpi3
   python3 depth_pipeline_python/depth_vpi.py
   ```

## Troubleshooting

- `Could not open CSI camera` / no frames: `sudo systemctl restart nvargus-daemon`.
- `ModuleNotFoundError: jetcam`: check that `/home/ircv16/HYU-ECL3003/rover/jetcam`
  exists, or vendor it in.
- VPI packages not found: confirm `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list`
  points to `r36.5`, then `sudo apt update`. On this Jetson the
  available versions are `vpi3` (not `vpi1`).
- Notebook 02 produces empty maps: usually means notebook 01 was run
  with too few valid chessboard detections — recapture more poses.

## Camera assignment

`Camera(0)` is **left**, `Camera(1)` is **right** in both the capture
script and the depth pipelines. If your physical wiring is opposite,
swap the indices in `calib/capture_stereo.py` and in the
`CameraThread` instantiations inside `depth_pipeline_python/*.py`.
