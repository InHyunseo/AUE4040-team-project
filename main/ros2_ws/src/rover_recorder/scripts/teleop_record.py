"""
Manual teleop + dual CSI camera recording (no ROS, SSH-friendly).

Reads keys directly from the controlling terminal in cbreak mode — no X
display, no pynput. Run it from a plain SSH session.

Keys (tap or hold; OS keyboard auto-repeat drives "hold"):
  w / s        : accelerate forward / reverse
  a / d        : steer left / right
  space        : hard stop (steer=0, speed=0)
  r            : toggle recording on/off
  q  or  ESC   : quit

If no input arrives within a tick, speed decays *0.9 and steer *0.5
(so letting go = coasting to stop).

Output:
  ~/rover_data/<session>_<ts>/left/<idx:06d>.jpg
  ~/rover_data/<session>_<ts>/right/<idx:06d>.jpg
  ~/rover_data/<session>_<ts>/annotation.csv   idx,timestamp,steer,speed

Run:
  python3 teleop_record.py --session test1
"""
import argparse
import csv
import json
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import serial

# Vendored jetcam under team/calibration/camera/jetcam (carries sync=false fix).
sys.path.insert(0, str(Path.home() / "team" / "calibration" / "camera"))
from jetcam.csi_camera import CSICamera  # noqa: E402


MAX_STEER = 0.8
MAX_SPEED = 0.5
STEP_STEER = 0.2
STEP_SPEED = 0.05
TICK_HZ = 10.0


class Motor:
    def __init__(self, dev: str, baud: int):
        try:
            self.ser = serial.Serial(dev, baud, timeout=1)
        except serial.SerialException as e:
            print(f"[motor] serial open failed ({e}); running dry")
            self.ser = None
        self.q: "queue.Queue[dict]" = queue.Queue()
        threading.Thread(target=self._writer, daemon=True).start()

    def _writer(self):
        while True:
            data = self.q.get()
            if self.ser is None:
                continue
            try:
                self.ser.write((json.dumps(data) + "\n").encode())
            except Exception as e:
                print(f"[motor] write error: {e}")

    def drive(self, L: float, R: float):
        self.q.put({"T": 1, "L": L, "R": R})

    def stop(self):
        self.drive(0.0, 0.0)


def clip(v, m):
    return max(min(v, m), -m)


def mix(steering: float, speed: float):
    s = clip(steering, MAX_STEER)
    base = abs(speed)
    L = clip(base * max(0.0, 1.0 - s), MAX_SPEED)
    R = clip(base * max(0.0, 1.0 + s), MAX_SPEED)
    if speed < 0:
        L, R = -L, -R
    # HYU rover wiring: invert to match physical drive direction.
    return -L, -R


def drain_stdin() -> str:
    """Return all characters available on stdin without blocking."""
    chars = []
    while select.select([sys.stdin], [], [], 0)[0]:
        c = sys.stdin.read(1)
        if not c:
            break
        chars.append(c)
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "rover_data")
    ap.add_argument("--uart", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start-recording", action="store_true",
                    help="begin recording immediately (otherwise press 'r')")
    args = ap.parse_args()

    if not sys.stdin.isatty():
        print("[err] stdin is not a TTY. Run this directly in an SSH terminal,"
              " not piped or backgrounded.")
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = args.out_root / f"{args.session}_{ts}"
    (session_dir / "left").mkdir(parents=True, exist_ok=True)
    (session_dir / "right").mkdir(parents=True, exist_ok=True)
    ann_path = session_dir / "annotation.csv"
    ann_file = open(ann_path, "w", newline="")
    ann = csv.writer(ann_file)
    ann.writerow(["idx", "timestamp", "steer", "speed"])

    print(f"[rec] session dir: {session_dir}")
    print(f"[rec] keys: w/s speed, a/d steer, space stop, r toggle rec, q/ESC quit")
    print(f"[rec] tip: hold keys (OS auto-repeat) or tap fast; let go to coast.")

    cam_l = cam_r = None
    try:
        cam_l = CSICamera(capture_device=0, capture_width=args.width,
                          capture_height=args.height,
                          downsample=args.downsample, capture_fps=args.fps)
        cam_r = CSICamera(capture_device=1, capture_width=args.width,
                          capture_height=args.height,
                          downsample=args.downsample, capture_fps=args.fps)
    except Exception as e:
        print(f"[cam] init failed: {e}")
        if cam_l is not None:
            try: cam_l.cap.release()
            except Exception: pass
        if cam_r is not None:
            try: cam_r.cap.release()
            except Exception: pass
        ann_file.close()
        sys.exit(2)
    motor = Motor(args.uart, args.baud)

    steering = 0.0
    speed = 0.0
    recording = bool(args.start_recording)
    idx = 0
    dt = 1.0 / TICK_HZ

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        quit_flag = False
        while not quit_flag:
            t0 = time.time()
            buf = drain_stdin()

            saw = {"w": False, "s": False, "a": False, "d": False}
            for c in buf:
                if c == "q" or c == "\x1b":  # q or ESC
                    quit_flag = True
                elif c == " ":
                    steering, speed = 0.0, 0.0
                elif c == "r":
                    recording = not recording
                    print(f"\n[rec] recording = {recording}")
                elif c in saw:
                    saw[c] = True

            if saw["w"]:
                speed += STEP_SPEED
            elif saw["s"]:
                speed -= STEP_SPEED
            else:
                speed *= 0.9
            if saw["a"]:
                steering -= STEP_STEER
            elif saw["d"]:
                steering += STEP_STEER
            else:
                steering *= 0.5
            speed = clip(speed, MAX_SPEED)
            steering = clip(steering, MAX_STEER)

            L, R = mix(steering, speed)
            motor.drive(L, R)

            try:
                frame_l = cam_l.read()
                frame_r = cam_r.read()
            except RuntimeError as e:
                print(f"\n[cam] {e}")
                break

            if recording:
                stamp = time.time()
                cv2.imwrite(str(session_dir / "left" / f"{idx:06d}.jpg"), frame_l)
                cv2.imwrite(str(session_dir / "right" / f"{idx:06d}.jpg"), frame_r)
                ann.writerow([idx, f"{stamp:.6f}",
                              f"{steering:.4f}", f"{speed:.4f}"])
                ann_file.flush()
                idx += 1

            sys.stdout.write(
                f"\r[{'REC' if recording else '   '}] "
                f"frames={idx:6d}  steer={steering:+.2f}  speed={speed:+.2f}  "
                f"L={L:+.2f} R={R:+.2f}   "
            )
            sys.stdout.flush()

            sleep_left = dt - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\n[exit] stopping motor")
        motor.stop()
        time.sleep(0.1)
        motor.stop()
        for c in (cam_l, cam_r):
            if c is not None:
                try: c.cap.release()
                except Exception: pass
        ann_file.close()
        print(f"[exit] saved {idx} frame pairs to {session_dir}")


if __name__ == "__main__":
    main()
