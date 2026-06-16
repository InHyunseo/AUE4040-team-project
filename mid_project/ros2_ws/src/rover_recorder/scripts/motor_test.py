"""
Minimal motor sanity test — no cameras, no keyboard, no ROS.

Sends explicit {"T":1,"L":..,"R":..} commands and prints what the board
replies (if anything). Use this to figure out whether the motor isn't
moving because of (a) wrong port, (b) wrong baud, (c) sign flip,
(d) dead-zone, or (e) board firmware not accepting the message.

Usage:
  python3 motor_test.py                     # default sweep on /dev/ttyUSB0
  python3 motor_test.py --uart /dev/ttyUSB1
  python3 motor_test.py --power 0.8         # try stronger drive
  python3 motor_test.py --invert            # flip sign (try if drive does nothing)
"""
import argparse
import json
import sys
import time

import serial


def send(ser, data: dict, verbose: bool = True):
    line = (json.dumps(data) + "\n").encode("utf-8")
    ser.write(line)
    if verbose:
        print(f"  TX: {line!r}")


def drain_rx(ser, label: str):
    time.sleep(0.05)
    n = ser.in_waiting
    if n:
        rx = ser.read(n)
        print(f"  RX ({label}): {rx!r}")
    else:
        print(f"  RX ({label}): <nothing>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uart", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--power", type=float, default=0.5,
                    help="L/R magnitude to try (0..1)")
    ap.add_argument("--dur", type=float, default=2.0,
                    help="duration each phase runs (s)")
    ap.add_argument("--invert", action="store_true",
                    help="negate L/R (try if no movement)")
    args = ap.parse_args()

    print(f"[open] {args.uart} @ {args.baud}")
    try:
        ser = serial.Serial(args.uart, args.baud, timeout=0.5)
    except Exception as e:
        print(f"[err] open failed: {e}")
        sys.exit(1)

    # Some firmwares wait for DTR/RTS toggle to wake up.
    ser.setDTR(False); ser.setRTS(False); time.sleep(0.1)
    ser.setDTR(True);  ser.setRTS(True);  time.sleep(0.5)
    drain_rx(ser, "boot")

    sign = -1.0 if args.invert else 1.0
    p = args.power * sign

    phases = [
        ("FORWARD  both wheels", p, p),
        ("STOP",                0.0, 0.0),
        ("REVERSE both wheels", -p, -p),
        ("STOP",                0.0, 0.0),
        ("PIVOT LEFT  (R only)", 0.0, p),
        ("STOP",                0.0, 0.0),
        ("PIVOT RIGHT (L only)", p, 0.0),
        ("STOP",                0.0, 0.0),
    ]

    try:
        for label, L, R in phases:
            print(f"\n[{label}]  L={L:+.2f} R={R:+.2f}  for {args.dur:.1f}s")
            t_end = time.time() + args.dur
            first = True
            while time.time() < t_end:
                send(ser, {"T": 1, "L": float(L), "R": float(R)},
                     verbose=first)
                first = False
                time.sleep(0.1)
            drain_rx(ser, label)
    except KeyboardInterrupt:
        print("\n[int] keyboard interrupt")
    finally:
        print("\n[exit] sending STOP x3")
        for _ in range(3):
            send(ser, {"T": 1, "L": 0.0, "R": 0.0}, verbose=False)
            time.sleep(0.05)
        ser.close()


if __name__ == "__main__":
    main()
