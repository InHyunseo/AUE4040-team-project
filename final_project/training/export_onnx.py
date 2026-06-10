"""E2ENet 체크포인트 → ONNX (PHASE2.md 5단계).

  python export_onnx.py --ckpt ../models/e2e_best.pt --out ../models/e2e.onnx

train_e2e.py 가 저장한 {"model": state_dict, ...} 또는 순수 state_dict 를 모두
받는다. 입력은 lane/front 두 개 (1,3,224,224), 출력은 steer/throttle/waypoints.
추론(실주행)에서 waypoints 출력은 시각화 전용 — engine 변환 후 무시.

이후 Jetson 에서만:
  /usr/src/tensorrt/bin/trtexec --onnx=e2e.onnx --fp16 --saveEngine=e2e.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import E2ENet  # noqa: E402


def load_e2e(ckpt_path: str) -> E2ENet:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model = E2ENet()
    model.load_state_dict(state)
    model.eval()
    if isinstance(obj, dict) and "val_total" in obj:
        print(f"loaded ckpt: epoch={obj.get('epoch')} val_total={obj['val_total']:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "models" / "e2e.onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--check", action="store_true",
                    help="onnxruntime 로 PyTorch vs ONNX 출력 일치 검증")
    args = ap.parse_args()

    model = load_e2e(str(args.ckpt))

    dummy_lane = torch.randn(1, 3, 224, 224)
    dummy_front = torch.randn(1, 3, 224, 224)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy_lane, dummy_front), str(args.out),
        input_names=["lane", "front"],
        output_names=["steer", "throttle", "waypoints"],
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"exported -> {args.out}")

    if args.check:
        import numpy as np
        import onnxruntime as ort
        with torch.no_grad():
            ts, tt, tw = model(dummy_lane, dummy_front)
        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        os, ot, ow = sess.run(
            None, {"lane": dummy_lane.numpy(), "front": dummy_front.numpy()})
        for name, a, b in [("steer", ts.numpy(), os),
                           ("throttle", tt.numpy(), ot),
                           ("waypoints", tw.numpy(), ow)]:
            md = float(np.abs(a.reshape(-1) - b.reshape(-1)).max())
            print(f"  {name:9s} max|Δ| = {md:.2e}")
        print("check done (expect ~1e-5 or less)")

    print("next (Jetson only): trtexec --onnx=%s --fp16 --saveEngine=e2e.engine" % args.out)


if __name__ == "__main__":
    main()
