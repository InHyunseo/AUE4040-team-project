"""Export a trained center-regression CNN to ONNX."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.center_cnn import build_center_cnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    model = build_center_cnn(pretrained=False).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    dummy = torch.zeros(1, 3, args.size, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["input"], output_names=["center_xy"],
        opset_version=17, dynamic_axes=None,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
