#!/usr/bin/env bash
# Convert an ONNX model to a TensorRT engine on the Jetson.
# Usage: ./export_to_trt.sh model.onnx model.engine [fp16|fp32]
set -euo pipefail

ONNX="${1:?onnx path}"
ENG="${2:?engine path}"
PREC="${3:-fp16}"

ARGS=("--onnx=$ONNX" "--saveEngine=$ENG")
if [[ "$PREC" == "fp16" ]]; then
    ARGS+=("--fp16")
fi

/usr/src/tensorrt/bin/trtexec "${ARGS[@]}"
