# YOLO26 + TensorRT Quantization (Jetson Orin Nano)

**Author**: Hyun-seo In (Hanyang Univ. BME/AE) — <inhsroy@hanyang.ac.kr>
**Date**: 2026.05

실습 코드는 [02_yolo_quantization.ipynb](02_yolo_quantization.ipynb) 참조. 이 문서는 개념과 주의사항만 정리.

---

## 파이프라인

PyTorch (`.pt`) → ONNX → TensorRT 엔진 (`.engine`)
→ FP32 (베이스라인) vs FP16 vs INT8 비교

## 환경 주의사항

- **OpenCV**: Jetson에서 `pip install opencv-python` 불가 → `sudo apt install python3-opencv` 로 JetPack 최적화 버전 설치 (GStreamer 지원 필수).
- **CSI 카메라**: `cv2.VideoCapture(0)` 직접 사용 불가 → `nvarguscamerasrc` GStreamer 파이프라인 필요.
- **모델 크기**: n < s < m < l < x. Orin Nano 8GB 기준 m까지 실시간 추론 가능.

## TensorRT 엔진 빌드 주의

- 첫 빌드는 5~10분 소요.
- 빌드된 `.engine` 은 **해당 Jetson 하드웨어 전용** — 다른 기기로 이식 불가.
- INT8 빌드 시 `--fp16` 도 같이 켜는 것이 일반적 (fallback 용).

## 예상 성능 (참고치 — 실측 필요)

| Backend | Latency (ms) | FPS | 비고 |
|---------|-------------|-----|------|
| PyTorch FP32 | ~35 | ~28 | 베이스라인 |
| TensorRT FP16 | ~12 | ~83 | 정확도 유지 |
| TensorRT INT8 | ~8 | ~125 | 정확도 소폭 감소 |
