# rover_training

ROS-외부 학습 스크립트와 데이터 디렉토리. `colcon`이 건드리지 않습니다.

## Layout

```
data/
  raw/                          # 세션별 원본 (recorder_node 출력)
  processed/
    common/  left/  right/     # segment별 분할
scripts/
  preprocess.py                 # raw -> processed split (segment label 기준)
  train_center.py               # center-regression CNN 학습 (3개 모델)
  train_yolo.py                 # YOLOv8 fine-tune
  export_to_onnx.py
  export_to_trt.sh
models/
  center_cnn.py                 # ResNet18-head (rover/cnn/center_dataset.py 호환)
notebooks/
  data_analysis.ipynb           # histogram / distribution checks
  label_centers.ipynb           # 수동 클릭 라벨링
  model_eval.ipynb              # cross-eval (left model on right images, etc.)
```

## Sources

- `center_cnn.py` / `train_center.py` 베이스:
  `~/HYU-ECL3003/rover/cnn/center_dataset.py` + `train_road_center_model.ipynb`
- `train_yolo.py` 베이스: `~/HYU-ECL3003/week07/YOLOv8/demo.ipynb`
