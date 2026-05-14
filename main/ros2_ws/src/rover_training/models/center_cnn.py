"""
Center-regression CNN. Output: (x, y) in [-1, +1].

Baseline: ImageNet-pretrained ResNet18 with a 2-unit regression head — matches
HYU-ECL3003/rover/cnn/center_dataset.py output shape and the training loop in
rover/train_road_center_model.ipynb.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


def build_center_cnn(pretrained: bool = True) -> nn.Module:
    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    return backbone


if __name__ == "__main__":
    model = build_center_cnn()
    x = torch.zeros(1, 3, 224, 224)
    print(model(x).shape)  # -> torch.Size([1, 2])
