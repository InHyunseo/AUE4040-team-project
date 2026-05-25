"""
Action-classification BC CNN with step input.

Input: image (1,3,224,224) + step (1,1) normalized to [0,1].
Output: 6 logits (one per action class).

ResNet18 ImageNet features -> concat step -> FC head.
Action lookup (idx -> steer, speed) lives in rover_lane.center_inference.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


NUM_CLASSES = 6
STEP_EMBED_DIM = 32


class ActionCNN(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES,
                 step_embed_dim: int = STEP_EMBED_DIM):
        super().__init__()
        backbone = tvm.resnet18(
            weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Raw step is a single scalar in [0,1] — direct concat to a 512-D image
        # feature lets the FC layer learn to ignore it. Project it up first so
        # it actually contributes to the decision.
        self.step_embed = nn.Sequential(
            nn.Linear(1, step_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(step_embed_dim, step_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + step_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        s = self.step_embed(step)
        return self.head(torch.cat([feats, s], dim=1))


def build_center_cnn(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    return ActionCNN(pretrained=pretrained, num_classes=num_classes)


if __name__ == "__main__":
    model = build_center_cnn()
    img = torch.zeros(1, 3, 224, 224)
    step = torch.zeros(1, 1)
    print(model(img, step).shape)  # -> torch.Size([1, 6])
