"""E2E 자율주행 네트워크 (차선 주행 + 정지 차량 추월).

입력:
  lane_img   (B, 3, 224, 224)  SegFormer 차선 마스크 오버레이된 Lane 이미지
  front_img  (B, 3, 224, 224)  YOLO 차량 bbox 오버레이된 Front 이미지
출력:
  steer      (B,)      [-1, 1]   → rover_control에서 angular.z로 변환  (메인)
  throttle   (B,)      [-1, 1]   → rover_control에서 linear.x로 변환   (메인)
  waypoints  (B, 5, 2)  meters   로봇 프레임 미래 궤적 (x_forward, y_left)  (보조)

구조: LaneEncoder/FrontEncoder(ResNet18×2) → concat → ControlHead + WaypointHead.
waypoint는 보조 출력(multi-task head): GT는 cmd_vel 적분 궤적(extract_labels.waypoint_gt)
으로, 공유 feature를 멀티스텝 의도 쪽으로 regularize해 steer/throttle 일반화를 돕는다.
추론 시에는 사용 안 함(디버깅/시연 때만 시각화). SegFormer/YOLO는 전처리 단계에서
freeze, ResNet18·헤드는 처음부터 학습.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# waypoint: 미래 2.5초를 5점으로 (extract_labels.WP_N, WP_HORIZON_S와 일치)
WP_N = 5


class LaneEncoder(nn.Module):
    """
    ResNet18-A — Lane 이미지 전용 인코더

    입력: SegFormer가 차선 마스크를 오버레이한 Lane 이미지 (B, 3, 224, 224)
          - 좌실선 / 우실선 / 중앙점선을 색상으로 구별해서 오버레이
          - 차선 곡률이 이미지에 그대로 나타나므로 코너링 여부를 모델이 직접 읽음
    출력: lane_feat (B, 256)

    학습: ImageNet pretrained 재사용, freeze 없이 전체 학습
          SegFormer는 이미 freeze되어 있으므로 gradient 경로 불균형 없음
    """
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # FC 레이어 제거, GAP(avgpool)까지만 사용
        # children 순서: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # output shape: (B, 512, 1, 1)

        # projection: 512 → 256, LayerNorm으로 스케일 정규화
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # closed-world 과적합 방지
        )

    def forward(self, x):
        feat = self.encoder(x)   # (B, 512, 1, 1)
        feat = feat.flatten(1)   # (B, 512)
        feat = self.proj(feat)   # (B, 256)
        return feat


class FrontEncoder(nn.Module):
    """
    ResNet18-B — Front 이미지 전용 인코더

    입력: YOLO가 차량 bbox를 오버레이한 Front 이미지 (B, 3, 224, 224)
          - YOLO는 별도 fine-tuning 후 freeze → bbox 오버레이 이미지 생성 전용
          - 모델은 "정지 차량이 이 위치/크기로 있을 때 어떻게 움직여야 하나"를 학습
            (bbox 위치 + 크기로 거리감을 암묵적으로 학습 → 회피/추월 타이밍)
    출력: front_feat (B, 256)

    학습: ImageNet pretrained 재사용, freeze 없이 전체 학습
    """
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        feat = self.encoder(x)   # (B, 512, 1, 1)
        feat = feat.flatten(1)   # (B, 512)
        feat = self.proj(feat)   # (B, 256)
        return feat


class SteerHead(nn.Module):
    """
    waypoints → steer, throttle  (cascade 제어 출력)

    입력: WaypointHead가 낸 waypoints 평탄화 (B, WP_N*2 = 10)
          ※ E2ENet.forward 에서 waypoints.detach() 로 넘긴다 — backbone/waypoint 는
            waypoint loss 로만 학습돼 "정직한 미래 경로"로 유지되고, SteerHead 는 그
            위를 읽는 깨끗한 downstream reader 가 된다("학습된 pure pursuit").
    출력: steer (B,), throttle (B,)  ∈ [-1, 1]

    weaving 회피: 입력 waypoint 가 cmd_vel 적분이라 저주파(부드러움) → steer 출력도
    부드럽다. 타깃은 시간 스무딩된 사람 steer(dataset steer_smooth)로 노이즈 제거.
    5점 전체를 보므로 단일 lookahead pure pursuit 보다 회피 path 모양을 더 잘 읽는다.

    역변환(추론): linear.x = -(0.20 + |steer|*0.05), angular.z = steer*1.2.
    입력 차원이 작아(10) BatchNorm 대신 Dropout 만. 처음부터 학습.
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(WP_N * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(64, 64),
            nn.ReLU(inplace=True),

            # 64 → 2 (steer, throttle)
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, wp_flat):
        return self.head(wp_flat)    # (B, 2)


class WaypointHead(nn.Module):
    """
    concat feature → waypoints (5점)  (보조 출력)

    입력: concat(lane_feat, front_feat) = 512
    출력: waypoints (B, 5, 2)  meters, 로봇 프레임 (x_forward, y_left)

    구조:
    - 512 → 512 → 256 → 64 → Linear(64, WP_N*2=10) → (B, 5, 2)로 reshape
    - Tanh 없음 (waypoint는 미터 단위 unbounded 회귀값)

    역할(cascade): backbone 을 멀티스텝 의도(미래 경로)로 학습시키는 주 신호이자,
          SteerHead 가 읽어 조향을 만드는 입력. GT는 cmd_vel 적분 궤적.

    학습: pretrained 없음, 처음부터 학습
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 64 → 10 (5점 × 2좌표), 활성화 없음
            nn.Linear(64, WP_N * 2),
        )

    def forward(self, x):
        out = self.head(x)            # (B, 10)
        return out.view(-1, WP_N, 2)  # (B, 5, 2)


class E2ENet(nn.Module):
    """
    전체 E2E 네트워크

    구조 (cascade):
      LaneEncoder(lane_img)     → lane_feat   (B, 256)
      FrontEncoder(front_img)   → front_feat  (B, 256)
                                   concat      (B, 512)
      WaypointHead(combined)    → waypoints (B, 5, 2)      [의도]
      SteerHead(waypoints.detach().flatten) → steer, throttle  [제어, 학습된 pursuit]

    gradient 흐름:
      backbone(encoders)+WaypointHead 는 waypoint loss 로 학습(정직한 경로 유지).
      SteerHead 는 detach 된 waypoints 를 입력으로 받아 steer loss 로만 학습 →
      waypoint 를 왜곡하지 않는 깨끗한 waypoint→steer 변환기.
      SegFormer, YOLO 는 이 네트워크 외부에서 freeze.
    """
    def __init__(self):
        super().__init__()
        self.lane_encoder   = LaneEncoder()
        self.front_encoder = FrontEncoder()
        self.waypoint_head = WaypointHead()
        self.steer_head    = SteerHead()

    def forward(self, lane_img, front_img):
        lane_feat   = self.lane_encoder(lane_img)       # (B, 256)
        front_feat = self.front_encoder(front_img)   # (B, 256)

        combined = torch.cat([lane_feat, front_feat], dim=1)  # (B, 512)

        waypoints = self.waypoint_head(combined)             # (B, 5, 2)
        # detach: waypoint 는 waypoint loss 로만 학습, steer 는 그 위를 읽기만.
        ctrl      = self.steer_head(waypoints.detach().flatten(1))  # (B, 2)
        steer     = ctrl[:, 0]                               # (B,)
        throttle  = ctrl[:, 1]                               # (B,)
        return steer, throttle, waypoints


class E2ELoss(nn.Module):
    """
    steer / throttle / waypoint MSE loss

    가중치:
      steer_weight=1.0     조향이 주행 안정성에 가장 중요
      throttle_weight=0.5  throttle은 steer에 coupling되므로 낮게
      waypoint_weight=0.5  보조 task — 메인을 regularize하되 압도하지 않게

    참고: throttle GT는 텔레옵 1D steering level coupling에 의해
          직선 ≈ -0.20, 회전 ≈ -0.25 분포.
          waypoint GT는 cmd_vel 적분 궤적이라 모터 응답 지연/슬립으로
          약간 부정확 → 가중치를 메인보다 낮게 둬서 noise 영향 제한.
    """
    def __init__(self, steer_weight=1.0, throttle_weight=0.5, waypoint_weight=0.5):
        super().__init__()
        self.steer_w    = steer_weight
        self.throttle_w = throttle_weight
        self.waypoint_w = waypoint_weight

    def forward(self, steer_pred, throttle_pred, wp_pred,
                steer_gt, throttle_gt, wp_gt):
        steer_loss    = F.mse_loss(steer_pred, steer_gt)
        throttle_loss = F.mse_loss(throttle_pred, throttle_gt)
        wp_loss       = F.mse_loss(wp_pred, wp_gt)
        total = (self.steer_w * steer_loss
                 + self.throttle_w * throttle_loss
                 + self.waypoint_w * wp_loss)
        return total, steer_loss, throttle_loss, wp_loss


if __name__ == "__main__":
    model = E2ENet()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터:  {total:,}")
    print(f"학습 파라미터:  {trainable:,}")

    B = 4
    lane_img   = torch.randn(B, 3, 224, 224)
    front_img = torch.randn(B, 3, 224, 224)

    steer, throttle, wp = model(lane_img, front_img)
    print(f"steer:     {tuple(steer.shape)}  {steer.min().item():.3f} ~ {steer.max().item():.3f}")
    print(f"throttle:  {tuple(throttle.shape)}  {throttle.min().item():.3f} ~ {throttle.max().item():.3f}")
    print(f"waypoints: {tuple(wp.shape)}  {wp.min().item():.3f} ~ {wp.max().item():.3f}")

    criterion   = E2ELoss()
    steer_gt    = torch.randn(B)
    throttle_gt = torch.randn(B)
    wp_gt       = torch.randn(B, WP_N, 2)
    loss, sl, tl, wl = criterion(steer, throttle, wp, steer_gt, throttle_gt, wp_gt)
    print(f"loss: total={loss:.4f}  steer={sl:.4f}  throttle={tl:.4f}  waypoint={wl:.4f}")
