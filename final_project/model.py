"""
AUE4040 자동차임베디드AI — E2E 자율주행 네트워크

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 파이프라인에서 이 네트워크의 위치
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BEV 카메라
    → SegFormer (별도 fine-tuning 후 freeze)
    → 차선 마스크 오버레이 이미지 생성
    → [BEVEncoder] ← 이 파일에서 학습

  Front 카메라
    → YOLO (별도 fine-tuning 후 freeze)
    → bbox 오버레이 이미지 생성
    → [FrontEncoder] ← 이 파일에서 학습

  FSM 상태 (rover_decision 토픽)
    → [FSMEncoder] ← 이 파일에서 학습

  → [ControlHead] → steer, throttle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  bev_img     (B, 3, 224, 224)  SegFormer 차선 마스크 오버레이된 BEV 이미지
  front_img   (B, 3, 224, 224)  YOLO bbox 오버레이된 Front 이미지
  fsm_scalars (B, 11)           FSM 스칼라 벡터

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  steer    (B,)  [-1, 1]  → rover_control에서 실제 angular.z로 변환
  throttle (B,)  [-1, 1]  → rover_control에서 실제 linear.x로 변환

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
학습 전략 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SegFormer:    트랙 데이터로 차선 세그 fine-tuning → freeze
                (이 파일과 무관, 전처리 파이프라인에서 처리)

  YOLO:         트랙 데이터로 객체인식 fine-tuning → freeze
                (이 파일과 무관, 전처리 파이프라인에서 처리)

  ResNet18 A,B: ImageNet pretrained → freeze 없이 처음부터 학습
                SegFormer/YOLO가 freeze되므로 gradient 경로 차이 문제 없음
                전처리된 이미지를 보고 제어값을 학습하는 핵심 백본

  FSMEncoder:   처음부터 학습 (pretrained 없음)

  ControlHead:  처음부터 학습

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
throttle GT 분포 (텔레옵 1D steering level 방식 기준)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  직선:  throttle ≈ -0.15, steer ≈ 0.0
  회전:  throttle ≈ -0.25, steer ≈ ±0.8
  → 모델이 steer 클수록 throttle 높이는 coupling을 자연스럽게 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ──────────────────────────────────────────────────────
# FSM 스칼라 구성 (총 11개)
# ──────────────────────────────────────────────────────
# [0]  stop_bbox_h        stop 표지판 bbox 높이 (없으면 0)
# [1]  stop_visible_step  stop 감지 후 경과 프레임 (타이머 역할)
#                         → "stop 보고 3초 지났으면 출발" 학습 가능
# [2]  red_visible        적신호 감지 여부 (0 or 1)
# [3]  green_visible      녹신호 감지 여부 (0 or 1)
# [4]  person_bbox_h      보행자 bbox 높이 (없으면 0)
# [5]  car_bbox_h         전방 차량 bbox 높이 (없으면 0)
#                         → 높을수록 차량 가까움 (d = K / bbox_h)
# [6]  left_bbox_h        left 표지판 bbox 높이
# [7]  right_bbox_h       right 표지판 bbox 높이
# [8]  current_fsm_state  현재 FSM 상태 정수
#                         COMMON=0, SLOW=1, WAITING=2,
#                         STOPPED=3, TURNING=4, ARRIVED=5
# [9]  mission            latch된 미션 (none=0, left=1, right=2)
#                         → 표지판이 사라진 후에도 유지
# [10] mission_step       미션 진입 후 경과 프레임
#                         → 회전교차로에서 얼마나 돌았는지 표현
FSM_DIM = 11


class BEVEncoder(nn.Module):
    """
    ResNet18-A — BEV 이미지 전용 인코더

    입력: SegFormer가 차선 마스크를 오버레이한 BEV 이미지 (B, 3, 224, 224)
          - 좌측 실선, 우측 실선, 중앙 점선을 색상으로 구별해서 오버레이
          - 중앙 점선: 회전교차로 내부 위치 판단에 활용
            (회전교차로 안에서는 점선이 곡선 형태로 나타남 → 모델이 패턴 학습)
    출력: bev_feat (B, 256)

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
        # LayerNorm 선택 이유: 배치 크기 무관하게 안정적, 다른 branch와 스케일 맞춤
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # closed-world 과적합 방지
        )

    def forward(self, x):
        feat = self.encoder(x)   # (B, 512, 1, 1)
        feat = feat.flatten(1)   # (B, 512)
        feat = self.proj(feat)   # (B, 256)
        return feat


class FrontEncoder(nn.Module):
    """
    ResNet18-B — Front 이미지 전용 인코더

    입력: YOLO가 bbox를 오버레이한 Front 이미지 (B, 3, 224, 224)
          - 클래스별 색상으로 bbox 표시: car/stop/red/green/left/right/person
          - YOLO는 별도 fine-tuning 후 freeze → bbox 오버레이 이미지 생성 전용
          - 모델은 "bbox가 이 위치에 있을 때 어떻게 움직여야 하나"를 학습
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
            nn.Dropout(0.3),
        )

    def forward(self, x):
        feat = self.encoder(x)   # (B, 512, 1, 1)
        feat = feat.flatten(1)   # (B, 512)
        feat = self.proj(feat)   # (B, 256)
        return feat


class FSMEncoder(nn.Module):
    """
    FSM 스칼라 → dense feature

    입력: fsm_scalars (B, 11)
    출력: fsm_feat (B, 64)

    역할:
    - 이미지만으로 표현할 수 없는 시간적 맥락 제공
    - stop_visible_step: "stop 보고 몇 초 지났는가" → 출발 타이밍 학습
    - mission + mission_step: 표지판 사라진 후에도 회전 지속
    - current_fsm_state: 지금 어떤 상태인지 명시적으로 전달

    학습: pretrained 없음, 처음부터 학습
    정규화: LayerNorm (입력 스케일이 스칼라마다 다르므로 정규화 중요)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FSM_DIM, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.encoder(x)   # (B, 64)


class ControlHead(nn.Module):
    """
    concat feature → steer, throttle

    입력: concat(bev_feat, front_feat, fsm_feat) = 256 + 256 + 64 = 576
    출력: steer (B,), throttle (B,)  ∈ [-1, 1]

    정규화: BatchNorm1d 사용
    - FC 레이어 간 internal covariate shift 방지
    - LayerNorm과 달리 BatchNorm은 FC 이후에 적합

    출력 활성화: Tanh
    - steer/throttle 범위를 [-1, 1]로 강제
    - rover_control에서 실제 값으로 역변환:
        linear.x  = -(0.15 + abs(steer) * 0.10)  # -0.15 ~ -0.25
        angular.z = steer * 0.8                   # -0.8 ~ +0.8

    학습: pretrained 없음, 처음부터 학습
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            # 576 → 512
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # 256 → 64
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 64 → 2
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.head(x)      # (B, 2)


class E2ENet(nn.Module):
    """
    전체 E2E 네트워크

    구조:
      BEVEncoder(bev_img)       → bev_feat   (B, 256)
      FrontEncoder(front_img)   → front_feat (B, 256)
      FSMEncoder(fsm_scalars)   → fsm_feat   (B,  64)
                                   concat    (B, 576)
      ControlHead(combined)     → steer, throttle

    gradient 흐름:
      loss.backward() 한 번으로 BEVEncoder, FrontEncoder, FSMEncoder,
      ControlHead 전체 동시 업데이트
      SegFormer, YOLO는 이 네트워크 외부에서 freeze되므로 영향 없음
    """
    def __init__(self):
        super().__init__()
        self.bev_encoder   = BEVEncoder()
        self.front_encoder = FrontEncoder()
        self.fsm_encoder   = FSMEncoder()
        self.control_head  = ControlHead()

    def forward(self, bev_img, front_img, fsm_scalars):
        bev_feat   = self.bev_encoder(bev_img)       # (B, 256)
        front_feat = self.front_encoder(front_img)   # (B, 256)
        fsm_feat   = self.fsm_encoder(fsm_scalars)   # (B, 64)

        combined = torch.cat(
            [bev_feat, front_feat, fsm_feat], dim=1  # (B, 576)
        )

        out      = self.control_head(combined)       # (B, 2)
        steer    = out[:, 0]                         # (B,)
        throttle = out[:, 1]                         # (B,)
        return steer, throttle


class E2ELoss(nn.Module):
    """
    steer / throttle MSE loss

    가중치:
      steer_weight=1.0    조향이 주행 안정성에 더 중요
      throttle_weight=0.5 throttle은 steer에 coupling되므로 낮게

    참고: throttle GT는 텔레옵 1D steering level coupling에 의해
          직선 ≈ -0.15, 회전 ≈ -0.25 분포 → 모델이 steer-throttle
          coupling을 자연스럽게 학습
    """
    def __init__(self, steer_weight=1.0, throttle_weight=0.5):
        super().__init__()
        self.steer_w    = steer_weight
        self.throttle_w = throttle_weight

    def forward(self, steer_pred, throttle_pred, steer_gt, throttle_gt):
        steer_loss    = F.mse_loss(steer_pred, steer_gt)
        throttle_loss = F.mse_loss(throttle_pred, throttle_gt)
        total = self.steer_w * steer_loss + self.throttle_w * throttle_loss
        return total, steer_loss, throttle_loss


if __name__ == "__main__":
    model = E2ENet()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터:  {total:,}")
    print(f"학습 파라미터:  {trainable:,}")

    B = 4
    bev_img     = torch.randn(B, 3, 224, 224)
    front_img   = torch.randn(B, 3, 224, 224)
    fsm_scalars = torch.randn(B, FSM_DIM)

    steer, throttle = model(bev_img, front_img, fsm_scalars)
    print(f"steer:    {steer.shape}  {steer.min().item():.3f} ~ {steer.max().item():.3f}")
    print(f"throttle: {throttle.shape}  {throttle.min().item():.3f} ~ {throttle.max().item():.3f}")

    criterion   = E2ELoss()
    steer_gt    = torch.randn(B)
    throttle_gt = torch.randn(B)
    loss, sl, tl = criterion(steer, throttle, steer_gt, throttle_gt)
    print(f"loss: total={loss:.4f}  steer={sl:.4f}  throttle={tl:.4f}")