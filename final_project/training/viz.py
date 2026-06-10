"""의도 시각화 — 학습/추론 공용 (켜고 끌 수 있음).

두 가지를 그린다. 둘 다 좌표 변환을 visualize_labels.draw_waypoints 와 동일하게
맞춰(같은 DEBUG_PPM, 같은 원점) 라벨/추론 좌표계가 안 갈라지게 한다:

  draw_intent(lane_bgr, wp, color)
      한 세트의 waypoint(미터, 로봇 프레임)를 lane 이미지에 점+선으로.
      추론 노드(rover_lane)가 예측 waypoint 를 :8080 디버그 화면에 그릴 때 이걸 쓴다.

  pred_vs_gt_panel(lane_bgr, front_bgr, pred_wp, gt_wp, pred_steer, pred_thr,
                   gt_steer, gt_thr)
      학습 모니터링용 패널: lane 에 GT waypoint(흰) vs 예측 waypoint(노랑) 겹쳐
      그리고, steer/throttle 의 예측 vs GT 를 텍스트로. lane+front 가로 결합.

입력 이미지는 합성된 BGR(데이터로더 to_input_tensor 직전 상태) 또는 정규화
텐서를 역변환한 것을 받는다 — viz 는 그리기만 하고 정규화는 모른다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# 좌표 변환 상수를 라벨 시각화와 한 소스로 공유 (좌표계 일치).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from visualize_labels import DEBUG_PPM  # noqa: E402

GT_COLOR   = (255, 255, 255)   # 흰 — GT waypoint
PRED_COLOR = (0, 255, 255)     # 노랑(BGR) — 예측 waypoint


def _wp_to_px(wp, W, H, ppm):
    """waypoint (N,2) 미터(로봇 프레임 x_forward,y_left) → 이미지 픽셀 (N,2).

    visualize_labels.draw_waypoints 와 동일: 원점은 이미지 하단 중앙 근처,
    +y(좌)→-u, +x(전방)→-v."""
    ox, oy = W // 2, H - H // 8
    pts = []
    for (x_m, y_m) in wp:
        u = int(ox - y_m * ppm)
        v = int(oy - x_m * ppm)
        pts.append((u, v))
    return (ox, oy), pts


def draw_intent(lane_bgr, wp, color=PRED_COLOR, ppm=DEBUG_PPM):
    """waypoint 한 세트를 lane 이미지(BGR)에 점+연결선으로. 추론 노드 공용."""
    out = lane_bgr.copy()
    H, W = out.shape[:2]
    (ox, oy), pts = _wp_to_px(wp, W, H, ppm)
    cv2.circle(out, (ox, oy), 4, (200, 200, 200), 1)
    for (u, v) in pts:
        cv2.circle(out, (u, v), 3, color, -1)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(out, a, b, color, 1)
    return out


def pred_vs_gt_panel(lane_bgr, front_bgr, pred_wp, gt_wp,
                     pred_steer, pred_thr, gt_steer, gt_thr,
                     ppm=DEBUG_PPM):
    """학습 모니터링 패널: lane 에 GT(흰)/예측(노랑) waypoint 겹치기 + 제어 텍스트.

    lane/front 는 합성된 BGR uint8 (224,224,3). 반환: 가로 결합 BGR 패널."""
    lane_vis = draw_intent(lane_bgr, gt_wp, color=GT_COLOR, ppm=ppm)
    lane_vis = draw_intent(lane_vis, pred_wp, color=PRED_COLOR, ppm=ppm)

    panel = np.hstack([lane_vis, front_bgr.copy()])
    H = panel.shape[0]
    cv2.putText(panel, f"steer  gt={gt_steer:+.2f} pred={pred_steer:+.2f}",
                (5, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(panel, f"throt  gt={gt_thr:+.2f} pred={pred_thr:+.2f}",
                (5, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # 범례
    cv2.putText(panel, "wp: white=GT  yellow=pred",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return panel
