# 자율주행 AI 개선 프로젝트

> AUE4040 자동차임베디드AI — 기존 BC 모델 개선

---

## 한 줄 요약

**"규칙 기반 FSM이 제공하는 상황 정보를 선생님 삼아, E2E AI가 맥락을 이해하고 주행을 학습한다"**

---

## 기존 BC 모델의 문제

```
이미지 + step 숫자 → ResNet18 → 6가지 키 중 하나 선택
```

- step이 트랙에 과적합 → 새 트랙에서 망가짐
- 키 6개로만 표현 → 부드러운 제어 불가
- 왜 이 행동인지 맥락 없이 패턴만 암기
- 차량 회피 / stop 판단을 FSM이 별도 규칙으로 처리 → AI가 이해 못함
- step 때문에 데이터 추가학습 어려움

---

## 이번에 하려는 것

```
전처리된 이미지 2장 + FSM 스칼라 11개
  → ResNet18 x2 + FSM Encoder
  → steer / throttle 직접 출력
```

### 입력 1 — 전처리된 이미지 (2장)

```
BEV 카메라
  → SegFormer로 차선 세그 후 오버레이
  → 좌측 실선 / 우측 실선 / 중앙 점선 구별해서 색상으로 표시
  → ResNet18-A에 입력

Front 카메라
  → YOLO로 객체 감지 후 bbox 오버레이
  → car / stop / red / green / left / right / person 표시
  → ResNet18-B에 입력
```

SegFormer와 YOLO는 별도 fine-tuning 후 freeze — 이미지 전처리 전용.
ResNet18 두 개가 전처리된 이미지를 보고 제어값을 학습.

### 입력 2 — FSM 스칼라 (11개)

```
stop_bbox_h        stop 표지판 크기 (없으면 0)
stop_visible_step  stop 감지 후 경과 프레임
red_visible        적신호 감지 여부 (0/1)
green_visible      녹신호 감지 여부 (0/1)
person_bbox_h      보행자 크기 (없으면 0)
car_bbox_h         전방 차량 크기 (없으면 0)
left_bbox_h        left 표지판 크기
right_bbox_h       right 표지판 크기
current_fsm_state  현재 FSM 상태 (0~5)
mission            latch된 미션 (0=없음, 1=좌회전, 2=우회전)
mission_step       미션 진입 후 경과 프레임
```

### 중앙 점선의 역할

회전교차로 내부에서는 점선이 곡선 형태로 나타나고 출구 구간에서 패턴이 바뀐다. 점선 오버레이가 이미지에 포함되면 모델이 "아직 회전교차로 안인지, 나가야 하는지"를 스스로 학습. 별도 FSM 규칙 없이 E2E로 해결.

---

## 왜 이게 더 나은가

| | 기존 BC | 이번 |
|---|---|---|
| 선생님 | 키 입력만 | 키 입력 + 상황 전체 |
| 차량 회피 | FSM 규칙 | 모델이 이미지 보고 학습 |
| stop 후 출발 | FSM 규칙 | stop_visible_step으로 학습 |
| 회전교차로 판단 | FSM + mission_step | 점선 패턴 + mission으로 학습 |
| 새 환경 일반화 | 약함 (step 과적합) | 상황 기반이라 강건 |
| 추가학습 | step 때문에 어려움 | 자유로움 |

---

## 비유

```
기존: 답만 보고 외우기
이번: 풀이 과정(FSM 상태) + 답(steer/throttle) 같이 보고 학습
```

사람이 운전 배울 때처럼 —
"stop 표지판이 보이고, 3초 지났고, 우회전 미션이니까 이렇게 움직여라"
를 맥락째로 학습. 규칙 기반 FSM이 정확한 선생님 역할.

---

## 기존 코드 변경사항

```
그대로 유지:  YOLO, FSM, rover_stereo, rover_control, UART
바뀌는 것:    rover_lane 내부 모델 교체
              텔레옵 방식 교체 (버튼식 → 1D steering level)
추가:         SegFormer BEV 전처리 노드, FSM 스칼라 수집 로직
```

### 텔레옵 변경 (데이터 수집 품질 향상)

기존 버튼식 텔레옵을 **1D steering level + throttle coupling** 방식으로 교체.

```
turn_level: -5 ~ +5  (좌우 버튼으로 1단계씩 조절)
직진:  linear.x = -0.15, angular.z = 0.0
회전:  linear.x = -0.25까지 자동 증가, angular.z = ±0.8
```

회전 시 throttle이 자동으로 높아져서 차동 모터 토크 부족 문제를 구조적으로 해결. 데이터 라벨도 단계적으로 안정적.

### throttle-steer coupling 값

```
직선:  throttle -0.15, steer 0.0
회전:  throttle -0.25, steer ±0.8  (토크 보상)
```

이 coupling이 학습 데이터에 그대로 반영되어 모델이 회전 시 자동으로 throttle을 높이는 것을 학습.

---

## 해야 할 일 순서

```
1. SegFormer fine-tuning (좌실선/우실선/중앙점선 세그)
2. YOLO fine-tuning (기존 재사용 가능, left 클래스 데이터 보강)
3. 텔레옵 교체
     1D steering level + throttle coupling 구현
     turn_level -5~+5, smoothing 적용
4. 데이터 수집
     rosbag compressed 토픽으로 수집
     FSM 상태도 같이 기록
     직선 → 코너링 → 직선 복귀 시퀀스 반복 수집 (코너 데이터 충분히)
     stop/신호/회전교차로 상황 의도적으로 반복
5. 라벨 파이프라인
     rosbag → labels_cache.h5
     FSM 스칼라 자동 생성 (YOLO 재실행 + FSM 재현)
6. E2E 모델 학습 (Colab)
     ResNet18 x2 + FSM Encoder + Control Head
7. TRT 변환 + rover_lane 노드 교체
8. 실차 테스트 및 비교
```