# 사전 학습 가이드

트랙2 지정주제(KITTI 객체인지 + Hailo-10H 엣지 배포)를 진행하기 전에 알아둬야 할 개념을 정리한 문서입니다.
**꼭 알아야 할 것 → 알면 좋은 것** 순서로 정리했습니다.

## 목차

1. [딥러닝 기초](#1-딥러닝-기초)
2. [객체 탐지 (Object Detection)](#2-객체-탐지-object-detection)
3. [평가 지표](#3-평가-지표-이번-대회-채점-기준)
4. [KITTI 데이터셋 다루기](#4-kitti-데이터셋-다루기)
5. [모델 경량화 / 최적화](#5-모델-경량화--최적화-대회의-핵심)
6. [성능 측정 방법론](#6-성능-측정-방법론-보고서-점수-35점)
7. [모델 포맷과 툴체인](#7-모델-포맷과-툴체인)
8. [임베디드 / 시스템](#8-임베디드--시스템)
9. [개발 도구](#9-개발-도구)
10. [추천 학습 순서](#추천-학습-순서)
11. [흔히 빠지는 함정](#흔히-빠지는-함정)

---

## 1. 딥러닝 기초

| 개념 | 알아야 하는 이유 |
| --- | --- |
| **CNN** (Convolution, Stride, Padding, Pooling) | YOLO 같은 탐지 모델의 기본 구성 요소예요. Hailo NPU는 Conv 연산에 최적화되어 있어요. |
| **BatchNorm, 활성화 함수** (ReLU, SiLU, LeakyReLU) | 양자화할 때 BN을 Conv에 합치는 **BN folding**이 쓰이고, 활성화 함수에 따라 양자화 난이도가 달라져요. |
| **Backbone / Neck / Head 구조** | 모델의 어느 부분을 줄이거나 바꿀지 판단할 때 필요해요. |
| **학습 흐름** (Loss, Optimizer, LR, Epoch, Overfitting) | KITTI에 맞춰 fine-tuning 해야 해요. |
| **전이학습 / Fine-tuning** | COCO로 사전학습된 가중치에서 시작해 KITTI 클래스에 맞게 재학습해요. |

- [ ] 공부 완료

## 2. 객체 탐지 (Object Detection)

| 개념 | 핵심 |
| --- | --- |
| **Bounding Box 표현** | `(x1,y1,x2,y2)` 형식과 `(cx,cy,w,h)` 형식 간 변환, 정규화 좌표 |
| **IoU** | 예측 박스와 정답 박스가 겹치는 정도로, 평가의 기준이 돼요. |
| **NMS** (Non-Maximum Suppression) | 겹치는 중복 박스를 제거해요. **NPU에서 돌릴지 CPU에서 돌릴지**가 속도에 영향을 줘요. |
| **Confidence threshold** | 이 값에 따라 Precision과 Recall이 서로 반대로 움직여요. |
| **One-stage vs Two-stage** | YOLO와 SSD는 one-stage라 빠르고, Faster R-CNN은 two-stage라 느려요. |
| **Anchor-based vs Anchor-free** | YOLOv8 이후 모델은 anchor-free예요. |
| **YOLO 계열 차이** (v8 / v10 / v11, 크기 n·s·m) | 베이스라인 모델을 고를 때 필요해요. v10은 NMS-free 구조예요. |

- [ ] 공부 완료

## 3. 평가 지표 (이번 대회 채점 기준)

- **Precision, Recall, PR 커브**
- **AP (Average Precision)**: PR 커브 아래 면적이에요.
  - **AP11 vs AP40**: Recall 지점을 11개 쓰느냐 40개 쓰느냐의 차이예요. 이번 대회는 **AP40**을 써요.
- **mAP**: 여러 클래스의 AP 평균이에요. 이번 대회는 Car, Pedestrian, Cyclist 세 클래스예요.
- **KITTI 클래스별 IoU 기준**: Car는 **0.7**, Pedestrian과 Cyclist는 **0.5**예요. COCO와 달라서 Car 기준이 꽤 엄격해요.
- **KITTI 난이도 구분 (Easy / Moderate / Hard)**: 박스 높이, 가려짐(occlusion), 잘림(truncation)으로 나눠요.

  | 난이도 | 최소 박스 높이 | 가려짐 | 잘림 |
  | --- | --- | --- | --- |
  | Easy | 40px | 완전히 보임 | 15% 이하 |
  | **Moderate** | **25px** | **부분 가려짐까지** | **30% 이하** |
  | Hard | 25px | 많이 가려짐까지 | 50% 이하 |

  Moderate에는 25px 높이의 작은 객체까지 포함돼요. 그래서 **입력 해상도를 너무 줄이면 점수가 크게 떨어질 수 있어요.**
- **mAP Drop Rate**: 양자화 전후 정확도 유지율로, 예선 채점 항목이에요.

- [ ] 공부 완료

## 4. KITTI 데이터셋 다루기

- **라벨 형식**: 한 줄에 15개 필드가 있어요 (`type, truncated, occluded, alpha, bbox(4), dimensions(3), location(3), rotation_y`). 2D 탐지에는 `type`과 `bbox`만 쓰고, 난이도 판정에 `truncated`와 `occluded`를 써요.
- **클래스 구성**: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, **DontCare**. Van이나 Person_sitting 같은 클래스를 어떻게 처리할지(합칠지, 버릴지) 정해야 해요.
- **DontCare 영역**: 평가할 때 이 영역에 있는 탐지는 무시돼요.
- **YOLO 포맷 변환**: KITTI 라벨을 YOLO txt 파일(정규화된 `cls cx cy w h`)로 바꿔야 해요.
- **Train/Val 분리**: training 이미지 7,481장 중 **`eval_val.txt`의 1,000장을 학습에서 반드시 빼야 해요** (Data Leakage 시 실격). KITTI test 셋에는 라벨이 없어서 쓸 수 없어요.
- **이미지 비율**: 원본이 약 1242x375로 가로가 길어요. 그래서 **Letterbox**와 **직사각형 입력**(1280x384, 640x192) 개념을 알아야 해요.

- [ ] 공부 완료

## 5. 모델 경량화 / 최적화 (대회의 핵심)

### 양자화 (Quantization) ⭐ 가장 중요

- **기본 원리**: FP32 값을 INT8이나 INT4로 바꾸는 매핑이에요. 공식은 `q = round(x / scale) + zero_point`예요.
- **Symmetric vs Asymmetric** 양자화
- **Per-tensor vs Per-channel** 양자화
- **PTQ (Post-Training Quantization)**: 재학습 없이 **Calibration 데이터**로 scale을 정해요.
- **QAT (Quantization-Aware Training)**: 학습 과정에서 양자화 오차를 시뮬레이션해요. 정확도는 더 잘 지키지만 시간이 오래 걸려요.
- **Weight 양자화 vs Activation 양자화**
- **Mixed Precision**: 민감한 레이어(보통 첫 레이어와 Head)만 높은 정밀도로 남겨요.
- **양자화 민감도 분석**: 어떤 레이어가 정확도를 떨어뜨리는지 찾는 작업이에요.
- **INT4 / INT8 / FP16 차이**: Hailo-10H는 셋 다 지원해요.

### 그 외 기법

- **Pruning (가지치기)**: 구조적(structured) 방식과 비구조적 방식이 있어요. NPU에서 실제로 빨라지는 건 **구조적 pruning**뿐이에요.
- **Knowledge Distillation**: 큰 모델(teacher)의 출력을 작은 모델(student)이 학습하는 방식이에요.
- **모델 스케일 조절**: n, s, m 크기 선택, 입력 해상도 조절
- **연산자 교체**: NPU가 지원하지 않는 레이어를 호환되는 레이어로 바꿔요.

- [ ] 공부 완료

## 6. 성능 측정 방법론 (보고서 점수 35점)

| 지표 | 주의할 점 |
| --- | --- |
| **Params** (파라미터 수) | 모델 크기와 관련돼요. |
| **FLOPs / MACs** | 1 MAC ≈ 2 FLOPs예요. 둘을 섞어 쓰지 마세요. |
| **Latency vs Throughput(FPS)** | batch=1 기준의 latency가 엣지 환경에 맞는 지표예요. |
| **측정 방식** | Warm-up 몇 회 후에 N회 반복해서 평균과 표준편차를 내요. 전처리와 후처리(NMS)를 포함했는지 **명시**해야 해요. |
| **메모리** | Peak RAM이나 VRAM, 모델 파일 크기 |
| **재현성** | Seed 고정, 라이브러리 버전 고정, 실험 환경 명세 |

심사 측이 제출한 모델을 직접 돌려서 결과를 재현하기 때문에, **측정 스크립트를 한 번에 실행할 수 있게** 만들어 두는 게 중요해요.

- [ ] 공부 완료

## 7. 모델 포맷과 툴체인

- **PyTorch** (`.pt`) → **ONNX** (`.onnx`) 변환
  - opset 버전, **static shape**(고정 입력 크기), 변환 후 그래프 확인 (도구: **Netron**)
- **ONNX Runtime**: 데스크탑에서 INT8 PTQ를 하고 추론 속도를 측정할 때 써요 (예선용).
- (선택) **TensorRT, OpenVINO**: GPU나 CPU 전용 최적화 엔진이에요.
- **Ultralytics 라이브러리**: YOLO 학습, 검증, export 기능을 한 번에 제공해요.

### Hailo 툴체인 ⭐ 본선의 핵심

```
ONNX → [Parse] → .har → [Optimize: 양자화 + Calibration] → .har → [Compile] → .hef → HailoRT로 추론
```

- **Hailo Dataflow Compiler (DFC)**: x86 **Linux(Ubuntu)** 환경에서 실행돼요. Windows에서는 WSL2나 Docker가 필요해요.
- **Hailo-10H 전용 버전을 써야 해요.** Hailo-8용 DFC나 HailoRT와는 버전 계열이 달라요. 설치 전에 Hailo Developer Zone에서 Hailo-10H 대응 버전을 꼭 확인하세요.
- **중요한 포인트**: 최종 양자화는 **DFC가 직접 수행**해요. PyTorch에서 만든 INT8 모델이 그대로 올라가는 게 아니에요. 그래서 **예선의 양자화 전략이 DFC 방식(PTQ + calibration, optimization level)과 이어지도록** 설계하는 게 좋아요.
- **CPU Fallback**: DFC가 지원하지 않는 연산이 있으면 그 부분이 Pi CPU에서 돌아가서 속도가 크게 떨어져요.
- **Hailo Model Zoo**: 검증된 모델과 설정(YAML)이 모여 있어요. 먼저 여기서 YOLO 예제가 돌아가는지 확인해 보세요.
- **HailoRT**: 보드에서 쓰는 추론 런타임이에요 (Python API와 CLI 제공).

- [ ] 공부 완료

## 8. 임베디드 / 시스템

- **Linux 기본**: 셸, 파일 권한, 패키지 설치, SSH 원격 접속
- **Raspberry Pi 5**: ARM64 아키텍처, Raspberry Pi OS, PCIe로 AI HAT 연결
- **발열과 Throttling**: 온도가 오르면 클럭이 떨어져서 측정값이 흔들려요. 쿨러를 쓰고, 측정할 때 온도를 같이 기록하세요.
- **파이프라인 병렬화**: 전처리, NPU 추론, 후처리를 스레드로 나누면 FPS가 올라가요.
- (선택) **카메라 입력**: picamera2, GStreamer로 실시간 시연할 때 필요해요.

- [ ] 공부 완료

## 9. 개발 도구

- **Git / GitHub**: 브랜치, 커밋, 협업
- **Python 가상환경** (venv 또는 conda), `requirements.txt`
- **CUDA / GPU 학습 환경**: Colab이나 학교 서버를 쓸 수도 있어요.
- **Docker**: Hailo DFC 환경을 구성할 때 거의 필수예요.
- **WSL2**: Windows에서 Linux 도구를 쓸 때 필요해요.

- [ ] 공부 완료

---

## 추천 학습 순서

예선 마감(11/1) 기준, 9/28부터 약 5주 일정입니다.

| 기간 | 할 일 |
| --- | --- |
| **1주차** | 2~4번 (객체 탐지, mAP/AP40, KITTI 포맷), Ultralytics로 YOLOv8n을 KITTI에 fine-tuning |
| **2주차** | 6번 측정 스크립트 만들기 (AP40 평가, latency, 메모리 측정), 베이스라인 수치 확보 |
| **3~4주차** | 5번 양자화 (ONNX Runtime PTQ부터 시작해 QAT 순으로), 해상도 비교 실험. **동시에 7번 Hailo DFC 환경을 미리 설치하고 컴파일까지 시험해 보기** |
| **5주차** | 보고서 작성, 재현 코드 정리, 제출 |

## 흔히 빠지는 함정

1. `eval_val.txt` 이미지를 학습에 섞는 실수 (**실격**)
2. 평가를 COCO mAP(IoU 0.5:0.95)로 해 버리는 실수. 이번 대회는 **KITTI AP40, Moderate 기준**이에요.
3. 정사각형 640x640 입력을 그대로 쓰는 실수 (연산 낭비)
4. Hailo-8용 모델이나 문서를 참고하는 실수 (`.hef` 호환 안 됨)
5. latency를 잴 때 warm-up을 빼거나, 측정 조건을 명시하지 않는 실수
