# embeded_ondeviceAI

2026학년도 모바일공학전공 하반기 공모전 **[트랙 2] 임베디드 환경 On-device AI 최적화 챌린지** 참가 저장소입니다.

> 주제 확정 전입니다. 현재는 **지정주제(자율주행 객체인지)** 로 진행하는 방향을 검토 중입니다.

## 대회 개요

자원이 제한된 임베디드 환경에서 오픈소스 AI 모델을 양자화·경량화하고, 실제 엣지 디바이스(NPU)에 탑재해 성능을 검증하는 대회입니다.

| 단계 | 기간 | 내용 |
| --- | --- | --- |
| 예선 | ~ 11/1(일) 23:59 | 자체 PC/서버 환경에서 양자화·최적화 후 보고서, 모델, 재현 코드 제출 |
| 예선 심사 | 11/2 ~ 11/5 | 제출 모델로 결과 재현 검증 |
| 본선 진출 발표 | 11/6(금) | 6팀 선발(지정 3 + 자유 3), 하드웨어 키트 지급 |
| 본선 | 11/6 ~ 11/30(월) 23:59 | Hailo-10H로 컴파일해 보드에 탑재하고 실측 |
| 최종 발표 | 12/4(금) | 발표·시연 및 시상 |

## 지정주제: 자율주행 객체인지 (Object Detection)

- **데이터셋**: KITTI Object Detection Evaluation 2012
  - `data_object_image_2.zip` (좌측 컬러 이미지, 약 12GB)
  - `data_object_label_2.zip` (라벨, 약 16MB)
  - Raw Data, Velodyne(LiDAR) 데이터는 받지 않습니다.
- **평가 메트릭**: KITTI 표준 **Moderate AP40** (Car, Pedestrian, Cyclist 3개 클래스 평균)
- **고정 검증셋**: [`eval_val.txt`](eval_val.txt) (1,000장)
  - ⚠️ 이 인덱스의 이미지는 **절대 학습 데이터에 포함하면 안 됩니다.** (Data Leakage 적발 시 실격)
- **예선 평가 요소**: 양자화 전후 Moderate mAP 유지율, 파라미터 수, FLOPs, 메모리 사용량, 데스크탑 추론 지연 시간

### 모델 설계 가이드

- 입력 해상도는 가로 1280 이하, 32의 배수, KITTI 원본 비율(약 1242x375)에 맞춘 직사각형을 권장합니다.
  - 정밀도 중심: `1280x384`
  - 속도 중심: `640x192`
- Hailo DFC가 지원하지 않는 레이어가 있으면 CPU Fallback이 일어나 속도가 크게 떨어집니다. Hailo Model Zoo에서 호환성이 검증된 Conv 기반 모델(YOLOv8 / v10 / v11 등)을 베이스라인으로 쓰는 것을 권장합니다.

## 타깃 하드웨어 (본선)

| 구분 | 품명 | 비고 |
| --- | --- | --- |
| 보드 | Raspberry Pi 5 (8GB) | |
| NPU | Raspberry Pi AI HAT+ 2 (Hailo-10H) | 40 TOPS(INT4) / 20 TOPS(INT8), INT4·INT8·FP16 지원 |

- `.hef` 파일은 칩끼리 호환되지 않습니다. Hailo-8/8L용 모델은 Hailo-10H에서 돌아가지 않으므로 **반드시 Hailo-10H 대상으로 컴파일**해야 합니다.
- 본선 기간이 약 3주로 짧기 때문에 예선 단계부터 Hailo-10H 이식을 염두에 두고 양자화 전략을 세웁니다.

## 예선 제출물 체크리스트

- [ ] 최적화 결과 보고서 (PDF)
- [ ] 베이스라인 모델 파일 (최적화 전 원본)
- [ ] 최적화 모델 파일 (포맷과 버전 명시, 예: `.pt`, `.onnx`, `.tflite`)
- [ ] 벤치마크 측정 및 추론 스크립트
- [ ] 의존성 목록(`requirements.txt`)과 실행 방법
- [ ] 데이터셋 경로 또는 다운로드 방법

## 저장소 구성

| 파일 | 설명 |
| --- | --- |
| [`2026_모바일_하반기공모전_트랙2_최종안내.docx`](2026_모바일_하반기공모전_트랙2_최종안내.docx) | 트랙2 대회 최종 안내문 |
| [`트랙2 지정주제 벤치마크 공지-1.hwpx`](트랙2%20지정주제%20벤치마크%20공지-1.hwpx) | 지정주제 벤치마크(KITTI) 공지 |
| [`eval_val.txt`](eval_val.txt) | 지정주제 고정 검증셋 이미지 인덱스 (1,000장) |
| [`docs/study_guide.md`](docs/study_guide.md) | 프로젝트 전 알아둘 개념 정리 (사전 학습 가이드) |

## 실행 방법

| 단계 | 노트북 | 내용 |
| --- | --- | --- |
| 01 | [`notebooks/01_prepare_data.ipynb`](notebooks/01_prepare_data.ipynb) ([Colab에서 열기](https://colab.research.google.com/github/gyu-young-04/embeded_ondeviceAI/blob/main/notebooks/01_prepare_data.ipynb)) | KITTI 다운로드, YOLO 형식 변환, train / dev / eval 분리 |
| 02 | [`notebooks/02_train.ipynb`](notebooks/02_train.ipynb) ([Colab에서 열기](https://colab.research.google.com/github/gyu-young-04/embeded_ondeviceAI/blob/main/notebooks/02_train.ipynb)) | 모델(v8n / v8s / 11n) · 입력 크기(640 / 960 / 1280) 비교 학습 후 최고 조합 선정, 결과는 Google Drive에 저장 |

### 데이터 분리 원칙

| 셋 | 장수 | 용도 |
| --- | --- | --- |
| train | 5,833 | 학습 |
| dev | 648 | 학습 중 점검 (eval_val 제외한 사진의 10%) |
| eval | 1,000 | `eval_val.txt` 고정 검증셋. **최종 채점에만 사용, 학습·튜닝에 쓰지 않음** |

클래스는 Van→Car, Person_sitting→Pedestrian 으로 합쳐 Car / Pedestrian / Cyclist 3개를 사용합니다.

## 진행 상황

- [x] 대회 안내 자료 정리
- [ ] [사전 학습](docs/study_guide.md)
- [x] 주제 확정: 지정주제
- [x] KITTI 데이터 준비 (YOLO 변환)
- [ ] 베이스라인 모델 선정 및 KITTI 평가 환경 구축
- [ ] 양자화·경량화 실험
- [ ] 예선 보고서 작성 및 제출
