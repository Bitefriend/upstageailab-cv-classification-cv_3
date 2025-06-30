# Document Image Classification Project

## Team

| [조의영](https://github.com/yuiyeong) | [김두환](https://github.com/korea202a) | [나주영](https://github.com/najuyoung) | [조재형](https://github.com/Bitefriend) |
|:----------------------------------:|:-----------------------------------:|:-----------------------------------:|:------------------------------------:|
|                 팀장                 |                 팀원                  |                 팀원                  |                  팀원                  |

## 0. Overview

### Environment

- **OS**: Linux (CUDA 12.1 지원)
- **Python**: 3.11+
- **Deep Learning Framework**: PyTorch 2.6.0
- **주요 라이브러리**: timm, albumentations, OpenCV, scikit-learn
- **실험 관리**: Weights & Biases (wandb)

### Requirements

본 프로젝트는 Poetry를 사용하여 의존성을 관리합니다.

**주요 의존성**

- `torch>=2.6.0` - PyTorch 딥러닝 프레임워크
- `timm>=1.0.16` - 사전 훈련된 컴퓨터 비전 모델
- `albumentations>=2.0.8` - 이미지 증강 라이브러리
- `opencv-python>=4.11.0` - 컴퓨터 비전 라이브러리
- `scikit-learn>=1.7.0` - 머신러닝 유틸리티
- `wandb>=0.20.1` - 실험 추적 및 관리
- `boto3>=1.38.46` - AWS SDK (데이터 다운로드용)

**설치 방법**

```bash
# script 로 instance 환경 설정 후
poetry install
```

## 1. Competition Info

### Overview

이번 대회는 **문서 타입 분류**를 위한 이미지 분류 대회입니다.

- **태스크**: 17개 클래스의 문서 이미지 분류
- **도메인**: Computer Vision - Image Classification
- **활용 분야**: 금융, 의료, 보험, 물류 등 산업 전반의 문서 자동화 처리
- **데이터**: 실제 현업에서 사용하는 데이터를 기반으로 제작
- **평가 지표**: Macro F1 Score

### Dataset Statistics

- **학습 데이터**: 1,570장의 이미지
- **평가 데이터**: 3,140장의 이미지
- **클래스 수**: 17개 문서 타입
- **이미지 형식**: JPG
- **특징**: 다양한 문서 상태 (회전, 뒤집힘, 훼손 등)

### Class Information

총 17개의 문서 클래스로 구성

- account_number (계좌번호)
- application_for_payment_of_pregnancy_medical_e (임신의료비 지급신청서)
- car_dashboard (차량 대시보드)
- confirmation_of_admission_and_discharge (입퇴원 확인서)
- diagnosis (진단서)
- driver_licence (운전면허증)
- medical_bill_receipts (의료비 영수증)
- medical_outpatient_certificate (의료 외래 증명서)
- national_id_card (주민등록증)
- passport (여권)
- payment_confirmation (결제 확인서)
- pharmaceutical_receipt (약국 영수증)
- prescription (처방전)
- resume (이력서)
- statement_of_opinion (의견서)
- vehicle_registration_certificate (차량등록증)
- vehicle_registration_plate (차량번호판)

### Timeline

- **Start Date**: 2025-06-30
- **Final submission deadline**: 2025-07-11

## 2. Components

### Directory

```
├── data
│     ├── fonts
│     │     └── NanumBarunGothic.ttf
│     └── raw
├── docs
│     └── img
├── notebooks
│     ├── duhwan
│     ├── jaehyeong
│     ├── juyoung
│     ├── notebook_template.ipynb
│     └── yuiyeong
│         └── hello_baseline_codes.ipynb
├── scripts
│     └── init-cloud-instance.sh
└── src
    └── __init__.py
```

## 3. Data Description

### Dataset Overview

본 대회의 데이터셋은 실제 현업에서 사용되는 문서 이미지를 기반으로 구성되었습니다.

**학습 데이터 구성**

- `train/`: 1,570장의 학습용 이미지
- `train.csv`: 학습 이미지의 파일명과 정답 클래스 정보
- `meta.csv`: 17개 클래스의 번호와 이름 매핑 정보

**평가 데이터 구성**

- `test/`: 3,140장의 평가용 이미지
- `sample_submission.csv`: 제출 형식 템플릿

### EDA

**클래스 분포 분석**

- 학습 데이터의 클래스별 이미지 수 분포 확인
- 클래스 불균형 여부 분석

**이미지 특성 분석**

- 이미지 크기 분포
- 색상 채널 분석 (RGB/Grayscale)
- 이미지 품질 및 노이즈 분석

**데이터 품질 이슈**

- 평가 데이터에 랜덤 회전(Rotation) 및 뒤집기(Flip) 적용
- 일부 훼손된 이미지 존재

### Data Processing

**전처리 파이프라인**

1. 이미지 리사이징 및 정규화
2. 데이터 증강 기법 적용
3. 클래스 불균형 해결을 위한 샘플링 전략

**데이터 증강**

- Albumentations 라이브러리 활용
- 회전, 뒤집기, 색상 변환, 노이즈 추가 등

## 4. Modeling

### Model Description

**베이스 모델**

- TIMM 라이브러리의 사전 훈련된 모델 활용
- EfficientNet, ResNet, Vision Transformer 등 다양한 아키텍처 실험

**모델 선택 기준**

- 문서 이미지의 특성을 고려한 모델 구조
- 17개 클래스 분류에 적합한 성능
- 추론 속도와 정확도의 균형

### Modeling Process

**실험 과정**

1. 베이스라인 모델 구축
2. 하이퍼파라미터 튜닝
3. 앙상블 모델 구성
4. 교차 검증을 통한 모델 검증

**최적화 기법**

- Learning Rate Scheduling
- 정규화 기법 (Dropout, Weight Decay)
- 손실 함수 최적화

## 5. Result

### Leader Board

W.I.P.

### Presentation

W.I.P.

## etc

### Meeting Log

W.I.P.

### Reference

- [TIMM Documentation](https://timm.fast.ai/)
- [Albumentations Documentation](https://albumentations.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
