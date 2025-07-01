# Document Image Classification Project

## Team

| [조의영](https://github.com/yuiyeong) | [김두환](https://github.com/korea202a) | [나주영](https://github.com/najuyoung) | [조재형](https://github.com/Bitefriend) |
|:----------------------------------:|:-----------------------------------:|:-----------------------------------:|:------------------------------------:|
|                 팀장                 |                 팀원                  |                 팀원                  |                  팀원                  |


## 0. Getting Started

### Quick Setup for Cloud Instance

- 본 프로젝트는 클라우드 인스턴스에서 진행됩니다.
- 다음 단계를 따라 환경을 설정하고 프로젝트를 시작하세요.

#### 1. 환경 설정

클라우드 인스턴스에 접속한 후, 다음 명령어를 실행하여 개발 환경을 자동으로 설정합니다,

```bash
# 환경 설정 스크립트 다운로드 및 실행
wget https://gist.githubusercontent.com/yuiyeong/8ae3f167e97aeff90785a4ccda41e5fe/raw/d5e030ea64bbd9c41ce2b4c825bc03c86f0c3dac/setup_env.sh
chmod +x setup_env.sh
./setup_env
```

**설정 내용:**
- Python 3.11 conda 환경 (py311) 생성
- Poetry 설치 및 PATH 설정
- /workspace 작업 디렉토리 생성
- SSH 로그인 시 자동으로 /workspace로 이동

#### 2. 환경 적용

스크립트 실행 후 SSH를 재접속하여 변경사항을 적용합니다.

```bash
# SSH 재접속 후 환경 확인
python --version  # Python 3.11.x 확인
poetry --version  # Poetry 설치 확인
pwd              # /workspace 확인
```
#### 3. Git Config

- 다음 명령어를 실행하여 git config 를 설정합니다.
- `{username}` 과 `{emailaddr}` 에 본인의 github name 과 email 을 적어주세요

```bash
git config --global user.name "{username}"
git config --global user.email "{emailaddr}"
git config --global core.editor "vim"
git config --global core.pager "cat"
```

- 설정된 내용은 `git config --list` 로 확인합니다.
- 수정이 필요할 경우, `vi ~/.gitconfig` 를 실행해서 값을 수정합니다.

#### 4. 프로젝트 복제 및 설정

```bash
# /workspace 디렉토리에서 프로젝트 복제
cd /workspace
git clone https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_3.git
cd upstageailab-cv-classification-cv-3

# Poetry를 사용하여 의존성 설치
poetry install
```

#### 5. 데이터 다운로드

```bash
# 데이터 다운로드 (대회 페이지에서 URL 확인)
wget [DATA_URL] -O data.tar.gz

# 압축 해제
tar -zxvf data.tar.gz

mv data/ upstageailab-cv-classification-cv-3/data/raw
```

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
├── data/                           # 데이터 저장 디렉토리
│   ├── fonts/                      # 문서 이미지 생성/처리용 폰트 파일
│   └── raw/                        # 원본 데이터 (train, test 이미지, CSV 파일)
├── docs/                           # 프로젝트 문서 및 자료
│   └── img/                        # 문서용 이미지
├── notebooks/                      # Jupyter 노트북 디렉토리
│   ├── duhwan/                     # 김두환 개인 실험 노트북
│   ├── jaehyeong/                  # 조재형 개인 실험 노트북
│   ├── juyoung/                    # 나주영 개인 실험 노트북
│   ├── yuiyeong/                   # 조의영 개인 실험 노트북
│   └── notebook_template.ipynb     # 노트북 작성을 위한 공통 템플릿
├── scripts/                        # 유틸리티 스크립트 모음
│   └── init-cloud-instance.sh      # 클라우드 인스턴스 환경 설정 스크립트
└── src/                            # 소스 코드 메인 디렉토리
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

W.I.P.

### Data Processing

W.I.P.

## 4. Modeling

W.I.P.

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
