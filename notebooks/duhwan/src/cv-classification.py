import os
import sys
import time
import random

import timm
import torch
import cv2

import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from dotenv import load_dotenv, dotenv_values

# 하이드라와 주피터 노트북은 아규먼트 관련 충돌이 발생하므로 초기화 해줌
sys.argv = ['']
# 환경변수 읽기
if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)

from src.data.FullAugraphyPipeline import FullAugraphyPipeline
from src.data.ImageDataset import ImageDataset
from src.model.CustomModel import CustomModel
from src.util import config
from src.util.utils import send_kakao_message

# 시드 고정
def random_seed(seed_num=42):

    """ SEED = seed_num
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True """
    
    # seed_everything 은 위의 내용 제어 + 밑에내용
    pl.seed_everything(seed_num)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

# 데이터 준비 함수
def prepare_data(batch_size=32, num_workers=4):
    
   # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = ImageDataset.get_datasets()

    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,  # 별도의 검증 데이터셋
        batch_size=batch_size,
        shuffle=False,  # 검증 시에는 셔플하지 않음
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def main():

    # model config
    model_name = 'efficientnet_b4' # 'resnet50' 'efficientnet_b4', ...

    # training config
    EPOCHS = 10
    BATCH_SIZE = 32
    num_workers = 0
    num_classes = 17
    learning_rate = 1e-3
    
    # 모델 초기화 전에 설정
    torch.set_float32_matmul_precision('medium')
    
    random_seed(42)

    # 데이터 로더 준비
    #train_loader, val_loader, test_loader = prepare_data(batch_size=BATCH_SIZE, num_workers=4)

    augmentation_pipeline = FullAugraphyPipeline(max_effects=2)
    
    # 커스텀 변환 클래스 정의
    class To_BGR(object):
        """PIL RGB 이미지를 numpy BGR 이미지로 변환"""
        def __call__(self, image):
            image_numpy = np.array(image)
            if len(image_numpy.shape) < 3:
                return cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2BGR)
            else:
                return cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

    # 수정된 변환 파이프라인
    dirty_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        To_BGR(),
        augmentation_pipeline,  # Augraphy 적용
        transforms.ToTensor(),  # numpy -> tensor
        # 정규화
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    clean_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 정규화
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 데이터셋 생성
    d1 = ImageDataset(config.CV_CLS_TRAIN_CSV, config.CV_CLS_TRAIN_DIR, transform=dirty_transforms)
    d2 = ImageDataset(config.CV_CLS_TRAIN_CSV, config.CV_CLS_TRAIN_DIR, transform=clean_transforms)

    train_dataset = ConcatDataset([d1, d2])

    # 전체 데이터셋을 8:2로 분할
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size]
        #generator=torch.Generator().manual_seed(42)  # 재현 가능성을 위한 시드
    )

    test_dataset = ImageDataset(
        config.CV_CLS_TEST_CSV,
        config.CV_CLS_TEST_DIR,
        transform=clean_transforms
    )

    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,  # 별도의 검증 데이터셋
        batch_size=BATCH_SIZE,
        shuffle=False,  # 검증 시에는 셔플하지 않음
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    model = CustomModel(
        model_name= model_name,
        num_classes=num_classes,
        learning_rate=learning_rate
    )

     # 콜백을 직접 생성
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            min_delta=0.001,
            verbose=True
        ),
        LearningRateMonitor(
            logging_interval='epoch',
            log_momentum=False
        )
    ]

    trainer = Trainer(default_root_dir=config.OUTPUTS_DIR, max_epochs=EPOCHS, accelerator='auto', callbacks=callbacks)
    
    # 훈련
    trainer.fit(model, train_loader, val_loader)
    
    # 테스트
    trainer.test(model, test_loader)

    print("테스트 갯수=",len(model.test_predictions))
    if len(model.test_predictions) > 0:
        # 모든 예측값과 실제값 합치기
        all_preds = model.test_predictions
        
        pred_df = pd.DataFrame(test_loader.dataset.df, columns=['ID', 'target'])
        pred_df['target'] = all_preds

        sample_submission_df = pd.read_csv(config.CV_CLS_TEST_CSV)
        assert (sample_submission_df['ID'] == pred_df['ID']).all()
        pred_df.to_csv(config.OUTPUTS_DIR + "/pred.csv", index=False)

    else:
        print("테스트 결과를 가져올 수 없습니다.")

    send_kakao_message("작업종료!\n\n" + str(trainer.callback_metrics))
if __name__ == "__main__":
    main()