# modules/evaluation.py
"""
[SpikeHunter v4.0] 모델 평가 및 분석 모듈
- 학습된 모델의 성능을 평가하고, 결과 리포트를 생성합니다.
- v4.0 변경: 구버전 함수 의존성 제거 및 새로운 데이터셋 구조 지원
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import warnings

# 내부 모듈 임포트
from .utils_io import read_yaml, ensure_dir
from .utils_logger import logger
# [수정] 구버전 함수 'run_derive_advanced_ml_dataset' 임포트 제거
from .derive import _get_feature_cols 

warnings.filterwarnings("ignore")

def load_test_data(cfg: dict, model_type='classification'):
    """평가를 위한 테스트 데이터 로드"""
    paths = cfg['paths']
    ml_dataset_path = paths['features'] # v4.0에서는 features 폴더에 dataset_v4.parquet 저장
    
    # v4.0 데이터셋 경로
    dataset_path = os.path.join(ml_dataset_path, "dataset_v4.parquet")
    
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋을 찾을 수 없습니다: {dataset_path}")
        return None, None
    
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 학습/테스트 분할 (최근 데이터 20%를 테스트셋으로 가정)
    # 실제로는 train.py의 분할 로직과 일치시켜야 하나, 여기서는 간이로 시간순 분할
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    if model_type == 'classification':
        # 타겟: label_class (트리플 배리어 라벨)
        feature_cols = _get_feature_cols(test_df.columns)
        X_test = test_df[feature_cols]
        y_test = test_df['label_class']
        return X_test, y_test
    else:
        # 회귀 모델 평가는 추후 구현 (현재는 분류 집중)
        return None, None

def evaluate_classification_model(settings_path="config/settings.yaml"):
    """분류 모델 평가 메인 함수"""
    if not os.path.exists(settings_path):
        logger.error("설정 파일이 없습니다.")
        return

    cfg = read_yaml(settings_path)
    paths = cfg['paths']
    
    # 모델 로드
    model_path = os.path.join(paths['models'], "lgbm_model.joblib")
    if not os.path.exists(model_path):
        logger.error(f"학습된 모델이 없습니다: {model_path}")
        return
    
    model = joblib.load(model_path)
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 데이터 로드
    X_test, y_test = load_test_data(cfg, 'classification')
    if X_test is None: return
    
    # 예측
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 임계값 설정 (기본 0.5, 필요 시 조정 가능)
    threshold = cfg.get('ml_params', {}).get('classification_threshold', 0.5)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 평가 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info("\n" + "="*40)
    logger.info("      [모델 평가 결과 (Test Set)]")
    logger.info("="*40)
    logger.info(f"정확도 (Accuracy) : {acc:.4f}")
    logger.info(f"정밀도 (Precision): {prec:.4f}")
    logger.info(f"재현율 (Recall)   : {rec:.4f}")
    logger.info(f"F1 점수 (F1-Score): {f1:.4f}")
    logger.info(f"AUC Score         : {auc:.4f}")
    logger.info("-" * 40)
    
    # 혼동 행렬 출력
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"혼동 행렬 (Confusion Matrix):\n{cm}")
    
    # 상세 리포트
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info(f"상세 리포트:\n{report}")
    logger.info("="*40)

if __name__ == "__main__":
    evaluate_classification_model()