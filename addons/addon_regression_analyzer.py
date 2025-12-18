# addons/addon_regression_analyzer.py

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from modules.utils_io import read_yaml, ensure_dir
from modules.utils_logger import logger
from addons.addon_shap_analysis import _set_korean_font

def run_regression_analysis(settings_path: str):
    """
    저장된 회귀 모델의 성능을 심층 분석하고 결과를 시각화합니다.
    (모델에 저장된 피처 목록을 사용하여 예측-훈련 간 피처 불일치 문제 해결)
    """
    logger.info("\n" + "="*60)
    logger.info("      <<< 목표가 예측 모델(회귀) 성능 분석기 >>>")
    logger.info("="*60)

    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    dataset_path = os.path.join(paths["ml_dataset"], "ml_regression_dataset.parquet")
    model_path = os.path.join(paths["models"], "target_model.joblib")

    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        logger.error("ML 회귀 데이터셋 또는 학습된 모델이 없습니다. '5. 고급 ML 데이터셋 생성' 및 '6. ML 모델 재학습'을 먼저 실행해주세요.")
        return

    logger.info(f"데이터셋 로드: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    logger.info(f"모델 로드: {model_path}")
    model = joblib.load(model_path)

    # --- [수정] 모델이 학습한 피처 목록을 직접 가져와 사용 ---
    try:
        # LightGBM 모델은 학습 시 사용한 피처 이름을 .feature_name_ 속성에 저장합니다.
        model_features = model.feature_name_
    except AttributeError:
        logger.error("로드된 모델에서 피처 목록을 찾을 수 없습니다. 모델이 올바르게 저장되었는지 확인해주세요.")
        return

    missing_features = [f for f in model_features if f not in df.columns]
    if missing_features:
        logger.error(f"데이터셋에 모델 학습에 사용된 피처가 없습니다: {missing_features}")
        return
        
    # 모델이 학습한 피처만으로 X, y를 구성합니다. 목표 변수는 target_log_ret 입니다.
    X, y = df[model_features], df['target_log_ret']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    logger.info(f"테스트 데이터 {len(X_test)}건에 대한 예측을 수행합니다... (사용된 피처 수: {len(X_test.columns)})")
    predictions = model.predict(X_test)
    
    # [개선] 로그 변환된 값을 실제 수익률로 역변환하여 분석
    results = pd.DataFrame({
        'actual': np.expm1(y_test), 
        'predicted': np.expm1(predictions)
    })
    results['residual'] = results['actual'] - results['predicted']
    # --- 수정 완료 ---
    
    _set_korean_font()
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(paths["backtest"], f"regression_analysis_{run_timestamp}")
    ensure_dir(output_dir)
    logger.info(f"분석 결과 그래프는 다음 폴더에 저장됩니다: {output_dir}")

    # 그래프 1: 예측-실제 값 분포 (Scatter Plot)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='actual', y='predicted', data=results, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('목표 수익률 예측 vs 실제 값 분포', fontsize=16)
    plt.xlabel('실제 최대 수익률', fontsize=12)
    plt.ylabel('모델 예측 수익률', fontsize=12)
    plt.grid(True)
    plot_path1 = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(plot_path1, dpi=150)
    plt.close()
    logger.info(f"  - 예측-실제 값 분포도 저장 완료: {plot_path1}")

    # 그래프 2: 오차(Residual) 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(results['residual'], kde=True, bins=50)
    plt.title('예측 오차(Residual) 분포', fontsize=16)
    plt.xlabel('오차 (실제 - 예측)', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.grid(True)
    plot_path2 = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    logger.info(f"  - 오차 분포도 저장 완료: {plot_path2}")
    
    logger.info("\n분석이 완료되었습니다.")