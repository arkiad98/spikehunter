# addons/addon_predict_target.py
# [수정] CSV 저장 시 발생하는 KeyError 해결

import os
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

from modules.utils_io import read_yaml, load_partition_day, get_stock_names
from modules.utils_logger import logger

def get_latest_prediction_file(pred_path: str):
    """predictions 폴더에서 가장 최신 추천 종목 파일을 찾습니다."""
    if not os.path.exists(pred_path):
        return None
    
    files = [os.path.join(pred_path, f) for f in os.listdir(pred_path) if f.endswith('.csv') and '_targets' not in f]
    if not files:
        return None
        
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def run_target_prediction(settings_path: str):
    """
    ML 회귀 모델을 사용하여 '오늘의 추천 종목'의 미래 기대 수익률과 목표가를 예측합니다.
    """
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # 1. 저장된 최신 추천 종목 CSV 파일을 로드합니다.
    pred_path = paths["predictions"]
    latest_pred_file = get_latest_prediction_file(pred_path)
    
    if not latest_pred_file:
        logger.error("예측할 추천 종목 파일이 없습니다. '9. 오늘의 추천 종목 생성'을 먼저 실행해주세요.")
        return
        
    logger.info(f"최신 추천 종목 파일을 로드합니다: {latest_pred_file}")
    
    try:
        recs_df = pd.read_csv(latest_pred_file, dtype={'code': str})
    except Exception as e:
        logger.error(f"추천 종목 파일 로드 중 오류 발생: {e}", exc_info=True)
        return

    latest_date = pd.to_datetime(os.path.basename(latest_pred_file).replace('.csv', ''))

    # 2. 추천 종목에 해당하는 날짜의 피처 데이터를 로드합니다.
    features_today = load_partition_day(paths["features"], latest_date, latest_date)
    if features_today.empty:
        logger.error(f"{latest_date.date()}의 피처 데이터가 없습니다.")
        return
        
    # 🔴 [수정] 컬럼명 충돌을 피하고 데이터 소스를 일원화하는 로직
    # ----------------------------------------------------------------------------------
    # 추천 종목 CSV에서는 'code'와 원본 'ml_score'만 사용하고,
    # 가격을 포함한 모든 데이터는 피처 데이터(features_today)를 기준으로 합니다.
    recs_subset = recs_df[['code', 'ml_score']].copy()
    recs_subset.rename(columns={'ml_score': 'score'}, inplace=True) # 원본 스코어 컬럼명 통일

    # 'inner' join을 사용하여 두 데이터에 모두 존재하는 종목만 안전하게 병합
    recs_with_features = pd.merge(recs_subset, features_today, on='code', how='inner')
    # ----------------------------------------------------------------------------------

    # 4. 저장된 회귀(Regression) 모델을 로드합니다.
    logger.info("저장된 목표가 예측 ML 모델(회귀)을 로드합니다...")
    model_path = paths.get("models")
    target_model_filename = os.path.join(model_path, "target_model.joblib")
    
    if not os.path.exists(target_model_filename):
        logger.error(f"학습된 목표가 예측 모델이 없습니다: {target_model_filename}")
        return
        
    target_model = joblib.load(target_model_filename)

    # 5. 모델이 학습한 피처만 순서대로 준비하여 수익률을 예측합니다.
    # 🔴 [수정] feature_names_in_을 사용하여 모델 호환성 확보
    try:
        features_for_regression = recs_with_features.set_index('code').loc[:, target_model.feature_names_in_]
    except AttributeError:
        logger.error("오래된 버전의 모델일 수 있습니다. 'feature_names_in_' 속성이 없습니다.")
        # 구버전 모델 호환을 위한 대체 로직 (필요 시)
        # features_for_regression = recs_with_features.set_index('code').loc[:, target_model.feature_name_]
        return
        
    predicted_returns_log = target_model.predict(features_for_regression)
    # 🔴 [수정] 로그 수익률을 일반 수익률로 변환
    recs_with_features['predicted_ret'] = np.expm1(predicted_returns_log)
    
    # 이제 'close' 컬럼이 recs_with_features에 확실히 존재하므로 오류가 발생하지 않습니다.
    recs_with_features['predicted_target_price'] = recs_with_features['close'] * (1 + recs_with_features['predicted_ret'])
    
    # 6. 최종 결과를 출력하고 CSV로 저장합니다.
    names = get_stock_names(recs_with_features['code'].tolist())
    recs_with_features['name'] = recs_with_features['code'].map(names)

    # --- 콘솔 출력용 데이터프레임 ---
    df_display = recs_with_features.copy()
    df_display['predicted_ret_pct'] = (df_display['predicted_ret'] * 100).map('{:,.2f}%'.format)
    df_display['current_price'] = df_display['close'].map('{:,.0f}'.format)
    df_display['predicted_target_price'] = df_display['predicted_target_price'].map('{:,.0f}'.format)
    df_display['ml_score_clf'] = (df_display['score'] * 100).map('{:,.2f}'.format)
    
    display_cols = ['name', 'current_price', 'ml_score_clf', 'predicted_ret_pct', 'predicted_target_price']
    final_df_display = df_display.sort_values('score', ascending=False).set_index('code')[display_cols]
    
    logger.info("="*70)
    logger.info(f"     <<< {latest_date.date()} 기준 다음 영업일 추천 종목 목표가 예측 >>>")
    logger.info("="*70)
    print(final_df_display.to_string())
    logger.info("="*70)

    # --- CSV 저장용 데이터프레임 ---
    df_export = recs_with_features.copy()
    df_export = df_export.sort_values('score', ascending=False)
    df_export['rank'] = range(1, len(df_export) + 1)
    
    export_cols = [
        'rank', 'code', 'name', 'close', 'predicted_target_price', 'predicted_ret', 'score'
    ]
    final_df_export = df_export[export_cols]
    
    final_df_export = final_df_export.rename(columns={
        'close': 'current_price',
        'predicted_ret': 'upside_potential',
        'score': 'ml_score_clf'
    })
    
    try:
        filename = f"{latest_date.date()}_targets.csv"
        output_path = os.path.join(pred_path, filename)
        
        final_df_export.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.4f')
        logger.info(f"목표가 예측 결과가 '{output_path}'에 저장되었습니다.")
    except Exception as e:
        logger.error(f"목표가 예측 결과 파일 저장 중 오류 발생: {e}")
