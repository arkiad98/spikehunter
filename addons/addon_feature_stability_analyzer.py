# addons/addon_feature_stability_analyzer.py
"""
시간의 흐름에 따른 피처 중요도의 안정성을 분석하는 애드온 모듈.
롤링 윈도우(Rolling Window) 방식을 사용하여 각 기간별 피처 순위를 측정한 뒤,
피처별 평균 순위와 순위 변동성(표준편차)을 계산하여 핵심 피처를 식별합니다.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime 

from modules.utils_io import read_yaml, ensure_dir
from modules.utils_logger import logger
from modules.derive import _get_feature_cols



def run_stability_analysis(settings_path: str):
    """롤링 윈도우 방식으로 피처 중요도의 안정성을 분석합니다."""
    logger.info("="*60)
    logger.info("      <<< 핵심 피처 안정성 분석 시작 >>>")
    logger.info("="*60)

    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    # 분석 파라미터 설정
    WINDOW_SIZE_MONTHS = 12  # 각 분석에 사용할 데이터 기간 (12개월)
    STEP_SIZE_MONTHS = 3     # 윈도우를 이동시킬 간격 (3개월)

    logger.info(f"분석 설정: 윈도우 크기={WINDOW_SIZE_MONTHS}개월, 이동 간격={STEP_SIZE_MONTHS}개월")

    # 1. 전체 데이터셋 로드
    dataset_file = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_file):
        logger.error(f"ML 데이터셋 파일({dataset_file})이 없습니다."); return

    logger.info(f"전체 ML 분류 데이터셋 로드 중...")
    df = pd.read_parquet(dataset_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # [Fix] Use Core features from registry
    from modules.train import _get_core_features_from_registry
    core_features = _get_core_features_from_registry("config/feature_registry.yaml")
    
    if core_features:
        feature_cols = [f for f in core_features if f in df.columns]
        logger.info(f"Feature Registry에서 {len(feature_cols)}개의 Core 피처를 로드했습니다.")
    else:
        logger.warning("Feature Registry를 로드할 수 없어 전체 숫자형 컬럼을 사용합니다.")
        feature_cols = _get_feature_cols(df.columns)

    lgbm_params = cfg.get("ml_params", {}).get("lgbm_params_classification", {})

    # 2. 롤링 윈도우 기간 설정
    all_results = []
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    current_start = start_date
    while current_start + pd.DateOffset(months=WINDOW_SIZE_MONTHS) <= end_date:
        current_end = current_start + pd.DateOffset(months=WINDOW_SIZE_MONTHS)
        period_str = f"{current_start.date()} ~ {current_end.date()}"
        
        # 3. 현재 윈도우 데이터로 모델 학습 및 피처 중요도 추출
        logger.info(f"  - 기간 분석 중: {period_str}")
        
        window_df = df[(df['date'] >= current_start) & (df['date'] < current_end)]
        if len(window_df) < 1000: # 데이터가 너무 적으면 건너뛰기
            current_start += pd.DateOffset(months=STEP_SIZE_MONTHS)
            continue
            
        X_window, y_window = window_df[feature_cols], window_df['label_class']

        # [Fix] Filter out param_space keys
        clean_params = {k: v for k, v in lgbm_params.items() if not k.startswith('param_space_')}
        model = lgb.LGBMClassifier(**clean_params)
        model.fit(X_window, y_window)
        
        importance_df = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_,
            'period': period_str
        }).sort_values('importance', ascending=False)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        all_results.append(importance_df)
        
        current_start += pd.DateOffset(months=STEP_SIZE_MONTHS)
        
    if not all_results:
        logger.error("분석할 기간이 충분하지 않습니다. 데이터 기간을 확인해주세요.")
        return

    # 4. 결과 집계 및 분석
    final_df = pd.concat(all_results, ignore_index=True)
    
    summary_df = final_df.groupby('feature').agg(
        mean_rank=('rank', 'mean'),
        std_rank=('rank', 'std'),
        avg_importance=('importance', 'mean'),
        zero_importance_count=('importance', lambda x: (x == 0).sum())
    ).sort_values('mean_rank', ascending=True).reset_index()

    summary_df['std_rank'] = summary_df['std_rank'].fillna(0) # 순위 변동이 없는 경우 NaN -> 0

    logger.info("\n" + "="*80)
    logger.info("                     <<< 피처 안정성 최종 분석 보고서 >>>")
    logger.info("="*80)
    print(tabulate(summary_df, headers='keys', tablefmt='psql', floatfmt=(".2f")))
    logger.info(
        "\n* mean_rank: 낮을수록 꾸준히 상위권에 있었다는 의미.\n"
        "* std_rank: 낮을수록 순위 변동이 적었다는 의미 (0에 가까울수록 안정적)."
    )
    logger.info("="*80)

    # 5. 결과 파일 저장
    output_dir = os.path.join(paths["backtest"], f"feature_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(output_dir)
    result_path = os.path.join(output_dir, "feature_stability_report.csv")
    summary_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    logger.info(f"상세 분석 결과가 '{result_path}'에 저장되었습니다.")