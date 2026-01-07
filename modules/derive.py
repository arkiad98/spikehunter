# modules/derive.py (v6.0 - Market Features Integration)
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# [수정] load_index_data 임포트 추가 및 features 함수 임포트
from .utils_io import read_yaml, to_date, load_partition_day, ensure_dir, load_index_data
from .utils_logger import logger
from .features import generate_features, add_cross_sectional_rank_features, calculate_market_features
from .labeling import add_labels

def _get_feature_cols(columns):
    """학습용 피처 컬럼 추출"""
    exclude_cols = {
        'date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'amount_bil',
        'target', 'target_log_ret', 'label_class',
        'year', 'month', 'day', 'weekday',
        'fwd_max_ret', 'fwd_ret',
        'inst_net_val', 'foreign_net_val',
        'value', 'change', 'market_index', 'regime',
        'kospi_close', 'kospi_value' # 원본 KOSPI 컬럼 제외
    }
    feature_cols = []
    for c in columns:
        if c in exclude_cols: continue
        if c.startswith('target_'): continue
        feature_cols.append(c)
    return feature_cols

def run_derive(settings_path="config/settings.yaml"):
    # 1. 설정 로드
    if not os.path.exists(settings_path):
        logger.error(f"설정 파일 없음: {settings_path}")
        return
        
    cfg = read_yaml(settings_path)
    paths = cfg['paths']
    ml_params = cfg.get('ml_params', {})
    
    # 2. 기간 설정
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = to_date(cfg.get('data_range', {}).get('start', '2020-01-01'))
    end_date = to_date(cfg.get('data_range', {}).get('end', today_str))
    
    logger.info(f"[Derive] 데이터 로드 시작: {start_date.date()} ~ {end_date.date()}")
    
    # 3. 종목 데이터 로드
    if 'merged' not in paths:
        logger.error("settings.yaml에 'merged' 경로 없음")
        return

    df_ohlcv = load_partition_day(paths['merged'], start_date, end_date)
    if df_ohlcv.empty:
        logger.error(f"데이터 없음: {paths['merged']}")
        return
    logger.info(f"로드된 종목 데이터: {len(df_ohlcv)} 행")

    # 4. [NEW] 시장(KOSPI) 데이터 로드 및 피처 생성
    logger.info("[Derive] KOSPI 시장 지표 생성 중...")
    # 이동평균 계산을 위해 시작일보다 400일 전부터 로드
    kospi_start = start_date - pd.DateOffset(days=400)
    kospi_df = load_index_data(kospi_start, end_date, paths['raw_index'])
    
    market_features = pd.DataFrame()
    if not kospi_df.empty:
        market_features = calculate_market_features(kospi_df)
        # 분석 기간에 맞게 자르기
        market_features = market_features[market_features['date'] >= start_date]
    else:
        logger.warning("KOSPI 데이터가 없어 시장 피처를 생성하지 못했습니다.")

    # 5. 종목별 피처 엔지니어링
    logger.info("[Derive] 개별 종목 피처 생성 중...")
    processed_dfs = []
    grouped = df_ohlcv.groupby('code')
    
    for code, group in tqdm(grouped, desc="Feature Engineering"):
        if len(group) < 60: continue 
        
        # 유동성 필터 (10억 미만 제외)
        avg_amount = (group['close'] * group['volume']).rolling(20).mean().iloc[-1]
        if avg_amount < 1_000_000_000: continue

        feat_df = generate_features(group)
        if not feat_df.empty:
            processed_dfs.append(feat_df)
            
    if not processed_dfs:
        logger.error("피처 생성 결과 없음")
        return
        
    df_features = pd.concat(processed_dfs).reset_index(drop=True)
    
    # 6. [NEW] 시장 피처 병합 (Market Features Merge)
    if not market_features.empty:
        logger.info("[Derive] 시장 피처(Trend/Vol) 병합 중...")
        df_features = pd.merge(df_features, market_features, on='date', how='left')
        # 시장 데이터 결측치는 전날 값으로 채움
        df_features['market_bullish'] = df_features['market_bullish'].fillna(method='ffill').fillna(0)
        df_features['market_volatility'] = df_features['market_volatility'].fillna(method='ffill').fillna(0)
    else:
        logger.warning("KOSPI 데이터 없음: 시장 피처를 0(기본값)으로 설정합니다.")
        df_features['market_bullish'] = 0
        df_features['market_volatility'] = 0

    # 7. [NEW] Cross-Sectional Rank 피처 생성
    logger.info("[Derive] 시장 대비 순위(Rank) 피처 생성 중...")
    df_features = add_cross_sectional_rank_features(df_features)
    
    # 8. 라벨링 (5일/13.5% 기준 [최적값], 손절 -5.7%)
    target_surge = ml_params.get('target_surge_rate', 0.135)
    stop_loss = ml_params.get('stop_loss_rate', -0.057)
    target_hold = ml_params.get('target_hold_period', 5)
    
    logger.info(f"[Derive] 라벨링 생성 중... (익절: {target_surge*100}%, 손절: {stop_loss*100}%, 기간: {target_hold}일)")
    
    df_labeled = add_labels(
        df_features, 
        profit_th=target_surge,
        loss_th=stop_loss,
        horizon=target_hold
    )
    
    # 9. 저장
    save_dir = paths.get('ml_dataset', 'data/proc/ml_dataset')
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, "ml_classification_dataset.parquet")
    
    df_labeled.to_parquet(save_path, compression='snappy')
    
    logger.info(f"[Derive] 데이터셋 저장 완료: {save_path}")
    logger.info(f"총 데이터 수: {len(df_labeled)} 행")

if __name__ == "__main__":
    run_derive()