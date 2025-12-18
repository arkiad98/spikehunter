import pandas as pd
from datetime import datetime, timedelta
import joblib
import os

from modules.utils_io import read_yaml, load_partition_day, load_index_data
from modules.utils_logger import logger
from modules.backtest import _determine_regime

def find_latest_feature_day(features_path: str) -> pd.Timestamp:
    """피처 데이터가 존재하는 가장 최근 날짜를 찾습니다."""
    date = pd.Timestamp.today().normalize()
    for _ in range(30): # 최대 30일 전까지 탐색
        df = load_partition_day(features_path, date, date)
        if not df.empty:
            return date
        date -= timedelta(days=1)
    return None

def run_strategy_debugger(settings_path: str):
    """
    SpikeHunter 전략의 필터링 과정을 단계별로 추적하여
    매수 추천 종목이 없는 원인을 진단합니다.
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< SpikeHunter 전략 디버거 v2.0 시작 >>>")
    logger.info("="*80)

    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    # 1. 분석할 최신 데이터 로드
    target_date = find_latest_feature_day(paths["features"])
    if target_date is None:
        logger.error("분석할 최신 피처 데이터가 없습니다. 피처 생성을 먼저 실행해주세요.")
        return
    
    logger.info(f"🔍 분석 대상 날짜: {target_date.date()}")
    df_today = load_partition_day(paths["features"], target_date, target_date)

    # 2. 해당일의 시장 국면(Regime) 및 전략 파라미터 결정
    kospi = load_index_data(target_date - timedelta(days=400), target_date, paths["raw_index"])
    kospi_today = kospi[kospi['date'] <= target_date]
    
    current_kospi_close = kospi_today['kospi_close'].iloc[-1]
    current_ma200 = kospi_today['kospi_close'].rolling(200).mean().iloc[-1]
    current_kospi_vol_20d = kospi_today['kospi_close'].pct_change().rolling(20).std().iloc[-1]
    
    vol_threshold = cfg["strategies"]["SpikeHunter_R1_BullStable"]["max_market_vol"]
    is_bull = current_kospi_close > current_ma200
    is_stable = current_kospi_vol_20d < vol_threshold
    current_regime = _determine_regime(is_bull, is_stable)
    
    strategy_key = f'SpikeHunter_{current_regime}'
    # 공통 파라미터와 체제별 파라미터를 모두 합칩니다.
    params = {**cfg, **cfg['strategies'][strategy_key]}
    logger.info(f"시장 국면: {current_regime} | 적용 파라미터 세트: {strategy_key}")

    # 3. 필터링 단계별 분석
    logger.info("\n--- [전략 필터링 단계별 추적] ---")
    
    # ... [0] ~ [3] 단계는 기존과 동일 ...
    num_stocks = len(df_today)
    logger.info(f"  [0] 총 분석 대상 종목 수: {num_stocks} 개")
    df_step1 = df_today[df_today['signal_spike_hunter'] == 1]
    logger.info(f"  [1] 'signal_spike_hunter == 1' 필터 후: {len(df_step1)} 개")
    # ... (상세 분석 로그는 생략) ...
    df_step2 = df_step1[
        (df_step1['dist_from_ma20'] < params['max_dist_from_ma']) &
        (df_step1["avg_value_20"] >= params['min_avg_value'])
    ]
    logger.info(f"  [2] 이격도 및 평균 거래대금 필터 후: {len(df_step2)} 개")
    df_step3 = df_step2[df_step2['daily_ret'] < params['max_daily_ret_entry']]
    logger.info(f"  [3] 진정 필터(당일 급등 제외) 후: {len(df_step3)} 개")

    # [수정] 4단계: ML 모델 로드 및 스코어 직접 계산
    if len(df_step3) > 0:
        model_path = os.path.join(paths["models"], "lgbm_model.joblib")
        if not os.path.exists(model_path):
            logger.error("  [4] ML 모델 파일이 없어 스코어 필터를 건너뜁니다.")
        else:
            model_clf = joblib.load(model_path)
            # [수정] .feature_name_ -> .feature_names_in_
            features_for_ml = df_step3[model_clf.feature_names_in_]
            pred_probs = model_clf.predict_proba(features_for_ml)[:, 1]
            df_step3['ml_score'] = pred_probs

            logger.info("\n  --- [ML 스코어 계산 결과 (상위 5개)] ---")
            logger.info(df_step3[['code', 'ml_score']].sort_values('ml_score', ascending=False).head().to_string())
            
            min_ml_score = params['min_ml_score']
            df_step4 = df_step3[df_step3['ml_score'] >= min_ml_score]
            logger.info(f"\n  [4] ML 스코어 필터 (>= {min_ml_score}) 후: {len(df_step4)} 개")
            if len(df_step3) > 0 and len(df_step4) == 0:
                logger.info("    [최종 진단] 모든 후보 종목의 ML 스코어가 설정된 최소 점수보다 낮아 최종 탈락했습니다.")
    else:
        logger.info("  [4] ML 스코어 필터: 이전 단계에서 살아남은 후보 종목이 없어 건너뜁니다.")

    logger.info("="*80)