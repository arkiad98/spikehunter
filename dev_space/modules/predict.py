# predict.py

import os
import pandas as pd
from datetime import datetime, timedelta
import joblib
from abc import ABC, abstractmethod
from typing import Optional
import lightgbm as lgb

# 프로젝트 유틸리티 모듈 임포트
from modules.utils_io import read_yaml, load_partition_day, ensure_dir, load_index_data
from modules.utils_logger import logger

# [복원] backtest.py와 동일한 Strategy 클래스 정의 포함
class Strategy(ABC):
    """모든 전략 클래스가 상속받아야 하는 추상 기본 클래스(Abstract Base Class)."""
    def __init__(self, params: dict):
        self.params = params
        self.name = params.get("name", "UnnamedStrategy")

        # 파라미터를 명시적으로 할당하여 안정성 확보
        self.target_r = params.get('target_r', 0.15)
        self.stop_r = params.get('stop_r', -0.05)
        self.max_hold = int(params.get('max_hold', 5))
        self.max_dist_from_ma = params.get('max_dist_from_ma', 0.2)
        self.entry_slip_r = params.get('entry_slip_r', 0.01)
        self.trail_k = params.get('trail_k', 2.0)
        self.max_market_vol = params.get('max_market_vol', 0.02)
        self.max_daily_ret_entry = params.get('max_daily_ret_entry', 0.1)
        self.min_ml_score = params.get('min_ml_score', 0.5)
        self.min_avg_value = params.get('min_avg_value', 1_000_000_000)
        self.top_n = int(params.get('top_n', 7))

    @abstractmethod
    def generate_recs(self, df_feat: pd.DataFrame, model_clf: Optional[lgb.LGBMClassifier] = None) -> pd.DataFrame:
        """매수 추천 종목을 생성하는 로직 (하위 클래스에서 구현)"""
        raise NotImplementedError("generate_recs() must be implemented by a subclass.")

class SpikeHunterStrategy(Strategy):
    """SpikeHunter 전략 로직을 구현한 클래스."""
    def __init__(self, params: dict):
        super().__init__(params)

    def generate_recs(self, df_feat: pd.DataFrame, model_clf: Optional[lgb.LGBMClassifier] = None) -> pd.DataFrame:
        """SpikeHunter 전략의 매수 추천 종목 생성 로직"""
        # backtest.py의 최종 로직과 동일하게 유지
        entry_conditions = (
            (df_feat['signal_spike_hunter'] == 1) &
            (df_feat['dist_from_ma20'] < self.max_dist_from_ma) &
            (df_feat["avg_value_20"] >= self.min_avg_value)
        )
        recs = df_feat[entry_conditions].copy()
        recs = recs[recs['daily_ret'] < self.max_daily_ret_entry]
        
        if model_clf is None or recs.empty:
            return pd.DataFrame()

        try:
            required_features = model_clf.feature_names_in_
            features_for_ml = recs.set_index('code').loc[:, required_features]
            pred_probs = model_clf.predict_proba(features_for_ml)[:, 1]
            recs['ml_score'] = pred_probs
        except Exception as e:
            logger.warning(f"ML score 예측 중 오류 발생: {e}")
            return pd.DataFrame()
        
        recs = recs[recs['ml_score'] >= self.min_ml_score]
        
        if recs.empty:
            return pd.DataFrame()

        recs = recs.sort_values(by="ml_score", ascending=False).copy()
        recs["rank"] = range(1, len(recs) + 1)
        
        final_recs = recs[recs["rank"] <= self.top_n]
        return final_recs

# [복원] 시장 국면 판단 함수
def _determine_regime(is_bull_market: bool, is_stable_market: bool) -> str:
    """시장 상황에 따라 4개의 국면(Regime) 중 하나를 결정합니다."""
    if is_bull_market and is_stable_market: return "R1_BullStable"
    if is_bull_market and not is_stable_market: return "R2_BullVolatile"
    if not is_bull_market and is_stable_market: return "R3_BearStable"
    return "R4_BearVolatile"

# [수정] 디버그 로그가 추가된 버전
def generate_recommendations(cfg: dict, strategy_key: str, features_today: pd.DataFrame, model_clf) -> pd.DataFrame:
    """주어진 전략과 피처 데이터로 오늘의 추천 종목을 생성합니다."""
    
    logger.info("\n--- [추천 종목 생성 디버그 시작] ---")
    
    # 1. 파라미터 로드
    common_params = {
        'min_avg_value': cfg.get('min_avg_value'),
        'top_n': cfg.get('top_n')
    }
    strategy_params = cfg.get("strategies", {}).get(strategy_key)
    if not strategy_params:
        logger.error(f"'{strategy_key}'에 해당하는 전략 파라미터를 찾을 수 없습니다.")
        return pd.DataFrame()
        
    final_params = {**common_params, **strategy_params}
    strategy = SpikeHunterStrategy(final_params)
    
    logger.info(f"  [0] 총 분석 대상 종목 수: {len(features_today)} 개")

    # 2. 필터링 로직 (SpikeHunterStrategy.generate_recs와 동일한 로직)
    #    각 단계별로 남는 종목 수를 추적합니다.
    
    # --- 단계 1: 기본 필터 (신호, 이격도, 거래대금) ---
    entry_conditions = (
        (features_today['signal_spike_hunter'] == 1) &
        (features_today['dist_from_ma20'] < strategy.max_dist_from_ma) &
        (features_today["avg_value_20"] >= strategy.min_avg_value)
    )
    recs = features_today[entry_conditions].copy()
    logger.info(f"  [1] 기본 필터 후: {len(recs)} 개")

    # --- 단계 2: 진정 필터 ---
    recs = recs[recs['daily_ret'] < strategy.max_daily_ret_entry]
    logger.info(f"  [2] 진정 필터(당일 급등 제외) 후: {len(recs)} 개")
    
    if recs.empty:
        logger.warning("모든 후보 종목이 필터링되었습니다. 추천 종목이 없습니다.")
        logger.info("--- [추천 종목 생성 디버그 종료] ---\n")
        return pd.DataFrame()

    # --- 단계 3: ML 스코어 계산 및 필터 ---
    try:
        required_features = model_clf.feature_names_in_
        features_for_ml = recs.set_index('code').loc[:, required_features]
        pred_probs = model_clf.predict_proba(features_for_ml)[:, 1]
        recs['ml_score'] = pred_probs
        
        logger.info(f"  [3] ML 스코어 계산 완료. (상위 점수: {recs['ml_score'].max():.4f}, 하위 점수: {recs['ml_score'].min():.4f})")
        
        recs_after_ml = recs[recs['ml_score'] >= strategy.min_ml_score]
        logger.info(f"  [4] ML 스코어 필터(>= {strategy.min_ml_score:.4f}) 후: {len(recs_after_ml)} 개")

        recs = recs_after_ml
    except Exception as e:
        logger.error(f"ML 스코어 계산 또는 필터링 중 오류 발생: {e}", exc_info=True)
        logger.info("--- [추천 종목 생성 디버그 종료] ---\n")
        return pd.DataFrame()

    if recs.empty:
        logger.warning("모든 후보 종목이 ML 스코어 필터에서 탈락했습니다. 추천 종목이 없습니다.")
        logger.info("--- [추천 종목 생성 디버그 종료] ---\n")
        return pd.DataFrame()
        
    # --- 단계 4: 최종 랭킹 및 선택 ---
    recs = recs.sort_values(by="ml_score", ascending=False).copy()
    recs["rank"] = range(1, len(recs) + 1)
    
    final_recs = recs[recs["rank"] <= strategy.top_n]
    logger.info(f"  [5] 최종 선택(top_n={strategy.top_n}) 후: {len(final_recs)} 개")
    logger.info("--- [추천 종목 생성 디버그 종료] ---\n")
    
    return final_recs

def generate_and_save_recommendations(settings_path: str):
    """데일리 운영을 위한 추천 종목 생성 및 저장을 총괄하는 메인 함수."""
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    # 1. 최신 피처 데이터 로드
    logger.info("최신 피처 데이터를 로드합니다...")
    today = datetime.today()
    start_of_period = today - timedelta(days=400) # MA200 계산 등을 위해 넉넉히
    latest_features = load_partition_day(paths["features"], start_of_period, today)
    
    if latest_features.empty:
        logger.error("추천 종목을 생성할 최신 피처 데이터가 없습니다.")
        return

    latest_date = latest_features['date'].max()
    logger.info(f"데이터 최신 기준일: {latest_date.date()}")
    
    features_today = latest_features[latest_features['date'] == latest_date].copy()

    # 2. 실시간 시장 상황 판단
    # [수정] KOSPI 데이터를 불러오는 함수를 load_index_data로 변경
    kospi = load_index_data(start_of_period, today, paths["raw_index"])
    kospi_past_and_today = kospi[kospi['date'] <= latest_date]
    
    if len(kospi_past_and_today) < 200:
        logger.error("시장 국면을 판단하기에 KOSPI 데이터가 부족합니다.")
        return
        
    current_kospi_close = kospi_past_and_today['kospi_close'].iloc[-1]
    current_ma200 = kospi_past_and_today['kospi_close'].rolling(200).mean().iloc[-1]
    current_kospi_vol_20d = kospi_past_and_today['kospi_close'].pct_change().rolling(20).std().iloc[-1]
    
    base_vol_threshold_params = cfg["strategies"].get("SpikeHunter_R1_BullStable", {})
    vol_threshold = base_vol_threshold_params.get("max_market_vol", 0.02)
    is_bull_market = current_kospi_close > current_ma200
    is_stable_market = current_kospi_vol_20d < vol_threshold
    
    current_regime = _determine_regime(is_bull_market, is_stable_market)
    logger.info(f"시장 진단 결과: {current_regime} (강세장={is_bull_market}, 안정장={is_stable_market})")

    strategy_key = f"SpikeHunter_{current_regime}"
    logger.info(f"'{strategy_key}' 전략으로 추천 종목을 생성합니다...")

    # 3. 모델 로드 및 추천 종목 생성
    model_clf_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if not os.path.exists(model_clf_path):
        logger.error(f"추천에 필요한 ML 모델({model_clf_path})이 없습니다.")
        return
    model_clf = joblib.load(model_clf_path)

    recommendations = generate_recommendations(cfg, strategy_key, features_today, model_clf)
    
    # 4. 결과 저장
    if recommendations.empty:
        logger.info("최종 추천 종목이 없습니다.")
    else:
        output_dir = paths["predictions"]
        ensure_dir(output_dir)
        filename = f"{latest_date.date()}.csv"
        output_path = os.path.join(output_dir, filename)
        
        # 종목명 추가
        from modules.utils_io import get_stock_names
        names = get_stock_names(recommendations['code'].tolist())
        recommendations['name'] = recommendations['code'].map(names)
        
        # --- [추가] 터미널에 추천 종목 리스트를 표 형태로 출력 ---
        display_cols = ['rank', 'name', 'code', 'close', 'ml_score']
        recommendations_display = recommendations[display_cols].copy()
        # 'close' 컬럼을 천 단위 콤마가 있는 정수 형태로 포맷팅
        recommendations_display['close'] = recommendations_display['close'].map('{:,.0f}'.format)

        logger.info("="*60)
        logger.info(f"     <<< {latest_date.date()} 오늘의 추천 종목 >>>")
        logger.info("="*60)
        print(recommendations_display.to_string(index=False))
        logger.info("="*60)
        # ------------------------- 출력 코드 추가 완료 -------------------------
        
        # 필요한 컬럼만 선택하여 저장
        cols_to_save = ['rank', 'code', 'name', 'close', 'ml_score']
        recommendations[cols_to_save].to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.4f')
        logger.info(f"{len(recommendations)}개 추천 종목을 '{output_path}'에 저장했습니다.")