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
from modules import collect, derive  # [추가] 데이터 수집 및 파생 모듈 임포트
from pykrx import stock  # [추가] 영업일 확인용

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

    # [수정] 필수 파생 피처 보정 (Calibration Step) - 리빌딩
    # predict.py가 기대하는 컬럼명과 features.py가 생성하는 컬럼명 간의 불일치를 해결하고
    # 누락된 파생 변수를 즉석에서 계산합니다.
    try:
        # 1. 이격도 매핑 (dist_ma20 -> dist_from_ma20)
        if 'dist_from_ma20' not in features_today.columns:
            if 'dist_ma20' in features_today.columns:
                features_today['dist_from_ma20'] = features_today['dist_ma20'].abs() # 절대값 처리 필요시 확인
            elif 'close' in features_today.columns and 'ma20' in features_today.columns:
                features_today['dist_from_ma20'] = (features_today['close'] / features_today['ma20'] - 1).abs()
            else:
                logger.warning("'dist_from_ma20' 계산 불가 (ma20 부재). 0으로 대체")
                features_today['dist_from_ma20'] = 0

        # 2. 평균 거래대금 (avg_value_20)
        if 'avg_value_20' not in features_today.columns:
            # 거래대금 컬럼이 있는지 확인
            if 'amount' not in features_today.columns:
                 features_today['amount'] = features_today['close'] * features_today['volume']
            
            # rolling mean은 전체 데이터가 필요하므로, 여기서는 단일 날짜 데이터만 있을 수 있음.
            # 하지만 features_today는 load_partition_day로 로드했으므로 history가 있을 수 있음?
            # 아니, predict.py 앞부분에서 features_today = df_all[df_all['date'] == latest_date].copy() 했으므로 1일치임.
            # 따라서 'avg_value_20'은 derive 단계에서 생성되어 있어야 함.
            # 만약 없다면, amount라도 씁니다 (임시 방편) 혹은 amount_ma5 등 대체재 찾기
            if 'amount_ma5' in features_today.columns:
                 # amount_bil 단위(억)일 수 있으므로 확인 필요. features.py 보니 amount_bil = close*vol/1억.
                 # avg_value_20은 min_avg_value(10억)와 비교되므로 원화 단위여야 함.
                 # amount_ma5rk가 억단위라면 1억 곱해줌.
                 features_today['avg_value_20'] = features_today['amount_ma5'] * 100_000_000
            elif 'amount' in features_today.columns:
                 features_today['avg_value_20'] = features_today['amount'] # 당일 거래대금으로 대체
            else:
                 features_today['avg_value_20'] = 0
        
        # 3. 등락률 (daily_ret)
        if 'daily_ret' not in features_today.columns:
            # change 컬럼 확인 (pykrx 등에서 줌)
            if 'change' in features_today.columns:
                features_today['daily_ret'] = features_today['change']
            else:
                # 전일 종가 정보가 없으면 계산 불가. 0으로 처리
                features_today['daily_ret'] = 0

        logger.info("  -> 필수 피처 보정 완료")

    except Exception as e:
        logger.error(f"피처 보정 중 오류 발생: {e}")
        return pd.DataFrame()
    #    각 단계별로 남는 종목 수를 추적합니다.
    
    # --- 단계 1: 기본 필터 (이격도, 거래대금) ---
    entry_conditions = (
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



def get_target_business_day() -> Optional[pd.Timestamp]:
    """
    현재 시각을 기준으로 추천에 사용할 목표 영업일을 결정합니다.
    - 장 마감(15:40) 전: 전일(또는 직전 영업일)
    - 장 마감(15:40) 후: 당일(또는 당일이 휴일이면 직전 영업일)
    """
    now = datetime.now()
    cutoff_time = now.replace(hour=17, minute=0, second=0, microsecond=0)
    
    # 최근 14일간의 영업일 조회 (넉넉하게)
    end_str = now.strftime("%Y%m%d")
    start_str = (now - timedelta(days=14)).strftime("%Y%m%d")
    
    try:
        # [수정] KOSPI 지수 데이터 수집(get_index_ohlcv) 오류 우회 (2026년 issue)
        # 삼성전자(005930)를 Proxy로 사용하여 영업일 확인
        df_index = stock.get_market_ohlcv_by_ticker(start_str, end_str, "005930")
        
        if df_index.empty:
            logger.error("영업일 목록을 가져올 수 없습니다.")
            return None
            
        valid_days = df_index.index
        last_day = valid_days[-1]
        
        is_today_market_open = (last_day.date() == now.date())
        
        if is_today_market_open:
            if now >= cutoff_time:
                return last_day # 장 마감 후: 오늘 데이터 사용
            else:
                return valid_days[-2] # 장 마감 전: 어제(직전 영업일) 데이터 사용
        else:
            return last_day # 오늘은 휴장일: 마지막 개장일 사용
            
    except Exception as e:
        logger.error(f"목표 영업일 결정 중 오류 발생: {e}")
        return None

def ensure_latest_data(settings_path: str, target_date: pd.Timestamp):
    """
    현재 데이터셋이 목표 영업일(target_date)까지 포함하고 있는지 확인하고,
    부족하다면 자동으로 수집(collect) 및 파생(derive)을 수행합니다.
    """
    cfg = read_yaml(settings_path)
    dataset_path = os.path.join(cfg["paths"].get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    
    current_latest_date = pd.Timestamp.min
    
    if os.path.exists(dataset_path):
        try:
            # 전체를 읽지 않고 날짜만 빠르게 확인하고 싶지만, parquet 특성상 메타데이터 확인이 복잡할 수 있음.
            # 여기서는 편의상 날짜 컬럼만 로드 (pandas >= 0.24)
            df_dates = pd.read_parquet(dataset_path, columns=['date'])
            if not df_dates.empty:
                current_latest_date = pd.to_datetime(df_dates['date']).max()
        except Exception as e:
            logger.warning(f"기존 데이터셋 날짜 확인 실패: {e}. 재구축을 시도합니다.")
    
    # 날짜 비교 (시간 정보 제거)
    target_date_normalized = target_date.normalize()
    current_latest_normalized = current_latest_date.normalize()
    
    if current_latest_normalized < target_date_normalized:
        logger.warning(f"데이터가 최신이 아닙니다. (보유: {current_latest_normalized.date()}, 목표: {target_date_normalized.date()})")
        logger.info(">> 최신 데이터 자동 수집 및 생성을 시작합니다...")
        
        target_date_str = target_date_normalized.strftime("%Y-%m-%d")
        
        # 1. 수집
        collect.run_collect(settings_path, end=target_date_str)
        
        # 2. 파생 (전체 기간 갱신)
        derive.run_derive(settings_path)
        
        logger.info(">> 데이터 최신화 완료.")
    else:
        logger.info(f"데이터가 이미 최신 상태입니다. ({current_latest_normalized.date()})")

def generate_and_save_recommendations(settings_path: str):
    """데일리 운영을 위한 추천 종목 생성 및 저장을 총괄하는 메인 함수."""
    
    # [추가] 0. 최신 영업일 확인 및 데이터 자동 갱신
    target_business_day = get_target_business_day()
    if target_business_day:
        logger.info(f"목표 기준 영업일: {target_business_day.date()}")
        try:
            ensure_latest_data(settings_path, target_business_day)
        except Exception as e:
            logger.error(f"데이터 자동 갱신 중 오류 발생: {e}. 기존 데이터로 진행합니다.")
    else:
        logger.warning("목표 영업일을 결정하지 못해 데이터 갱신을 건너뜁니다.")

    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    # 1. 최신 피처 데이터 로드
    # 1. 최신 피처 데이터 로드
    logger.info("최신 피처 데이터를 로드합니다...")
    
    # [복구] KOSPI 로드 등에 필요한 날짜 변수 정의
    today = datetime.today()
    start_of_period = today - timedelta(days=400) # MA200 계산 등을 위해 넉넉히
    
    # [수정] 단일 파일 로드 방식 적용 (ml_dataset 경로 사용)
    dataset_path = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    
    if not os.path.exists(dataset_path):
        logger.error(f"피처 데이터 파일이 없습니다: {dataset_path}\n'피처 생성(Derive)' 단계를 먼저 실행해주세요.")
        return

    try:
        # 전체 데이터 로드 (필요시 날짜 필터링 최적화 가능)
        df_all = pd.read_parquet(dataset_path)
        df_all['date'] = pd.to_datetime(df_all['date'])
        
        if df_all.empty:
            logger.error("피처 데이터 파일이 비어있습니다.")
            return
            
        # [수정] latest_date를 target_business_day가 있으면 우선 사용, 없으면 데이터상 최신일
        if target_business_day:
             latest_date = target_business_day
        else:
             latest_date = df_all['date'].max()

        logger.info(f"데이터 최신 기준일: {latest_date.date()}")
        
        # 특정 날짜의 데이터만 필터링
        features_today = df_all[df_all['date'].dt.date == latest_date.date()].copy()
        
        if features_today.empty:
            logger.error(f"{latest_date.date()} 에 해당하는 피처 데이터가 없습니다.")
            return
        
    except Exception as e:
        logger.error(f"피처 데이터 로드 중 오류 발생: {e}")
        return

    except Exception as e:
        logger.error(f"피처 데이터 로드 중 오류 발생: {e}")
        return

    # [NEW] Apply Date-Specific Exclusion (Same as Backtest)
    exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
    if os.path.exists(exclude_path):
        # from modules.utils_io import read_yaml # [Removed] Shadowing fix
        ex_cfg = read_yaml(exclude_path)
        exclusions = ex_cfg.get('exclusions', [])
        
        if exclusions and not features_today.empty:
            original_len = len(features_today)
            exclude_set = set()
            for item in exclusions:
                c = item['code']
                for d in item['dates']:
                     # For 'predict', we only care if today's date matches the exclusion
                     exclude_set.add((c, pd.Timestamp(d).date()))
            
            # Check if any row in features_today matches exclusion
            # features_today usually has 1 date, but iterate to be safe
            keep_indices = []
            for idx, row in features_today.iterrows():
                if (row['code'], row['date'].date()) not in exclude_set:
                    keep_indices.append(idx)
            
            features_today = features_today.loc[keep_indices]

            if len(features_today) < original_len:
                logger.info(f"이상치 데이터 제외 적용: {original_len - len(features_today)}개 종목 제외됨.")

    if features_today.empty:
        logger.warning("제외 목록 적용 후 추천 대상 종목이 없습니다.")
        return

    # 2. 실시간 시장 상황 판단
    # [수정] KOSPI 데이터를 불러오는 함수를 load_index_data로 변경
    kospi = load_index_data(start_of_period, today, paths["raw_index"])
    
    # [Safety] KOSPI 데이터 부재 시 기본값 처리
    if kospi.empty:
        logger.warning("KOSPI 데이터가 없어 시장 국면을 'R4_BearVolatile'로 가정합니다.")
        is_bull_market = False
        is_stable_market = False
    else:
        kospi_past_and_today = kospi[kospi['date'] <= latest_date]
        
        if len(kospi_past_and_today) < 200:
            logger.warning("시장 국면을 판단하기에 KOSPI 데이터가 부족합니다 (<200). 'R4_BearVolatile'로 가정합니다.")
            is_bull_market = False
            is_stable_market = False
        else:
            current_kospi_close = kospi_past_and_today['kospi_close'].iloc[-1]
            current_ma200 = kospi_past_and_today['kospi_close'].rolling(200).mean().iloc[-1]
            current_kospi_vol_20d = kospi_past_and_today['kospi_close'].pct_change().rolling(20).std().iloc[-1]
            
            base_vol_threshold_params = cfg["strategies"].get("SpikeHunter_R1_BullStable", {})
            vol_threshold = base_vol_threshold_params.get("max_market_vol", 0.02)
            
            is_bull_market = current_kospi_close > current_ma200
            is_stable_market = current_kospi_vol_20d < vol_threshold

    
    current_regime = _determine_regime(is_bull_market, is_stable_market)
    logger.info(f"시장 진단 결과: {current_regime} (강세장={is_bull_market}, 안정장={is_stable_market})")
    
    # [수정] R2~R4 미사용 정책 반영 (Single Strategy Mode)
    # 시장 국면과 관계없이 메인 전략(R1)을 강제 적용합니다.
    strategy_key = "SpikeHunter_R1_BullStable" 
    logger.info(f"[System] Single Strategy Mode Enforced. Using '{strategy_key}'")
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
        
        # 5. DB 저장 (검증용)
        try:
            from modules.utils_db import create_tables, insert_daily_signals
            create_tables() # ensure table exists
            insert_daily_signals(recommendations, strategy_key)
        except Exception as e:
            logger.warning(f"추천 종목 DB 저장 실패: {e}")