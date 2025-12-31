import pandas as pd
from datetime import datetime, timedelta
import joblib
import os

from modules.utils_io import read_yaml, load_partition_day, load_index_data
from modules.utils_logger import logger

def find_latest_date_in_parquet(file_path: str) -> pd.Timestamp:
    """Parquet 파일 내 가장 최근 날짜를 반환합니다."""
    if not os.path.exists(file_path):
        return None
    try:
        # date 컬럼만 읽어서 최대값 확인 (효율성)
        df_dates = pd.read_parquet(file_path, columns=['date'])
        return df_dates['date'].max()
    except Exception as e:
        logger.error(f"날짜 확인 중 오류 발생: {e}")
        return None

def run_strategy_debugger(settings_path: str):
    """
    SpikeHunter 전략(v4.0)의 필터링 과정을 추적하여
    매수 추천 종목이 없는 원인을 진단합니다. (ML 스코어 중심)
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< SpikeHunter 전략 디버거 v4.0 (ML Focus) >>>")
    logger.info("="*80)

    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    # 1. 분석할 최신 데이터 로드 (ML Dataset 사용)
    # derive.py가 생성한 최종 데이터셋을 사용해야 모든 피처가 포함되어 있음
    dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    
    if not os.path.exists(dataset_path):
        logger.error(f"ML 데이터셋 파일이 없습니다: {dataset_path}")
        logger.error("메인 메뉴에서 '2. 피처 생성 및 라벨링 (Derive)'을 먼저 실행해주세요.")
        return

    logger.info("데이터셋을 로드하여 최신 날짜를 확인합니다...")
    target_date = find_latest_date_in_parquet(dataset_path)
    
    if target_date is None:
        logger.error("데이터셋에서 날짜 정보를 읽을 수 없습니다.")
        return
    
    logger.info(f"🔍 분석 대상 날짜: {target_date.date()}")
    
    # 해당 날짜의 데이터만 로드
    df_all = pd.read_parquet(dataset_path)
    df_today = df_all[df_all['date'] == target_date].copy()

    
    if df_today.empty:
        logger.error("데이터를 로드했으나 비어있습니다.")
        return

    # 2. 파라미터 로드 (SpikeHunter_R1_BullStable 기준)
    # v4.0 전략은 Regime 구분 없이 ML Score를 메인으로 사용합니다.
    strategy_name = "SpikeHunter_R1_BullStable"
    if 'strategies' in cfg and strategy_name in cfg['strategies']:
        params = cfg['strategies'][strategy_name]
    else:
        logger.warning(f"전략 '{strategy_name}' 설정이 없어 기본값을 사용합니다.")
        params = {}
        
    ml_params = cfg.get("ml_params", {})
    threshold = params.get('min_ml_score', ml_params.get('classification_threshold', 0.4))
    
    logger.info(f"기준 임계값(Threshold): {threshold}") # min_ml_score

    # 3. 모델 로드
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if not os.path.exists(model_path):
        logger.error(f"ML 모델 파일이 없습니다: {model_path}")
        return
        
    try:
        model_clf = joblib.load(model_path)
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return

    # 4. ML 스코어 계산
    # feature_names_in_ 확인
    if not hasattr(model_clf, 'feature_names_in_'):
        logger.error("모델에 'feature_names_in_' 속성이 없습니다. 호환되지 않는 모델입니다.")
        return

    features_needed = model_clf.feature_names_in_
    missing_cols = [c for c in features_needed if c not in df_today.columns]
    
    if missing_cols:
        logger.warning(f"데이터에 일부 피처가 누락되어 0으로 채웁니다: {missing_cols[:5]}...")
        for c in missing_cols:
            df_today[c] = 0
            
    X = df_today[features_needed].fillna(0)
    scores = model_clf.predict_proba(X)[:, 1]
    df_today['ml_score'] = scores
    
    # 5. 결과 분석
    logger.info("\n--- [ML 스코어 분석 결과] ---")
    logger.info(f"전체 대상 종목 수: {len(df_today)} 개")
    logger.info(f"ML Score 평균: {scores.mean():.4f}, 최대: {scores.max():.4f}, 최소: {scores.min():.4f}")
    
    passed_candidates = df_today[df_today['ml_score'] >= threshold].sort_values('ml_score', ascending=False)
    num_passed = len(passed_candidates)
    
    logger.info(f"임계값({threshold}) 이상 통과 종목: {num_passed} 개")
    
    if num_passed > 0:
        logger.info("\n[상위 후보 종목 TOP 10]")
        print(passed_candidates[['code', 'close', 'ml_score']].head(10).to_string(index=False))
        
        # 추가 진단: 보유 기간 내 매도되었을 경우 추정 (백테스트 로직 일부 차용)
        # 여기서는 단순히 목록만 보여줌
    else:
        logger.warning("\n[진단] 임계값을 넘는 종목이 하나도 없습니다.")
        logger.info("  - 시장 상황이 좋지 않거나, 모델이 매우 보수적일 수 있습니다.")
        logger.info("  - '최적 임계값 탐색(Add-on 7)'을 실행하여 임계값을 조정해보세요.")
        
        # 아쉽게 탈락한 종목들
        logger.info("\n[아쉽게 탈락한 상위 종목 TOP 5]")
        logger.info(df_today[['code', 'close', 'ml_score']].sort_values('ml_score', ascending=False).head(5).to_string(index=False))

    logger.info("="*80)