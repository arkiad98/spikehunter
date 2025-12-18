# modules/optimization.py (v5.9 - Parallel Optimization Restored)
"""
[SpikeHunter v5.9] 전략 파라미터 최적화 모듈
- Optuna를 사용하여 백테스트 성능을 최대화하는 파라미터 조합을 탐색합니다.
- 복원: 병렬 처리(n_jobs) 기능 복구 (사용자 입력 지원)
"""

import optuna
import os
import pandas as pd
from datetime import datetime
import joblib

from .utils_io import read_yaml, to_date, get_user_input, update_yaml
from .utils_logger import logger
from .backtest import run_backtest

def objective(trial, settings_path, strategy_name, start_date, end_date, preloaded_features):
    """Optuna 목적 함수: 특정 파라미터 조합으로 백테스트 실행 후 성과 반환"""
    
    cfg = read_yaml(settings_path)
    opt_cfg = cfg.get("optimization", {}).get(strategy_name, {})
    param_space = opt_cfg.get("param_space", {})
    
    # 1. 파라미터 제안 (Suggest)
    params = {}
    for name, config in param_space.items():
        if config['type'] == 'int':
            params[name] = trial.suggest_int(name, config['low'], config['high'])
        elif config['type'] == 'float':
            step = config.get('step', None)
            params[name] = trial.suggest_float(name, config['low'], config['high'], step=step)
        elif config['type'] == 'categorical':
            params[name] = trial.suggest_categorical(name, config['choices'])
            
    # 2. 백테스트 실행
    # 임시 결과 폴더 (결과 저장 안 함)
    # 병렬 실행 시 폴더명 충돌을 방지하기 위해 trial 번호를 포함
    temp_run_dir = os.path.join(cfg["paths"]["cache"], f"opt_trial_{trial.number}")
    
    try:
        result = run_backtest(
            run_dir=temp_run_dir,
            strategy_name=strategy_name,
            settings_cfg=cfg,
            start=start_date,
            end=end_date,
            param_overrides=params,
            quiet=True,
            preloaded_features=preloaded_features,
            save_to_db=False # 최적화 중에는 DB 저장 생략
        )
        
        if not result: return -999.0 # 실패 시 페널티
        
        metrics = result['metrics']
        optimize_on = opt_cfg.get("optimize_on", "Sharpe_raw")
        
        score = metrics.get(optimize_on, -999.0)
        
        # 안전장치: 거래 횟수가 너무 적으면 페널티 (과적합 방지)
        if metrics.get('총거래횟수', 0) < 10:
            return -999.0
            
        return score
        
    except Exception as e:
        # logger.warning(f"Trial {trial.number} failed: {e}")
        return -999.0

def run_optimization_pipeline(settings_path: str, strategy_name: str = "SpikeHunter", 
                              use_ml_target: bool = True, regime_to_optimize: str = None, 
                              warm_start: bool = True):
    """전략 최적화 메인 파이프라인"""
    logger.info("\n" + "="*60)
    logger.info("      <<< 전략 파라미터 최적화 (Strategy Optimization) >>>")
    logger.info("="*60)
    
    # [복구] 사용자 입력으로 병렬 작업 수 설정
    n_jobs_input = get_user_input("병렬 작업 수(n_jobs)를 입력하세요 (엔터: 1, -1: 전체): ")
    try:
        n_jobs = int(n_jobs_input) if n_jobs_input.strip() else 1
    except ValueError:
        n_jobs = 1
        
    logger.info(f" >> 설정: n_jobs={n_jobs}")
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # 최적화 기간 설정
    start_date = "2020-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"최적화 데이터 로드 중... ({start_date} ~ {end_date})")
    
    # 데이터 미리 로드 (속도 향상 및 I/O 병목 제거)
    dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋이 없습니다: {dataset_path}")
        return

    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 컬럼명 통일 (안전장치)
    rename_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Optuna Study 생성
    study_name = f"Opt-Strategy-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    n_trials = cfg.get("optimization", {}).get(strategy_name, {}).get("n_trials", 20)
    
    logger.info(f"최적화 시작: {n_trials}회 시도")
    
    # 실행
    study.optimize(
        lambda trial: objective(trial, settings_path, strategy_name, start_date, end_date, df),
        n_trials=n_trials,
        n_jobs=n_jobs, # 사용자 입력값 적용
        show_progress_bar=True
    )
    
    logger.info("="*60)
    logger.info(f"최적화 완료. Best Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info("="*60)
    
    # 결과 저장 여부
    choice = get_user_input("최적 파라미터를 settings.yaml에 저장하시겠습니까? (y/n): ")
    if choice.lower() == 'y':
        update_yaml(settings_path, "strategies", strategy_name, study.best_params)
        logger.info("설정 파일이 업데이트되었습니다.")