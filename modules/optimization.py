import os
import optuna
import pandas as pd
from datetime import datetime
from ruamel.yaml import YAML

from modules.utils_logger import logger
from modules.utils_io import read_yaml, update_yaml, get_user_input
from modules.backtest import run_backtest

def objective(trial, settings_path, strategy_name, start_date, end_date, preloaded_features=None):
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
            save_to_db=False, # 최적화 중에는 DB 저장 생략
            skip_exclusion=True # [Optim] 이미 필터링된 데이터 사용
        )
        
        if not result: return -999.0 # 실패 시 페널티
        
        metrics = result['metrics']
        optimize_on = opt_cfg.get("optimize_on", "Sharpe_raw")
        
        if optimize_on == "Hybrid_Stability":
            # [Refined V2] 안정성 + 리스크 관리 (Fixed MDD Constraint)
            # 동적 판단(Look-ahead Bias) 대신 고정된 완화값(-30%) 사용
            sharpe = metrics.get('Sharpe_raw', -999.0)
            win_rate = metrics.get('win_rate_raw', 0.0)
            mdd = metrics.get('MDD_raw', -1.0) 
            
            # Constraint 1: 승률 < 20%
            # Constraint 2: MDD < -30% (Bear Market 수용 가능한 현실적 하한선)
            if win_rate < 0.2 or mdd < -0.30:
                # [Debug] rejection reason logging
                logger.debug(f"Trial {trial.number} Rejected: WR={win_rate:.2f}, MDD={mdd:.2f}")
                score = -999.0 # 탈락
            else:
                # [Robust Metric] 단순 Sharpe가 아닌, MDD와 승률을 반영한 안정성 점수
                # Score = Sharpe * (승률 가중치) - (MDD 페널티)
                # MDD가 0에 가까울수록(음수) 좋음. mdd가 낮을수록(더 큰 음수) 페널티.
                stability_score = sharpe * (1 + win_rate) * (1 + 1/abs(mdd-0.01)) 
                score = stability_score
        else:
            score = metrics.get(optimize_on, -999.0)

        
        # 안전장치: 거래 횟수가 너무 적으면 페널티 (과적합 방지, WFO 안정성 위해 완화)
        if metrics.get('총거래횟수', 0) < 3:
            # [Debug] rejection reason
            logger.debug(f"Trial {trial.number} Rejected: Too few trades ({metrics.get('총거래횟수', 0)})")
            return -999.0
            
        return score
        
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return -999.0

def optimize_strategy_headless(settings_path: str, strategy_name: str, 
                             start_date: str, end_date: str, 
                             n_trials: int = 20, n_jobs: int = 1, dataset_path: str = None):
    """비대화형 전략 최적화 (Walk-Forward용)"""
    logger.info(f" >> [WFO] 전략 최적화 시작 ({start_date} ~ {end_date})")
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # 데이터셋 로드
    if dataset_path is None:
        # Default logic for WFO (Train split prioritized)
        train_split_path = os.path.join(paths["features"], "dataset_v4.parquet")
        if os.path.exists(train_split_path):
            dataset_path = train_split_path
        else:
            dataset_path = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
        
    if not os.path.exists(dataset_path):
        logger.error(f"[WFO] 데이터셋 없음: {dataset_path}")
        return {}

    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Rename columns safety
    rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df.rename(columns=rename_map, inplace=True)
    
    # Study
    study_name = f"WFO-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    # Warm Start (현재 설정값)
    current_params = cfg.get("strategies", {}).get(strategy_name, {})
    opt_cfg = cfg.get("optimization", {}).get(strategy_name, {})
    param_space = opt_cfg.get("param_space", {})
    
    warm_params = {}
    for k, v in current_params.items():
        if k in param_space:
             space = param_space[k]
             # Clamp value to be within [low, high]
             low = space.get('low', float('-inf'))
             high = space.get('high', float('inf'))
             
             if v < low: v = low
             if v > high: v = high
             
             warm_params[k] = v
             
    if warm_params:
        study.enqueue_trial(warm_params)
        
    # [Optimization Speedup] Pre-calculate ML Scores for WFO
    # WFO runs on Train Data (dataset_path), so we can pre-calc scores.
    # Note: ensure we use the model trained for THIS period (which should be in paths['models'])
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if os.path.exists(model_path):
        import joblib
        logger.info(f" >> [WFO] ML Score 사전 계산 중... ({model_path})")
        try:
            model = joblib.load(model_path)
            feature_names = getattr(model, 'feature_names_in_', None)
            
            if feature_names is not None:
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    for c in missing: df[c] = 0
                
                # Predict
                X_temp = df[feature_names].fillna(0)
                scores = model.predict_proba(X_temp)[:, 1]
                df['ml_score'] = scores
                logger.info(f" >> [WFO] ML Score 계산 완료. (Mean: {scores.mean():.4f}, Max: {scores.max():.4f})")
            else:
                df['ml_score'] = 0
        except Exception as e:
            logger.error(f"[WFO] 모델 로드/예측 오류: {e}")
            df['ml_score'] = 0
    else:
        df['ml_score'] = 0

    # [Constraint] 동적 국면 판단(Adaptive Regime) 제거
    # 이유: 백테스트 시점(Start~End)의 전체 수익률을 보고 제약을 거는 것은 Look-ahead Bias(미래 편향) 소지 있음
    # 따라서 고정된 완화값(-30%)을 Objective 함수 내에 하드코딩하여 사용함.

    # Execute
    study.optimize(
        lambda trial: objective(trial, settings_path, strategy_name, start_date, end_date, df),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False 
    )
    
    logger.info(f" >> [WFO] 최적화 완료. Best Score: {study.best_value:.4f}")
    return study.best_params

def run_optimization_pipeline(settings_path: str, strategy_name: str = "SpikeHunter", 
                              use_ml_target: bool = True, regime_to_optimize: str = None, 
                              warm_start: bool = True):
    """전략 최적화 메인 파이프라인"""
    logger.info("\n" + "="*60)
    logger.info("      <<< 전략 파라미터 최적화 (Strategy Optimization) >>>")
    logger.info("="*60)
    
    # 사용자 입력으로 병렬 작업 수 설정
    n_jobs_input = get_user_input("병렬 작업 수(n_jobs)를 입력하세요 (엔터: 1, -1: 전체): ")
    try:
        n_jobs = int(n_jobs_input) if n_jobs_input.strip() else 1
    except ValueError:
        n_jobs = 1
        
    logger.info(f" >> 설정: n_jobs={n_jobs}")
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # Strategy Name Auto-Detection
    if strategy_name == "SpikeHunter": # Default value check
        strategies = cfg.get("strategies", {})
        if strategies:
            strategy_name = list(strategies.keys())[0]
            logger.info(f" >> 감지된 전략: {strategy_name}")
        else:
            logger.warning("전략 설정이 없어 기본값 'SpikeHunter'를 사용합니다.")
    
    # 최적화 기간 설정
    start_date = "2020-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"최적화 데이터 로드 중... ({start_date} ~ {end_date})")
    
    # Correct Dataset Path
    dataset_path = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_path):
        # Fallback to feature dir if not in ml_dataset
        dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
        
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋이 없습니다: {dataset_path}")
        return

    try:
        df = pd.read_parquet(dataset_path)
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        logger.error(f"데이터셋 파일을 읽을 수 없습니다 (손상됨): {dataset_path}")
        logger.error(f"오류 내용: {e}")
        logger.info(">> 해결 방법: 메인 메뉴 '1. 데이터 관리' -> '2. 피처 생성 (Derive)'를 실행하여 데이터셋을 재생성해주세요.")
        return
    
    # 컬럼명 통일 (안전장치)
    rename_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # [Optim Efficiency] Pre-filter exclusions ONCE here
    exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
    if os.path.exists(exclude_path):
        ex_cfg = read_yaml(exclude_path)
        exclusions = ex_cfg.get('exclusions', [])
        
        if exclusions:
            logger.info(" >> 최적화 전 이상 데이터 일괄 제거 중...")
            original_len = len(df)
            
            exclude_set = set()
            for item in exclusions:
                c = item['code']
                for d in item['dates']:
                    exclude_set.add((c, pd.Timestamp(d).date()))
            
            exclude_records = []
            for c, d in exclude_set:
                exclude_records.append({'code': c, 'date': pd.Timestamp(d)})
            
            if exclude_records:
                exclude_df = pd.DataFrame(exclude_records)
                exclude_df['exclude'] = True
                exclude_df['date'] = pd.to_datetime(exclude_df['date'])
                
                df = df.merge(exclude_df, on=['code', 'date'], how='left')
                df = df[df['exclude'].isna()].drop(columns=['exclude'])
            
            filtered_len = len(df)
            if original_len > filtered_len:
                logger.info(f" >> 통합 필터링 완료: {original_len - filtered_len}건 제거됨.")
    
    # [Optimization Speedup] Pre-calculate ML Scores ONCE
    # This prevents running predict_proba in every single trial (massive bottleneck)
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if os.path.exists(model_path):
        import joblib
        logger.info(" >> [Optimization] ML Score 사전 계산 중... (Speed Optimization)")
        try:
            model = joblib.load(model_path)
            feature_names = getattr(model, 'feature_names_in_', None)
            
            if feature_names is not None:
                # Ensure columns exist
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    for c in missing: df[c] = 0
                
                # Predict
                X_temp = df[feature_names].fillna(0)
                scores = model.predict_proba(X_temp)[:, 1]
                df['ml_score'] = scores
                logger.info(f" >> ML Score 계산 완료. (Mean: {scores.mean():.4f}, Max: {scores.max():.4f})")
            else:
                df['ml_score'] = 0
        except Exception as e:
            logger.error(f"모델 로드/예측 중 오류 (무시됨): {e}")
            df['ml_score'] = 0 # Fallback
    else:
        logger.warning(f"모델 파일이 없어 스코어를 0으로 설정합니다: {model_path}")
        df['ml_score'] = 0

    # Optuna Study 생성
    study_name = f"Opt-Strategy-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    n_trials = cfg.get("optimization", {}).get(strategy_name, {}).get("n_trials", 20)
    
    logger.info(f"최적화 시작: {n_trials}회 시도")

    # [NEW] Warm Start: 현재 파라미터를 초기 큐에 추가
    baseline_value = -999.0
    if warm_start:
        current_strategy_params = cfg.get("strategies", {}).get(strategy_name, {})
        opt_cfg = cfg.get("optimization", {}).get(strategy_name, {})
        param_space = opt_cfg.get("param_space", {})
        
        # 최적화 대상 파라미터만 추출하여 초기값으로 설정
        warm_params = {}
        for key in param_space.keys():
            if key in current_strategy_params:
                warm_params[key] = current_strategy_params[key]
        
        if warm_params:
            study.enqueue_trial(warm_params)
            logger.info(f" >> Warm Start 설정됨: 현재 파라미터를 첫 번째 시도로 예약함 ({warm_params})")
    
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
    
    # [NEW] Baseline(Warm Start) 결과 존재 시 비교
    improvement_msg = ""
    is_improved = True
    
    if warm_start and len(study.trials) > 0:
        # 첫 번째 Trial(Warm Start)의 결과를 Baseline으로 간주
        # (주의: 병렬 처리 시 순서가 보장되지 않을 수 있으나, enqueue된 것은 보통 가장 먼저 할당됨)
        try:
            baseline_value = study.trials[0].value
            if baseline_value is not None:
                logger.info(f"Baseline Score (Current): {baseline_value:.4f}")
                diff = study.best_value - baseline_value
                if diff > 0.0001:
                    improvement_msg = f" (▲ {diff:.4f} 개선됨)"
                    is_improved = True
                else:
                    improvement_msg = " (개선 없음)"
                    is_improved = False
        except:
            pass
            
    logger.info(f"성능 비교: {improvement_msg}")
    logger.info("="*60)
    
    # 결과 저장 여부
    if is_improved:
        prompt_msg = "최적 파라미터를 settings.yaml에 저장하시겠습니까? (y/n): "
    else:
        prompt_msg = "성능 개선이 없거나 미미합니다. 그래도 저장하시겠습니까? (y/n): "
        
    choice = get_user_input(prompt_msg)
    if choice.lower() == 'y':
        update_yaml(settings_path, "strategies", strategy_name, study.best_params)
        
        # Consistency: Update ml_params as well if target_r or stop_r changed
        ml_updates = {}
        if 'target_r' in study.best_params:
            ml_updates['target_surge_rate'] = float(study.best_params['target_r']) # Cast to float for safety
        if 'stop_r' in study.best_params:
            ml_updates['stop_loss_rate'] = float(study.best_params['stop_r'])
        
        # XGB/LGBM thresholds mapping
        if 'min_ml_score' in study.best_params:
            ml_updates['classification_threshold'] = float(study.best_params['min_ml_score'])
            
        if ml_updates:
            # 직접 YAML 로드 및 수정 (update_yaml의 한계 극복)
            yaml = YAML()
            yaml.preserve_quotes = True
            
            with open(settings_path, 'r', encoding='utf-8') as f:
                config_data = yaml.load(f)
            
            if 'ml_params' in config_data:
                for k, v in ml_updates.items():
                    config_data['ml_params'][k] = v
                
                with open(settings_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f)
                logger.info(" >> ml_params 동기화 완료 (target_r/stop_r/threshold)")
            
        logger.info("설정 파일이 업데이트되었습니다.")