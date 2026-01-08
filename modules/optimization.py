import os
import joblib
import optuna
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from ruamel.yaml import YAML
from joblib import Parallel, delayed

from modules.utils_logger import logger
from modules.utils_io import read_yaml, update_yaml, get_user_input
from modules.utils_system import get_optimal_cpu_count
from modules.backtest import run_backtest

# [NEW] Optuna Distributions Helper
def get_optuna_distributions(param_space):
    """YAML 설정의 param_space를 Optuna Distribution 객체로 변환"""
    dists = {}
    for name, config in param_space.items():
        p_type = config.get('type')
        if p_type == 'int':
            dists[name] = optuna.distributions.IntDistribution(
                config['low'], config['high'], step=config.get('step', 1)
            )
        elif p_type == 'float':
            dists[name] = optuna.distributions.FloatDistribution(
                config['low'], config['high'], step=config.get('step', None)
            )
        elif p_type == 'categorical':
            dists[name] = optuna.distributions.CategoricalDistribution(config['choices'])
    return dists

# [NEW] Worker Function (Process-based)
def _worker_backtest(params, data_path, strategy_name, settings_path, start_date, end_date):
    """
    워커 프로세스용 백테스트 실행 함수
    - data_path: 멤맵(mmap) 가능한 파일 경로 (Zero-Copy)
    - params: Optuna가 제안한 파라미터 셋
    """
    try:
        # 1. Load Data (Memory Mapped - Zero Copy Read)
        # mmap_mode='r'로 로드하면 힙 메모리를 복제하지 않고 파일 포인터만 공유함
        df = joblib.load(data_path, mmap_mode='r')
        
        # 2. Load Config (Lightweight)
        # 각 프로세스에서 설정을 개별 로드 (Side-Effect 방지)
        cfg = read_yaml(settings_path)
        
        # 3. Run Backtest
        # preloaded_features에 df(mmap) 전달 -> run_backtest 내부에서 필요한 기간만 .copy()하여 사용
        result = run_backtest(
            run_dir=None, # Temp run (no artifacts save)
            strategy_name=strategy_name,
            settings_cfg=cfg,
            start=start_date,
            end=end_date,
            param_overrides=params,
            quiet=True,
            preloaded_features=df, 
            save_to_db=False,
            skip_exclusion=True # 이미 메인에서 필터링됨
        )
        
        if not result: return -999.0
        
        # 4. Calculate Score & Constraints
        metrics = result['metrics']
        opt_cfg = cfg.get("optimization", {}).get(strategy_name, {})
        optimize_on = opt_cfg.get("optimize_on", "Sharpe_raw")
        
        # Hybrid Stability Logic (Copied from original logic)
        sharpe = metrics.get('Sharpe_raw', -999.0)
        win_rate = metrics.get('win_rate_raw', 0.0)
        mdd = metrics.get('MDD_raw', -1.0) 
        
        score = -999.0
        
        if optimize_on == "Hybrid_Stability":
            # Constraint: WR < 20% or MDD < -30%
            if win_rate < 0.2 or mdd < -0.30:
                # Rejected
                return -999.0
            
            # Score Calculation
            # Score = Sharpe * (승률 가중치) - (MDD 페널티)
            score = sharpe * (1 + win_rate) * (1 + 1/abs(mdd-0.01))
            
        else:
            score = metrics.get(optimize_on, -999.0)
            
        # Min Trades Check
        if metrics.get('총거래횟수', 0) < 3:
            return -999.0
            
        return score
        
    except Exception as e:
        # Worker log might be lost or interleaved, safe to just return fail
        import traceback
        print(f"\n[Worker Error] Trail failed: {e}")
        traceback.print_exc()
        return -999.0

# [Refactored] Optimization Pipeline (In-Memory Batch Processing)
def run_optimization_pipeline(settings_path: str, strategy_name: str = "SpikeHunter", 
                              use_ml_target: bool = True, regime_to_optimize: str = None, 
                              warm_start: bool = True, n_jobs: int = None, auto_approve: bool = False):
    """전략 최적화 메인 파이프라인 (In-Memory Multiprocessing)"""
    logger.info("\n" + "="*60)
    logger.info("      <<< 전략 파라미터 최적화 (Strategy Optimization - Multiprocess) >>>")
    logger.info("="*60)
    
    # 1. Parallel Config
    default_jobs = get_optimal_cpu_count(0.75)
    if n_jobs is None:
        n_jobs_input = get_user_input(f"병렬 작업 수(n_jobs) (엔터: {default_jobs} [75%], -1: 전체): ")
        try:
            n_jobs = int(n_jobs_input) if n_jobs_input.strip() else default_jobs
            if n_jobs == -1: n_jobs = os.cpu_count()
        except ValueError:
            n_jobs = default_jobs
    
    # Windows Joblib Safety (Avoid excessive overhead)
    # joblib on Windows uses 'loky' backend which spawns new processes.
    # 20+ processes is fine with mmap.
    logger.info(f" >> 설정: n_jobs={n_jobs}")
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # Strategy Detection
    if strategy_name == "SpikeHunter": 
        strategies = cfg.get("strategies", {})
        if strategies:
            strategy_name = list(strategies.keys())[0]
            logger.info(f" >> 감지된 전략: {strategy_name}")
        else:
            logger.warning("전략 설정이 없어 기본값 'SpikeHunter'를 사용합니다.")
    
    # 2. Date Setup
    now = datetime.now()
    ml_params = cfg.get("ml_params", {})
    train_months = ml_params.get("classification_train_months", 36)
    offset_months = ml_params.get("classification_train_end_offset", 6)
    
    end_dt = now - pd.DateOffset(months=offset_months)
    start_dt = end_dt - pd.DateOffset(months=train_months)
    start_date = start_dt.strftime('%Y-%m-%d')
    end_date = end_dt.strftime('%Y-%m-%d')
    
    logger.info(f"최적화 데이터 로드 중... ({start_date} ~ {end_date})")
    
    # 3. Data Loading & Preprocessing
    dataset_path = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
        
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋이 없습니다: {dataset_path}")
        return

    try:
        # Load FULL dataset (to be mmapped)
        df = pd.read_parquet(dataset_path)
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        logger.error(f"데이터셋 로드 오류: {e}")
        return
    
    # Normalize Columns
    rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df.rename(columns=rename_map, inplace=True)
    
    # Apply Exclusions (Data Cleaning)
    exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
    if os.path.exists(exclude_path):
        ex_cfg = read_yaml(exclude_path)
        exclusions = ex_cfg.get('exclusions', [])
        if exclusions:
            logger.info(" >> 최적화 전 이상 데이터 일괄 제거 중...")
            exclude_set = set()
            for item in exclusions:
                for d in item['dates']:
                    exclude_set.add((item['code'], pd.Timestamp(d).date()))
            
            # Vectorized filter (Fast)
            # Create a lookup column
            df['temp_key'] = list(zip(df['code'], df['date'].dt.date))
            df = df[~df['temp_key'].isin(exclude_set)].drop(columns=['temp_key'])
            logger.info(f" >> 필터링 완료. (현재 데이터: {len(df)}행)")

    # Pre-calculate ML Scores
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if os.path.exists(model_path):
        logger.info(" >> [Optimization] ML Score 사전 계산 중... (Speed Optimization)")
        try:
            model = joblib.load(model_path)
            feature_names = getattr(model, 'feature_names_in_', None)
            if feature_names is not None:
                missing = [c for c in feature_names if c not in df.columns]
                for c in missing: df[c] = 0
                X_temp = df[feature_names].fillna(0)
                df['ml_score'] = model.predict_proba(X_temp)[:, 1]
                logger.info(f" >> ML Score 계산 완료. (Mean: {df['ml_score'].mean():.4f})")
            else:
                df['ml_score'] = 0
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}")
            df['ml_score'] = 0
    else:
        df['ml_score'] = 0

    # 4. Prepare Shared Memory (Temp File)
    # 16GB RAM 이슈 해결: 데이터를 임시 파일로 덤프 후 워커들은 mmap으로 읽음
    temp_dir = tempfile.mkdtemp()
    data_dump_path = os.path.join(temp_dir, 'opt_shared_data.joblib')
    
    logger.info(f" >> 공유 메모리용 데이터 덤프 중... ({data_dump_path})")
    # compress=0으로 해야 mmap 가능 (joblib default is 0/None for dump of numpy arrays inside objects?)
    # DataFrame dump via joblib might use pickle. For mmap, joblib handles numpy arrays separately.
    joblib.dump(df, data_dump_path) 
    # Clear original df from memory to free up space (since it's on disk now)
    del df 
    import gc; gc.collect()

    try:
        # 5. Setup Optuna Study
        study_name = f"Opt-Strategy-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(direction="maximize", study_name=study_name) # In-Memory Storage
        
        opt_cfg = cfg.get("optimization", {}).get(strategy_name, {})
        n_trials = opt_cfg.get("n_trials", 20)
        param_space = opt_cfg.get("param_space", {})
        dists = get_optuna_distributions(param_space)

        # Warm Start
        if warm_start:
            curr_params = cfg.get("strategies", {}).get(strategy_name, {})
            # Only keep keys in param_space
            warm_p = {k: v for k, v in curr_params.items() if k in param_space}
            if warm_p:
                study.enqueue_trial(warm_p)
                logger.info(f" >> Warm Start 예약: {warm_p}")

        logger.info(f"최적화 시작: {n_trials}회 시도 (병렬 {n_jobs} 프로세스)")
        
        # 6. Batch Execution Loop
        pbar = tqdm(total=n_trials, desc="Optimization Progress")
        
        # joblib Context
        with Parallel(n_jobs=n_jobs) as parallel:
            while len(study.trials) < n_trials:
                # Ask (Generate batch of params)
                batch_size = min(n_jobs, n_trials - len(study.trials))
                
                # Ask multiple trials
                trials_in_batch = []
                for _ in range(batch_size):
                    trials_in_batch.append(study.ask(dists))
                
                # Execute Batch in Parallel
                scores = parallel(
                    delayed(_worker_backtest)(
                        trial.params, 
                        data_dump_path, 
                        strategy_name, 
                        settings_path, 
                        start_date, 
                        end_date
                    ) for trial in trials_in_batch
                )
                
                # Tell (Report results)
                for trial, score in zip(trials_in_batch, scores):
                    study.tell(trial, score)
                    pbar.update(1)
                    
                    # Log (Console) - Only Show Passed or Every N
                    if score > -900: # Passed or valid
                         logger.info(f" >> [Finished] Trial {trial.number}: Score={score:.4f} | Params={trial.params}")
                    else:
                         logger.debug(f" >> [Rejected] Trial {trial.number}")

        pbar.close()
        
        # 7. Finalize & Save
        logger.info("="*60)
        best_trial = study.best_trial
        logger.info(f"최적화 완료. Best Score: {best_trial.value:.4f}")
        logger.info(f"Best Params: {best_trial.params}")
        
        # [NEW] Baseline(Warm Start) 결과 존재 시 비교
        improvement_msg = ""
        is_improved = True
        
        if warm_start and len(study.trials) > 0:
            # 첫 번째 Trial(Warm Start)의 결과를 Baseline으로 간주
            try:
                # Ask-and-Tell에서는 병렬 실행 순서가 섞일 수 있으나, 
                # enqueue된 trial은 보통 가장 먼저 ask() 되므로 trial number가 0일 확률이 높음.
                baseline_trial = study.trials[0]
                baseline_value = baseline_trial.value
                
                if baseline_value is not None:
                    logger.info(f"Baseline Score (Current): {baseline_value:.4f}")
                    diff = best_trial.value - baseline_value
                    if diff > 0.0001:
                        improvement_msg = f" (▲ {diff:.4f} 개선됨)"
                        is_improved = True
                    else:
                        improvement_msg = " (개선 없음)"
                        is_improved = False
            except Exception as e:
                logger.debug(f"Baseline 비교 실패: {e}")
                pass
                
        logger.info(f"성능 비교: {improvement_msg}")
        logger.info("="*60)
        
        # 결과 저장 여부
        if auto_approve:
            logger.info(" >> [Auto-Approve] 결과를 자동으로 저장합니다.")
            choice = 'y'
        else:
            if is_improved:
                prompt_msg = "최적 파라미터를 settings.yaml에 저장하시겠습니까? (y/n): "
            else:
                prompt_msg = "성능 개선이 없거나 미미합니다. 그래도 저장하시겠습니까? (y/n): "
                
            choice = get_user_input(prompt_msg)
             
        if choice.lower() == 'y':
            update_yaml(settings_path, "strategies", strategy_name, best_trial.params)
            
            # Sync to ml_params
            ml_updates = {}
            if 'target_r' in best_trial.params: ml_updates['target_surge_rate'] = float(best_trial.params['target_r'])
            if 'stop_r' in best_trial.params: ml_updates['stop_loss_rate'] = float(best_trial.params['stop_r'])
            if 'min_ml_score' in best_trial.params: ml_updates['classification_threshold'] = float(best_trial.params['min_ml_score'])
            
            if ml_updates:
                yaml = YAML()
                yaml.preserve_quotes = True
                with open(settings_path, 'r', encoding='utf-8') as f: config_data = yaml.load(f)
                if 'ml_params' in config_data:
                    for k, v in ml_updates.items(): config_data['ml_params'][k] = v
                    with open(settings_path, 'w', encoding='utf-8') as f: yaml.dump(config_data, f)
            logger.info("설정 저장 완료.")
            
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"최적화 중 치명적 오류: {e}", exc_info=True)
        
    finally:
        # Cleanup Temp Files
        # [Fix] Windows file lock issue: mmap objects need to be GC'd before file deletion
        import gc
        import time
        from joblib.externals.loky import get_reusable_executor
        
        # Force GC to release mmap handles
        df = None
        study = None
        gc.collect()
        
        # [Critical] Force Kill Worker Processes (reclaims ~3.5GB RAM)
        try:
            get_reusable_executor().shutdown(wait=True)
        except: pass
        
        if os.path.exists(temp_dir):
            try:
                # Slight delay to allow OS to release locks
                time.sleep(1.0) 
                shutil.rmtree(temp_dir)
                logger.debug("임시 파일 정리 완료.")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패 (Memory Leak 가능성): {e}")
                logger.warning(f"경로: {temp_dir} - 수동으로 삭제해 주세요.")
            
def optimize_strategy_headless(settings_path: str, strategy_name: str, 
                             start_date: str, end_date: str, 
                             n_trials: int = 20, n_jobs: int = 1, dataset_path: str = None):
     # WFO용 별도 함수 (유지하되 내부 로직 업데이트 필요 시 참조, 일단 기존 구조 유지하거나 통합 고려)
     # 현재 Task 범위는 'Strategy Optimization' 메뉴 개선이므로 일단 둠.
     # WFO도 이 로직을 쓰도록 차후 리팩토링 권장.
     pass