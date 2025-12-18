# addons/addon_single_period_wf.py
import pandas as pd
import os
import joblib
from tqdm import tqdm
from ruamel.yaml import YAML
import shutil
import psutil
from dask.distributed import Client
import optuna
from modules.utils_io import read_yaml, ensure_dir, to_date, update_yaml
from modules.utils_logger import logger
from modules import train, derive, optimization
from modules.backtest import run_backtest
from datetime import datetime

def run_single_period_wf(settings_path: str):
    """
    사용자가 지정한 단일 기간에 대해서만 Walk-Forward 스타일의
    (학습-최적화-테스트) 전체 과정을 수행합니다.
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< 단일 기간 Walk-Forward 테스터 시작 >>>")
    logger.info("="*80)

    # 1. 사용자로부터 테스트 시작일 입력받기
    try:
        test_start_str = input("테스트를 시작할 날짜를 입력하세요 (예: 2022-01-01): ")
        test_start_date = to_date(test_start_str)
        if test_start_date is None:
            raise ValueError
    except (ValueError, TypeError):
        logger.error("잘못된 날짜 형식입니다. 'YYYY-MM-DD' 형식으로 입력해주세요.")
        return

    # 2. 설정 파일 로드 및 기간 계산
    base_cfg = read_yaml(settings_path)
    strategy_name = "SpikeHunter"
    wf_cfg = base_cfg.get("walk_forward", {})
    train_months = wf_cfg.get("train_months", 24)
    test_months = wf_cfg.get("test_months", 6)

    test_start = test_start_date
    test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
    train_end = test_start - pd.Timedelta(days=1)
    train_start = train_end - pd.DateOffset(months=train_months) + pd.Timedelta(days=1)

    logger.info("\n" + "="*80)
    logger.info(f"  Train Period: {train_start.date()} ~ {train_end.date()}")
    logger.info(f"  Test Period:  {test_start.date()} ~ {test_end.date()}")
    logger.info("="*80)

    # 3. 임시 실행 환경 설정
    paths = base_cfg["paths"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(paths["backtest"], f"SingleWF-{strategy_name}-{timestamp}")
    ensure_dir(run_dir)
    logger.info(f"테스트 결과 저장 폴더: {run_dir}")
    
    temp_settings_path = os.path.join(run_dir, 'temp_settings.yaml')
    temp_models_dir = os.path.join(run_dir, 'temp_models')
    ensure_dir(temp_models_dir)

    num_cores = max(1, int(psutil.cpu_count(logical=True) * 0.5))
    with Client(n_workers=num_cores, threads_per_worker=1, silence_logs='ERROR') as client:
        
        # --- [PASS 1] 기간 한정 '순수' 임시 분류 모델 생성 ---
        logger.info("PASS 1: 순수 임시 분류 모델 생성을 시작합니다...")
        
        pass1_cfg = base_cfg.copy()
        pass1_dataset_path = os.path.join(run_dir, 'basic_ml_dataset')
        pass1_model_path = os.path.join(temp_models_dir, 'temp_clf.joblib')
        pass1_cfg['paths']['ml_dataset'] = pass1_dataset_path
        
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(temp_settings_path, 'w', encoding='utf-8') as f:
            yaml.dump(pass1_cfg, f)
        
        derive.run_derive_ml_dataset(settings_path=temp_settings_path, overwrite=True,
                                    start_date_override=train_start, end_date_override=train_end)
        
        temp_clf_dataset_path = os.path.join(pass1_dataset_path, "ml_classification_dataset.parquet")
        if os.path.exists(temp_clf_dataset_path):
            from modules.train import _train_and_evaluate_classification_model
            df_clf_temp = pd.read_parquet(temp_clf_dataset_path)
            feature_cols = [c for c in df_clf_temp.columns if c not in ['date','code','target']]
            X, y = df_clf_temp[feature_cols], df_clf_temp['target']
            temp_clf_params = pass1_cfg.get("ml_params", {}).get("lgbm_params_classification", {})
            temp_clf_result = _train_and_evaluate_classification_model(temp_clf_params, X, y, X, y)
            joblib.dump(temp_clf_result['model'], pass1_model_path)
            logger.info(f"PASS 1: 순수 임시 분류 모델이 '{pass1_model_path}'에 저장되었습니다.")
        else:
            logger.error("PASS 1: 임시 모델 학습을 위한 데이터셋 생성에 실패했습니다.")
            return

        # --- [PASS 2] 최종 모델 생성 및 전략 최적화/테스트 ---
        logger.info("PASS 2: 최종 모델 생성 및 테스트를 시작합니다...")

        pass2_cfg = base_cfg.copy()
        pass2_dataset_path = os.path.join(run_dir, 'advanced_ml_dataset')
        pass2_cfg['paths']['models'] = temp_models_dir
        pass2_cfg['paths']['ml_dataset'] = pass2_dataset_path
        
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(temp_settings_path, 'w', encoding='utf-8') as f:
            yaml.dump(pass2_cfg, f)

        derive.run_derive_advanced_ml_dataset(settings_path=temp_settings_path, overwrite=True,
                                              start_date_override=train_start, end_date_override=train_end,
                                              clf_model_path_override=pass1_model_path)
        
        train.run_train_pipeline(settings_path=temp_settings_path, dataset_dir_override=pass2_dataset_path)
        
        storage = optuna.storages.InMemoryStorage()
        for regime in ["R1_BullStable", "R2_BullVolatile", "R3_BearStable"]:
            logger.info(f"Optimizing for {regime} in training period...")
            best_result = optimization.run_optimization_pipeline(
                temp_settings_path, strategy_name, start=str(train_start.date()), end=str(train_end.date()),
                is_walk_forward=True, use_ml_target=True,
                dask_client=client, storage=storage, study_name_prefix=f"SingleWF_{regime}_",
                regime_to_optimize=regime
            )
            if best_result and best_result.get('params'):
                strategy_key = f"{strategy_name}_{regime}"
                current_params = read_yaml(temp_settings_path)["strategies"][strategy_key]
                current_params.update(best_result['params'])
                update_yaml(temp_settings_path, "strategies", strategy_key, current_params)
                logger.info(f" >> Best Params for {regime} updated.")
            else:
                logger.info(f" >> No new optimal params found for {regime}.")
        
        logger.info("Running backtest on test period with optimized parameters...")
        run_backtest(
            settings_path=temp_settings_path, run_dir=run_dir, strategy_name=strategy_name,
            start=str(test_start.date()), end=str(test_end.date()),
            quiet=False, use_ml_target=True
        )
        
        shutil.rmtree(pass1_dataset_path, ignore_errors=True)
        shutil.rmtree(pass2_dataset_path, ignore_errors=True)

    logger.info("\n단일 기간 Walk-Forward 테스트가 완료되었습니다.")
    logger.info(f"상세 결과는 '{run_dir}' 폴더를 확인해주세요.")
    logger.info("="*80)