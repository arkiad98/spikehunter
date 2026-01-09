# modules/train.py (v5.9 - CatBoost & Ensemble Support)
"""
[SpikeHunter v5.9] 모델 학습 및 ML 최적화 모듈
- 기능: LightGBM, XGBoost, CatBoost 3종 앙상블 지원
- 업데이트: CatBoost 추가 및 앙상블 로직 개선
- Fix: Sequential Folds + Multi-threading for stability
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
import joblib
from joblib import Parallel, delayed
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                             average_precision_score, balanced_accuracy_score)

from .utils_io import read_yaml, optimize_memory_usage, get_user_input, update_yaml
from .utils_logger import logger
from .utils_system import get_optimal_cpu_count # [New] CPU Utility
from .derive import _get_feature_cols

# 앙상블 클래스 임포트
try:
    from .ensemble import SpikeHunterEnsemble
except ImportError:
    SpikeHunterEnsemble = None

# -----------------------------------------------------------------------------
# 1. Helper Functions & Training Logic
# -----------------------------------------------------------------------------
def calculate_top_n_precision(y_true, y_scores, n=5):
    """
    Calculate Precision of Top-N items per day.
    For optimization speed, we use top-k percentile as proxy for top-n strategy.
    """
    if len(y_true) == 0: return 0.0
    
    # Simply take top 5% as proxy for "Trade Candidates"
    k = max(int(len(y_true) * 0.05), 5) 
    
    try:
        top_k_indices = np.argsort(y_scores)[-k:]
        y_true_top_k = y_true.iloc[top_k_indices] if hasattr(y_true, 'iloc') else y_true[top_k_indices]
        return np.mean(y_true_top_k)
    except:
        return 0.0

def analyze_importance(study):
    """
    Analyze and print hyperparameter importances.
    """
    try:
        logger.info("\n" + "="*40)
        logger.info("   [Hyperparameter Importance Analysis]")
        logger.info("="*40)
        
        importance = optuna.importance.get_param_importances(study)
        
        for i, (param, score) in enumerate(importance.items()):
            logger.info(f" {i+1}. {param:<20}: {score*100:.1f}%")
            
        logger.info("="*40 + "\n")
            
    except Exception as e:
        logger.warning(f"Importance analysis failed: {e}")

def logging_callback(study, frozen_trial):
    """
    Optuna Trial 완료 시 호출되는 콜백.
    logger를 사용하여 Trial 결과를 JSON 및 콘솔에 기록합니다.
    """
    try:
        # Best Trial 여부 확인
        is_best = (study.best_trial.number == frozen_trial.number)
        best_mark = "[BEST]" if is_best else ""
        
        logger.info(
            f"[Trial {frozen_trial.number}] {best_mark} Value: {frozen_trial.value:.6f} | "
            f"Params: {frozen_trial.params}"
        )
    except Exception as e:
        logger.warning(f"Logging callback failed: {e}")

def _get_core_features_from_registry(registry_path: str) -> list:
    if not os.path.exists(registry_path): return []
    try:
        reg = read_yaml(registry_path)
        return [f['name'] for f in reg.get('features', []) if f.get('status') == 'core']
    except: return []

def _train_single_fold(model_type, params, X_train, y_train, X_test, y_test, threshold=0.5):
    """단일 모델, 단일 Fold 학습 함수"""
    # Fix: Filter out param_space keys that might be in the config
    p = {k: v for k, v in params.items() if not k.startswith('param_space_')}

    if model_type == 'lgbm':
        p['verbose'] = -1
        model = lgb.LGBMClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    elif model_type == 'xgb':
        if 'n_jobs' not in p: p['n_jobs'] = 1
        model = xgb.XGBClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif model_type == 'cat':
        if 'verbose' not in p: p['verbose'] = 0
        p['allow_writing_files'] = False
        # p['thread_count'] = 1 # Removed to allow multi-threading
        model = CatBoostClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    scores = {
        'average_precision': average_precision_score(y_test, probs),
        'auc': roc_auc_score(y_test, probs),
        'f1': f1_score(y_test, preds, zero_division=0),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0)
    }
    return model, scores, probs

def _run_classification_training(cfg: dict, return_results_only: bool = False, benchmark_mode: bool = False):
    """분류 모델 학습 메인 로직 (앙상블 지원)"""
    if not return_results_only:
        print("\n" + "="*60)
        print("   [Phase 3] SpikeHunter v5.9 앙상블 모델 학습")
        print("="*60)
    
    paths = cfg.get("paths", {})
    ml_params_cfg = cfg.get("ml_params", {})
    
    # 1. 데이터 로드 (Fix: WF 호환성 - dataset_v4.parquet 존재 시 최우선 로드)
    dataset_dir = paths.get("ml_dataset", "data/proc/ml_dataset")
    
    # [Fix] Walk-Forward 등에서 생성한 학습 전용 데이터셋이 있으면 우선 사용
    # run_pipeline.py에서 'features' 경로에 dataset_v4.parquet를 생성함
    train_split_path = os.path.join(paths.get("features", ""), "dataset_v4.parquet")
    
    if os.path.exists(train_split_path):
        dataset_path = train_split_path
        if not return_results_only:
             print(f" >> [Data] 학습 전용 데이터셋 로드 (WF Mode): {dataset_path}")
    else:
        dataset_path = os.path.join(dataset_dir, "ml_classification_dataset.parquet")
        if not return_results_only:
             print(f" >> [Data] 전체 데이터셋 로드: {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋이 없습니다: {dataset_path}")
        return

    df = pd.read_parquet(dataset_path)
    df = optimize_memory_usage(df)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. 기간 필터링 (Start Offset ~ End Offset)
    train_months = ml_params_cfg.get("classification_train_months", 36)
    end_offset = ml_params_cfg.get("classification_train_end_offset", 0)
    
    if train_months > 0:
        # 기준 종료일: 전체 데이터 끝 - Offset
        effective_end_date = df['date'].max() - pd.DateOffset(months=end_offset)
        # 기준 시작일: 기준 종료일 - Train Months
        effective_start_date = effective_end_date - pd.DateOffset(months=train_months)
        
        if not return_results_only:
            print(f" >> 학습 데이터 구간: {effective_start_date.date()} ~ {effective_end_date.date()} (Train: {train_months}m, Gap: {end_offset}m)")
            
        df = df[(df['date'] >= effective_start_date) & (df['date'] <= effective_end_date)].copy()

    # 3. 피처 선택
    target_col = 'label_class'
    registry_path = "config/feature_registry.yaml"
    core_features = _get_core_features_from_registry(registry_path)
    available_cols = set(df.columns)
    feature_cols = [f for f in core_features if f in available_cols]
    if not feature_cols: feature_cols = _get_feature_cols(df.columns)
    
    if not return_results_only:
        print(f" >> 학습 피처 수: {len(feature_cols)}개")
        cnt = df[target_col].value_counts()
        print(f" >> 타겟 분포: 성공(1) {cnt.get(1,0)}개, 실패(0) {cnt.get(0,0)}개")

    X = df[feature_cols]
    y = df[target_col]

    # 4. 학습 파라미터
    lgbm_params = ml_params_cfg.get("lgbm_params_classification", {})
    xgb_params = ml_params_cfg.get("xgb_params_classification")
    cat_params = ml_params_cfg.get("cat_params_classification")
    threshold = ml_params_cfg.get("classification_threshold", 0.5)
    
    # 모델 선택 로직 (Settings > active_model 우선)
    active_model = ml_params_cfg.get("active_model", "lgbm").lower()
    
    if benchmark_mode:
        # 벤치마크: 가능한 모든 모델 활성화
        use_lgbm = True
        use_xgb = (xgb_params is not None)
        use_cat = (cat_params is not None) and (CatBoostClassifier is not None)
    else:
        # 프로덕션: 선택된 단일 모델만 활성화
        use_lgbm = (active_model == 'lgbm')
        use_xgb = (active_model == 'xgb') and (xgb_params is not None)
        use_cat = (active_model == 'cat') and (cat_params is not None) and (CatBoostClassifier is not None)
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_indices = list(tscv.split(X))
    
    if not return_results_only:
        print(f"\n >> {n_splits}-Fold 시계열 교차검증 시작... (Sequential Folds)")
        models_str = []
        if use_lgbm: models_str.append("LGBM")
        if use_xgb: models_str.append("XGB")
        if use_cat: models_str.append("CatBoost")
        print(f"    * 앙상블 모델: {', '.join(models_str)}")

    # 병렬 학습 (Parallel Folds)
    # n_jobs=-1 uses all available cores. Assuming model n_jobs is set appropriately.
    lgbm_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_train_single_fold)('lgbm', lgbm_params, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], threshold)
        for tr, te in fold_indices
    )
    
    xgb_results = []
    if use_xgb:
        xgb_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(_train_single_fold)('xgb', xgb_params, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], threshold)
            for tr, te in fold_indices
        )

    cat_results = []
    if use_cat:
        cat_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(_train_single_fold)('cat', cat_params, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], threshold)
            for tr, te in fold_indices
        )

    # 결과 집계
    ap_scores = []
    for i in range(n_splits):
        _, test_idx = fold_indices[i]
        y_test_fold = y.iloc[test_idx]
        
        preds_list = []
        p_lgbm = lgbm_results[i][2]
        preds_list.append(p_lgbm)
        
        desc_parts = [f"L={average_precision_score(y_test_fold, p_lgbm):.4f}"]
        
        if use_xgb:
            p_xgb = xgb_results[i][2]
            preds_list.append(p_xgb)
            desc_parts.append(f"X={average_precision_score(y_test_fold, p_xgb):.4f}")
            
        if use_cat:
            p_cat = cat_results[i][2]
            preds_list.append(p_cat)
            desc_parts.append(f"C={average_precision_score(y_test_fold, p_cat):.4f}")
            
        # 평균 앙상블
        p_ens = np.mean(preds_list, axis=0)
        score = average_precision_score(y_test_fold, p_ens)
        
        desc = f"Ensemble AP: {score:.4f} ({', '.join(desc_parts)})"
        ap_scores.append(score)
        
        if not return_results_only:
            print(f" [Fold {i+1}] {desc}")

    if not return_results_only:
        print(f"\n >> 전체 평균 AP: {np.mean(ap_scores):.4f}")

    if return_results_only:
        return {}, np.mean(ap_scores)

    # 5. 전체 학습 및 저장
    print("\n >> 전체 데이터로 최종 모델 학습 중...")
    final_models_list = []
    
    # LGBM
    lgbm_p = {k: v for k, v in lgbm_params.items() if not k.startswith('param_space_')}
    model_lgbm_final = lgb.LGBMClassifier(**lgbm_p)
    model_lgbm_final.fit(X, y)
    final_models_list.append(model_lgbm_final)
    
    # XGB
    if use_xgb:
        xgb_p = {k: v for k, v in xgb_params.items() if not k.startswith('param_space_')}
        model_xgb_final = xgb.XGBClassifier(**xgb_p)
        model_xgb_final.fit(X, y)
        final_models_list.append(model_xgb_final)
        
    # CatBoost
    if use_cat:
        cat_p = {k: v for k, v in cat_params.items() if not k.startswith('param_space_')}
        cat_p['allow_writing_files'] = False
        # Final training runs sequentially, so we can use full threads
        model_cat_final = CatBoostClassifier(**cat_p)
        model_cat_final.fit(X, y, verbose=False)
        final_models_list.append(model_cat_final)
        
    save_path = os.path.join(cfg['paths']['models'], "lgbm_model.joblib")
    if len(final_models_list) > 1:
        ensemble_model = SpikeHunterEnsemble(final_models_list)
        joblib.dump(ensemble_model, save_path)
        print(f" >> 앙상블 모델 저장 완료: {save_path}")
    else:
        joblib.dump(model_lgbm_final, save_path)
        print(f" >> LightGBM 모델 저장 완료: {save_path}")

# -----------------------------------------------------------------------------
# 2. Optuna Optimization
# -----------------------------------------------------------------------------
class Objective:
    def __init__(self, model_type: str, param_space: dict, optimize_on: str, n_jobs: int):
        self.model_type = model_type
        self.param_space = param_space
        self.n_jobs = n_jobs
        
        cfg = read_yaml("config/settings.yaml")
        
        # 데이터 로드 (동일 로직)
        # 데이터 로드 (Fix: WF/Optim 호환성)
        dataset_path = os.path.join(cfg["paths"]["ml_dataset"], "ml_classification_dataset.parquet")
        
        # [Fix] Check for split dataset in optimization as well
        train_split_path = os.path.join(cfg["paths"]["features"], "dataset_v4.parquet")
        if os.path.exists(train_split_path):
             dataset_path = train_split_path

        df = pd.read_parquet(dataset_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        registry_path = "config/feature_registry.yaml"
        core_features = _get_core_features_from_registry(registry_path)
        available_cols = set(df.columns)
        self.feature_cols = [f for f in core_features if f in available_cols]
        if not self.feature_cols: self.feature_cols = _get_feature_cols(df.columns)
        
        ml_params = cfg.get("ml_params", {})
        train_months = ml_params.get("classification_train_months", 24)
        end_offset = ml_params.get("classification_train_end_offset", 0)
        
        # Calculate Effective Data Range
        effective_end_date = df['date'].max() - pd.DateOffset(months=end_offset)
        effective_start_date = effective_end_date - pd.DateOffset(months=train_months)
        
        df = df[(df['date'] >= effective_start_date) & (df['date'] <= effective_end_date)]
        
        logger.info(f" >> 최적화 데이터 구간: {effective_start_date.date()} ~ {effective_end_date.date()} (Train: {train_months}m, Gap: {end_offset}m)")
        logger.info(f" >> 학습 데이터 수: {len(df)} samples")
        
        self.X = df[self.feature_cols]
        self.y = df['label_class']

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = {
            'n_jobs': self.n_jobs,
            'random_state': 42
        }
        
        # 파라미터 매핑
        for name, config in self.param_space.items():
            if config['type'] == 'int':
                params[name] = trial.suggest_int(name, config['low'], config['high'])
            elif config['type'] == 'float':
                params[name] = trial.suggest_float(name, config['low'], config['high'], log=config.get('log', False))
            elif config['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, config['choices'])
        
        # 검증 (Hold-out)
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
        
        if self.model_type == 'lgbm':
            params.update({'objective': 'binary', 'metric': 'average_precision', 'verbosity': -1})
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        elif self.model_type == 'xgb':
            params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist'})
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif self.model_type == 'cat':
            # Fix: Handle n_jobs for CatBoost (rename to thread_count)
            if 'n_jobs' in params:
                params['thread_count'] = params.pop('n_jobs')
            params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0, 'allow_writing_files': False})
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # [New] Auxiliary Metric: Top-N Precision (Proxy)
        top_prec = calculate_top_n_precision(y_val, y_pred_proba)
        trial.set_user_attr("top_n_precision", top_prec)
        
        return average_precision_score(y_val, y_pred_proba)

def run_ml_optimization_pipeline(settings_path: str, model_type: str = None, n_trials: int = 20, n_jobs: int = 1, save: bool = False):
    """ML 모델 최적화 (LGBM / XGB / CatBoost 선택 가능) - 통합 버전"""
    logger.info("="*60)
    logger.info("      <<< ML 모델 하이퍼파라미터 최적화 >>>")
    logger.info("="*60)
    
    if model_type:
        choice = {'lgbm': '1', 'xgb': '2', 'cat': '3'}.get(model_type, '1')
    else:
        print("최적화할 모델을 선택하세요:")
        print("1. LightGBM (추천)")
        print("2. XGBoost")
        print("3. CatBoost")
        choice = get_user_input("선택 (1/2/3): ")
    
    if choice == '1':
        target_model = 'lgbm'
        target_key = 'lgbm_params_classification'
        space_key = 'param_space_lgbm'
    elif choice == '2':
        target_model = 'xgb'
        target_key = 'xgb_params_classification'
        space_key = 'param_space_xgb'
    elif choice == '3':
        target_model = 'cat'
        target_key = 'cat_params_classification'
        space_key = 'param_space_cat'
    else:
        print("잘못된 선택입니다.")
        return
    
    # [New] Trial Mode Selection (if n_trials default)
    if n_trials == 20: # Assuming 20 is the default passed from menu
        print("\n[Optimization Mode]")
        print("1. Fast Scan (20 trials)")
        print("2. Deep Search (100 trials)")
        print("3. Custom Input")
        mode = get_user_input("선택 (1/2/3): ")
        if mode == '2': n_trials = 100
        elif mode == '3':
             try: n_trials = int(get_user_input("Trials 입력: "))
             except: n_trials = 20
    
    # 병렬 설정
    if n_jobs is None or n_jobs == 1:
        # Ask user but default to 1 (Serial Optuna) to allow Model parallelism
        n_jobs_input = get_user_input("병렬 작업 수(Optuna Trials) (엔터=1 [권장/모델병렬], -1=전체): ")
        try:
            optuna_n_jobs = int(n_jobs_input) if n_jobs_input.strip() else 1
        except: optuna_n_jobs = 1
    else:
        optuna_n_jobs = n_jobs

    # Model Parallelism Calculation
    if optuna_n_jobs == 1:
        # Serial Trials -> Maximize Model Parallelism (75% of Cores)
        model_n_jobs = get_optimal_cpu_count(0.75)
        logger.info(f" >> [CPU] Serial Optimization Mode: Model n_jobs set to {model_n_jobs} (75% Cores)")
    else:
        # Parallel Trials -> Minimize Model Parallelism to avoid congestion
        model_n_jobs = 1
        logger.info(f" >> [CPU] Parallel Optimization Mode: Model n_jobs set to 1")
    
    cfg = read_yaml(settings_path)
    
    # 파라미터 공간 로드 (ml_params 내부에서 탐색)
    ml_params = cfg.get("ml_params", {})
    target_params = ml_params.get(target_key, {})
    
    if space_key in target_params:
        param_space = target_params[space_key]
    else:
        print(f"설정 파일({target_key})에 {space_key}가 없습니다.")
        return
        
    logger.info(f" >> 모델: {target_model.upper()}, Trials: {n_trials}, Optuna Jobs: {optuna_n_jobs}")
    
    # [Mod] Optuna 기본 로깅 억제 (커스텀 콜백 사용)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    objective = Objective(target_model, param_space, "AP", model_n_jobs)
    
    try:
        # Warm Start Logic
        warm_params = {}
        for k, v in target_params.items():
            if k in param_space and not k.startswith('param_space'):
                 space = param_space[k]
                 # Clamp value to be within [low, high] for numerical params
                 if space.get('type') in ['int', 'float']:
                     low = space.get('low', float('-inf'))
                     high = space.get('high', float('inf'))
                     
                     if v < low: v = low
                     if v > high: v = high
                 
                 warm_params[k] = v
                 
        if warm_params:
            logger.info(f" >> Warm Start 설정됨: {len(warm_params)} params")
            study.enqueue_trial(warm_params)

        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=optuna_n_jobs, 
            show_progress_bar=True,
            callbacks=[logging_callback] # [Mod] Callback 추가
        )
        
        logger.info("\n" + "="*60)
        logger.info(f" [Optimization Result]")
        logger.info(f" Best AP: {study.best_value:.4f}")
        
        best_trial = study.best_trial
        top_n_prec = best_trial.user_attrs.get("top_n_precision", 0.0)
        logger.info(f" Top-N Precision (Proxy): {top_n_prec:.4f}")
        logger.info(f" Best Params: {study.best_params}")
        logger.info("="*60)

        # [New] Importance Analysis
        analyze_importance(study)
        
        if save:
            do_save = 'y'
        else:
            do_save = get_user_input("설정 파일에 저장하시겠습니까? (y/n): ")
            
        if do_save.lower() == 'y':
            # 기존 파라미터에 업데이트
            base_params = cfg['ml_params'].get(target_key, {})
            base_params.update(study.best_params)
            update_yaml(settings_path, "ml_params", target_key, base_params)
            logger.info(" >> 저장 완료. '모델 학습' 메뉴를 통해 재학습해주세요.")
            
    except Exception as e:
        logger.error(f"최적화 중 오류: {e}")

def run_train_pipeline(settings_path: str):
    cfg = read_yaml(settings_path)
    _run_classification_training(cfg)
def run_benchmark_mode(settings_path: str):
    """LGBM, XGB, CatBoost 3종 모델 성능 비교 벤치마크"""
    print("\n" + "="*60)
    print("      <<< ML Model Benchmark Mode >>>")
    print("      LightGBM vs XGBoost vs CatBoost")
    print("="*60)
    
    cfg = read_yaml(settings_path)
    ml_params = cfg.get("ml_params", {})
    
    # 벤치마크용 기본 파라미터 주입 (설정에 없으면 기본값)
    if "xgb_params_classification" not in ml_params:
        logger.info("XGBoost 파라미터가 없어 기본값을 로드합니다.")
        ml_params["xgb_params_classification"] = {
            'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 6, 
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': 1, 'tree_method': 'hist'
        }
        
    if "cat_params_classification" not in ml_params:
        logger.info("CatBoost 파라미터가 없어 기본값을 로드합니다.")
        ml_params["cat_params_classification"] = {
            'iterations': 1000, 'learning_rate': 0.01, 'depth': 6, 
            'verbose': 0, 'allow_writing_files': False, 'thread_count': 1
        }
    
    # 중요: train.py 내부 로직이 params 존재 여부를 체크하므로, cfg를 갱신
    cfg["ml_params"] = ml_params
    
    # _run_classification_training 호출
    # (주의: 내부에서 import된 train.py 함수가 아니라 이 파일의 함수를 써야 함)
    _run_classification_training(cfg, benchmark_mode=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmark") # [ADD]
    parser.add_argument("--model", type=str, default="lgbm", help="Model type (lgbm, xgb, cat)")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--jobs", type=int, default=1, help="Number of jobs (-1 for all)")
    parser.add_argument("--save", action="store_true", help="Auto save results")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_mode("config/settings.yaml")
    elif args.optimize:
        run_ml_optimization_pipeline("config/settings.yaml", model_type=args.model, n_trials=args.trials, n_jobs=args.jobs, save=args.save)
    else:
        run_train_pipeline("config/settings.yaml")