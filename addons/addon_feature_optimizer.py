# addons/addon_feature_optimizer.py (최종 간소화 버전)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import gc
import time
from datetime import datetime
from typing import List, Optional
import logging

from sklearn.model_selection import TimeSeriesSplit
from imblearn.combine import SMOTEENN
from sklearn.metrics import average_precision_score

from modules.utils_io import read_yaml, ensure_dir
from modules.derive import _get_feature_cols


class FeatureObjective:
    def __init__(self, core_features: list, candidate_features: list, X: pd.DataFrame, y: pd.Series, lgbm_params: dict):
        self.core_features = core_features
        self.candidate_features = candidate_features
        self.X = X
        self.y = y
        self.lgbm_params = lgbm_params
        self.lgbm_params['n_jobs'] = -1
        self.n_splits = 5
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.indices = list(self.tscv.split(X))

    def __call__(self, trial: optuna.trial.Trial) -> float:
        logger = logging.getLogger("QuantPipeline")
        trial_start_time = time.time()
        
        # 두 개의 하이퍼파라미터를 동시에 탐색합니다.
        num_additional_features = trial.suggest_int('num_additional_features', 0, len(self.candidate_features))
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 2, 50) # 비용 민감도 탐색 추가

        logger.info(f"\n>>>>> Trial #{trial.number} 시작 | 추가 피처: {num_additional_features}개 | scale_pos_weight: {scale_pos_weight} <<<<<")
        
        try:
            features_to_use = self.core_features + self.candidate_features[:num_additional_features]
            added_features = self.candidate_features[:num_additional_features]
            logger.info(f"  - Core({len(self.core_features)}) + Added({num_additional_features}) = Total({len(features_to_use)})")
            if added_features:
                logger.info(f"  - Added Features: {added_features}")

            X_subset = self.X[features_to_use]
            scores = []
            
            # CV 로직에서 SMOTEENN을 제거하고, 비용 민감 학습을 적용합니다.
            for i, (train_index, test_index) in enumerate(self.indices):
                X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                # lgbm_params에 trial에서 제안된 scale_pos_weight를 설정합니다.
                params_for_trial = self.lgbm_params.copy()
                # [Fix] Filter param_space
                params_for_trial = {k: v for k, v in params_for_trial.items() if not k.startswith('param_space_')}
                params_for_trial['scale_pos_weight'] = scale_pos_weight
                
                # 원본 데이터로 바로 학습합니다.
                model = lgb.LGBMClassifier(**params_for_trial)
                model.fit(X_train, y_train) 
                
                preds = model.predict_proba(X_test)[:, 1]
                scores.append(average_precision_score(y_test, preds))
                del X_train, X_test, y_train, y_test, model, preds; gc.collect()
                
            final_score = np.mean(scores)
            trial_duration = time.time() - trial_start_time
            logger.info(f">>>>> Trial #{trial.number} 성공. 최종 점수(AP): {final_score:.4f} (소요 시간: {trial_duration:.2f}초) <<<<<")
            return final_score

        except Exception as e:
            logger.error(f">>>>> Trial #{trial.number} 에러 발생! 에러: {e} <<<<<", exc_info=True)
            raise optuna.exceptions.TrialPruned()

def run_feature_combination_optimization(
    settings_path: str,
    candidate_features_override: Optional[List[str]] = None
):
    from addons.addon_feature_engineering_suite import load_feature_registry
    from modules.utils_logger import logger
    
    logger.info("="*60 + "\n      <<< (지능형) 최적 피처 조합 탐색 시작 >>>\n" + "="*60)
    
    # ... (피처 및 데이터 로딩 로직은 동일) ...
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    core_features, registry_candidates, _ = load_feature_registry()
    df = pd.read_parquet(os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet"))
    df = df.sort_values('date').reset_index(drop=True)
    X, y = df, df['label_class']
    lgbm_params = cfg.get("ml_params", {}).get("lgbm_params_classification", {})

    if candidate_features_override:
        candidate_features = candidate_features_override
    else:
        candidate_features = [f for f in registry_candidates if f in df.columns and f not in core_features]
    logger.info(f"탐색 대상 후보 피처: {len(candidate_features)}개")

    study = optuna.create_study(direction='maximize')
    objective = FeatureObjective(core_features, candidate_features, X, y, lgbm_params)

    n_trials = 100
    logger.info(f"Optuna 최적화를 '단일 코어'로 시작합니다 (총 {n_trials} Trials)...")
    
    # [핵심] n_jobs=1로 고정하여 순차 실행 및 로그 안정성 확보
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # ... (최종 결과 보고 로직은 동일) ...
    from modules.utils_logger import setup_global_logger
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_global_logger(run_timestamp)
    logger.info("\n" + "="*60 + "\n          <<< 지능형 피처 조합 최적화 최종 결과 >>>\n" + "="*60)
    best_trial = study.best_trial
    best_num_additional = best_trial.params['num_additional_features']
    logger.info(f"최고 점수(AP): {best_trial.value:.4f}")
    logger.info(f"최적 피처 개수: {len(core_features) + best_num_additional}개 (핵심 {len(core_features)}개 + 추가 {best_num_additional}개)")
    logger.info(f"추천되는 추가 피처: {candidate_features[:best_num_additional]}")

