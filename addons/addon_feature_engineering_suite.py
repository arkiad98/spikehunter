# addons/addon_feature_engineering_suite.py

import os
import pandas as pd
import numpy as np
from ruamel.yaml import YAML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from imblearn.combine import SMOTEENN
import lightgbm as lgb
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import gc
from modules.utils_io import optimize_memory_usage # 메모리 최적화 함수 임포트
#from joblib import Parallel, delayed # <-- joblib 임포트 추가


# --- Constants and Configurations ---
FEATURE_REGISTRY_PATH = "config/feature_registry.yaml"
ML_DATASET_PATH = "data\proc\features\dataset_v4.parquet"
ANALYSIS_OUTPUT_DIR = "data/proc/backtest/feature_analysis"
N_SPLITS_CV = 5

# --- Part 1: Feature Registry Management ---

def load_feature_registry() -> Tuple[List[str], List[str], Dict[str, Dict]]:
    """
    feature_registry.yaml 파일을 로드하여 status별 피처 목록과
    전체 피처의 메타데이터를 반환합니다.
    """
    yaml = YAML()
    try:
        with open(FEATURE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = yaml.load(f)
    except FileNotFoundError:
        print(f"오류: {FEATURE_REGISTRY_PATH} 파일을 찾을 수 없습니다.")
        return [], [], {}
        
    if not registry or 'features' not in registry:
        return [], [], {}

    all_features_meta = {feat['name']: feat for feat in registry['features']}
    
    core_features = [f['name'] for f in registry['features'] if f['status'] == 'core']
    candidate_features = [f['name'] for f in registry['features'] if f['status'] in ['candidate', 'experimental']]
    
    print(f"피처 레지스트리 로드 완료. Core: {len(core_features)}개, Candidates: {len(candidate_features)}개")
    return core_features, candidate_features, all_features_meta

def add_new_feature_to_registry(feature_name: str, category: str, description: str, status: str = "experimental"):
    """
    신규 피처 정보를 feature_registry.yaml 파일에 안전하게 추가합니다.
    이미 존재하는 피처인 경우, 중복 추가를 방지합니다.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=6, offset=4)
    try:
        with open(FEATURE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = yaml.load(f)
    except FileNotFoundError:
        print(f"오류: {FEATURE_REGISTRY_PATH} 파일을 찾을 수 없습니다.")
        return

    for feature in registry.get('features', []):
        if feature.get('name') == feature_name:
            print(f"경고: '{feature_name}' 피처는 이미 레지스트리에 존재합니다. 추가하지 않습니다.")
            return

    new_feature = {
        'name': feature_name,
        'status': status,
        'category': category,
        'description': description,
        'last_validated_on': None,
        'latest_lift': None
    }
    registry['features'].append(new_feature)

    with open(FEATURE_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(registry, f)
    print(f"성공: 신규 피처 '{feature_name}'가 {FEATURE_REGISTRY_PATH}에 추가되었습니다.")

# --- Part 2: Individual Feature Analyzer ---

# [수정] 함수 시그니처 변경

# 이 함수는 병렬로 실행될 작업 단위이며, 로그를 리스트에 저장합니다.
def _process_single_fold(train_index, test_index, X, y, features_to_use, lgbm_params, fold_num):
    """
    단일 CV Fold를 처리하고, 결과와 함께 로그 메시지 리스트를 반환합니다.
    """
    import pandas as pd
    import lightgbm as lgb
    from imblearn.combine import SMOTEENN
    import gc
    import time
    from sklearn.metrics import average_precision_score

    log_messages = [] # 로그를 저장할 리스트
    fold_start_time = time.time()
    log_prefix = f"[Fold {fold_num}/{N_SPLITS_CV}]"
    log_messages.append(f"  {log_prefix} 시작...")

    X_subset = X[features_to_use]
    X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    smote_start_time = time.time()
    log_messages.append(f"    - SMOTEENN 적용 시작... (원본 shape: {X_train.shape})")
    sampler = SMOTEENN(random_state=42, n_jobs=-1)
    X_train_sm, y_train_sm = sampler.fit_resample(X_train, y_train)
    smote_duration = time.time() - smote_start_time
    log_messages.append(f"    - SMOTEENN 적용 완료. (재조정 shape: {X_train_sm.shape}, 소요 시간: {smote_duration:.2f}초)")

    fit_start_time = time.time()
    log_messages.append("    - 모델 학습(fit) 시작...")
    
    # [Fix] Filter param_space
    clean_params = {k: v for k, v in lgbm_params.items() if not k.startswith('param_space_')}
    model = lgb.LGBMClassifier(**clean_params)
    
    model.fit(X_train_sm, y_train_sm, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
    fit_duration = time.time() - fit_start_time
    log_messages.append(f"    - 모델 학습(fit) 완료. (소요 시간: {fit_duration:.2f}초)")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    result_tuple = (y_test, pd.Series(y_pred_proba, index=y_test.index))
    
    fold_duration = time.time() - fold_start_time
    score_fold = average_precision_score(y_test, y_pred_proba)
    log_messages.append(f"  {log_prefix} 종료. Avg Precision: {score_fold:.4f} (총 소요 시간: {fold_duration:.2f}초)")

    del X_train, X_test, y_train, X_train_sm, y_train_sm, model, sampler, y_pred_proba
    gc.collect()
    
    # 결과와 함께 수집된 로그 메시지 리스트를 반환
    return result_tuple[0], result_tuple[1], log_messages


# 이 함수는 병렬 처리를 지휘하고, 결과로 받은 로그를 순서대로 출력합니다.
def _run_cv_for_feature_set(
    X: pd.DataFrame, 
    y: pd.Series, 
    features_to_use: List[str], 
    lgbm_params: dict,
    trial_number: Optional[int] = None
) -> float:
    from modules.utils_logger import logger
    import psutil
    from joblib import Parallel, delayed
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import average_precision_score
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    
    # 안정성을 위해 n_jobs=1로 우선 설정 (단일 코어)
    # 추후 메모리 사용량 확인 후 2, 4 등으로 늘려 테스트 가능
    n_jobs = -1
    logger.info(f"  -> {N_SPLITS_CV}개 Fold 교차검증을 {n_jobs}개 코어로 처리합니다...")
    
    X_subset = X[features_to_use]
    
    tasks = []
    for i, (train_index, test_index) in enumerate(tscv.split(X_subset)):
        tasks.append(delayed(_process_single_fold)(
            train_index, test_index, X, y, features_to_use, lgbm_params, fold_num=i+1
        ))

    # 병렬(또는 순차) 실행. 결과는 (y_test, y_pred, logs) 튜플의 리스트
    results = Parallel(n_jobs=n_jobs)(tasks)

    all_y_test = []
    all_y_pred_proba = []
    
    # 모든 작업이 끝난 후, 결과를 순회하며 로그를 순차적으로 출력
    for res in sorted(results, key=lambda x: x[2][0]): # Fold 번호 순으로 정렬
        y_test_fold, y_pred_proba_fold, log_messages = res
        for msg in log_messages:
            logger.info(msg) # 저장된 로그 메시지를 순서대로 출력
        all_y_test.append(y_test_fold)
        all_y_pred_proba.append(y_pred_proba_fold)

    y_test_all = pd.concat(all_y_test)
    y_pred_proba_all = pd.concat(all_y_pred_proba)
    
    final_score = average_precision_score(y_test_all, y_pred_proba_all)
    return final_score


def run_individual_feature_analysis():
    """
    Part 2의 메인 실행 함수. 피처 레지스트리를 기반으로 개별 피처의 성능 기여도를 측정합니다.
    """
    # 전역 로거 사용
    from modules.utils_logger import logger
    
    logger.info("\n" + "="*80)
    logger.info("      <<< Part 2: 개별 피처 유의성 분석기(Analyzer) 시작 >>>")
    logger.info("="*80)

    # 1. 레지스트리 및 데이터 로딩
    core_features, candidate_features, _ = load_feature_registry()
    if not core_features:
        logger.error("분석의 기준이 될 Core 피처가 레지스트리에 없습니다.")
        return
        
    logger.info(f"ML 분류 데이터셋 로딩: {ML_DATASET_PATH}")
    df = pd.read_parquet(ML_DATASET_PATH)

    logger.info("데이터프레임 메모리 사용량 최적화를 시작합니다...")
    df = optimize_memory_usage(df)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    from modules.utils_io import read_yaml
    cfg = read_yaml('config/settings.yaml')
    lgbm_params = cfg.get("ml_params", {}).get("lgbm_params_classification", {})

    X, y = df, df['label_class']

    # 2. 베이스라인 성능 측정 (Core 피처만 사용)
    logger.info(f"\n[1] 베이스라인 성능을 측정합니다 (Core Features: {len(core_features)}개)...")
    baseline_score = _run_cv_for_feature_set(X, y, core_features, lgbm_params)
    logger.info(f"  -> 베이스라인 Average Precision 점수: {baseline_score:.4f}")

    # 3. 후보 피처 성능 측정 루프
    logger.info(f"\n[2] 후보 피처 성능 기여도 측정 (Candidates: {len(candidate_features)}개)...")
    results = []
    if candidate_features:
        for feature in tqdm(candidate_features, desc="Analyzing Candidates"):
            try:
                score = _run_cv_for_feature_set(X, y, core_features + [feature], lgbm_params)
                results.append({"feature_name": feature, "score_with_feature": score, "lift": score - baseline_score})
            except Exception as e:
                logger.error(f"'{feature}' 분석 중 오류: {e}", exc_info=True)
                results.append({"feature_name": feature, "score_with_feature": 0.0, "lift": -baseline_score})
    else:
        logger.warning("분석할 후보(Candidate) 피처가 없습니다. 측정을 건너뜁니다.")


    # 4. 결과 집계 및 리포팅
    logger.info("\n[3] 분석 결과 집계 및 리포팅...")
    
    # Core 피처에 대한 DataFrame 생성
    core_df = pd.DataFrame({
        'feature_name': core_features,
        'baseline_score': baseline_score,
        'score_with_feature': baseline_score,
        'lift': 'CORE'  # Core 피처임을 명시
    })
    
    # Candidate 피처에 대한 DataFrame 생성
    if results:
        candidate_df = pd.DataFrame(results).sort_values("lift", ascending=False)
        candidate_df['baseline_score'] = baseline_score
    else:
        candidate_df = pd.DataFrame(columns=['feature_name', 'score_with_feature', 'lift', 'baseline_score'])

    # Core 피처와 Candidate 피처 결과를 하나로 합치고 컬럼 순서 정리
    final_report_df = pd.concat([core_df, candidate_df], ignore_index=True)
    final_report_df = final_report_df[['feature_name', 'baseline_score', 'score_with_feature', 'lift']]
    
    logger.info("\n" + "="*80 + "\n                      <<< 개별 피처 성능 분석 결과 >>>\n" + "="*80)
    print(tabulate(final_report_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".4f"))
    logger.info("="*80)
    
    # 최종 리포트 저장
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, f"individual_feature_analysis_{timestamp}.csv")
    final_report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n상세 분석 결과가 '{output_path}'에 저장되었습니다.")
    
    return output_path


# --- Part 3: Optimal Combination Explorer ---
def run_optimal_combination_analysis(min_lift: float = 0.0):
    """
    Part 3의 메인 실행 함수. Part 2의 결과를 바탕으로 유망 피처를 선별하여
    자동으로 최적의 피처 조합을 탐색합니다.
    """
    from addons.addon_feature_optimizer import run_feature_combination_optimization
    from modules.utils_logger import logger
    
    logger.info("\n" + "="*80 + "\n      <<< Part 3: 최적 조합 탐색기(Explorer) 시작 >>>\n" + "="*80)

    # 1. Part 2 결과 파일 찾기
    logger.info("[1/3] 개별 피처 분석(Part 2)의 최신 결과 파일을 찾습니다...")
    try:
        analysis_files = [os.path.join(ANALYSIS_OUTPUT_DIR, f) for f in os.listdir(ANALYSIS_OUTPUT_DIR) if f.startswith('individual_feature_analysis') and f.endswith('.csv')]
        if not analysis_files:
            logger.error(f"'{ANALYSIS_OUTPUT_DIR}'에서 Part 2의 분석 결과 파일을 찾을 수 없습니다. Part 2를 먼저 실행해야 합니다.")
            return
        latest_analysis_file = max(analysis_files, key=os.path.getctime)
        logger.info(f"  -> 최신 분석 결과 로드: {latest_analysis_file}")
        analysis_result_df = pd.read_csv(latest_analysis_file)
    except Exception as e:
        logger.error(f"Part 2 결과 파일 로딩 중 오류 발생: {e}", exc_info=True)
        return

    # 2. 유망 후보 피처 선별
    # --- [핵심 수정] ---
    # 'lift' 컬럼을 숫자로 변환합니다. 'CORE' 같은 문자열은 NaN으로 바뀌어 이후 비교 연산에서 자동으로 제외됩니다.
    analysis_result_df['lift'] = pd.to_numeric(analysis_result_df['lift'], errors='coerce')
    # --- [수정 끝] ---
    
    promising_candidates = analysis_result_df[analysis_result_df['lift'] > min_lift]['feature_name'].tolist()
    
    if not promising_candidates:
        logger.warning(f"성능 기여도(lift) > {min_lift} 기준을 만족하는 유망 후보 피처가 없습니다. 조합 탐색을 중단합니다.")
        return

    logger.info(f"[2/3] 성능 기여도 > {min_lift} 기준, {len(promising_candidates)}개의 유망 피처를 선별했습니다.")
    logger.info(f"  -> 탐색 대상: {promising_candidates}")

    # 3. 선별된 후보군으로 최적 조합 탐색기(feature_optimizer) 실행
    logger.info("[3/3] 선별된 후보군으로 최적 조합 탐색을 시작합니다...")
    
    settings_path = 'config/settings.yaml'
    
    run_feature_combination_optimization(
        settings_path=settings_path,
        candidate_features_override=promising_candidates
    )

    logger.info("\n최적 조합 탐색이 완료되었습니다.")