# addons/addon_feature_selection.py (v5.0)
"""
[SpikeHunter v5.0] 최적 피처 조합 탐색기 (RFE Renewal)
- 기능: 재귀적 피처 제거(RFE)를 통해 모델 성능을 극대화하는 최소/최적 피처셋 발굴
- 연동: v5.0 데이터셋 경로 및 feature_registry.yaml 상태 표시 지원
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

from modules.utils_io import read_yaml, ensure_dir
from modules.utils_logger import logger
from modules.derive import _get_feature_cols

def _set_korean_font():
    font_path = 'c:/Windows/Fonts/malgun.ttf'
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

def _get_feature_status(registry_path: str) -> dict:
    """레지스트리에서 피처 상태(Core/Candidate) 로드"""
    if not os.path.exists(registry_path): return {}
    try:
        reg = read_yaml(registry_path)
        return {f['name']: f.get('status', 'unknown') for f in reg.get('features', [])}
    except: return {}

def run_feature_selection(settings_path: str):
    """RFE 실행 메인 함수"""
    logger.info("="*60)
    logger.info("      <<< 최적 피처 조합 탐색(RFE) 시작 >>>")
    logger.info("="*60)
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # 1. 데이터 로드 (표준 경로)
    dataset_file = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_file):
        logger.error(f"데이터셋이 없습니다: {dataset_file}\n먼저 '데이터 관리 -> 데이터셋 생성'을 실행하세요.")
        return
    
    logger.info(f"데이터 로드 중... {dataset_file}")
    df = pd.read_parquet(dataset_file)
    
    # 2. 피처 및 타겟 설정
    # 데이터셋에 있는 모든 유효 피처를 후보로 사용
    valid_features = _get_feature_cols(df.columns)
    
    # 레지스트리 정보 로드 (리포팅용)
    feat_status = _get_feature_status("config/feature_registry.yaml")
    
    logger.info(f"분석 대상 피처: {len(valid_features)}개")
    
    # 샘플링 (속도 향상) - 최근 데이터 위주로 30만개만 사용
    if len(df) > 300000:
        df = df.sort_values('date').tail(300000)
    
    X = df[valid_features]
    y = df['label_class']
    
    # 학습/검증 분리 (Hold-out)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. LightGBM 파라미터 (속도 최적화)
    lgbm_params = cfg.get("ml_params", {}).get("lgbm_params_classification", {}).copy()
    lgbm_params.update({'n_estimators': 500, 'verbose': -1, 'n_jobs': -1})
    
    # 4. RFE Loop
    features_curr = valid_features.copy()
    history = []
    
    pbar = tqdm(total=len(features_curr)-1, desc="Eliminating Features")
    
    while len(features_curr) >= 1:
        # 모델 학습
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_train[features_curr], y_train)
        
        # 평가 (Average Precision 기준)
        y_pred = model.predict_proba(X_test[features_curr])[:, 1]
        score = average_precision_score(y_test, y_pred)
        
        history.append({
            'n_features': len(features_curr),
            'score': score,
            'features': features_curr.copy()
        })
        
        if len(features_curr) == 1: break
        
        # 중요도 하위 제거
        importances = pd.Series(model.feature_importances_, index=features_curr)
        worst_feature = importances.idxmin()
        features_curr.remove(worst_feature)
        pbar.update(1)
        
    pbar.close()
    
    # 5. 결과 분석
    res_df = pd.DataFrame(history).sort_values('score', ascending=False)
    best_res = res_df.iloc[0]
    
    logger.info("\n" + "="*60)
    logger.info(f"   [RFE 최종 결과]")
    logger.info(f"   최고 성능(AP): {best_res['score']:.4f}")
    logger.info(f"   최적 피처 개수: {best_res['n_features']}개")
    logger.info("="*60)
    
    logger.info("\n[추천 피처 조합 (중요도순 정렬)]")
    # 최적 조합으로 다시 학습해서 중요도 순서대로 출력
    final_feats = best_res['features']
    final_model = lgb.LGBMClassifier(**lgbm_params)
    final_model.fit(X_train[final_feats], y_train)
    final_imp = pd.Series(final_model.feature_importances_, index=final_feats).sort_values(ascending=False)
    
    for i, (feat, imp) in enumerate(final_imp.items()):
        status = feat_status.get(feat, 'unknown')
        logger.info(f" {i+1:2d}. {feat:<20} (Status: {status}) | Imp: {imp}")
        
    # 6. 시각화
    _set_korean_font()
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['n_features'], res_df['score'], marker='o')
    plt.axvline(x=best_res['n_features'], color='r', linestyle='--', label=f"Best: {best_res['n_features']}")
    plt.title("피처 개수에 따른 모델 성능(AP) 변화")
    plt.xlabel("피처 개수")
    plt.ylabel("Average Precision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = "analysis_rfe_result.png"
    plt.savefig(plot_path)
    logger.info(f"\n[!] 결과 차트 저장됨: {plot_path}")