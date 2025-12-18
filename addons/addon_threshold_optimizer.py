# addons/addon_threshold_optimizer.py (v5.2.4 - Fix YAML Update Error)
"""
[SpikeHunter] 최적 임계값(Threshold) 정밀 탐색기
- 기능: 모델 예측 확률을 기반으로 0.15 ~ 0.35 구간을 정밀 스캔하여 최적의 매매 타점을 찾습니다.
- 수정: 설정 파일 업데이트 시 float 타입 처리 오류 해결
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils_io import read_yaml, ensure_dir, get_user_input
from modules.utils_logger import logger
from modules.train import _get_core_features_from_registry
from modules.derive import _get_feature_cols

def run_threshold_optimization(settings_path: str):
    logger.info("\n" + "="*80)
    logger.info("      <<< 분류 모델 최적 임계값 정밀 탐색기 (Fine-Tuning) >>>")
    logger.info("="*80)
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # 1. 데이터 및 모델 로드
    dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        logger.error("데이터셋 또는 모델 파일이 없습니다.")
        return

    logger.info(f"데이터 로드: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 최근 데이터만 사용
    train_months = cfg.get("ml_params", {}).get("classification_train_months", 24)
    cutoff_date = df['date'].max() - pd.DateOffset(months=train_months)
    df = df[df['date'] >= cutoff_date]

    logger.info(f"모델 로드: {model_path}")
    model = joblib.load(model_path)
    
    # 피처 준비
    registry_path = "config/feature_registry.yaml"
    core_features = _get_core_features_from_registry(registry_path)
    available_cols = set(df.columns)
    feature_cols = [f for f in core_features if f in available_cols]
    if not feature_cols:
        feature_cols = _get_feature_cols(df.columns)
        
    X = df[feature_cols]
    y = df['label_class']
    
    # 테스트 셋 분리 (Shuffle=False 필수)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    logger.info(f"테스트 데이터 {len(X_test)}건에 대해 정밀 분석을 시작합니다...")
    
    # 2. 예측 확률 계산
    try:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
    except Exception as e:
        logger.error(f"예측 확률 계산 중 오류: {e}")
        return

    # 3. 정밀 탐색 범위 설정 (0.15 ~ 0.35, 0.01 단위)
    thresholds = np.arange(0.15, 0.50, 0.01)
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if y_pred.sum() == 0: continue
            
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        num_trades = y_pred.sum()
        
        results.append({
            "Threshold": thresh,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Trades": num_trades
        })

    # 4. 결과 출력
    if not results:
        logger.warning("해당 구간에서 매수 신호가 발생하지 않았습니다.")
        return

    res_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("             <<< 정밀 탐색 결과 (0.15 ~ 0.35) >>>")
    print("="*80)
    print(tabulate(res_df, headers="keys", tablefmt="psql", floatfmt=".4f", showindex=False))
    
    # 5. 최적값 제안
    best_f1 = res_df.loc[res_df['F1-Score'].idxmax()]
    
    # 승률 30% 이상이면서 F1이 가장 높은 값 찾기
    valid_prec = res_df[res_df['Precision'] >= 0.3]
    if not valid_prec.empty:
        best_balanced = valid_prec.loc[valid_prec['F1-Score'].idxmax()]
    else:
        best_balanced = best_f1
    
    print("\n[AI 추천 설정]")
    print(f"1. 이론적 최적값 (F1 Max): Threshold {best_f1['Threshold']:.2f}")
    print(f"   - 승률: {best_f1['Precision']*100:.2f}% | 포착률: {best_f1['Recall']*100:.2f}% | 거래수: {int(best_f1['Trades'])}")
    
    if not valid_prec.empty:
        print(f"2. 실전 권장값 (승률 30% 이상 중 F1 Max): Threshold {best_balanced['Threshold']:.2f}")
        print(f"   - 승률: {best_balanced['Precision']*100:.2f}% | 포착률: {best_balanced['Recall']*100:.2f}% | 거래수: {int(best_balanced['Trades'])}")
    
    print("="*80)
    
    # 설정 파일 업데이트
    apply_val = float(best_balanced['Threshold'])
    choice = get_user_input(f"\n'실전 권장값 ({apply_val:.2f})'을 설정 파일에 적용하시겠습니까? (y/n): ")
    
    if choice.lower() == 'y':
        # [수정] update_yaml 대신 직접 ruamel.yaml을 사용하여 단일 값 업데이트
        from ruamel.yaml import YAML
        
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                data = yaml.load(f)
            
            if 'ml_params' not in data:
                data['ml_params'] = {}
            
            # 단일 값 직접 할당
            data['ml_params']['classification_threshold'] = apply_val
            
            with open(settings_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
                
            logger.info(f"설정 파일 업데이트 완료: classification_threshold = {apply_val:.2f}")
            
        except Exception as e:
            logger.error(f"설정 파일 업데이트 실패: {e}")

if __name__ == "__main__":
    run_threshold_optimization("config/settings.yaml")