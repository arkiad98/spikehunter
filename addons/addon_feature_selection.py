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

from modules.utils_io import read_yaml, ensure_dir, get_user_input
from modules.utils_logger import logger
from modules.derive import _get_feature_cols
from ruamel.yaml import YAML # Safe processing

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
    # [수정] CommentedMap -> dict 변환 및 param_space_ 제거
    raw_params = cfg.get("ml_params", {}).get("lgbm_params_classification", {})
    if hasattr(raw_params, 'items'): # CommentedMap or dict
        lgbm_params = dict(raw_params)
    else:
        lgbm_params = {}

    # param_space_ 로 시작하는 키 제거 (모델 인자로 전달되지 않도록)
    keys_to_remove = [k for k in lgbm_params.keys() if k.startswith('param_space_')]
    for k in keys_to_remove:
        lgbm_params.pop(k)

    lgbm_params.update({'n_estimators': 500, 'verbose': -1, 'n_jobs': -1})
    
    # [NEW] 4. Baseline 평가 (현재 Core 피처)
    core_feats = [f for f, s in feat_status.items() if s == 'core']
    valid_core_feats = [f for f in core_feats if f in valid_features]
    
    baseline_score = 0.0
    if valid_core_feats:
        base_model = lgb.LGBMClassifier(**lgbm_params)
        base_model.fit(X_train[valid_core_feats], y_train)
        y_pred_base = base_model.predict_proba(X_test[valid_core_feats])[:, 1]
        baseline_score = average_precision_score(y_test, y_pred_base)
        logger.info(f" >> [Baseline] 현재 Core 피처 ({len(valid_core_feats)}개) 성능(AP): {baseline_score:.4f}")
    else:
        logger.info(" >> [Baseline] Core 피처가 없어 평가를 건너뜁니다.")

    # 5. RFE Loop
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
    
    # 6. 결과 분석
    res_df = pd.DataFrame(history).sort_values('score', ascending=False)
    best_res = res_df.iloc[0]
    
    logger.info("\n" + "="*60)
    logger.info(f"   [RFE 최종 결과]")
    logger.info(f"   Baseline (현재): {baseline_score:.4f}")
    logger.info(f"   Best RFE (추천): {best_res['score']:.4f} (피처 {best_res['n_features']}개)")
    
    diff = best_res['score'] - baseline_score
    if diff > 0:
        logger.info(f"   >> 성능 개선: +{diff:.4f} (개선됨)")
    else:
        logger.info(f"   >> 성능 변화: {diff:.4f} (현재가 더 좋거나 비슷함)")
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
        
    # 7. 시각화
    _set_korean_font()
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['n_features'], res_df['score'], marker='o')
    plt.axvline(x=best_res['n_features'], color='r', linestyle='--', label=f"Best: {best_res['n_features']}")
    plt.axhline(y=baseline_score, color='g', linestyle=':', label=f"Baseline: {baseline_score:.4f}")
    plt.title("피처 개수에 따른 모델 성능(AP) 변화")
    plt.xlabel("피처 개수")
    plt.ylabel("Average Precision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = "analysis_rfe_result.png"
    plt.savefig(plot_path)
    logger.info(f"\n[!] 결과 차트 저장됨: {plot_path}")

    # [NEW] 8. 레지스트리 업데이트 (사용자 선택)
    print("\n" + "="*60)
    if diff > 0:
        print("💡 RFE 결과가 현재보다 우수합니다. 추천 조합을 적용하시겠습니까?")
    else:
        print("⚠️ RFE 결과가 현재보다 좋지 않습니다. 적용을 권장하지 않습니다.")
    
    choice = get_user_input("최적 피처 조합을 'config/feature_registry.yaml'에 반영하시겠습니까? (y/n): ")
    
    if choice.lower() == 'y':
        reg_path = "config/feature_registry.yaml"
        
        # 1. 백업
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"config/feature_registry_backup_{timestamp}.yaml"
        ensure_dir("config/old") # 백업 폴더가 있으면 좋은데 없으니 그냥 config에 or config/old?
        # 그냥 같은 폴더에
        import shutil
        shutil.copy(reg_path, backup_path)
        logger.info(f"백업 완료: {backup_path}")
        
        # 2. 업데이트
        try:
            yaml = YAML()
            yaml.preserve_quotes = True
            
            with open(reg_path, 'r', encoding='utf-8') as f:
                data = yaml.load(f)
            
            # 피처 상태 업데이트
            updated_count = 0
            best_feat_set = set(final_feats)
            
            if 'features' in data:
                for item in data['features']:
                    fname = item.get('name')
                    if fname:
                        if fname in best_feat_set:
                            item['status'] = 'core'
                        else:
                            # 기존에 core였다면 candidate로 강등? 아니면 unused?
                            # 사용자 요청상 "선택된 피처들만 core로"
                            # 나머지는 unused 또는 candidate. 보통 unused가 안전.
                            item['status'] = 'unused'
                        updated_count += 1
            
            with open(reg_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
                
            logger.info(f"레지스트리 업데이트 완료! ({updated_count}개 피처 상태 변경됨)")
            logger.info("이제 '모델 학습' 메뉴를 실행하면 새로운 피처 조합이 사용됩니다.")
            
        except Exception as e:
            logger.error(f"레지스트리 업데이트 실패: {e}")