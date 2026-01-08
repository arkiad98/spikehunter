# run_pipeline.py (v5.2.3)
"""
[SpikeHunter v5.2.3] 통합 파이프라인 실행 스크립트
- 통합: 데이터 관리, 학습, 전략(백테스트/최적화/WF), 심층 분석(애드온) 모두 포함
- 업데이트: '최적 임계값 탐색(Threshold Optimizer)' 애드온 메뉴 추가
"""

import os
import sys
import os
import sys
# [Fix] Apply Pykrx Patch (PR #249)
try:
    from modules.patch_pykrx import patch_pykrx_referer
    patch_pykrx_referer()
except ImportError:
    pass

import argparse
import glob
import copy
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import joblib
import numpy as np
from ruamel.yaml import YAML

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# [System] Apply Pykrx Patch (Critical for 2026+ Data)
try:
    from modules.patch_pykrx import patch_pykrx_referer
    patch_pykrx_referer()
    print("[System] Pykrx Patched Successfully.")
except Exception as e:
    print(f"[Warning] Pykrx Patch Failed: {e}")

# -----------------------------------------------------------------------------
# Core Modules Import
# -----------------------------------------------------------------------------
from modules import collect, derive, train, backtest, optimization, predict
from modules import utils_io, utils_logger, utils_db
from modules.utils_io import get_user_input, read_yaml, ensure_dir
from modules.backtest import run_backtest, _calculate_full_metrics

# -----------------------------------------------------------------------------
# Add-ons Import (Graceful Import)
# -----------------------------------------------------------------------------
try:
    from addons import (
        addon_feature_selection,          # 1. 피처 조합 탐색
        addon_feature_stability_analyzer, # 2. 피처 안정성 분석
        addon_pre_spike_analyzer,         # 3. 급등 전조 분석
        addon_shap_analysis,              # 4. SHAP 모델 해석
        addon_tradelog_analyzer,          # 5. 매매 로그 분석
        addon_cs_weight_finder,           # 6. 비용 민감도 탐색
        addon_threshold_optimizer         # 7. 임계값 탐색 [추가됨]
    )
    ADDONS_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] 일부 애드온을 로드할 수 없습니다: {e}")
    ADDONS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Setup & Helpers
# -----------------------------------------------------------------------------
def setup_system(settings_path):
    """로거 설정 및 DB 초기화"""
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    utils_logger.setup_global_logger(run_timestamp)
    utils_db.create_tables()
    
    if not os.path.exists(settings_path):
        utils_logger.logger.error(f"설정 파일이 없습니다: {settings_path}")
        return False
    return True

def _print_header():
    print("\n" + "="*60)
    print("      <<< SpikeHunter v5.2 Integrated Pipeline >>>")
    print("="*60)

def _get_latest_backtest_run(base_dir):
    """가장 최근 백테스트 결과 폴더 찾기"""
    runs = glob.glob(os.path.join(base_dir, "run_*"))
    if not runs: return None
    return max(runs, key=os.path.getctime)

# -----------------------------------------------------------------------------
# Walk-Forward Logic
# -----------------------------------------------------------------------------
def _calculate_optimal_threshold(settings_path: str) -> float:
    """[Headless] 최적 임계값 자동 계산 (for WFO)"""
    try:
        from addons import addon_threshold_optimizer
        # Redirect stdout to suppress print output if needed, or just let it log
        # But addon assumes interactive user input at the end.
        # So we better reimplement core logic or genericize the addon.
        # For safety/speed, let's implement a lightweight version here using the same logic.
        
        cfg = read_yaml(settings_path)
        paths = cfg["paths"]
        dataset_path = os.path.join(paths["features"], "dataset_v4.parquet") # Use Train-split dataset
        model_path = os.path.join(paths["models"], "lgbm_model.joblib")
        
        if not os.path.exists(dataset_path) or not os.path.exists(model_path):
            utils_logger.logger.warning("[WFO-Thres] 데이터/모델 없음. 기본값 0.5 반환")
            return 0.5
            
        model = joblib.load(model_path)
        df = pd.read_parquet(dataset_path)
        
        # Prepare Data
        feature_cols = getattr(model, 'feature_name_', [])
        if not len(feature_cols): # Fallback
             # Try to infer from derive
             from modules.derive import _get_feature_cols
             feature_cols = _get_feature_cols(df.columns)
             
        # Filter available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].fillna(0)
        y = df['label_class']
        
        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X)
            
        # Optimize F1
        from sklearn.metrics import precision_score, f1_score
        
        best_f1 = -1
        best_thresh = 0.5
        
        # Scan 0.20 ~ 0.80
        for th in np.arange(0.20, 0.80, 0.01):
            pred = (probs >= th).astype(int)
            if pred.sum() < 5: continue # Too few trades
            
            # Use F1 but prioritize Precision >= 0.3 for stability (SpikeHunter rule)
            prec = precision_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)
            
            if prec >= 0.3: # Valid candidate
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = th
                    
        utils_logger.logger.info(f" >> [WFO-Thres] 최적 임계값 발견: {best_thresh:.2f} (F1: {best_f1:.4f})")
        return float(best_thresh)
        
    except Exception as e:
        utils_logger.logger.error(f"[WFO-Thres] 임계값 계산 오류: {e}")
        return 0.49 # Safe default
        
def run_walk_forward_pipeline(settings_path: str, strategy_name: str):
    utils_logger.logger.info(f">>> Walk-Forward 최적화를 시작합니다: '{strategy_name}'")
    
    # 원본 설정 로드
    original_cfg = read_yaml(settings_path)
    wf_cfg = original_cfg.get("walk_forward", {})
    
    total_start_date = pd.Timestamp(wf_cfg.get("total_start_date", "2020-01-01"))
    total_end_date = pd.Timestamp.today().normalize()
    train_months = wf_cfg.get("train_months", 24)
    test_months = wf_cfg.get("test_months", 6)
    
    paths = original_cfg["paths"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wf_run_dir = os.path.join(paths["backtest"], f"WF-{strategy_name}-{timestamp}")
    ensure_dir(wf_run_dir)
    
    utils_logger.logger.info(f"결과 저장 경로: {wf_run_dir}")
    
    temp_settings_path = os.path.join(wf_run_dir, 'temp_settings.yaml')
    
    # 구간 생성
    wf_periods = []
    current_date = total_start_date
    while current_date + pd.DateOffset(months=train_months + test_months) <= total_end_date:
        train_start = current_date
        train_end = current_date + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        wf_periods.append((train_start, train_end, test_start, test_end))
        current_date += pd.DateOffset(months=test_months)

    all_test_equities = []
    all_trade_logs = []
    next_initial_cash = original_cfg.get("backtest_defaults", {}).get("initial_cash", 10000000.0)

    # [NEW] Adaptive Parameters Carry-over
    # 이전 Period에서 찾은 최적 파라미터를 다음 Period의 '라벨링' 및 '초기 탐색'에 반영
    current_best_ml_params = copy.deepcopy(original_cfg.get('ml_params', {}))

    for i, (train_start, train_end, test_start, test_end) in enumerate(tqdm(wf_periods, desc="Walk-Forward Periods")):
        utils_logger.logger.info(f"\n[Period {i+1}] Train: {train_start.date()}~{train_end.date()} | Test: {test_start.date()}~{test_end.date()}")
        
        # 1. 임시 폴더 설정
        period_dir = os.path.join(wf_run_dir, f"period_{i+1}")
        period_feat_dir = os.path.join(period_dir, "features")
        period_model_dir = os.path.join(period_dir, "models")
        ensure_dir(period_feat_dir)
        ensure_dir(period_model_dir)
        
        # 2. 설정 객체 깊은 복사 (Deepcopy)
        period_cfg = copy.deepcopy(original_cfg)
        period_cfg['paths']['features'] = period_feat_dir
        period_cfg['paths']['models'] = period_model_dir
        period_cfg['paths']['ml_dataset'] = period_feat_dir # [Fix] derive.py가 여기로 저장하도록 강제
        period_cfg['data_range'] = {'start': str(train_start.date()), 'end': str(test_end.date())}
        
        # [NEW] Adaptive Apply: 이전 최적값 적용
        if i > 0: # 첫 번째 구간은 원본 설정 사용
            utils_logger.logger.info(f" >> [Adaptive] 이전 구간 최적 파라미터 적용 (Target: {current_best_ml_params.get('target_surge_rate', '?'):.4f})")
            period_cfg['ml_params'].update(current_best_ml_params)
            
            # 전략 파라미터 초기값도 업데이트 (Warm Start 효과)
            if 'target_surge_rate' in current_best_ml_params:
                # strategy param name is 'target_r'
                 if strategy_name in period_cfg['strategies']:
                    period_cfg['strategies'][strategy_name]['target_r'] = current_best_ml_params['target_surge_rate']
            if 'stop_loss_rate' in current_best_ml_params:
                 if strategy_name in period_cfg['strategies']:
                    period_cfg['strategies'][strategy_name]['stop_r'] = current_best_ml_params['stop_loss_rate']

        
        # 임시 설정 파일 저장
        yaml_obj = YAML()
        with open(temp_settings_path, 'w', encoding='utf-8') as f:
            yaml_obj.dump(period_cfg, f)
            
        # 3. 데이터 생성 (Train + Test 기간)
        try:
            # derive에서 labeling 수행 시 period_cfg(수정된 ml_params)를 사용함 -> Adaptive Labeling 구현됨
            derive.run_derive(settings_path=temp_settings_path)
        except Exception as e:
            utils_logger.logger.error(f"데이터 생성 실패 (Period {i+1}): {e}")
            continue
            
        # 4. 학습 데이터 분리 (Train Only)
        full_dataset_path = os.path.join(period_feat_dir, "ml_classification_dataset.parquet") # [Fix] 파일명 변경
        if not os.path.exists(full_dataset_path):
            utils_logger.logger.error(f"데이터셋 파일 생성 실패: {full_dataset_path}")
            continue
            
        df_full = pd.read_parquet(full_dataset_path)
        df_full['date'] = pd.to_datetime(df_full['date'])
        df_train = df_full[df_full['date'] <= train_end].copy()
        
        train_feat_dir = os.path.join(period_dir, "features_train_only")
        ensure_dir(train_feat_dir)
        df_train.to_parquet(os.path.join(train_feat_dir, "dataset_v4.parquet"))
        
        # 5. 학습용 설정 생성
        train_cfg = copy.deepcopy(period_cfg)
        train_cfg['paths']['features'] = train_feat_dir
        
        # [CRITICAL FIX] WFO에서는 이미 날짜별로 잘라서(Train Only) 전달하므로
        # 내부적인 offset(Gap)을 0으로 강제해야 "Double Offset" 문제를 방지함.
        # 기존: Train End(12/31) - Offset(6m) = 6/30 (최근 6개월 데이터 누락 발생)
        # 수정: Train End(12/31) - Offset(0m) = 12/31 (정상 학습)
        train_cfg['ml_params']['classification_train_end_offset'] = 0
        
        train_settings_path = os.path.join(period_dir, "settings_train.yaml")
        with open(train_settings_path, 'w', encoding='utf-8') as f:
            yaml_obj.dump(train_cfg, f)
            
        # 6. 모델 학습 (Train Data)
        utils_logger.logger.info(f" >> 모델 학습 진행 (Train Data Only)...")
        train.run_train_pipeline(settings_path=train_settings_path)
        
        # [NEW] 6.1 최적 임계값 탐색 (Threshold Optimization)
        # 학습된 모델을 기반으로 최적 임계값을 찾고, 이를 설정에 반영 (Fix)
        optimal_threshold = _calculate_optimal_threshold(train_settings_path)
        utils_logger.logger.info(f" >> [Adaptive] Period {i+1} 최적 임계값 적용: {optimal_threshold:.4f}")
        
        # Update settings for optimization
        train_cfg['ml_params']['classification_threshold'] = optimal_threshold
        if strategy_name in train_cfg['strategies']:
            train_cfg['strategies'][strategy_name]['min_ml_score'] = optimal_threshold # Force Update
            
        # Save updated settings
        with open(train_settings_path, 'w', encoding='utf-8') as f:
            yaml_obj.dump(train_cfg, f)
        
        # 6.5. 전략 최적화 (Train Data) - WFO 적용
        utils_logger.logger.info(f" >> 전략 파라미터 최적화 (Train Data)...")
        try:
            # Note: min_ml_score might be re-optimized if it's in param_space.
            # If user wants to FIX it to this optimal value, we should remove it from param_space dynamically?
            # User said: "최적값을 찾아 고정하고" -> Yes, we should likely fix it.
            # Let's verify if we should modify param_space.
            # Strategy: Let Optuna search AROUND this value? or FIX it?
            # User request: "최적값을 찾아 고정하고 백테스트 거래를 진행"
            # So let's FIX it by creating a tight range or removing from space.
            pass # Logic implemented above updates the BASE value. Optuna range overrides it.
            
            # To FIX it in Optuna, we need to modify param_space in the YAML
            # Remove 'min_ml_score' from param_space if it exists, to treat it as fixed constant.
            if strategy_name in train_cfg.get('optimization', {}):
                p_space = train_cfg['optimization'][strategy_name].get('param_space', {})
                if 'min_ml_score' in p_space:
                    del train_cfg['optimization'][strategy_name]['param_space']['min_ml_score']
                    utils_logger.logger.info(f" >> [Optuna] min_ml_score 최적화 제외 (Calculated Fixed Value: {optimal_threshold})")
                    # Save again
                    with open(train_settings_path, 'w', encoding='utf-8') as f:
                        yaml_obj.dump(train_cfg, f)

            best_params = optimization.optimize_strategy_headless(
                settings_path=train_settings_path,
                strategy_name=strategy_name,
                start_date=str(train_start.date()),
                end_date=str(train_end.date()),
                n_trials=50, 
                n_jobs=-1
            )
            
            if best_params:
                utils_logger.logger.info(f" >> 적용된 최적 파라미터: {best_params}")
                # period_cfg 업데이트 (Test 기간 백테스트용)
                if strategy_name not in period_cfg['strategies']:
                    period_cfg['strategies'][strategy_name] = {}
                period_cfg['strategies'][strategy_name].update(best_params)
                
                # ML Parameter Sync (중요: 백테스트 시 일관성 유지)
                if 'target_r' in best_params:
                    period_cfg['ml_params']['target_surge_rate'] = float(best_params['target_r'])
                if 'stop_r' in best_params:
                    period_cfg['ml_params']['stop_loss_rate'] = float(best_params['stop_r'])
                if 'min_ml_score' in best_params:
                    period_cfg['ml_params']['classification_threshold'] = float(best_params['min_ml_score'])
                
                # [NEW] Update Current Best for NEXT Period
                # 다음 루프의 Adaptive Labeling을 위해 저장
                current_best_ml_params.update(period_cfg['ml_params'])

                # 임시 설정 파일 재저장 (Optimized Params for Test)
                with open(temp_settings_path, 'w', encoding='utf-8') as f:
                    yaml_obj.dump(period_cfg, f)
        except Exception as e:
            utils_logger.logger.error(f"전략 최적화 중 오류 무시하고 진행: {e}")

        # 7. 백테스트 (Test Data)
        utils_logger.logger.info(f" >> 백테스트 진행 (Test Period)...")
        test_result = run_backtest(
            run_dir=os.path.join(period_dir, "result"),
            strategy_name=strategy_name,
            settings_cfg=period_cfg,
            settings_path=settings_path, # [Fix] Pass explicit path for exclude_dates.yaml resolution
            start=str(test_start.date()),
            end=str(test_end.date()),
            initial_cash=next_initial_cash,
            use_ml_target=True,
            quiet=True,
            save_to_db=False
        )
        
        if test_result and not test_result['daily_equity'].empty:
            eq = test_result['daily_equity']
            all_test_equities.append(eq)
            next_initial_cash = eq['equity'].iloc[-1]
            
            tl = pd.read_parquet(test_result['tradelog_path']) if os.path.exists(test_result['tradelog_path']) else pd.DataFrame()
            if not tl.empty:
                all_trade_logs.append(tl)
            utils_logger.logger.info(f" >> Period {i+1} 종료 자산: {int(next_initial_cash):,}원")
        else:
            utils_logger.logger.warning(f"Period {i+1} 백테스트 결과 없음.")

    # 최종 리포트
    if all_test_equities:
        final_equity = pd.concat(all_test_equities).sort_index()
        final_tradelog = pd.concat(all_trade_logs).reset_index(drop=True) if all_trade_logs else pd.DataFrame()
        metrics = _calculate_full_metrics(final_equity, final_tradelog)
        
        final_equity.to_parquet(os.path.join(wf_run_dir, "wf_daily_equity.parquet"))
        if not final_tradelog.empty:
            final_tradelog.to_parquet(os.path.join(wf_run_dir, "wf_tradelog.parquet"))
            final_tradelog.to_csv(os.path.join(wf_run_dir, "wf_tradelog.csv"), index=False, encoding='utf-8-sig')
            
        utils_logger.logger.info("\n" + "="*60)
        utils_logger.logger.info(f"   [Walk-Forward 최종 결과] {total_start_date.date()} ~ {final_equity.index.max().date()}")
        utils_logger.logger.info(f"   - 최종 자산: {int(final_equity['equity'].iloc[-1]):,} 원")
        utils_logger.logger.info(f"   - 수익률(CAGR): {metrics['CAGR_raw']*100:.2f}%")
        utils_logger.logger.info(f"   - MDD: {metrics['MDD_raw']*100:.2f}%")
        utils_logger.logger.info(f"   - 승률: {metrics['win_rate_raw']*100:.2f}% ({metrics['총거래횟수']}회)")
        utils_logger.logger.info("="*60)
    else:
        utils_logger.logger.error("Walk-Forward 결과가 생성되지 않았습니다.")

# -----------------------------------------------------------------------------
# Menus
# -----------------------------------------------------------------------------
def _menu_data_management(settings_path):
    while True:
        print("\n--- [1. 데이터 관리] ---")
        print("1. 데이터 수집 (Collect)")
        print("2. 피처 생성 및 라벨링 (Derive)")
        print("3. 데이터 무결성 점검 (Check)")
        print("4. 데이터 정제 및 이상치 보정 (Clean) [NEW]")
        print("0. 이전 메뉴로")
        
        sel = get_user_input("선택: ")
        if sel == '1': 
            print("  [데이터 수집 옵션]")
            print("  1. 자동 (최근 수집일 이후 ~ 오늘)")
            print("  2. 전체 초기화 및 재수집 (2020-01-01 ~ 오늘)")
            print("  3. 기간 지정 (직접 입력)")
            sub_sel = get_user_input("  선택 (1/2/3): ")

            if sub_sel == '1':
                collect.run_collect(settings_path)
            elif sub_sel == '2':
                confirm = get_user_input("  정말 전체 데이터를 삭제하고 다시 수집하시겠습니까? (y/n): ")
                if confirm.lower() == 'y':
                    collect.run_collect(settings_path, overwrite=True)
            elif sub_sel == '3':
                s_date = get_user_input("  시작일 (YYYY-MM-DD): ")
                e_date = get_user_input("  종료일 (YYYY-MM-DD): ")
                collect.run_collect(settings_path, start=s_date, end=e_date, use_meta=False)

        elif sel == '2': derive.run_derive(settings_path)
        elif sel == '3': 
            try:
                from addons import addon_data_integrity_check
                addon_data_integrity_check.run_data_integrity_check(settings_path)
            except (ImportError, AttributeError):
                print("데이터 점검 모듈을 찾을 수 없습니다.")
        elif sel == '4':
            from modules import clean
            clean.run_clean_pipeline(settings_path)
        elif sel == '0': break

def _menu_training(settings_path):
    while True:
        print("\n--- [2. 모델 학습 및 최적화] ---")
        print("1. 모델 학습 (Train)")
        print("2. ML 하이퍼파라미터 최적화 (Optuna)")
        print("0. 이전 메뉴로")
        
        sel = get_user_input("선택: ")
        if sel == '1': train.run_train_pipeline(settings_path)
        elif sel == '2': train.run_ml_optimization_pipeline(settings_path)
        elif sel == '0': break

def _menu_strategy(settings_path):
    while True:
        print("\n--- [3. 전략 백테스트 및 최적화] ---")
        print("1. 전략 백테스트 실행 (Backtest)")
        print("2. 전략 파라미터 최적화 (Strategy Optimization)")
        print("3. Walk-Forward 검증 (전진 분석)")
        print("0. 이전 메뉴로")
        
        sel = get_user_input("선택: ")
        
        # 전략 이름 로드
        cfg = read_yaml(settings_path)
        strategy_name = list(cfg.get('strategies', {}).keys())[0] if cfg.get('strategies') else "SpikeHunter"

        if sel == '1': 
            print("  1. 전체 기간 (Full History)")
            print("  2. 최근 검증 기간 (Test Period Only: After Training)")
            sub_sel = get_user_input("  기간 선택 (1/2): ")
            
            start_date = None
            if sub_sel == '2':
                # Calculate Test Start Date
                try:
                    dataset_path = os.path.join(cfg['paths']['ml_dataset'], 'ml_classification_dataset.parquet')
                    if not os.path.exists(dataset_path): # Fallback
                         dataset_path = os.path.join(cfg['paths']['features'], 'dataset_v4.parquet')
                         
                    if os.path.exists(dataset_path):
                        df_tmp = pd.read_parquet(dataset_path, columns=['date'])
                        max_date = pd.to_datetime(df_tmp['date']).max()
                        end_offset = cfg.get('ml_params', {}).get('classification_train_end_offset', 6)
                        
                        # Test Start = MaxDate - Offset
                        test_start = max_date - pd.DateOffset(months=end_offset)
                        start_date = test_start.strftime("%Y-%m-%d")
                        print(f"  >> 검증 기간 자동 설정: {start_date} ~ {max_date.date()}")
                except Exception as e:
                    print(f"  >> 날짜 계산 실패 (전체 기간으로 수행): {e}")

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"data/proc/backtest/run_{run_id}"
            print(f" >> 전략 '{strategy_name}' 백테스트 시작...")
            backtest.run_backtest(
                run_dir=run_dir,
                strategy_name=strategy_name,
                settings_path=settings_path,
                start=start_date,
                save_to_db=True
            )
        elif sel == '2': 
            optimization.run_optimization_pipeline(settings_path)
        elif sel == '3':
            # Walk-Forward 실행
            run_walk_forward_pipeline(settings_path, strategy_name)
        elif sel == '0': break

def _menu_analysis(settings_path):
    if not ADDONS_AVAILABLE:
        print("\n[Error] 애드온 모듈이 로드되지 않아 사용할 수 없습니다.")
        return

    while True:
        print("\n--- [4. 진단 및 심층 분석 (Add-ons)] ---")
        print("1. 최적 피처 조합 탐색 (RFE Analysis)")
        print("2. 피처 안정성 분석 (Stability Analyzer)")
        print("3. 급등 전조 현상 분석 (Pre-Spike Analyzer)")
        print("4. SHAP 모델 해석 (Model Explainability)")
        print("5. 매매 로그 정밀 분석 (TradeLog Analyzer)")
        print("6. 최적 비용 민감도 탐색 (CS Weight Finder)")
        print("7. 최적 임계값 탐색 (Threshold Optimizer) [추가]")
        print("0. 이전 메뉴로")
        
        sel = get_user_input("선택: ")
        
        try:
            if sel == '1': addon_feature_selection.run_feature_selection(settings_path)
            elif sel == '2': addon_feature_stability_analyzer.run_stability_analysis(settings_path)
            elif sel == '3': 
                if hasattr(addon_pre_spike_analyzer, 'main'): addon_pre_spike_analyzer.main()
                else: print("Pre-Spike Analyzer 실행 함수를 찾을 수 없습니다.")
            elif sel == '4': addon_shap_analysis.run_shap_analysis(settings_path)
            elif sel == '5':
                if hasattr(addon_tradelog_analyzer, 'run_tradelog_analysis'):
                    addon_tradelog_analyzer.run_tradelog_analysis(settings_path)
                else:
                    print("TradeLog Analyzer 실행 함수를 찾을 수 없습니다.")
            elif sel == '6':
                if hasattr(addon_cs_weight_finder, 'run_find_optimal_cs_weight'):
                    addon_cs_weight_finder.run_find_optimal_cs_weight(settings_path)
                else:
                    print("CS Weight Finder 실행 함수를 찾을 수 없습니다.")
            elif sel == '7': # [추가된 기능]
                if hasattr(addon_threshold_optimizer, 'run_threshold_optimization'):
                    addon_threshold_optimizer.run_threshold_optimization(settings_path)
                else:
                    print("Threshold Optimizer 실행 함수를 찾을 수 없습니다.")
            elif sel == '0': break
        except Exception as e:
            utils_logger.logger.error(f"애드온 실행 중 오류 발생: {e}")
            print(f"!!! 실행 중 오류가 발생했습니다: {e}")

# -----------------------------------------------------------------------------
# Menus (Continued)
# -----------------------------------------------------------------------------
def _menu_utils(settings_path: str):
    while True:
        print("\n--- [5. 유틸리티 (Utils)] ---")
        print("1. 수정주가 데이터 정제 (Price Sanitizer)")
        print("2. 전략 로직 디버거 (Strategy Debugger)")
        print("3. WF 로그 통합 (Log Aggregator)")
        print("4. 데이터 로직 검증 (Logic Verifier)")
        print("0. 이전 메뉴로")
        
        sel = get_user_input("선택: ")
        
        if sel == '1':
            try:
                from addons import addon_price_sanitizer
                addon_price_sanitizer.run_price_sanitizer(settings_path)
            except ImportError as e:
                print(f"모듈을 찾을 수 없습니다: {e}")
        elif sel == '2':
            try:
                from addons import addon_strategy_debugger
                addon_strategy_debugger.run_strategy_debugger(settings_path)
            except ImportError as e:
                print(f"모듈을 찾을 수 없습니다: {e}")
        elif sel == '3':
            try:
                from addons import addon_wf_log_aggregator
                addon_wf_log_aggregator.run_aggregate_wf_logs(settings_path)
            except ImportError as e:
                print(f"모듈을 찾을 수 없습니다: {e}")
        elif sel == '4':
            try:
                from addons import verify_data_logic
                verify_data_logic.verify_single_case()
            except ImportError as e:
                print(f"모듈을 찾을 수 없습니다: {e}")
        elif sel == '0': break
        else:
            print("잘못된 선택입니다.")

def _menu_field_test(settings_path: str):
    """실전 검증 및 운영 관련 메뉴"""
    while True:
        print("\n" + "="*60)
        print("      <<< 6. 실전 검증 (Field Test) >>>")
        print("="*60)
        print("1. 금일 추천 종목 생성 (Predict)")
        print("2. 추천 종목 성과 검증 (Verify)")
        print("0. 이전 메뉴로")
        
        choice = get_user_input("선택 (1/2/0): ")
        
        if choice == '1':
            predict.generate_and_save_recommendations(settings_path)
        elif choice == '2':
            # Lazy import to avoid circular dependencies or heavy loading if not needed
            try:
                from modules import verify_daily_signals
                verify_daily_signals.verify_signals(settings_path)
                verify_daily_signals.print_verification_summary()
                verify_daily_signals.print_detailed_verification_report(settings_path) # [추가] 상세 리포트 출력 및 저장
            except ImportError:
                print("검증 모듈을 찾을 수 없습니다.")
        elif choice == '0':
            break
        else:
            print("잘못된 선택입니다.")

# -----------------------------------------------------------------------------
# Maintenance Menu
# -----------------------------------------------------------------------------
def _menu_maintenance(settings_path: str):
    """데이터 삭제 등 유지보수 메뉴"""
    while True:
        print("\n" + "="*60)
        print("      <<< 9. 유지보수 및 데이터 관리 (Maintenance) >>>")
        print("="*60)
        print("1. 특정 일자 데이터 일괄 삭제 (Rollback Date)")
        print("0. 이전 메뉴로")
        
        choice = get_user_input("선택: ")
        
        if choice == '1':
            target_date = get_user_input("삭제할 날짜 입력 (YYYY-MM-DD): ")
            confirm = get_user_input(f"정말로 {target_date} 데이터를 삭제하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                try:
                    from modules import data_manager
                    data_manager.delete_data_by_date(settings_path, target_date)
                except ImportError:
                    print("data_manager 모듈을 찾을 수 없습니다.")
            else:
                print("취소되었습니다.")
                
        elif choice == '0':
            break
        else:
            print("잘못된 선택입니다.")

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, default="config/settings.yaml")
    args = parser.parse_args()
    
    if not setup_system(args.settings):
        return

    while True:
        _print_header()
        print("1. 데이터 관리 (수집/생성/점검)")
        print("2. 모델 학습 및 ML 최적화")
        print("3. 전략 백테스트 / 최적화 / WF검증")
        print("4. 분석 메뉴 (Feature Analysis)")
        print("5. 유틸리티 (Utils)")
        print("6. 실전 검증 (Field Test)")
        print("9. 유지보수 및 데이터 관리 (Maintenance)")
        print("0. 종료")
        
        choice = get_user_input("\n메뉴를 선택하세요: ")
        
        if choice == '1': _menu_data_management(args.settings)
        elif choice == '2': _menu_training(args.settings)
        elif choice == '3': _menu_strategy(args.settings)
        elif choice == '4': _menu_analysis(args.settings)
        elif choice == '5': _menu_utils(args.settings) # Assuming _menu_utils exists or will be added
        elif choice == '6': _menu_field_test(args.settings)
        elif choice == '9': _menu_maintenance(args.settings)
        elif choice == '0': 
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다.")

if __name__ == "__main__":
    main()