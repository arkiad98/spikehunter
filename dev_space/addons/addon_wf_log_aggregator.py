import os
import pandas as pd
import joblib
from tqdm import tqdm
import shutil

# [수정 1] get_user_input 함수를 추가로 import 합니다.
from modules.utils_io import read_yaml, to_date, get_user_input
from modules.utils_logger import logger
from modules.backtest import run_backtest

# [수정 2] '거래내역 분석기'의 폴더 선택 함수를 그대로 가져옵니다.
def _select_backtest_run(backtest_path: str) -> str:
    """사용자가 분석할 백테스트 실행 기록을 선택하도록 안내합니다."""
    if not os.path.exists(backtest_path):
        logger.warning(f"백테스트 결과 폴더({backtest_path})가 없습니다.")
        return None
        
    runs = sorted(
        [d for d in os.listdir(backtest_path) if os.path.isdir(os.path.join(backtest_path, d))], 
        reverse=True
    )
    if not runs:
        logger.warning("분석할 백테스트 실행 결과가 없습니다.")
        return None
    
    logger.info("\n분석할 Walk-Forward 실행 기록을 선택하세요:")
    for i, run_name in enumerate(runs[:15]): # 최근 15개만 표시
        print(f"  {i+1:2d}. {run_name}")
    
    try:
        choice_str = get_user_input("번호를 선택하세요: ")
        choice = int(choice_str) - 1
        if 0 <= choice < len(runs):
            return os.path.join(backtest_path, runs[choice])
    except (ValueError, IndexError):
        pass
    
    logger.error("잘못된 선택입니다.")
    return None

def run_aggregate_wf_logs(settings_path: str):
    """
    사용자가 선택한 Walk-Forward 테스트 결과를 재구성하여 전체 거래 기록을 집계합니다.
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< Walk-Forward 전체 거래 기록 집계기 시작 >>>")
    logger.info("="*80)
    
    base_cfg = read_yaml(settings_path)
    paths = base_cfg["paths"]
    strategy_name = "SpikeHunter"

    # [수정 3] 자동 탐색 대신 사용자 선택 방식으로 변경합니다.
    wf_run_dir = _select_backtest_run(paths["backtest"])
    if not wf_run_dir:
        return

    logger.info(f"분석 대상 폴더: {wf_run_dir}")

    # [수정 4] 선택된 폴더가 Walk-Forward 실행 폴더가 맞는지 검증합니다.
    temp_settings_path = os.path.join(wf_run_dir, 'temp_settings.yaml')
    temp_models_dir = os.path.join(wf_run_dir, 'temp_models')
    if not os.path.exists(temp_settings_path) or not os.path.exists(temp_models_dir):
        logger.error(f"선택하신 폴더는 Walk-Forward 실행 결과 폴더가 아닙니다 ('temp_settings.yaml' 또는 'temp_models' 없음).")
        return
        
    wf_period_cfg = read_yaml(temp_settings_path)

    # ... 이하 함수의 나머지 코드는 모두 동일합니다 ...
    # 2. Walk-Forward 기간 정보 계산
    wf_cfg = base_cfg.get("walk_forward", {})
    total_start_date = pd.Timestamp(wf_cfg.get("total_start_date", "2020-01-01"))
    train_months = wf_cfg.get("train_months", 24)
    test_months = wf_cfg.get("test_months", 6)
    total_end_date = pd.Timestamp.today().normalize()
    
    wf_periods = []
    current_date = total_start_date
    while current_date + pd.DateOffset(months=train_months + test_months) <= total_end_date:
        train_start = current_date
        train_end = current_date + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        wf_periods.append((train_start, train_end, test_start, test_end))
        current_date += pd.DateOffset(months=test_months)

    # 3. 각 기간별 백테스트만 다시 실행하여 거래 기록 수집
    all_trade_logs = []
    next_initial_cash = base_cfg.get("backtest_defaults", {}).get("initial_cash", 10000000.0)
    
    temp_backtest_dir = os.path.join(wf_run_dir, "temp_backtests_for_agg")
    os.makedirs(temp_backtest_dir, exist_ok=True)

    for i, (_, _, test_start, test_end) in enumerate(tqdm(wf_periods, desc="Aggregating Period Logs")):
        period_model_path = os.path.join(wf_run_dir, 'temp_models', f'period_{i+1}_temp_clf.joblib')
        if not os.path.exists(period_model_path):
            logger.warning(f"Period {i+1}의 모델 파일이 없어 해당 기간을 건너뜁니다.")
            continue
            
        wf_period_cfg['paths']['models'] = os.path.join(wf_run_dir, 'temp_models')
        
        period_run_dir = os.path.join(temp_backtest_dir, f"period_{i+1}")
        
        # run_backtest에 전달할 임시 설정 객체를 만듭니다.
        temp_cfg_for_run = wf_period_cfg.copy()
        # lgbm_model.joblib 대신 기간별 임시 모델을 사용하도록 모델 파일명을 덮어씁니다.
        # 이 정보는 run_backtest 함수 내부에서 사용됩니다.
        if 'lgbm_params_classification' not in temp_cfg_for_run:
            temp_cfg_for_run['lgbm_params_classification'] = {}
        temp_cfg_for_run['lgbm_params_classification']['model_filename_override'] = f'period_{i+1}_temp_clf.joblib'

        test_result = run_backtest(
            run_dir=period_run_dir, 
            strategy_name=strategy_name,
            settings_cfg=temp_cfg_for_run,
            start=str(test_start.date()), end=str(test_end.date()),
            quiet=True, use_ml_target=True,
            initial_cash=next_initial_cash, save_to_db=False
        )
        
        tradelog_path = test_result.get("tradelog_path") if test_result else None
        if tradelog_path and os.path.exists(tradelog_path):
            tradelog_df = pd.read_parquet(tradelog_path)
            if not tradelog_df.empty:
                all_trade_logs.append(tradelog_df)
        
        daily_equity_df = test_result.get("daily_equity") if test_result else None
        if daily_equity_df is not None and not daily_equity_df.empty:
            next_initial_cash = daily_equity_df['equity'].iloc[-1]

    # 4. 전체 거래 기록 통합 및 저장
    if not all_trade_logs:
        logger.error("집계할 거래 기록이 없습니다.")
        shutil.rmtree(temp_backtest_dir) # 임시 폴더 삭제
        return

    final_tradelog_df = pd.concat(all_trade_logs, ignore_index=True)
    output_path = os.path.join(wf_run_dir, "consolidated_tradelog.parquet")
    final_tradelog_df.to_parquet(output_path)
    
    logger.info("\n" + "="*80)
    logger.info("      <<< 전체 거래 기록 집계 완료 >>>")
    logger.info(f"총 {len(final_tradelog_df)}건의 거래 기록을 '{output_path}'에 저장했습니다.")
    logger.info("이제 '거래내역 상세 분석기'를 이 폴더에 대해 실행하여 전체 기록을 분석할 수 있습니다.")
    logger.info("="*80)
    
    shutil.rmtree(temp_backtest_dir)