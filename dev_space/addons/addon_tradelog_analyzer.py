# addons/addon_tradelog_analyzer.py

"""
특정 백테스트 실행 결과의 거래내역(tradelog)을 심층 분석하는 애드온 모듈.
"""
import os
import pandas as pd
from tabulate import tabulate
from modules.utils_io import read_yaml, get_stock_names, get_user_input
from modules.utils_logger import logger

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
    
    logger.info("\n분석할 백테스트 실행 기록을 선택하세요:")
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

def run_tradelog_analysis(settings_path: str):
    """거래내역 분석 파이프라인을 실행하고, 결과를 엑셀 파일로 저장하는 옵션을 제공합니다."""
    logger.info("\n" + "="*60)
    logger.info("      <<< 백테스트 거래내역 상세 분석기 >>>")
    logger.info("="*60)
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    selected_run_path = _select_backtest_run(paths["backtest"])
    if not selected_run_path:
        return

    # --- [핵심 수정] ---
    # 1. Walk-Forward 전체 로그 파일 경로를 먼저 확인합니다.
    consolidated_log_path = os.path.join(selected_run_path, "consolidated_tradelog.parquet")
    
    # 2. 없다면, 기존의 단일 실행 로그 파일 경로를 확인합니다.
    standard_log_path = os.path.join(selected_run_path, "tradelog.parquet")

    tradelog_file = None
    if os.path.exists(consolidated_log_path):
        tradelog_file = consolidated_log_path
        logger.info("Walk-Forward 전체 통합 거래 기록을 분석합니다.")
    elif os.path.exists(standard_log_path):
        tradelog_file = standard_log_path
        logger.info("단일 실행 거래 기록을 분석합니다.")
    else:
        logger.error(f"선택한 폴더에 분석할 거래내역 파일('consolidated_tradelog.parquet' 또는 'tradelog.parquet')이 없습니다.")
        return
    # --- [수정 완료] ---

    logger.info(f"\n'{tradelog_file}' 파일을 로드하여 분석을 시작합니다...")
    df = pd.read_parquet(tradelog_file)
    
    if df.empty:
        logger.warning("거래내역이 비어있습니다. 분석을 종료합니다.")
        return

    # 데이터 보강
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['holding_period'] = (df['exit_date'] - df['entry_date']).dt.days
    stock_names = get_stock_names(df['code'].unique().tolist())
    df['name'] = df['code'].map(stock_names)

    # --- 1. 종합 요약 ---
    total_trades = len(df)
    win_rate = (df['return'] > 0).sum() / total_trades if total_trades > 0 else 0
    profit = df[df['return'] > 0]['return'].sum()
    loss = abs(df[df['return'] <= 0]['return'].sum())
    profit_factor = profit / loss if loss > 0 else float('inf')
    avg_return = df['return'].mean()
    avg_hold = df['holding_period'].mean()

    summary_data = [
        ["총 거래 횟수", f"{total_trades} 회"], ["승률", f"{win_rate:.2%}"],
        ["평균 수익률", f"{avg_return:.2%}"], ["수익/손실 비율 (Profit Factor)", f"{profit_factor:.2f}"],
        ["평균 보유 기간", f"{avg_hold:.2f} 일"],
    ]
    summary_df = pd.DataFrame(summary_data, columns=["지표", "값"])
    logger.info("\n" + "="*60 + "\n          <<< 거래내역 종합 요약 >>>\n" + "="*60)
    print(tabulate(summary_data, tablefmt="pretty"))

    # --- 2. 종료 사유별 성과 분석 ---
    by_reason_df = df.groupby('reason')['return'].agg(['count', 'mean', 'sum']).sort_values('count', ascending=False)
    by_reason_df.columns = ["거래 횟수", "평균 수익률", "누적 수익률"]
    logger.info("\n" + "="*60 + "\n        <<< 종료 사유별 성과 분석 >>>\n" + "="*60)
    print(tabulate(by_reason_df, headers='keys', tablefmt="grid"))

    # --- 3. Best / Worst Trades ---
    display_cols = ['name', 'code', 'entry_date', 'exit_date', 'holding_period', 'return', 'reason']
    best_trades_df = df.sort_values('return', ascending=False).head(5)[display_cols]
    worst_trades_df = df.sort_values('return', ascending=True).head(5)[display_cols]
    
    for df_slice, title in [(best_trades_df, "수익률 상위 5개 거래"), (worst_trades_df, "수익률 하위 5개 거래")]:
        if df_slice.empty: continue
        logger.info("\n" + "="*80 + f"\n        <<< {title} >>>\n" + "="*80)
        print(tabulate(df_slice, headers='keys', tablefmt="grid", showindex=False, floatfmt=".2%"))

    # --- [추가] 4. 파일로 저장 ---
    save_choice = get_user_input("\n분석 결과를 엑셀 파일로 저장하시겠습니까? (y/N): ").lower()
    if save_choice == 'y':
        run_name = os.path.basename(selected_run_path)
        output_filename = f"analysis_tradelog_{run_name}.xlsx"
        output_path = os.path.join(selected_run_path, output_filename)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                by_reason_df.to_excel(writer, sheet_name='By_Exit_Reason')
                best_trades_df.to_excel(writer, sheet_name='Best_Trades', index=False)
                worst_trades_df.to_excel(writer, sheet_name='Worst_Trades', index=False)
                df.to_excel(writer, sheet_name='Full_Tradelog', index=False)
            logger.info(f"분석 결과가 '{output_path}'에 성공적으로 저장되었습니다.")
        except Exception as e:
            logger.error(f"엑셀 파일 저장 중 오류 발생: {e}")

    logger.info("\n분석이 완료되었습니다.")