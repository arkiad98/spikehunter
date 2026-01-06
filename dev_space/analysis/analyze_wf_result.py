import pandas as pd
import numpy as np
import os

# Target Directory
TARGET_DIR = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260101_134244"

def analyze_wf():
    print(f"Analyzing WF Results in: {TARGET_DIR}")
    
    # 1. Load Tradelog
    tradelog_path = os.path.join(TARGET_DIR, "wf_tradelog.parquet")
    if not os.path.exists(tradelog_path):
        tradelog_path = os.path.join(TARGET_DIR, "wf_tradelog.csv")
        if os.path.exists(tradelog_path):
            tradelog = pd.read_csv(tradelog_path)
            # convert dates
            if 'entry_date' in tradelog.columns: tradelog['entry_date'] = pd.to_datetime(tradelog['entry_date'])
            if 'exit_date' in tradelog.columns: tradelog['exit_date'] = pd.to_datetime(tradelog['exit_date'])
        else:
            print("Tradelog file not found.")
            return
    else:
        tradelog = pd.read_parquet(tradelog_path)

    # 2. Load Equity
    equity_path = os.path.join(TARGET_DIR, "wf_daily_equity.parquet")
    if not os.path.exists(equity_path):
        print("Equity file not found.")
        return
    equity = pd.read_parquet(equity_path)
    
    # --- Metrics Calculation ---
    
    # 1. Trade Metrics
    total_trades = len(tradelog)
    win_rate = (tradelog['return'] > 0).mean() * 100
    avg_return = tradelog['return'].mean() * 100
    
    # Profit Factor
    gross_win = tradelog[tradelog['return'] > 0]['return'].sum()
    gross_loss = abs(tradelog[tradelog['return'] <= 0]['return'].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0
    
    # 2. Equity Metrics
    initial_equity = equity['equity'].iloc[0]
    final_equity = equity['equity'].iloc[-1]
    total_return_pct = (final_equity / initial_equity - 1) * 100
    
    days = (equity.index.max() - equity.index.min()).days
    if days > 0:
        cagr = (final_equity / initial_equity) ** (365 / days) - 1
        cagr_pct = cagr * 100
    else:
        cagr_pct = 0.0
        
    # MDD
    roll_max = equity['equity'].cummax()
    drawdown = equity['equity'] / roll_max - 1.0
    mdd_pct = drawdown.min() * 100
    
    print("\n" + "="*50)
    print(f" [WF Analysis Result]")
    print("="*50)
    print(f" Period: {equity.index.min().date()} ~ {equity.index.max().date()} ({days} days)")
    print(f" Initial Equity:: {initial_equity:,.0f}")
    print(f" Final Equity  : {final_equity:,.0f}")
    print("-" * 30)
    print(f" CAGR          : {cagr_pct:.2f}%")
    print(f" Total Return  : {total_return_pct:.2f}%")
    print(f" MDD           : {mdd_pct:.2f}%")
    print("-" * 30)
    print(f" Total Trades  : {total_trades}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Avg Profit    : {avg_return:.2f}%")
    print(f" Profit Factor : {profit_factor:.2f}")
    print("="*50)

if __name__ == "__main__":
    analyze_wf()
