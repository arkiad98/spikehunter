import pandas as pd
import numpy as np
import os

path_wf = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260108_150918\period_8\result\tradelog.parquet"
path_run = r"D:\spikehunter\data\proc\backtest\run_20260108_155731\tradelog.parquet"

def analyze_trades(df, name):
    print(f"\n--- Analysis for: {name} ---")
    if df.empty:
        print("No trades in this period.")
        return
    
    # Filter Oct-Dec 2025
    if 'exit_date' in df.columns:
        df['date_obj'] = pd.to_datetime(df['exit_date'])
    elif 'date' in df.columns:
        df['date_obj'] = pd.to_datetime(df['date'])
    else:
        print(f"Error: No date column found. Cols: {df.columns}")
        return

    mask = (df['date_obj'] >= '2025-10-01') & (df['date_obj'] <= '2025-12-31')
    period_df = df[mask].copy()
    
    print(f"Trades in period (Oct-Dec 2025): {len(period_df)}")
    
    if len(period_df) == 0:
        return

    # Metrics
    # 'return' is the column name for profit rate (e.g., 0.05 for 5%)
    profit_col = 'return'
    
    wins = period_df[period_df[profit_col] > 0]
    win_rate = len(wins) / len(period_df) * 100
    avg_return = period_df[profit_col].mean() * 100
    total_return_simple = period_df[profit_col].sum() * 100
    md = period_df[profit_col].min() * 100
    mx = period_df[profit_col].max() * 100

    print(f"Trade Count: {len(period_df)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Return per Trade: {avg_return:.2f}%")
    print(f"Sum of Returns (Simple): {total_return_simple:.2f}%")
    print(f"Max Profit: {mx:.2f}%")
    print(f"Max Loss: {md:.2f}%")
    
    print("\nSample Trades (First 5):")
    cols_to_show = ['code', 'entry_date', 'exit_date', 'return', 'reason']
    print(period_df[cols_to_show].head().to_string())

print("Loading WF (Period 8)...")
try:
    df_wf = pd.read_parquet(path_wf)
    analyze_trades(df_wf, "WF Period 8")
except Exception as e:
    print(f"Error loading WF: {e}")

print("\nLoading New Run...")
try:
    df_run = pd.read_parquet(path_run)
    analyze_trades(df_run, "New Run (run_20260108_155731)")
except Exception as e:
    print(f"Error loading New Run: {e}")
