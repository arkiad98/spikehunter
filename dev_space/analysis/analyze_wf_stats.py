
import pandas as pd
import os
import sys

log_path = r"d:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20251230_090213\consolidated_tradelog.parquet"

def analyze():
    print(f"Reading: {log_path}")
    if not os.path.exists(log_path):
        print("File not found.")
        return

    df = pd.read_parquet(log_path)
    if df.empty:
        print("Tradelog is empty.")
        return
        
    # Ensure datetime
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date')
    
    # Group by 6-Month Periods
    df['Period'] = df['exit_date'].dt.to_period('6M')
    
    print("\n[Period-wise Performance Breakdown]")
    print(f"{'Period':<10} | {'Trades':<6} | {'Win Rate':<8} | {'Avg Return':<10} | {'Total Return (Simple)':<20}")
    print("-" * 70)
    
    overall_equity = 1.0
    
    for period, group in df.groupby('Period'):
        trades = len(group)
        wins = (group['return'] > 0).sum()
        win_rate = wins / trades if trades > 0 else 0.0
        avg_ret = group['return'].mean()
        
        # Simple cumulative return within period
        period_ret = (1 + group['return']).prod() - 1.0
        
        overall_equity *= (1 + period_ret)
        
        print(f"{str(period):<10} | {trades:<6} | {win_rate*100:6.2f}% | {avg_ret*100:8.2f}% | {period_ret*100:8.2f}%")

    print("-" * 70)
    print(f"Total Cumulative Return: {(overall_equity - 1)*100:.2f}% (Approx)")

if __name__ == "__main__":
    analyze()
