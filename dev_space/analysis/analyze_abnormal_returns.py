
import pandas as pd
import os
import glob

def analyze_abnormal():
    # Find latest run
    base_dir = r"data/proc/backtest"
    runs = glob.glob(os.path.join(base_dir, "run_*"))
    latest_run = max(runs, key=os.path.getctime)
    
    print(f"Analyzing Run: {latest_run}")
    
    log_path = os.path.join(latest_run, "tradelog.parquet")
    if not os.path.exists(log_path):
        print("Trade log not found.")
        return

    df = pd.read_parquet(log_path)
    
    print(f"Total Trades: {len(df)}")
    print(f"Avg Return: {df['return'].mean()*100:.2f}%")
    print(f"Max Return: {df['return'].max()*100:.2f}%")
    print(f"Min Return: {df['return'].min()*100:.2f}%")
    
    # Check Outliers
    outliers = df[df['return'] > 0.20] # > 20%
    if not outliers.empty:
        print(f"\n[Outliers > 20%]: {len(outliers)}")
        print(outliers[['code', 'entry_date', 'exit_date', 'return', 'reason']].head())
        
    # Check Holding Period
    df['hold_days'] = (pd.to_datetime(df['exit_date']) - pd.to_datetime(df['entry_date'])).dt.days
    print(f"\nAvg Hold Days: {df['hold_days'].mean():.1f}")
    
    # Check Compounding
    init_cash = 10000000
    curr = init_cash
    
    # Simple compounding check
    # Note: Backtester simulates portfolio management (slots).
    # If Avg return is 2% and we trade 500 times with full leverage, 
    # (1.02)^500 is huge.
    # Check 'Turnover' or 'Exposure'.
    
    pos_returns = df[df['return'] > 0]['return']
    neg_returns = df[df['return'] <= 0]['return']
    print(f"Win/Loss Ratio: {len(pos_returns)} / {len(neg_returns)}")
    print(f"Avg Win: {pos_returns.mean()*100:.2f}%")
    print(f"Avg Loss: {neg_returns.mean()*100:.2f}%")

if __name__ == "__main__":
    analyze_abnormal()
