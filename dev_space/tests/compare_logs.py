
import pandas as pd
import os
import yaml

wf_path = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260102_130851\wf_tradelog.csv"
std_path = r"D:\spikehunter\data\proc\backtest\run_20260102_133019\tradelog.parquet"
wf_config_path = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260102_130851\temp_settings.yaml"

def analyze_log(df, name):
    print(f"--- Analysis for {name} ---")
    if df.empty:
        print("Empty DataFrame")
        return
    
    # Ensure datetime
    if 'exit_time' in df.columns:
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df = df.sort_values('exit_time')
    
    total_trades = len(df)
    
    # Check for profit column
    if 'profit_rate' in df.columns:
        avg_profit = df['profit_rate'].mean()
        win_rate = (df['profit_rate'] > 0).mean() * 100
        # Cumulative return (simple sum for now or compound if equity is tracked)
        # Assuming simple compounding for estimation: (1+r).prod() - 1
        cum_return = (1 + df['profit_rate']).prod() - 1
        
        # Max Drawdown estimation (on equity curve)
        df['equity_curve'] = (1 + df['profit_rate']).cumprod()
        df['peak'] = df['equity_curve'].cummax()
        df['drawdown'] = (df['equity_curve'] - df['peak']) / df['peak']
        mdd = df['drawdown'].min() * 100
        
        start_date = df['exit_time'].min() if 'exit_time' in df.columns else "N/A"
        end_date = df['exit_time'].max() if 'exit_time' in df.columns else "N/A"
        
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Avg Profit: {avg_profit:.4f}")
        print(f"Cumulative Return: {cum_return:.4f} ({cum_return*100:.2f}%)")
        print(f"MDD: {mdd:.2f}%")
    else:
        print("Column 'profit_rate' not found.")
    
    print("\n")

try:
    print("Reading WF Log (CSV)...")
    df_wf = pd.read_csv(wf_path)
    analyze_log(df_wf, "Walk-Forward")
    
    print("Reading Standard Log (Parquet)...")
    df_std = pd.read_parquet(std_path)
    analyze_log(df_std, "Standard Run")
    
    # Compare Settings if possible
    if os.path.exists(wf_config_path):
        with open(wf_config_path, 'r', encoding='utf-8') as f:
             wf_config = yaml.safe_load(f)
             print("WF Config snippet (backtest):")
             if 'backtest' in wf_config:
                 print(wf_config['backtest'])
             else:
                 print("No 'backtest' key in WF config")

except Exception as e:
    print(f"Error: {e}")
