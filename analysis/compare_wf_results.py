import pandas as pd
import numpy as np
import os

def load_data(base_path):
    tradelog_path = os.path.join(base_path, 'consolidated_tradelog.csv')
    equity_path = os.path.join(base_path, 'wf_daily_equity.parquet')
    
    try:
        tradelog = pd.read_csv(tradelog_path)
        equity = pd.read_parquet(equity_path)
    except Exception as e:
        print(f"Error loading data from {base_path}: {e}")
        return None, None
    
    return tradelog, equity

def calculate_metrics(tradelog, equity):
    if tradelog is None or equity is None:
        return {}
    
    # Trade Metrics
    num_trades = len(tradelog)
    win_trades = tradelog[tradelog['return'] > 0]
    loss_trades = tradelog[tradelog['return'] <= 0]
    
    win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
    avg_return = tradelog['return'].mean()
    
    gross_profit = win_trades['return'].sum()
    gross_loss = abs(loss_trades['return'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Equity Metrics
    if 'total_value' in equity.columns:
        equity_col = 'total_value'
    elif 'equity' in equity.columns:
        equity_col = 'equity'
    else:
        # Fallback: assume last column or check columns
        equity_col = equity.columns[-1] 
    
    # CAGR
    start_val = equity[equity_col].iloc[0]
    end_val = equity[equity_col].iloc[-1]
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0
    
    # MDD
    rolling_max = equity[equity_col].cummax()
    drawdown = (equity[equity_col] - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    total_return = (end_val - start_val) / start_val
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'MDD': mdd,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Num Trades': num_trades,
        'Avg Return': avg_return
    }

def main():
    path_before = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260109_125038"
    path_after = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260109_165509"
    
    print(f"Comparing:")
    print(f"Before: {os.path.basename(path_before)}")
    print(f"After : {os.path.basename(path_after)}")
    
    tl_before, eq_before = load_data(path_before)
    tl_after, eq_after = load_data(path_after)
    
    metrics_before = calculate_metrics(tl_before, eq_before)
    metrics_after = calculate_metrics(tl_after, eq_after)
    
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Before':<12} | {'After':<12} | {'Diff':<12}")
    print("-" * 50)
    
    metrics_list = ['Total Return', 'CAGR', 'MDD', 'Win Rate', 'Profit Factor', 'Num Trades', 'Avg Return']
    
    for m in metrics_list:
        val_b = metrics_before.get(m, 0)
        val_a = metrics_after.get(m, 0)
        
        diff = val_a - val_b
        
        # Formatting
        if m in ['Num Trades']:
            fmt = "{:.0f}"
            diff_fmt = "{:+.0f}"
        elif m in ['Profit Factor']:
            fmt = "{:.4f}"
            diff_fmt = "{:+.4f}"
        else:
            fmt = "{:.2%}"
            diff_fmt = "{:+.2%}"
            
        val_b_str = fmt.format(val_b)
        val_a_str = fmt.format(val_a)
        diff_str = diff_fmt.format(diff)
        
        print(f"{m:<20} | {val_b_str:<12} | {val_a_str:<12} | {diff_str:<12}")
    print("="*50)

if __name__ == "__main__":
    main()
