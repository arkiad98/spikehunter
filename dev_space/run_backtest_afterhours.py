import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm

def run_afterhours_backtest():
    print(">> Starting After-Hours (Close-to-Open/High) Backtest...")
    
    # 1. Load Settings
    model_path = "data/models/lgbm_model_2023.joblib"
    data_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    threshold = 0.40
    target_r = 0.10 # Reduced target for overnight gap capture
    stop_r = -0.05
    fee = 0.0015
    initial_cash = 10_000_000
    top_n = 5
    
    # 2. Load Model & Data
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = joblib.load(model_path)
    
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return
    df = pd.read_parquet(data_path)
    
    # Ensure columns
    req_cols = ['date', 'code', 'open', 'high', 'low', 'close']
    col_map = {c.lower(): c for c in df.columns}
    for r in req_cols:
        if r not in col_map and r.capitalize() in df.columns:
            df.rename(columns={r.capitalize(): r}, inplace=True)
            
    # Filter 2024
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2024-01-01'].sort_values('date').reset_index(drop=True)
    
    # 3. Predict (Score at T Close)
    print(">> Predicting...")
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None: return
    
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        for c in missing: df[c] = 0
        
    X = df[feature_names].fillna(0)
    df['score'] = model.predict_proba(X)[:, 1]
    
    # 4. Simulation (Buy T Close -> Sell T+1)
    print(">> Simulating...")
    cash = initial_cash
    portfolio = {} # code -> {entry_price, shares, entry_date}
    trade_log = []
    equity_curve = []
    
    dates = sorted(df['date'].unique())
    
    for date in tqdm(dates):
        daily_data = df[df['date'] == date].set_index('code')
        
        # 1. Sell (Positions from T-1)
        sold_codes = []
        for code, pos in portfolio.items():
            if code not in daily_data.index:
                # Delisted or missing? Force sell at prev close (entry price) if possible or 0
                # For simplicity, keep holding or sell at 0
                continue
                
            row = daily_data.loc[code]
            open_p = row['open']
            high = row['high']
            low = row['low']
            close = row['close']
            
            # Entry was T-1 Close.
            # We are now at T.
            # Check Gap (Open vs Entry)
            
            exit_price = None
            reason = ""
            
            # Strategy:
            # If Gap Up > Target: Sell at Open
            # Else: Watch High/Low
            
            target_price = pos['entry_price'] * (1 + target_r)
            stop_price = pos['entry_price'] * (1 + stop_r)
            
            if open_p >= target_price:
                exit_price = open_p
                reason = "GapTP"
            elif open_p <= stop_price:
                exit_price = open_p
                reason = "GapSL"
            else:
                # Intraday monitoring
                if high >= target_price:
                    exit_price = target_price
                    reason = "TP"
                elif low <= stop_price:
                    exit_price = stop_price
                    reason = "SL"
                else:
                    # Time Exit (1 Day Hold) - Sell at Close
                    exit_price = close
                    reason = "Time"
            
            if exit_price:
                ret = (exit_price - pos['entry_price']) / pos['entry_price'] - fee
                cash += exit_price * pos['shares'] * (1 - fee)
                trade_log.append({'date': date, 'code': code, 'return': ret, 'reason': reason})
                sold_codes.append(code)
                
        for code in sold_codes:
            del portfolio[code]
            
        # 2. Buy (At T Close)
        # Use Score(T)
        candidates = daily_data[daily_data['score'] >= threshold].sort_values('score', ascending=False)
        
        available_slots = top_n - len(portfolio)
        if available_slots > 0 and not candidates.empty:
            slot_cash = cash / available_slots if available_slots > 0 else 0
            
            for code, row in candidates.iterrows():
                if available_slots <= 0: break
                if code in portfolio: continue
                
                # Buy at Close (After-Hours Proxy)
                price = row['close']
                if price <= 0: continue
                
                shares = int(slot_cash // price)
                if shares > 0:
                    cost = shares * price * (1 + fee)
                    if cash >= cost:
                        cash -= cost
                        portfolio[code] = {'entry_price': price, 'shares': shares, 'entry_date': date}
                        available_slots -= 1
        
        # Equity Calc
        equity = cash
        for code, pos in portfolio.items():
            # Mark to Market (at Close)
            if code in daily_data.index:
                equity += pos['shares'] * daily_data.loc[code]['close']
            else:
                equity += pos['shares'] * pos['entry_price']
        
        equity_curve.append({'date': date, 'equity': equity})

    # Report
    final_equity = equity_curve[-1]['equity']
    cagr = (final_equity / initial_cash) ** (365 / (dates[-1] - dates[0]).days) - 1
    
    print("\n" + "="*40)
    print(f" [After-Hours Strategy Result]")
    print(f" Final Equity: {final_equity:,.0f} KRW")
    print(f" CAGR: {cagr*100:.2f}%")
    print(f" Total Trades: {len(trade_log)}")
    if trade_log:
        win_rate = len([t for t in trade_log if t['return'] > 0]) / len(trade_log)
        print(f" Win Rate: {win_rate*100:.2f}%")
        avg_ret = np.mean([t['return'] for t in trade_log])
        print(f" Avg Return: {avg_ret*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_afterhours_backtest()
