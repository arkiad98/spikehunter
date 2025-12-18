import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm

def run_simple_backtest():
    print(">> Starting Simplified Backtest...")
    
    # 1. Load Settings & Parameters
    model_path = "data/models/lgbm_model_2023.joblib"
    data_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    threshold = 0.40
    target_r = 0.13
    stop_r = -0.07
    fee = 0.0015
    initial_cash = 10_000_000
    top_n = 5
    
    # 2. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = joblib.load(model_path)
    print(f">> Model loaded: {type(model)}")
    
    # 3. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return
    df = pd.read_parquet(data_path)
    print(f">> Data loaded: {len(df)} rows")
    
    # Ensure columns
    req_cols = ['date', 'code', 'open', 'high', 'low', 'close']
    # Map columns if needed (case insensitive)
    col_map = {c.lower(): c for c in df.columns}
    for r in req_cols:
        if r not in col_map:
            # Try Capitalized
            if r.capitalize() in df.columns:
                df.rename(columns={r.capitalize(): r}, inplace=True)
            elif r.upper() in df.columns:
                df.rename(columns={r.upper(): r}, inplace=True)
    
    # Filter for 2024
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2024-01-01'].sort_values('date').reset_index(drop=True)
    print(f">> 2024 Data: {len(df)} rows")
    
    # 4. Predict
    print(">> Predicting...")
    # Prepare features
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        print("Error: Model missing feature_names_in_")
        return
        
    # Check missing columns
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing}, filling with 0")
        for c in missing: df[c] = 0
        
    X = df[feature_names].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    df['score'] = probs
    
    # [Fix Lookahead Bias]
    # Model predicts at T Close for T+1 Entry.
    # So for Date T, we must use Score from T-1.
    df['prev_score'] = df.groupby('code')['score'].shift(1)
    
    # 5. Simulation
    print(">> Simulating...")
    cash = initial_cash
    portfolio = {} # code -> {entry_price, shares, entry_date}
    trade_log = []
    equity_curve = []
    
    dates = df['date'].unique()
    for date in tqdm(dates):
        # Update Portfolio (Sell)
        daily_data = df[df['date'] == date].set_index('code')
        
        # Sell Logic
        sold_codes = []
        for code, pos in portfolio.items():
            if code not in daily_data.index: continue
            
            row = daily_data.loc[code]
            high = row['high']
            low = row['low']
            close = row['close']
            open_p = row['open']
            
            exit_price = None
            reason = ""
            
            # Target
            target_price = pos['entry_price'] * (1 + target_r)
            stop_price = pos['entry_price'] * (1 + stop_r)
            
            if high >= target_price:
                exit_price = target_price
                # Gap check
                if open_p > target_price: exit_price = open_p
                reason = "TP"
            elif low <= stop_price:
                exit_price = stop_price
                if open_p < stop_price: exit_price = open_p
                reason = "SL"
            elif (date - pos['entry_date']).days >= 5: # Max hold 5
                exit_price = close
                reason = "Time"
                
            if exit_price:
                ret = (exit_price - pos['entry_price']) / pos['entry_price'] - fee
                cash += exit_price * pos['shares'] * (1 - fee)
                trade_log.append({'date': date, 'code': code, 'return': ret, 'reason': reason})
                sold_codes.append(code)
                
        for code in sold_codes:
            del portfolio[code]
            
        # Buy Logic
        # Use prev_score (T-1) to buy at T Open
        candidates = daily_data[daily_data['prev_score'] >= threshold].sort_values('prev_score', ascending=False)
        
        # Gap Protection: Skip if Open > PrevClose * 1.07 (Approximation using today's open vs yesterday close not avail here easily without shift. 
        # Simplified: Skip if Open is very high relative to something? 
        # Actually, let's just trust the score for now or implement simple gap check if we had prev close.
        # We can assume 'open' is the entry price.)
        
        available_slots = top_n - len(portfolio)
        if available_slots > 0 and not candidates.empty:
            slot_cash = cash / available_slots if available_slots > 0 else 0
            # Limit slot cash to 1/top_n of total equity to avoid compounding too fast on one? 
            # Simple: cash / available_slots
            
            for code, row in candidates.iterrows():
                if available_slots <= 0: break
                if code in portfolio: continue
                
                # Simple Gap Check (Intraday proxy: if Open is > 20% of Low? No, that's volatility. 
                # Without prev close, we can't do exact gap check. 
                # But we can check if Open is not Ceiling (30%). 
                # Let's proceed without strict gap check for this simplified run, relying on the model.)
                
                price = row['open']
                if price <= 0: continue
                
                shares = int(slot_cash // price)
                if shares > 0:
                    cost = shares * price * (1 + fee)
                    if cash >= cost:
                        cash -= cost
                        portfolio[code] = {'entry_price': price, 'shares': shares, 'entry_date': date}
                        available_slots -= 1
                        
        # Calculate Equity
        equity = cash
        for code, pos in portfolio.items():
            if code in daily_data.index:
                equity += pos['shares'] * daily_data.loc[code]['close']
            else:
                equity += pos['shares'] * pos['entry_price'] # Fallback
        
        equity_curve.append({'date': date, 'equity': equity})

    # 6. Report
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    final_equity = equity_df['equity'].iloc[-1]
    cagr = (final_equity / initial_cash) ** (365 / (equity_df.index[-1] - equity_df.index[0]).days) - 1
    
    print("\n" + "="*40)
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
    run_simple_backtest()
