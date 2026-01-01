# modules/backtest.py (v4.0 - SpikeHunter Final)
"""
[SpikeHunter v4.0] Final Strategy Backtest Module
- Strategy: LightGBM Score > 0.40 -> Buy at T Close -> Hold Max 5 Days
- Logic:
  1. Entry: T Day Close (After-Hours/Closing Auction)
  2. Exit: Target (+10%), Stop (-5%), or Time (5 Days)
  3. Gap Protection: Immediate exit if Open Gap hits TP/SL
"""

import os
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from typing import Optional
import lightgbm as lgb

from .utils_io import read_yaml, to_date, ensure_dir, load_index_data
from .utils_logger import logger
from . import utils_db

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def _calculate_full_metrics(equity: pd.DataFrame, tradelog: pd.DataFrame) -> dict:
    """Calculate CAGR, MDD, Sharpe, Win Rate"""
    if isinstance(equity, pd.Series): equity = equity.to_frame('equity')
    eq_series = equity['equity'].dropna()
    
    if len(eq_series) < 2: 
        return {"CAGR_raw": 0.0, "Sharpe_raw": 0.0, "MDD_raw": 0.0, "총거래횟수": 0, "win_rate_raw": 0.0}
    
    full_idx = pd.date_range(start=eq_series.index.min(), end=eq_series.index.max(), freq='D')
    daily_eq = eq_series.reindex(full_idx).ffill()
    
    days = (daily_eq.index[-1] - daily_eq.index[0]).days
    cagr = (daily_eq.iloc[-1] / daily_eq.iloc[0]) ** (365 / days) - 1 if days > 0 else 0.0
    
    dd = daily_eq / daily_eq.cummax() - 1.0
    mdd = dd.min()
    
    ret = daily_eq.pct_change().fillna(0.0)
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0
    
    total_trades = len(tradelog)
    win_rate = (tradelog['return'] > 0).sum() / total_trades if total_trades > 0 else 0.0
    
    metrics = {
        "CAGR_raw": cagr, 
        "Sharpe_raw": sharpe, 
        "MDD_raw": mdd, 
        "총거래횟수": total_trades, 
        "win_rate_raw": win_rate
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in metrics.items()}

def run_backtest(run_dir: str, strategy_name: str, settings_path: str = None, settings_cfg: dict = None,
                 start: str = None, end: str = None, initial_cash: Optional[float] = None,
                 param_overrides: Optional[dict] = None, quiet: bool = False, use_ml_target: bool = False,
                 trial_number: int = None, temp_model: Optional[lgb.LGBMRegressor] = None,
                 preloaded_features: Optional[pd.DataFrame] = None, save_to_db: bool = True,
                 skip_exclusion: bool = False):
    
    # 1. Load Settings
    if settings_cfg: cfg = settings_cfg
    elif settings_path: cfg = read_yaml(settings_path)
    else: return None

    paths = cfg["paths"]
    fee = float(cfg.get("fee_rate", 0.0015))
    cash = initial_cash if initial_cash is not None else 10000000.0
    
    # Strategy Params (Defaults from v4.0 spec)
    top_n = int(cfg.get("top_n", 5))
    ml_params = cfg.get("ml_params", {})
    threshold = ml_params.get("classification_threshold", 0.40)
    
    # [수정] 전략/ML 파라미터 우선순위 적용 (하드코딩 제거)
    target_r = ml_params.get('target_surge_rate', 0.10)
    stop_r = ml_params.get('stop_loss_rate', -0.07)
    max_hold = ml_params.get('target_hold_period', 5)

    # Strategy override if exists
    if strategy_name in cfg.get('strategies', {}):
        st_cfg = cfg['strategies'][strategy_name]
        target_r = st_cfg.get('target_r', target_r)
        stop_r = st_cfg.get('stop_r', stop_r)
        stop_r = st_cfg.get('stop_r', stop_r)
        max_hold = st_cfg.get('max_hold', max_hold)
    
    # [Fix] Apply Optimization Overrides
    if param_overrides:
        target_r = param_overrides.get('target_r', target_r)
        stop_r = param_overrides.get('stop_r', stop_r)
        max_hold = param_overrides.get('max_hold', max_hold)
        threshold = param_overrides.get('min_ml_score', threshold) # Map min_ml_score to threshold
    
    # Load Model (Lazy Load)
    # [Optimization Speedup] Skip loading model if ml_score is already pre-calculated
    is_opt_mode = preloaded_features is not None
    has_precalc_score = is_opt_mode and 'ml_score' in preloaded_features.columns
    
    model_clf_path = os.path.join(paths["models"], "lgbm_model.joblib")
    # Only load model if we don't have scores
    model_clf = None
    if not has_precalc_score and os.path.exists(model_clf_path):
        model_clf = joblib.load(model_clf_path)
    
    ensure_dir(run_dir)
    is_opt_mode = preloaded_features is not None
    
    if not quiet: print(f"\n[SpikeHunter v4.0] Backtest Start (Close Entry + 5-Day Hold)")
    
    # 2. Prepare Data
    start_d = to_date(start) if start else pd.Timestamp("2020-01-01")
    end_d = to_date(end) if end else pd.Timestamp.today()
    
    if is_opt_mode: 
        all_features = preloaded_features
    else:
        dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
        # Fallback to ml_dataset if dataset_v4 not found (as seen in dev_space)
        if not os.path.exists(dataset_path):
             dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
        
        if not os.path.exists(dataset_path):
            logger.error(f"Data not found: {dataset_path}")
            return None
            
        all_features = pd.read_parquet(dataset_path)
        all_features['date'] = pd.to_datetime(all_features['date'])
        mask = (all_features['date'] >= start_d) & (all_features['date'] <= end_d)
        all_features = all_features.loc[mask].copy()

    if all_features.empty: return None

    # [NEW] Apply Date-Specific Exclusion
    # [NEW] Apply Date-Specific Exclusion
    if not skip_exclusion:
        exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
        if os.path.exists(exclude_path):
            # from modules.utils_io import read_yaml # [Removed] Shadowing fix
            ex_cfg = read_yaml(exclude_path)
            exclusions = ex_cfg.get('exclusions', [])
            
            if exclusions:
                original_len = len(all_features)
                
                # Efficient filtering using set of (code, date)
                exclude_set = set()
                for item in exclusions:
                    c = item['code']
                    for d in item['dates']:
                        exclude_set.add((c, pd.Timestamp(d).date()))
                
                # Create excluded dataframe
                exclude_records = []
                for c, d in exclude_set:
                    exclude_records.append({'code': c, 'date': pd.Timestamp(d)})
                
                if exclude_records:
                    exclude_df = pd.DataFrame(exclude_records)
                    exclude_df['exclude'] = True
                    
                    # Normalize date format
                    exclude_df['date'] = pd.to_datetime(exclude_df['date'])
                    
                    # Merge
                    all_features = all_features.merge(exclude_df, on=['code', 'date'], how='left')
                    all_features = all_features[all_features['exclude'].isna()].drop(columns=['exclude'])
                
                filtered_len = len(all_features)
                if original_len > filtered_len:
                    msg = f" >> Anomaly Exclusion applied: Removed {original_len - filtered_len} rows."
                    if not quiet: print(msg)
                    logger.info(msg)

    if all_features.empty: return None
    
    # Normalize columns
    req_cols = ['open', 'high', 'low', 'close']
    col_map = {c.lower(): c for c in all_features.columns}
    for r in req_cols:
        if r not in col_map and r.capitalize() in all_features.columns:
            all_features.rename(columns={r.capitalize(): r}, inplace=True)
            
    all_dates = pd.Series(pd.to_datetime(all_features['date'].unique())).sort_values().reset_index(drop=True)
    
    # 3. Predict (Batch Prediction for Speed)
    if 'ml_score' in all_features.columns:
        # [Optimization Speedup] Use pre-calculated scores
        pass
    elif model_clf:
        feature_names = getattr(model_clf, 'feature_names_in_', None)
        if feature_names is not None:
            # Ensure features exist
            missing = [c for c in feature_names if c not in all_features.columns]
            if missing:
                for c in missing: all_features[c] = 0
            
            X = all_features[feature_names].fillna(0)
            all_features['ml_score'] = model_clf.predict_proba(X)[:, 1]
        else:
            all_features['ml_score'] = 0
    else:
        all_features['ml_score'] = 0

    # 4. Simulation Loop
    portfolio = {} # code -> {entry_price, shares, entry_date}
    daily_equity = []
    trade_log = []
    
    iterator = tqdm(all_dates, desc=f"Simulating", disable=quiet)
    
    for date in iterator:
        today_data = all_features[all_features['date'] == date].set_index('code')
        
        # --- Sell Logic ---
        sold_codes = []
        for code, pos in portfolio.items():
            if code not in today_data.index: continue
            
            row = today_data.loc[code]
            open_p = row['open']
            high = row['high']
            low = row['low']
            close = row['close']
            
            exit_price = None
            reason = ""
            
            target_price = pos['entry_price'] * (1 + target_r)
            stop_price = pos['entry_price'] * (1 + stop_r)
            
            # Gap Check
            # Gap Check
            if open_p >= target_price:
                exit_price = open_p
                reason = "GapTP"
            elif open_p <= stop_price:
                exit_price = open_p
                reason = "GapSL"
            else:
                # Intraday
                if high >= target_price:
                    exit_price = target_price
                    reason = "TP"
                elif low <= stop_price:
                    exit_price = stop_price
                    reason = "SL"
                elif (date - pos['entry_date']).days >= max_hold:
                    exit_price = close
                    reason = "Time"
            
            if exit_price:
                ret = (exit_price - pos['entry_price']) / pos['entry_price'] - fee
                cash += exit_price * pos['shares'] * (1 - fee)
                trade_log.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'code': code,
                    'return': ret,
                    'reason': reason
                })
                sold_codes.append(code)
                
        for code in sold_codes:
            del portfolio[code]
            
        # --- Buy Logic (T Close) ---
        candidates = today_data[today_data['ml_score'] >= threshold].sort_values('ml_score', ascending=False)
        
        available_slots = top_n - len(portfolio)
        if available_slots > 0 and not candidates.empty:
            slot_cash = cash / available_slots if available_slots > 0 else 0
            
            for code, row in candidates.iterrows():
                if available_slots <= 0: break
                if code in portfolio: continue
                
                price = row['close']
                if price <= 0: continue
                
                shares = int(slot_cash // price)
                if shares > 0:
                    cost = shares * price * (1 + fee)
                    if cash >= cost:
                        cash -= cost
                        portfolio[code] = {'entry_price': price, 'shares': shares, 'entry_date': date}
                        available_slots -= 1
        
        # --- Equity Calculation ---
        current_equity = cash
        for code, pos in portfolio.items():
            if code in today_data.index:
                current_equity += pos['shares'] * today_data.loc[code]['close']
            else:
                current_equity += pos['shares'] * pos['entry_price']
        
        daily_equity.append({'date': date, 'equity': current_equity})

    # [Fix] Force Close at End (to show recent trades in log)
    if portfolio and not all_features.empty:
        last_date = all_features['date'].iloc[-1]
        last_close_map = all_features[all_features['date'] == last_date].set_index('code')['close'].to_dict()
        
        for code, pos in portfolio.items():
            exit_price = last_close_map.get(code, pos['entry_price']) # Fallback to entry if price missing
            ret = (exit_price - pos['entry_price']) / pos['entry_price'] - fee
            
            trade_log.append({
                'entry_date': pos['entry_date'],
                'exit_date': last_date,
                'code': code,
                'return': ret,
                'reason': "End"
            })
    
    # 5. Finalize
    if not daily_equity: return None
    equity_df = pd.DataFrame(daily_equity).set_index('date')
    tradelog_df = pd.DataFrame(trade_log)
    metrics = _calculate_full_metrics(equity_df, tradelog_df)
    
    equity_path = os.path.join(run_dir, "daily_equity.parquet")
    tradelog_path = os.path.join(run_dir, "tradelog.parquet")
    equity_df.to_parquet(equity_path)
    if not tradelog_df.empty: tradelog_df.to_parquet(tradelog_path)

    if save_to_db and not is_opt_mode:
        bid = f"{strategy_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # Pass dummy params for compatibility
        dummy_params = {"threshold": threshold, "hold": max_hold}
        utils_db.insert_backtest_results(bid, metrics, dummy_params, tradelog_df, equity_path, 
                                         start_d.strftime('%Y-%m-%d'), end_d.strftime('%Y-%m-%d'), strategy_name)

    if not quiet:
        print("\n" + "="*60)
        print(f"   [백테스트 결과] 최종 자산: {int(equity_df['equity'].iloc[-1]):,} 원")
        print(f"   - 수익률(CAGR): {metrics['CAGR_raw']*100:.2f}%")
        print(f"   - MDD: {metrics['MDD_raw']*100:.2f}%")
        print(f"   - 승률: {metrics['win_rate_raw']*100:.2f}% ({metrics['총거래횟수']}회)")
        print("="*60)

    return {"metrics": metrics, "daily_equity": equity_df, "tradelog_path": tradelog_path}