# modules/backtest.py (v5.0 - Clean & Gap Protected)
"""
[SpikeHunter v5.0] 스마트 머니 백테스트 (Gap Protection & Intraday Exit)
- 전략: T일 장 마감 후 분석 -> T+1일 시가(Open) 진입 -> 급등 시 즉시 매도
- 핵심 변경:
  1. Gap Filter: 시가 갭 7% 이상 시 진입 금지 (추격 매수 방지)
  2. Intraday Exit: 당일 매수 후 당일 매도 허용
  3. Smart Money: 수급(MFI, OBV) 및 유동성 필터 적용
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import lightgbm as lgb

from .utils_io import read_yaml, to_date, ensure_dir, load_index_data
from .utils_logger import logger
from . import utils_db

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# -----------------------------------------------------------------------------
# 1. Helper Functions (지표 계산 및 국면 판단)
# -----------------------------------------------------------------------------

def _determine_regime(current_val, ma200, vol_20, vol_threshold=0.015) -> str:
    """시장 국면(Regime)을 4단계로 분류"""
    if pd.isna(current_val) or pd.isna(ma200): return "R1_BullStable"
    
    is_bull = current_val > ma200
    is_stable = vol_20 < vol_threshold if pd.notna(vol_20) else True
    
    if is_bull and is_stable: return "R1_BullStable"
    elif is_bull and not is_stable: return "R2_BullVolatile"
    elif not is_bull and is_stable: return "R3_BearStable"
    else: return "R4_BearVolatile"

def _calculate_full_metrics(equity: pd.DataFrame, tradelog: pd.DataFrame) -> dict:
    """수익률, MDD, 샤프지수 등 성과 지표 계산"""
    if isinstance(equity, pd.Series): equity = equity.to_frame('equity')
    eq_series = equity['equity'].dropna()
    
    if len(eq_series) < 2: 
        return {"CAGR_raw": 0.0, "Sharpe_raw": 0.0, "MDD_raw": 0.0, "총거래횟수": 0, "win_rate_raw": 0.0}
    
    # 일별 데이터로 리샘플링 (빈 날짜 채움)
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
    
    # numpy 타입을 python native 타입으로 변환 (DB 저장 호환성)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in metrics.items()}

# -----------------------------------------------------------------------------
# 2. Strategy Classes (전략 로직)
# -----------------------------------------------------------------------------

class Strategy(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.name = params.get("name", "UnnamedStrategy")
        self.vbo_k = params.get('vbo_k', 0.5)           # 변동성 돌파 계수 (0이면 시가 매수)
        self.target_r = params.get('target_r', 0.10)    # 목표 수익률
        self.stop_r = params.get('stop_r', -0.05)       # 손절률
        self.max_hold = int(params.get('max_hold', 3))  # 최대 보유일
        self.min_ml_score = params.get('min_ml_score', 0.7) # ML 점수 임계값
        self.top_n = int(params.get('top_n', 5))        # 일별 최대 매수 종목 수
        
        # 수급 필터
        self.min_mfi = params.get('min_mfi', 50.0)
        self.min_obv_slope = params.get('min_obv_slope', 0.0)
        self.min_avg_value = params.get('min_avg_value', 1_000_000_000)

    @abstractmethod
    def generate_recs(self, df_feat: pd.DataFrame, model_clf: Optional[lgb.LGBMClassifier] = None) -> pd.DataFrame:
        pass

class SpikeHunterStrategy(Strategy):
    def __init__(self, params: dict):
        super().__init__(params)

    def generate_recs(self, df_feat: pd.DataFrame, model_clf: Optional[lgb.LGBMClassifier] = None) -> pd.DataFrame:
        # 유동성(거래대금) 필터
        if "amount_ma5" not in df_feat.columns:
            if "amount_bil" in df_feat.columns:
                df_feat["amount_ma5"] = df_feat["amount_bil"].rolling(5).mean()
            elif "close" in df_feat.columns and "volume" in df_feat.columns:
                df_feat["amount_ma5"] = (df_feat['close'] * df_feat['volume'] / 100000000).rolling(5).mean()
# -----------------------------------------------------------------------------
# 3. Main Backtest Loop (핵심 실행 로직)
# -----------------------------------------------------------------------------

def run_backtest(run_dir: str, strategy_name: str, settings_path: str = None, settings_cfg: dict = None,
                 start: str = None, end: str = None, initial_cash: Optional[float] = None,
                 param_overrides: Optional[dict] = None, quiet: bool = False, use_ml_target: bool = False,
                 trial_number: int = None, temp_model: Optional[lgb.LGBMRegressor] = None,
                 preloaded_features: Optional[pd.DataFrame] = None, save_to_db: bool = True):
    
    # 1. 설정 로드
    if settings_cfg: cfg = settings_cfg
    elif settings_path: cfg = read_yaml(settings_path)
    else: return None

    paths = cfg["paths"]
    fee = float(cfg.get("fee_rate", 0.0015))
    cash = initial_cash if initial_cash is not None else 10000000.0
    
    # 모델 로드
    model_clf_path = os.path.join(paths["models"], "lgbm_model.joblib")
    model_clf = joblib.load(model_clf_path) if os.path.exists(model_clf_path) else None
    
    ensure_dir(run_dir)
    is_opt_mode = preloaded_features is not None
    
    if not quiet: print(f"\n[백테스트 시작] {strategy_name} (Clean Version with Gap Protection)")
    
    # 2. 데이터 준비
    start_d = to_date(start) if start else pd.Timestamp("2020-01-01")
    end_d = to_date(end) if end else pd.Timestamp.today()
    
    if is_opt_mode: 
        all_features = preloaded_features
    else:
        dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
        if not os.path.exists(dataset_path): return None
        all_features = pd.read_parquet(dataset_path)
        all_features['date'] = pd.to_datetime(all_features['date'])
        mask = (all_features['date'] >= start_d) & (all_features['date'] <= end_d)
        all_features = all_features.loc[mask].copy()

    if all_features.empty: return None
    all_features.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
    all_dates = pd.Series(pd.to_datetime(all_features['date'].unique())).sort_values().reset_index(drop=True)
    
    # 지수 데이터 로드 (국면 판단용)
    kospi = None
    if not is_opt_mode:
        try:
            raw_kospi = load_index_data(start_d - pd.DateOffset(days=400), end_d, paths["raw_index"])
            if raw_kospi is not None and not raw_kospi.empty:
                raw_kospi = raw_kospi.sort_values('date').set_index('date')
                if 'kospi_close' in raw_kospi.columns:
                    raw_kospi['ma200'] = raw_kospi['kospi_close'].rolling(200).mean()
                    raw_kospi['vol20'] = raw_kospi['kospi_close'].pct_change().rolling(20).std()
                    kospi = raw_kospi
        except Exception: pass

    # 포트폴리오 초기화
    portfolio = {} 
    daily_equity = []
    trade_log = []
    pending_orders = [] # {code, prev_range, prev_close}
    
    regime_counts = {"Bull": 0, "Bear": 0}

    # 3. 일별 시뮬레이션
    iterator = tqdm(all_dates, desc=f"Simulating", disable=quiet)
    
    for date in iterator:
        today_data = all_features[all_features['date'] == date]
        if today_data.empty: continue
        today_prices = today_data.set_index('code')
        
        # (1) 국면 판단
        current_regime = "R1_BullStable"
        is_bear_market = False
        
        if kospi is not None and date in kospi.index:
            try:
                k_row = kospi.loc[date]
                if pd.notna(k_row['ma200']):
                    current_regime = _determine_regime(k_row['kospi_close'], k_row['ma200'], k_row['vol20'])
            except Exception: pass
        
        if "Bear" in current_regime:
            is_bear_market = True
            regime_counts["Bear"] += 1
        else:
            regime_counts["Bull"] += 1

        # (2) 전략 파라미터 로드
        regime_strategy_key = f"{strategy_name}_{current_regime}"
        strategies_cfg = cfg.get("strategies", {})
        
        if regime_strategy_key in strategies_cfg:
            base_params = strategies_cfg[regime_strategy_key].copy()
        else:
            base_params = strategies_cfg.get(strategy_name, {}).copy()

        if param_overrides: base_params.update(param_overrides)
        
        strategy = SpikeHunterStrategy(base_params)
        
        # --- [Phase 1] 매수 집행 (Gap Filter 적용) ---
        if pending_orders and not is_bear_market:
            # 현재 자산 기반 슬롯당 금액 계산
            current_total_value = cash + sum(p['entry_price']*p['shares'] for p in portfolio.values())
            slot_value = current_total_value / strategy.top_n if strategy.top_n > 0 else 0
            
            for order in pending_orders:
                code = order['code']
                prev_range = order['prev_range']
                prev_close = order.get('prev_close', None) # 전일 종가
                
                if code not in today_prices.index: continue
                if code in portfolio: continue # 이미 보유 중이면 패스
                
                row = today_prices.loc[code]
                open_price = row['open']
                high_price = row['high']
                
                # [★ 핵심] 갭 상승 방어 (Gap Protection)
                # 전일 종가 대비 시가가 7% 이상 높으면 추격 매수 금지
                if prev_close is not None and prev_close > 0:
                    gap_ratio = (open_price - prev_close) / prev_close
                    if gap_ratio > 0.07: # 7% 갭 제한
                        continue 
                else:
                    # 데이터 방어: 전일 종가 없으면 일단 보수적으로 시가가 전일 고가보다 너무 높으면 패스 (생략 가능)
                    pass

                # 진입 가격 결정 (VBO or 시가)
                # vbo_k가 0이면 시가 매수, 양수면 변동성 돌파
                breakout_price = open_price + (prev_range * strategy.vbo_k)
                
                if high_price >= breakout_price:
                    # 체결 가격: 시가가 이미 돌파가보다 높으면 시가 체결, 아니면 돌파가 체결
                    entry_price = max(open_price, breakout_price)
                    
                    if slot_value > entry_price:
                        shares = int(slot_value // entry_price)
                        cost = shares * entry_price * (1 + fee)
                        
                        if cash >= cost:
                            cash -= cost
                            portfolio[code] = {
                                'entry_date': date,
                                'entry_price': entry_price,
                                'shares': shares,
                                'target_price': entry_price * (1 + strategy.target_r),
                                'stop_loss': entry_price * (1 + strategy.stop_r),
                                'regime': current_regime
                            }
            pending_orders = [] # 주문 처리 완료 후 초기화

        # --- [Phase 2] 매도 관리 (당일 청산 허용) ---
        sold_codes = []
        for code, pos in portfolio.items():
            if code not in today_prices.index: continue
            
            # [변경] 당일 매수 종목도 조건 도달 시 즉시 매도 허용 (라인 삭제됨)
            
            row = today_prices.loc[code]
            high, low, close = row['high'], row['low'], row['close']
            
            exit_type, exit_price = None, 0
            
            # 1. 손절 (Stop Loss)
            if low <= pos['stop_loss']:
                exit_type = "SL"
                exit_price = pos['stop_loss']
                # 시가부터 갭하락으로 손절가 아래면 시가 청산
                # (단, 당일 진입 종목은 시가가 진입가이므로, 시가 갭하락은 발생 불가. 장중 하락만 해당)
                if pos['entry_date'] != date and row['open'] < pos['stop_loss']:
                    exit_price = row['open']
            
            # 2. 익절 (Take Profit)
            elif high >= pos['target_price']:
                exit_type = "TP"
                exit_price = pos['target_price']
                # 시가부터 갭상승으로 목표가 위면 시가 청산
                if pos['entry_date'] != date and row['open'] > pos['target_price']:
                    exit_price = row['open']
            
            # 3. 시간 만기 (Time Exit)
            elif (date - pos['entry_date']).days >= strategy.max_hold:
                exit_type = "TIME"
                exit_price = close
            
            if exit_type:
                sold_codes.append(code)
                cash += exit_price * pos['shares'] * (1 - fee)
                ret = (exit_price / pos['entry_price']) - 1
                trade_log.append({
                    'entry_date': pos['entry_date'], 'exit_date': date, 'code': code,
                    'return': ret, 'reason': exit_type, 'regime': pos['regime']
                })
        
        # 매도된 종목 포트폴리오에서 제거
        for code in sold_codes:
            del portfolio[code]

        # 일별 자산 평가
        current_equity = cash + sum(today_prices.loc[c]['close'] * p['shares'] for c, p in portfolio.items() if c in today_prices.index)
        daily_equity.append({'date': date, 'equity': current_equity})

        # --- [Phase 3] 내일의 매수 후보 선정 ---
        if not is_bear_market and model_clf:
            recs = strategy.generate_recs(today_data, model_clf)
            for _, rec in recs.iterrows():
                daily_range = rec['high'] - rec['low']
                # [중요] 내일 아침 갭 계산을 위해 '오늘 종가(close)'를 prev_close로 저장
                pending_orders.append({
                    'code': rec['code'], 
                    'prev_range': daily_range,
                    'prev_close': rec['close'] 
                })

    if not quiet:
        print(f"  [Regime] Bull: {regime_counts['Bull']}, Bear: {regime_counts['Bear']}")

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
        utils_db.insert_backtest_results(bid, metrics, strategy.params, tradelog_df, equity_path, 
                                         start_d.strftime('%Y-%m-%d'), end_d.strftime('%Y-%m-%d'), strategy_name)

    if not quiet:
        print("\n" + "="*60)
        print(f"   [백테스트 결과] 최종 자산: {int(equity_df['equity'].iloc[-1]):,} 원")
        print(f"   - 수익률(CAGR): {metrics['CAGR_raw']*100:.2f}%")
        print(f"   - MDD: {metrics['MDD_raw']*100:.2f}%")
        print(f"   - 승률: {metrics['win_rate_raw']*100:.2f}% ({metrics['총거래횟수']}회)")
        print("="*60)

    return {"metrics": metrics, "daily_equity": equity_df, "tradelog_path": tradelog_path}