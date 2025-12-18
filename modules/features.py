# modules/features.py (v6.0 - Final Integration)
"""
[SpikeHunter v6.0] 피처 엔지니어링 모듈
- 통합: 개별 종목 패턴(수급+수렴) + 시장 국면(Regime) + 상대적 순위(Rank)
- 구성:
  1. Volatility/Squeeze (변동성 축소)
  2. Smart Money (수급)
  3. Momentum/Trend (추세)
  4. Interaction (상호작용)
  5. Market Regime (시장 지표) - [NEW]
  6. Cross-Sectional Rank (상대 순위) - [NEW]
"""
import pandas as pd
import numpy as np

def add_volatility_contraction_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """변동성 및 수렴 지표"""
    # 1. 기본 이평선 및 볼린저 밴드
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (df['std20'] * 2)
    df['lower_band'] = df['ma20'] - (df['std20'] * 2)
    
    # Bandwidth (절대적 수렴)
    df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['ma20']
    # Percent B (상대적 위치)
    df['percent_b'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
    
    # 2. NR (절대 변동폭)
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # 3. Range Ratio (상대적 수렴) - 핵심 피처
    df['range_ma20'] = df['daily_range'].rolling(20).mean()
    df['range_ratio_20'] = df['daily_range'] / (df['range_ma20'] + 1e-9)
    
    # (보조) 5일 기준
    df['range_ma5'] = df['daily_range'].rolling(5).mean()
    df['range_ratio_5'] = df['daily_range'] / (df['range_ma5'] + 1e-9)
    
    return df

def add_smart_money_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """수급 및 거래량 지표"""
    # 거래대금 (억 단위)
    df['amount_bil'] = (df['close'] * df['volume']) / 100000000
    df['amount_ma5'] = df['amount_bil'].rolling(5).mean()

    # MFI (자금 흐름)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    delta = typical_price.diff()
    pos_flow = np.where(delta > 0, money_flow, 0)
    neg_flow = np.where(delta < 0, money_flow, 0)
    pos_mf_14 = pd.Series(pos_flow).rolling(14).sum()
    neg_mf_14 = pd.Series(neg_flow).rolling(14).sum()
    mfi_ratio = pos_mf_14 / (neg_mf_14 + 1e-9)
    df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
    
    # OBV (누적 매집)
    obv_change = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                          np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['obv'] = pd.Series(obv_change).cumsum()
    # OBV Slope (매집 강도)
    df['obv_slope_5'] = (df['obv'] - df['obv'].shift(5)) / 5
    
    # Volume Ratio (거래량 급증)
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio_5'] = df['volume'] / (df['vol_ma5'].shift(1) + 1e-9)
    
    return df

def add_interaction_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """상호작용 피처 (시너지 효과)"""
    # Squeeze Energy: 수렴 상태(Low Range)에서 거래량이 터질 때(High Vol) 점수 높음
    # (1 / (RangeRatio + 0.1)) * VolRatio
    squeeze_score = (1 / (df['range_ratio_20'] + 0.1)) 
    df['squeeze_energy'] = squeeze_score * df['vol_ratio_5']
    
    # PV Trend: 가격 상승폭 * 거래량 비율 (거래량 실린 장대양봉)
    df['pv_trend'] = df['close'].pct_change() * df['vol_ratio_5']
    
    return df

def add_technical_momentum(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """모멘텀 지표"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # RSI ROC (모멘텀 가속도)
    df['rsi_roc_3d'] = df['rsi_14'].diff(3)
    
    # 이격도
    df['dist_ma20'] = df['close'] / df['ma20'] - 1
    
    return df

def add_trend_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """추세 지표 (ADX, DMI)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 1. TR (True Range)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 2. DM (Directional Movement)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # 3. Smoothed TR, DM (Wilder's Smoothing)
    tr_s = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    pdm_s = pd.Series(pdm).ewm(alpha=1/14, adjust=False).mean()
    mdm_s = pd.Series(mdm).ewm(alpha=1/14, adjust=False).mean()
    
    # 4. DI (+DI, -DI)
    df['pdi_14'] = 100 * (pdm_s / (tr_s + 1e-9))
    df['mdi_14'] = 100 * (mdm_s / (tr_s + 1e-9))
    
    # 5. DX & ADX
    dx = 100 * (df['pdi_14'] - df['mdi_14']).abs() / (df['pdi_14'] + df['mdi_14'] + 1e-9)
    df['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    return df

def add_long_term_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """중장기 시계열 피처 (T-10일 이상)"""
    # 1. Momentum (60일)
    df['mom_60'] = df['close'].pct_change(60)
    
    # 2. Volume Momentum (20일)
    df['vol_mom_20'] = df['volume'].pct_change(20)
    
    # 3. Slope (20일 추세 기울기)
    # 간단하게 (현재가 - 20일전가) / 20일전가 사용 (정규화된 기울기)
    df['slope_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # 4. MA60 이격도
    df['ma60'] = df['close'].rolling(60).mean()
    df['dist_ma60'] = df['close'] / (df['ma60'] + 1e-9) - 1
    
    # 5. Upper Shadow (윗꼬리) 빈도/강도 (20일)
    # 윗꼬리 길이 = 고가 - max(시가, 종가)
    body_top = df[['open', 'close']].max(axis=1)
    upper_shadow = (df['high'] - body_top) / body_top
    df['upper_shadow_20'] = upper_shadow.rolling(20).mean()
    
    return df

def add_ma_convergence_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """이동평균 수렴/확산 및 골든크로스"""
    ma5 = df['close'].rolling(5).mean()
    ma20 = df['close'].rolling(20).mean()
    ma60 = df['close'].rolling(60).mean()
    
    # 1. MA Convergence (밀집도)
    # (Max(MA) - Min(MA)) / Min(MA) -> 작을수록 밀집
    ma_max = pd.concat([ma5, ma20, ma60], axis=1).max(axis=1)
    ma_min = pd.concat([ma5, ma20, ma60], axis=1).min(axis=1)
    df['ma_convergence'] = (ma_max - ma_min) / (ma_min + 1e-9)
    
    # 2. Golden Cross Score (정배열 강도)
    # 5 > 20 > 60 이면 1, 아니면 0 (혹은 점수화)
    is_aligned = (ma5 > ma20) & (ma20 > ma60)
    df['golden_cross_score'] = is_aligned.astype(int)
    
    # 3. MA Alignment Duration (정배열 지속 기간)
    # 연속된 1의 개수를 셈
    df['ma_alignment'] = df['golden_cross_score'].groupby((df['golden_cross_score'] != df['golden_cross_score'].shift()).cumsum()).cumcount()
    # 정배열이 아니면 0으로 초기화
    df.loc[df['golden_cross_score'] == 0, 'ma_alignment'] = 0
    
    return df

def add_chart_theory_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """차트 이론 보완 (VWAP, MACD, Stochastic)"""
    # 1. VWAP (Volume Weighted Average Price)
    # 일별 VWAP 근사치 = (Typical Price * Volume).cumsum() / Volume.cumsum()
    # 여기서는 20일 Rolling VWAP으로 계산 (단기 추세 반영)
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = tp * df['volume']
    df['vwap_20'] = tp_vol.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
    df['price_vs_vwap'] = df['close'] / (df['vwap_20'] + 1e-9) - 1
    
    # 2. MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    
    # 3. Stochastic Slow
    # Fast %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    # Slow %K = Fast %K smoothed with SMA (usually 3)
    n = 14
    low_n = df['low'].rolling(n).min()
    high_n = df['high'].rolling(n).max()
    fast_k = 100 * (df['close'] - low_n) / (high_n - low_n + 1e-9)
    df['stoch_k'] = fast_k.rolling(3).mean()
    
    return df

# [NEW] 시장 국면 피처 계산 함수 (필수 추가)
def calculate_market_features(kospi_df: pd.DataFrame) -> pd.DataFrame:
    """KOSPI 데이터를 받아 시장 국면(Regime) 피처를 계산합니다."""
    if kospi_df.empty: return pd.DataFrame()
    
    df = kospi_df.copy()
    # kospi_close 컬럼 사용
    close = df['kospi_close']
    
    # 1. Market Trend (장기 추세)
    # 200일 이동평균선 대비 현재 주가 위치 (1=정배열/상승장, 0=역배열/하락장)
    ma200 = close.rolling(200).mean()
    df['market_bullish'] = (close > ma200).astype(int)
    
    # 2. Market Volatility (시장 공포 지수)
    # 20일 변동성 (높을수록 시장이 불안정)
    df['market_volatility'] = close.pct_change().rolling(20).std()
    
    return df[['date', 'market_bullish', 'market_volatility']]

# [NEW] Cross-Sectional Rank 피처 생성 함수
def add_cross_sectional_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Cross-Sectional] 특정 날짜의 전 종목 대비 순위(Rank) 피처를 생성합니다.
    """
    # 순위 매길 대상 피처 목록
    target_cols = [
        'rsi_14', 'mfi_14', 'obv_slope_5', 
        'vol_ratio_5', 'range_ratio_20', 
        'amount_ma5', 'squeeze_energy'
    ]
    
    # 데이터프레임에 존재하는 컬럼만 선택
    cols_to_rank = [c for c in target_cols if c in df.columns]
    
    if not cols_to_rank:
        return df
        
    # 날짜별 그룹핑하여 순위 계산 (0.0 ~ 1.0 사이값으로 정규화)
    # pct=True: 백분위수 (1등 = 1.0, 꼴등 = 0.0)
    for col in cols_to_rank:
        rank_col_name = f"{col}_rank"
        # 변동성이 없는 경우(모두 0인 경우 등)를 대비해 method='min' 사용
        df[rank_col_name] = df.groupby('date')[col].rank(pct=True, ascending=True, method='min')
        
    return df

# 파이프라인 등록
FEATURE_CALCULATION_PIPELINE = [
    (add_volatility_contraction_features, {}),
    (add_smart_money_features, {}),
    (add_technical_momentum, {}),
    (add_interaction_features, {}),
    (add_trend_features, {}),
    (add_long_term_features, {}),
    (add_ma_convergence_features, {}),
    (add_chart_theory_features, {}),
]

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.sort_values('date').reset_index(drop=True)
    for func, kwargs in FEATURE_CALCULATION_PIPELINE:
        df = func(df, **kwargs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    return df