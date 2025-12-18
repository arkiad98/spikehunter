# modules/labeling.py (v5.8 - Fixed Target Restored)
"""
[SpikeHunter v5.8] 데이터 라벨링 모듈 (Rollback)
- 복구: ATR 동적 라벨링 실패 -> '고정 수익률(Fixed Return)' 방식으로 회귀
- 설정: 시가 진입 기준, 15% 상승 시 성공 (확실한 급등만 학습)
"""
import numpy as np
import pandas as pd
# modules/labeling.py (v5.8 - Fixed Target Restored)
"""
[SpikeHunter v5.8] 데이터 라벨링 모듈 (Rollback)
- 복구: ATR 동적 라벨링 실패 -> '고정 수익률(Fixed Return)' 방식으로 회귀
- 설정: 시가 진입 기준, 15% 상승 시 성공 (확실한 급등만 학습)
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_triple_barrier_label_fixed(close_series: pd.Series, 
                                   open_series: pd.Series, 
                                   high_series: pd.Series, 
                                   low_series: pd.Series,
                                   horizon: int = 2, 
                                   profit_th: float = 0.15, 
                                   loss_th: float = -0.07
                                   ) -> pd.Series:
    labels = np.zeros(len(close_series))
    
    closes = close_series.values
    opens = open_series.values
    highs = high_series.values
    lows = low_series.values
    
    limit = len(closes) - horizon - 1
    
    for t in range(limit):
        entry_price = opens[t+1] # T+1 시가 진입
        if entry_price == 0 or np.isnan(entry_price): continue

        # 목표가/손절가 설정 (고정 비율)
        target_price = entry_price * (1 + profit_th)
        stop_price = entry_price * (1 + loss_th)
        
        # 갭상승 필터: 시가가 이미 전일 종가 대비 10% 이상 떴으면 진입 포기
        if entry_price > closes[t] * 1.10:
             labels[t] = 0
             continue

        future_highs = highs[t+1 : t+1+horizon]
        future_lows = lows[t+1 : t+1+horizon]
        
        tp_indices = np.where(future_highs >= target_price)[0]
        sl_indices = np.where(future_lows <= stop_price)[0]
        
        has_tp = len(tp_indices) > 0
        has_sl = len(sl_indices) > 0
        
        if has_tp and not has_sl:
            # 목표가 도달 AND 손절 안함
            labels[t] = 1
        elif has_tp and has_sl:
            if tp_indices[0] <= sl_indices[0]:
                # 목표가 먼저 도달
                labels[t] = 1
            else:
                labels[t] = 0
        else:
            labels[t] = 0

    return pd.Series(labels, index=close_series.index)

def add_labels(df: pd.DataFrame, profit_th=0.15, loss_th=-0.07, horizon=2) -> pd.DataFrame:
    tqdm.pandas(desc="Generating Labels (Price Only)")
    
    def _apply(group):
        group = group.sort_values('date')
            
        return get_triple_barrier_label_fixed(
            group['close'], 
            group['open'], 
            group['high'], 
            group['low'],
            horizon=horizon,
            profit_th=profit_th,
            loss_th=loss_th
        )

    labeled_series = df.groupby('code', group_keys=False).apply(_apply)
    df['label_class'] = labeled_series
        
    return df