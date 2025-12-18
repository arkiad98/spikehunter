# addons/addon_pre_spike_analyzer.py (v5.0)
"""
[SpikeHunter v5.0] 급등 전조 현상 분석기 (Renewal)
- 목표: "상대적 수렴(Relative Squeeze) + 스마트 머니(Smart Money)" 가설 검증
- 분석 대상: 
  1. Range Ratio 20 (평소 대비 변동성 축소 정도)
  2. OBV Slope 5 (매집 강도)
  3. MFI 14 (자금 유입)
  4. Bandwidth (거시적 위치)
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils_io import load_partition_day, read_yaml, to_date
from modules.utils_logger import logger

# ==============================================================================
# 설정 (Configuration)
# ==============================================================================
CONFIG = {
    "spike_condition": {
        "window": 2,        # 2일간
        "threshold": 0.10   # 10% 이상 상승 시 '급등' (현실적 기준)
    },
    "pre_spike_window": 5,  # 급등 직전 5일 관찰
    "analysis_period": {
        "start": "2022-01-01",
        "end": "2024-12-31"
    }
}

def _set_korean_font():
    """한글 폰트 설정"""
    font_path = 'c:/Windows/Fonts/malgun.ttf'
    if not os.path.exists(font_path):
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

def calculate_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """v5.0 핵심 피처 계산 (features.py 로직 복제)"""
    # 1. 기본 지표
    df['prev_close'] = df['close'].shift(1)
    df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['vol_ma20'] = df['volume'].rolling(20).mean() # [수정] 거래량 이동평균 미리 계산
    
    # 2. Range Ratio (상대적 수렴)
    df['range_ma20'] = df['daily_range'].rolling(20).mean()
    df['range_ratio_20'] = df['daily_range'] / (df['range_ma20'] + 1e-9)
    
    # 3. Bandwidth (절대적 수렴)
    df['bandwidth'] = (df['std20'] * 4) / df['ma20']
    
    # 4. OBV Slope (세력 매집)
    obv_change = np.where(df['close'] > df['prev_close'], df['volume'], 
                          np.where(df['close'] < df['prev_close'], -df['volume'], 0))
    df['obv'] = pd.Series(obv_change).cumsum()
    df['obv_slope_5'] = (df['obv'] - df['obv'].shift(5)) / 5
    
    # 5. MFI (자금 유입)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    delta = typical_price.diff()
    pos_flow = np.where(delta > 0, money_flow, 0)
    neg_flow = np.where(delta < 0, money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfi_ratio = pos_mf / (neg_mf + 1e-9)
    df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
    
    # 6. Spike 여부 (미래 수익률)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=CONFIG['spike_condition']['window'])
    df['fwd_max_ret'] = df['close'].rolling(window=indexer).max() / df['close'] - 1.0
    
    # 7. 갭상승 필터용 (내일 시가)
    df['next_open_ret'] = df['open'].shift(-1) / df['close'] - 1.0
    
    return df.dropna()

def analyze_pre_spike_patterns(df_all: pd.DataFrame):
    """급등주와 일반주의 패턴 비교"""
    spike_events = []
    control_group = []
    
    grouped = df_all.groupby('code')
    logger.info(f"총 {len(grouped)}개 종목 분석 시작...")
    
    for code, df in tqdm(grouped, desc="Processing"):
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 60: continue
        
        df = calculate_new_features(df)
        if df.empty: continue

        # 급등 조건: 10% 이상 상승 & 갭 7% 미만 (진입 가능성 고려)
        is_spike = (df['fwd_max_ret'] >= CONFIG['spike_condition']['threshold']) & \
                   (df['next_open_ret'] < 0.07)
        
        spike_indices = df.index[is_spike].tolist()
        
        # 중복 제거
        filtered_indices = []
        last_idx = -999
        for idx in spike_indices:
            if idx - last_idx > CONFIG['pre_spike_window']:
                filtered_indices.append(idx)
                last_idx = idx
                
        # (1) 급등 그룹 수집
        for idx in filtered_indices:
            start_idx = idx - CONFIG['pre_spike_window']
            if start_idx < df.index[0]: continue
            
            pre_data = df.loc[start_idx : idx-1] # 급등 직전 5일치
            if len(pre_data) < CONFIG['pre_spike_window']: continue
            
            # **[핵심]** 급등 직전일(D-1)의 상태 추출
            last_day = pre_data.iloc[-1]
            
            stats = {
                'range_ratio_20': last_day['range_ratio_20'], # 상대적 수렴도
                'obv_slope_normalized': last_day['obv_slope_5'] / (last_day['vol_ma20'] + 1), # [수정] 미리 계산된 vol_ma20 사용
                'mfi_14': last_day['mfi_14'],
                'bandwidth': last_day['bandwidth'],
                'label': 'Spike (급등)'
            }
            spike_events.append(stats)
            
        # (2) 대조군 수집 (랜덤 샘플링)
        non_spike_indices = df.index[~is_spike].tolist()
        if len(non_spike_indices) > len(filtered_indices):
            sample_indices = np.random.choice(non_spike_indices, size=len(filtered_indices), replace=False)
        else:
            sample_indices = non_spike_indices
            
        for idx in sample_indices:
            start_idx = idx - CONFIG['pre_spike_window']
            if start_idx < df.index[0]: continue
            pre_data = df.loc[start_idx : idx-1]
            if len(pre_data) < CONFIG['pre_spike_window']: continue
            
            last_day = pre_data.iloc[-1]
            stats = {
                'range_ratio_20': last_day['range_ratio_20'],
                'obv_slope_normalized': last_day['obv_slope_5'] / (last_day['vol_ma20'] + 1), # [수정] 미리 계산된 vol_ma20 사용
                'mfi_14': last_day['mfi_14'],
                'bandwidth': last_day['bandwidth'],
                'label': 'Normal (일반)'
            }
            control_group.append(stats)

    return pd.DataFrame(spike_events + control_group)

def visualize_results(df_res):
    if df_res.empty:
        logger.warning("분석할 데이터가 없습니다.")
        return

    # 1. 요약 통계
    print("\n" + "="*80)
    print("   [v5.0 신규 피처 검증 결과 (Summary Statistics)]")
    print("="*80)
    summary = df_res.groupby('label').mean()
    print(summary)
    print("-" * 80)
    
    # Insight 출력
    spike_ratio = summary.loc['Spike (급등)', 'range_ratio_20']
    normal_ratio = summary.loc['Normal (일반)', 'range_ratio_20']
    
    if spike_ratio < normal_ratio:
        print(f"✅ [검증 성공] 급등주는 일반 종목보다 변동성이 {normal_ratio - spike_ratio:.2f}만큼 더 수렴합니다.")
        print(f"   (Spike Ratio: {spike_ratio:.2f} vs Normal: {normal_ratio:.2f})")
    else:
        print("⚠️ [특이 사항] 급등주의 변동성이 더 큽니다. (돌파 매매 관점 필요)")

    # 2. 시각화
    _set_korean_font()
    metrics = ['range_ratio_20', 'obv_slope_normalized', 'mfi_14', 'bandwidth']
    titles = ['Range Ratio (Relative Squeeze)', 'OBV Slope (Accumulation)', 'MFI (Money Flow)', 'Bandwidth (Absolute)']
    
    plt.figure(figsize=(20, 5))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 4, i+1)
        # 이상치 제거 (시각화용)
        q_low = df_res[metric].quantile(0.01)
        q_high = df_res[metric].quantile(0.99)
        data_filtered = df_res[(df_res[metric] >= q_low) & (df_res[metric] <= q_high)]
        
        sns.boxplot(x='label', y=metric, data=data_filtered, showfliers=False)
        plt.title(titles[i])
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    save_path = "analysis_v5_new_features.png"
    plt.savefig(save_path)
    print(f"\n[!] 시각화 차트가 저장되었습니다: {save_path}")
    print("="*80)

def main():
    cfg = read_yaml("config/settings.yaml")
    data_dir = cfg['paths']['merged'] # Raw Data 사용
    
    start_date = to_date(CONFIG['analysis_period']['start'])
    end_date = to_date(CONFIG['analysis_period']['end'])
    
    logger.info("데이터 로드 및 분석 준비...")
    df_all = load_partition_day(data_dir, start_date, end_date)
    
    if df_all.empty:
        logger.error("데이터가 없습니다.")
        return

    result_df = analyze_pre_spike_patterns(df_all)
    visualize_results(result_df)

if __name__ == "__main__":
    main()