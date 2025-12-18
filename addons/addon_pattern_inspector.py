# addons/addon_pattern_inspector.py
"""
[SpikeHunter] 급등주 10일 궤적 심층 분석기 (Extended Features)
- 업데이트: 스마트 머니(MFI, OBV, VWAP) 및 심리 지표(StochRSI) 추가 분석
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils_io import read_yaml
from modules.utils_logger import logger

# 한글 폰트 설정
def _set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

def run_pattern_inspection(settings_path="config/settings.yaml"):
    print("\n" + "="*80)
    print("      <<< 급등주 10일 궤적(Trajectory) 심층 분석 (Extended) >>>")
    print("="*80)

    if not os.path.exists(settings_path):
        print("설정 파일이 없습니다.")
        return

    cfg = read_yaml(settings_path)
    # [주의] 반드시 데이터 생성을 먼저 다시 해야 함
    dataset_path = os.path.join(cfg['paths']['features'], "dataset_v4.parquet")
    
    if not os.path.exists(dataset_path):
        print(f"데이터셋이 없습니다: {dataset_path}")
        print("먼저 [데이터 관리] -> [5. 데이터셋 생성]을 실행해주세요.")
        return

    print(" >> 데이터 로드 중...")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 분석 대상 피처 대폭 확대 (총 12개)
    features_to_analyze = [
        # [A] 스마트 머니 (세력의 흔적)
        'mfi_14',          # 자금 흐름 (거래량 실린 상승)
        'obv_slope_5',     # 매집 강도
        'dist_vwap',       # 세력 평단가 대비 위치
        
        # [B] 모멘텀 & 심리
        'rsi_14',          # 상대 강도
        'stoch_rsi',       # RSI의 위치 (침체권 탈출 여부)
        'dist_ma20',       # 단기 추세
        
        # [C] 변동성 & 거래량
        'bandwidth',       # 응축 여부
        'percent_b',       # 밴드 내 위치
        'range_ratio',     # 변동폭 축소
        'vol_ratio_5',     # 거래량 급증
        'vol_ratio_20',    
        'amount_ma5'       # 거래대금
    ]
    
    valid_features = [c for c in features_to_analyze if c in df.columns]
    print(f" >> 분석 대상 피처 ({len(valid_features)}개):")
    print(f"    {valid_features}")

    # 급등 정의: 2일 내 10% 이상 상승 (데이터 확보를 위해 10%로 설정)
    print(" >> 급등(Spike, +10%) 사례 추출 중...")
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    df['fwd_max_high'] = df.groupby('code')['high'].rolling(window=indexer).max().reset_index(level=0, drop=True)
    df['fwd_max_ret'] = (df['fwd_max_high'] / df['close']) - 1.0
    
    spike_threshold = 0.10
    lookback = 10

    spike_indices = df[df['fwd_max_ret'] >= spike_threshold].index
    normal_indices_pool = df[(df['fwd_max_ret'] < 0.05) & (df['fwd_max_ret'] > -0.05)].index
    
    sample_size = min(len(spike_indices), 3000) 
    if len(spike_indices) > sample_size:
        spike_indices = np.random.choice(spike_indices, size=sample_size, replace=False)
    if len(normal_indices_pool) > sample_size:
        normal_indices = np.random.choice(normal_indices_pool, size=sample_size, replace=False)
    else:
        normal_indices = normal_indices_pool

    def collect_trajectories(indices, label):
        temp_list = []
        valid_codes = set(df.loc[indices, 'code'])
        
        for code, sub_df in tqdm(df[df['code'].isin(valid_codes)].groupby('code'), desc=f"Extracting {label}"):
            sub_indices = [idx for idx in indices if idx in sub_df.index]
            for idx in sub_indices:
                curr_pos = sub_df.index.get_loc(idx)
                if curr_pos < lookback: continue
                
                trajectory = sub_df.iloc[curr_pos - lookback : curr_pos + 1].copy()
                if len(trajectory) < lookback + 1: continue

                trajectory['days_before'] = range(-lookback, 1)
                trajectory['group'] = label
                trajectory['id'] = idx
                
                temp_list.append(trajectory[['days_before', 'group', 'id'] + valid_features])
        return temp_list

    spike_data = collect_trajectories(spike_indices, 'Spike (급등주)')
    normal_data = collect_trajectories(normal_indices, 'Normal (일반주)')
    
    if not spike_data:
        print("분석할 데이터가 충분하지 않습니다.")
        return

    full_data = pd.concat(spike_data + normal_data, ignore_index=True)
    print(f" >> 데이터 수집 완료: 총 {len(full_data)} 행")

    # 시각화
    _set_korean_font()
    n_cols = 3
    n_rows = (len(valid_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    print(" >> 궤적 차트 생성 중...")
    for i, feature in enumerate(valid_features):
        ax = axes[i]
        sns.lineplot(
            data=full_data, x='days_before', y=feature, hue='group', style='group',
            markers=False, dashes=False, ax=ax,
            palette={'Spike (급등주)': '#FF3333', 'Normal (일반주)': '#999999'},
            linewidth=2, errorbar=('ci', 95)
        )
        ax.set_title(f"[{feature}]", fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
        
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
        
    plt.tight_layout()
    save_path = "analysis_10day_trajectory_smart_money.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n >> [완료] 차트 저장됨: {save_path}")
    print("    'mfi_14', 'obv_slope_5', 'stoch_rsi' 차트를 집중해서 확인하세요.")

if __name__ == "__main__":
    run_pattern_inspection()