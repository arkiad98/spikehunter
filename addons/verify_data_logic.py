# addons/verify_data_logic.py
import os
import sys
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils_io import load_partition_day, read_yaml, to_date
from modules.utils_logger import logger

def verify_single_case():
    """
    분석 로직의 정합성을 검증하기 위해
    '급등(Spike)'으로 감지된 첫 번째 사례의 상세 데이터를 출력합니다.
    """
    # 1. 설정 및 데이터 로드
    CONFIG = read_yaml("config/settings.yaml")
    data_dir = CONFIG['paths']['features']
    
    # 최근 데이터만 로드 (속도 위함)
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    print(f" >> 데이터 로드 중... ({start_date} ~ {end_date})")
    df_all = load_partition_day(data_dir, to_date(start_date), to_date(end_date))
    
    if df_all.empty:
        print("!! 데이터가 없습니다.")
        return

    # 2. 검증 로직 수행
    grouped = df_all.groupby('code')
    
    print(" >> 급등 사례 탐색 및 검증 시작...\n")
    
    for code, df in grouped:
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 30: continue
        
        # --- [로직 복제 시작] ---
        # 1. 원본 데이터 보존
        raw_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # 2. 피처 계산 (분석기와 동일한 공식)
        df['ma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        
        # Bandwidth 공식: (4 * std) / ma
        df['bandwidth'] = (df['std20'] * 4) / df['ma20']
        
        # NR (Daily Range) 공식: (High - Low) / Prev Close
        df['prev_close'] = df['close'].shift(1)
        df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
        df['nr5'] = df['daily_range'].rolling(5).min()
        
        # 급등 조건 확인 (10% 상승)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
        df['fwd_max_ret'] = df['close'].rolling(window=indexer).max() / df['close'] - 1.0
        
        # 갭상승 체크 (다음날 시가)
        df['next_open_ret'] = df['open'].shift(-1) / df['close'] - 1.0
        
        # 조건: 10% 이상 상승 AND 시가 갭 10% 미만
        is_spike = (df['fwd_max_ret'] >= 0.10) & (df['next_open_ret'] < 0.10)
        # --- [로직 복제 끝] ---

        if not is_spike.any():
            continue
            
        # 첫 번째 급등 사례 발견 시 상세 출력 후 종료
        spike_idx = df.index[is_spike][0]
        
        # 전조 기간 (D-5 ~ D-0) 데이터 추출
        start_idx = spike_idx - 5
        if start_idx < 0: continue
        
        debug_df = df.loc[start_idx : spike_idx].copy()
        
        print(f"★ 검증 대상 종목: {code}")
        print(f"★ 급등 감지일(D-Day): {df.loc[spike_idx, 'date']}")
        print(f"★ 수익률(Forward Max): {df.loc[spike_idx, 'fwd_max_ret']*100:.2f}%")
        print(f"★ 다음날 시가갭: {df.loc[spike_idx, 'next_open_ret']*100:.2f}%\n")
        
        print("=== [원본 데이터 vs 파생 피처 대조] ===")
        # 보기 좋게 출력 포맷 설정
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # 검증 포인트 컬럼 선택
        cols = ['date', 'close', 'ma20', 'std20', 'bandwidth', 'high', 'low', 'prev_close', 'daily_range']
        
        print(debug_df[cols].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        
        print("\n=== [검증 가이드] ===")
        print("1. Bandwidth = (std20 * 4) / ma20 인가?")
        print("2. Daily_Range = (High - Low) / Prev_Close 인가?")
        print("3. 이 수치들이 급격히 튀는 이상한 값(Outlier)은 없는가?")
        
        # 하나만 보고 종료
        break

if __name__ == "__main__":
    verify_single_case()