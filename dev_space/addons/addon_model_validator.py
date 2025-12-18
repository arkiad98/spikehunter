# addons/addon_model_validator.py
"""
[SpikeHunter v4.0] 모델 예측력 검증기 (Model Validator)
- 목적: 백테스트(매매 전략)를 배제하고, 모델 자체의 '종목 선별 능력'을 검증
- 기능:
  1. 학습된 모델과 데이터셋을 로드
  2. 전체 기간에 대해 예측 점수(Score) 생성
  3. 점수 구간별(Top 10%, 20%...) 미래 수익률 분석
  4. Information Coefficient (IC) 계산
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils_io import read_yaml, ensure_dir
from modules.utils_logger import logger
from modules.derive import _get_feature_cols

def run_model_validation(settings_path="config/settings.yaml"):
    print("\n" + "="*60)
    print("      <<< 모델 순수 예측력 검증 (Model Validator) >>>")
    print("="*60)

    # 1. 설정 및 경로 로드
    if not os.path.exists(settings_path):
        print("설정 파일이 없습니다.")
        return

    cfg = read_yaml(settings_path)
    paths = cfg['paths']
    
    # 2. 데이터셋 및 모델 로드
    dataset_path = os.path.join(paths['features'], "dataset_v4.parquet")
    model_path = os.path.join(paths['models'], "lgbm_model.joblib")
    
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print("데이터셋 또는 모델 파일이 없습니다.")
        return

    print(" >> 데이터 및 모델 로드 중...")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    model = joblib.load(model_path)
    
    # 3. 예측 수행
    print(" >> 전체 데이터에 대한 예측 수행 중...")
    feature_cols = _get_feature_cols(df.columns)
    X = df[feature_cols].fillna(0)
    
    # 예측 점수(확률) 계산
    df['pred_score'] = model.predict_proba(X)[:, 1]
    
    # 4. 미래 수익률 계산 (1일 후, 3일 후, 5일 후 최고가 기준)
    # (단순 종가 수익률이 아니라, 급등 전략이므로 '기간 내 최고가' 달성 여부 확인)
    print(" >> 미래 수익률 지표 계산 중...")
    df = df.sort_values(['code', 'date'])
    
    # 미래 N일 최고 수익률 (Forward Max Return)
    for horizon in [1, 3, 5]:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        # 미래 N일간의 고가(High) 중 최고가 / 현재 종가(Close) - 1
        df[f'fwd_max_ret_{horizon}d'] = df.groupby('code')['high'].rolling(window=indexer).max().reset_index(level=0, drop=True) / df['close'] - 1.0
        # 미래 N일 후 종가 수익률 (단순 보유 시)
        df[f'fwd_close_ret_{horizon}d'] = df.groupby('code')['close'].shift(-horizon) / df['close'] - 1.0

    df = df.dropna() # 미래 데이터 없는 마지막 구간 제거

    # 5. 분석 1: 분위수별 성과 (Quantile Analysis)
    print("\n[분석 1] 점수 구간별 평균 성과 (Decile Analysis)")
    # 날짜별로 점수 순위 매겨서 10분위로 나눔 (Q10: 상위 10%, Q1: 하위 10%)
    df['score_decile'] = df.groupby('date')['pred_score'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # 분위별 평균 수익률 계산
    perf_by_decile = df.groupby('score_decile')[
        ['fwd_max_ret_1d', 'fwd_max_ret_3d', 'fwd_max_ret_5d', 'fwd_close_ret_1d']
    ].mean() * 100
    
    print(perf_by_decile.sort_index(ascending=False).to_string(float_format="{:.2f}%".format))
    
    # 6. 분석 2: 상위 Top N 종목의 적중률 (Hit Rate)
    print("\n[분석 2] 매일 상위 Top 5 종목의 실제 급등 여부")
    # 매일 점수 상위 5개 추출
    top5_df = df.groupby('date').apply(lambda x: x.nlargest(5, 'pred_score')).reset_index(drop=True)
    
    # 5% 이상 급등한 비율
    hit_rate_1d = (top5_df['fwd_max_ret_1d'] > 0.05).mean()
    hit_rate_3d = (top5_df['fwd_max_ret_3d'] > 0.10).mean()
    avg_ret_1d = top5_df['fwd_close_ret_1d'].mean()
    
    print(f" - 상위 5개 종목이 익일 장중 5% 이상 급등할 확률: {hit_rate_1d*100:.2f}%")
    print(f" - 상위 5개 종목이 3일 내 10% 이상 급등할 확률 : {hit_rate_3d*100:.2f}%")
    print(f" - 상위 5개 종목의 익일 종가 평균 수익률      : {avg_ret_1d*100:.2f}%")

    # 7. 분석 3: 정보 계수 (Information Coefficient)
    # 점수와 미래 수익률 간의 상관계수 (높을수록 좋음, 0.05 이상이면 유의미)
    ic_1d, _ = spearmanr(df['pred_score'], df['fwd_close_ret_1d'])
    print(f"\n[분석 3] Information Coefficient (IC): {ic_1d:.4f}")
    if ic_1d < 0.02:
        print(" >> 경고: 모델의 예측력이 매우 낮습니다. (Random 수준)")
    elif ic_1d > 0.05:
        print(" >> 양호: 모델이 유의미한 예측력을 가지고 있습니다.")

    # 8. 시각화 저장
    plt.figure(figsize=(12, 6))
    perf_by_decile['fwd_max_ret_5d'].plot(kind='bar')
    plt.title("Average Max Return by Score Decile (Next 5 Days)")
    plt.xlabel("Score Decile (0=Lowest, 9=Highest)")
    plt.ylabel("Avg Max Return (%)")
    plt.grid(axis='y', alpha=0.3)
    
    output_img = "analysis_model_lift_chart.png"
    plt.savefig(output_img)
    print(f"\n >> 리프트 차트 저장됨: {output_img}")

if __name__ == "__main__":
    run_model_validation()