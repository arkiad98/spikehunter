# check_data.py
import pandas as pd
import os
from modules.utils_io import read_yaml

def check():
    cfg = read_yaml("config/settings.yaml")
    path = os.path.join(cfg['paths']['features'], "dataset_v4.parquet")
    
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        print("   -> [1. 데이터 관리] > [5. 데이터셋 생성]을 실행하세요.")
        return

    df = pd.read_parquet(path)
    print(f"✅ 데이터 로드 성공: {len(df)} 행")
    print(f"   기간: {df['date'].min()} ~ {df['date'].max()}")
    
    cols = df.columns.tolist()
    check_cols = ['amount_ma5', 'mfi_14', 'obv_slope_5', 'dist_vwap']
    
    print("\n[필수 컬럼 확인]")
    for c in check_cols:
        if c in cols:
            print(f"   O {c:<15} (평균: {df[c].mean():.4f})")
        else:
            print(f"   X {c:<15} (누락됨!)")
            
    if 'amount_ma5' not in cols:
        print("\n🚨 경고: 거래대금 컬럼이 누락되었습니다. 백테스트 시 거래가 안 됩니다.")
        print("   -> modules/features.py 교체 후 데이터셋을 재생성하세요.")

if __name__ == "__main__":
    check()