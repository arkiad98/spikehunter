# addons/addon_price_sanitizer.py
import pandas as pd
import os
from tqdm import tqdm
import time
from pykrx import stock

from modules.utils_io import read_yaml, load_partition_day, save_parquet_partitioned_monthly, yyyymmdd, retry_request
from modules.utils_logger import logger
from modules.collect import _rename_ohlcv_cols, REQUIRED_RAW_COLS
# 기존의 run_price_sanitizer 함수를 찾아 아래 코드로 전체를 교체해주세요.

def run_price_sanitizer(settings_path: str):
    """
    수집된 전체 가격 데이터에서 액면분할/병합 등으로 인한 이상치를 탐지하고,
    해당 종목의 전체 기간 데이터를 수정주가로 교체하여 데이터를 정제합니다.
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< 수정주가 데이터 정제기(Sanitizer) 시작 >>>")
    logger.info("="*80)
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # 1. 전체 원본 데이터 로드
    logger.info("전체 원본 데이터를 로드하여 분석합니다. (시간이 소요될 수 있습니다)")
    start_date, end_date = "2020-01-01", "2025-12-31"
    all_merged = load_partition_day(paths["merged"], start_date, end_date) # 모든 컬럼을 로드
    if all_merged.empty:
        logger.error("분석할 원본 데이터가 없습니다. '데이터 수집'을 먼저 실행해주세요.")
        return
        
    all_merged['date'] = pd.to_datetime(all_merged['date'])
    all_merged = all_merged.sort_values(['code', 'date'])

    # 2. 가격 이상치 탐지
    logger.info("가격 급변(전일 종가 대비 당일 시가 3배 이상) 종목을 탐지합니다...")
    price_cols = ['date', 'code', 'open', 'close']
    df_price_check = all_merged[price_cols].copy()
    df_price_check['prev_close'] = df_price_check.groupby('code')['close'].shift(1)
    df_price_check['open_close_ratio'] = (df_price_check['open'] / df_price_check['prev_close']).fillna(1)
    
    threshold = 3.0
    anomalies = df_price_check[(df_price_check['open_close_ratio'] >= threshold) | (df_price_check['open_close_ratio'] <= 1/threshold)]
    
    if anomalies.empty:
        logger.info("가격 데이터에서 수정주가 이벤트로 의심되는 이상치를 찾지 못했습니다. 데이터가 깨끗합니다.")
        return

    problem_tickers = anomalies['code'].unique()
    logger.warning(f"총 {len(problem_tickers)}개 종목에서 수정주가 이벤트로 의심되는 가격 변동을 탐지했습니다.")
    for _, row in anomalies.iterrows():
        logger.warning(f"  - 탐지: {row['date'].date()} / {row['code']} (비율: {row['open_close_ratio']:.2f})")

    # 3. 문제 종목 데이터 재수집
    logger.info(f"\n탐지된 {len(problem_tickers)}개 종목의 전체 기간 데이터를 수정주가로 다시 수집합니다...")
    
    corrected_data = []
    s_str, e_str = yyyymmdd(pd.to_datetime(start_date)), yyyymmdd(pd.to_datetime(end_date))

    for ticker in tqdm(problem_tickers, desc="수정주가 데이터 교체 중"):
        try:
            df_corrected_px = retry_request(stock.get_market_ohlcv, fromdate=s_str, todate=e_str, ticker=ticker, adjusted=True)
            if not df_corrected_px.empty:
                df_corrected_px['code'] = ticker
                corrected_data.append(df_corrected_px.reset_index())
        except Exception as e:
            logger.error(f"'{ticker}' 종목 데이터 재수집 중 오류 발생: {e}")
        time.sleep(0.2)

    if not corrected_data:
        logger.error("수정주가 데이터를 가져오지 못했습니다. 정제 작업을 중단합니다.")
        return
        
    df_final_corrected_px = pd.concat(corrected_data)
    df_final_corrected_px = _rename_ohlcv_cols(df_final_corrected_px)
    df_final_corrected_px['date'] = pd.to_datetime(df_final_corrected_px['date'])
    
    # [수정] 하드코딩 대신 collect 모듈에서 임포트한 표준 컬럼 목록을 사용
    standard_cols = REQUIRED_RAW_COLS
    df_final_corrected_px = df_final_corrected_px[[col for col in standard_cols if col in df_final_corrected_px.columns]]

    
    # 4. [로직 수정] 데이터 교체 작업
    # 4-1. 기존 데이터에서 문제 종목들의 데이터를 모두 제거
    logger.info(f"\n기존 데이터에서 탐지된 {len(problem_tickers)}개 종목의 데이터를 제거합니다...")
    df_clean_original = all_merged[~all_merged['code'].isin(problem_tickers)].copy()
    
    # 4-2. 새로 수집한 수정주가 데이터에 기존 수급 데이터를 다시 붙여줌
    ff_cols = ['date', 'code', 'inst_net_val', 'foreign_net_val']
    df_corrected_with_ff = pd.merge(df_final_corrected_px, all_merged[ff_cols], on=['date', 'code'], how='left')
    df_corrected_with_ff.fillna({'inst_net_val': 0, 'foreign_net_val': 0}, inplace=True)
    
    # 4-3. 문제없는 기존 데이터와, 새로 수집하여 보정한 데이터를 하나로 합침
    logger.info("문제없는 기존 데이터와 수정된 데이터를 병합하여 최종 정제 데이터를 생성합니다...")
    df_sanitized = pd.concat([df_clean_original, df_corrected_with_ff], ignore_index=True)

    # 5. 최종 정제된 전체 데이터를 Parquet 파일로 저장
    logger.info(f"총 {len(df_sanitized)}건의 정제된 데이터를 기존 Parquet 파일에 덮어씁니다...")
    
    # 기존 merged 폴더를 삭제하여 완전한 덮어쓰기 보장
    merged_path = paths["merged"]
    if os.path.exists(merged_path):
        import shutil
        shutil.rmtree(merged_path)
    
    save_parquet_partitioned_monthly(df_sanitized, merged_path)

    logger.info("="*80)
    logger.info("      <<< 데이터 정제 완료 >>>")
    logger.info(f"총 {len(problem_tickers)}개 종목의 데이터가 수정주가로 교체되었습니다.")
    logger.info("이제 '피처 데이터 재생성' -> 'ML 데이터셋 생성' -> '모델 재학습'을 진행해주세요.")
    logger.info("="*80)