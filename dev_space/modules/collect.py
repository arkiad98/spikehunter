# modules/collect.py (ver 2.8.17, API 호출 제한 방지 로직 적용)
"""데이터 수집 모듈.

[v2.8.17] 대한민국 공휴일 처리 최종 로직 적용
- 문제: pd.date_range(freq="B")가 주말만 제외하고 공휴일은 포함하여,
  공휴일에 데이터 수집을 시도하며 불필요한 오류 로그가 발생하는 문제 확인.
- 해결: 데이터 수집 시작 시, pykrx를 통해 전체 수집 기간의 실제 '영업일'
  목록을 미리 조회하는 로직을 run_collect 함수에 추가. 이제 수집 루프는
  이 영업일 목록을 기준으로만 동작하여 공휴일에는 API 호출 자체를
  시도하지 않도록 근본적으로 수정.
- 효과: 모든 종류의 비영업일(주말, 공휴일, 대체공휴일, 임시공휴일)을
  완벽하게 건너뛰어 데이터 수집 효율성을 극대화하고, 오류 로그 발생을
  원천적으로 차단하여 파이프라인의 안정성을 완성.

[수정] API 호출 제한 방지를 위해 time.sleep() 추가
- 문제: 짧은 시간 안에 과도한 API 요청으로 IP가 차단되거나 요청이 거부될 수 있는 문제.
- 해결: 일별 데이터 수집 루프에 `time.sleep()`을 추가하여 각 요청 사이에
  의도적인 지연 시간을 부여. `run_collect` 함수에 `delay_seconds` 파라미터를
  추가하여 지연 시간을 유연하게 조절할 수 있도록 개선.
- 효과: 안정적인 데이터 수집 환경을 구축하고 API 서버 부하를 최소화.
"""
import os
import gc
import shutil
import time  # 🔴 API 호출 지연을 위해 time 모듈 임포트
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from pykrx import stock

# 프로젝트 유틸리티 모듈 임포트
from modules.utils_io import (
    ensure_dir, read_yaml, to_date, yyyymmdd, retry_request,
    save_parquet_partitioned_monthly, write_meta, read_meta,
    load_partition_day,
    downcast_numeric
)
from modules.utils_logger import logger

REQUIRED_RAW_COLS = [
    "date", "code", "open", "high", "low", "close", "volume", "value",
    "inst_net_val", "foreign_net_val"
]

# [추가] 월별 데이터의 완결성을 검사하는 헬퍼 함수
def _check_month_completeness(ym_period: pd.Period, base_dir: str, all_trading_days: pd.DatetimeIndex) -> bool:
    """
    주어진 월(YYYY-MM)의 데이터가 마지막 영업일까지 수집되었는지 확인합니다.
    """
    path = os.path.join(base_dir, f"Y={ym_period.year}", f"M={ym_period.month:02d}", "part.parquet")
    
    if not os.path.exists(path):
        return False  # 파일이 없으면 미완결

    try:
        # 해당 월의 마지막 영업일 찾기
        last_trading_day_of_month = all_trading_days[
            (all_trading_days.year == ym_period.year) & (all_trading_days.month == ym_period.month)
        ].max()

        if pd.isna(last_trading_day_of_month):
            return True # 해당 월에 영업일이 없으면 완료된 것으로 간주

        # 저장된 데이터의 마지막 날짜 확인
        df = pd.read_parquet(path, columns=['date'])
        last_date_in_file = pd.to_datetime(df['date']).max()

        # 데이터의 마지막 날짜가 해당 월의 마지막 영업일과 같거나 크면 완결
        return last_date_in_file >= last_trading_day_of_month

    except Exception as e:
        logger.warning(f"{ym_period} 완결성 검사 중 오류 발생: {e}. 미완결로 처리합니다.")
        return False

def date_blocks(start: pd.Timestamp, end: pd.Timestamp, months: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """주어진 시작일과 종료일을 n개월 단위의 블록으로 나눕니다."""
    s = pd.Timestamp(start).normalize().replace(day=1)
    e = pd.Timestamp(end).normalize()
    out = []
    cur = s
    while cur <= e:
        nxt = (cur + pd.DateOffset(months=months)) - pd.Timedelta(days=1)
        if nxt > e:
            nxt = e
        out.append((cur, nxt))
        cur = nxt + pd.Timedelta(days=1)
    return out

def _rename_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """pykrx OHLCV 데이터프레임의 컬럼명을 표준 영문명으로 변경합니다."""
    rename_map = {
        "날짜": "date", "Date": "date",
        "시가": "open", "고가": "high", "저가": "low", "종가": "close",
        "거래량": "volume", "거래대금": "value",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Value": "value"
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

def _fetch_and_merge_daily_data(d: pd.Timestamp, required_cols: list) -> pd.DataFrame:
    """[v2.8.11 신규] 특정일의 모든 가격과 수급 데이터를 가져와 병합하는 통합 함수."""
    day_str = yyyymmdd(d)

    # 1. 가격 데이터(OHLCV) 수집
    try:
        df_kospi_px = retry_request(stock.get_market_ohlcv, date=day_str, market="KOSPI")
        df_kosdaq_px = retry_request(stock.get_market_ohlcv, date=day_str, market="KOSDAQ")
        df_px = pd.concat([df_kospi_px, df_kosdaq_px])
        
        if df_px.empty:
            return pd.DataFrame() 

        df_px.index.name = "code"
        df_px = df_px.reset_index()
    except Exception as e:
        logger.warning(f"{day_str} 가격 데이터 수집 중 오류 발생: {e}")
        return pd.DataFrame()

    df_px = _rename_ohlcv_cols(df_px)

    # 2. 수급 데이터 수집 (효율적인 최종 로직)
    try:
        all_fund_flow_data = []
        for market in ["KOSPI", "KOSDAQ"]:
            df_foreign = retry_request(
                stock.get_market_net_purchases_of_equities, day_str, day_str, market, "외국인"
            )
            df_foreign = df_foreign[['순매수거래대금']].rename(columns={'순매수거래대금': 'foreign_net_val'})

            df_inst = retry_request(
                stock.get_market_net_purchases_of_equities, day_str, day_str, market, "기관합계"
            )
            df_inst = df_inst[['순매수거래대금']].rename(columns={'순매수거래대금': 'inst_net_val'})
            
            df_market_ff = pd.merge(df_foreign, df_inst, left_index=True, right_index=True, how='outer')
            all_fund_flow_data.append(df_market_ff)

        df_ff = pd.concat(all_fund_flow_data)
        df_ff.index.name = 'code'
        df_ff = df_ff.reset_index()

    except Exception as e:
        logger.warning(f"{day_str} 수급 데이터 수집 중 오류 발생: {e}. 가격 데이터만 처리합니다.")
        df_ff = pd.DataFrame()

    # 3. 데이터 처리 및 병합
    if not df_ff.empty:
        df_px['code'] = df_px['code'].astype(str)
        df_ff['code'] = df_ff['code'].astype(str)
        df_merged = pd.merge(df_px, df_ff, on="code", how="left")
    else:
        df_merged = df_px.copy()

    for col in ['inst_net_val', 'foreign_net_val']:
        if col not in df_merged.columns:
            df_merged[col] = 0.0
    df_merged[['inst_net_val', 'foreign_net_val']] = df_merged[['inst_net_val', 'foreign_net_val']].fillna(0)

    # 4. 최종 데이터 정제
    df_merged["date"] = pd.to_datetime(d.date())
    df_merged["code"] = df_merged["code"].astype(str)
    
    final_cols = [col for col in required_cols if col in df_merged.columns]
    df_final = df_merged[final_cols]
    
    df_final = df_final.dropna(subset=["date", "code", "open", "close", "value"])
    df_final = df_final[(df_final["open"] > 0) & (df_final["close"] > 0) & (df_final["value"] >= 0)]
    
    return downcast_numeric(df_final, 
                            price_cols=["open", "high", "low", "close"], 
                            value_cols=["value", "inst_net_val", "foreign_net_val"], 
                            vol_cols=["volume"])
    # [수정] 하드코딩된 리스트 대신, 모듈 상단에 정의된 상수를 사용
    #final_cols = [col for col in REQUIRED_RAW_COLS if col in df_merged.columns]
    #df_final = df_merged[final_cols]
    
    #df_final = df_final.dropna(subset=["date", "code", "open", "close", "value"])
    
# [추가] KOSPI 지수 데이터를 수집하고 저장하는 내부 함수
def _collect_and_save_index_data(start_d: pd.Timestamp, end_d: pd.Timestamp, index_path: str):
    """KOSPI 지수 데이터를 증분 방식으로 수집하여 단일 Parquet 파일로 저장합니다."""
    ensure_dir(index_path)
    index_file = os.path.join(index_path, "kospi.parquet")
    
    # 기존 데이터가 있으면 마지막 날짜를 확인하여 이후 데이터만 요청
    if os.path.exists(index_file):
        existing_df = pd.read_parquet(index_file)
        last_date = existing_df['date'].max()
        fetch_start_d = last_date + pd.Timedelta(days=1)
    else:
        existing_df = pd.DataFrame()
        fetch_start_d = start_d

    if fetch_start_d > end_d:
        logger.info("KOSPI 지수 데이터는 이미 최신 상태입니다.")
        return

    logger.info(f"KOSPI 지수 데이터 수집: {fetch_start_d.date()} ~ {end_d.date()}")
    try:
        # 넉넉하게 이전 데이터를 포함하여 요청 후 필터링 (MA 계산 등을 위해)
        s_date_str = (fetch_start_d - pd.DateOffset(days=365)).strftime('%Y%m%d')
        e_date_str = end_d.strftime('%Y%m%d')
        
        new_df = retry_request(stock.get_index_ohlcv, s_date_str, e_date_str, "1001").reset_index()
        new_df = _rename_ohlcv_cols(new_df)
        new_df['date'] = pd.to_datetime(new_df['date'])

        # 기존 데이터와 병합 후 중복 제거
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date'], keep='last').sort_values('date')
        combined_df.to_parquet(index_file, index=False, compression="zstd")
        logger.info(f"KOSPI 지수 데이터가 '{index_file}'에 저장되었습니다.")

    except Exception as e:
        logger.error(f"KOSPI 지수 데이터 수집/저장 중 오류 발생: {e}", exc_info=True)

# [수정] run_collect 함수를 아래 코드로 전체 교체합니다.
# modules/collect.py
def run_collect(settings_path: str, start: str = None, end: str = None, use_meta: bool = True, overwrite: bool = False, delay_seconds: float = 0.5):
    """
    데이터 수집 파이프라인의 메인 실행 함수.
    """
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    if overwrite:
        for key in ["raw_prices", "raw_fundflow", "merged", "raw_index"]:
            if paths.get(key) and os.path.exists(paths[key]):
                logger.warning(f"Overwrite: 기존 '{paths[key]}' 디렉터리를 삭제합니다.")
                shutil.rmtree(paths[key])
        use_meta = False

    for p in paths.values(): ensure_dir(p)

    today = pd.Timestamp.today().normalize()
    start_d = to_date(start) if start else to_date('2020-01-01')
    end_d = to_date(end) if end else today

    if use_meta and not overwrite:
        last = read_meta(paths["meta"], "last_collected_date")
        if last: start_d = max(start_d, to_date(last) + pd.Timedelta(days=1))

    if start_d > end_d:
        logger.info("수집할 신규 데이터 구간이 없습니다.")
        return True

    logger.info(f"데이터 수집 시작: {start_d.date()} ~ {end_d.date()}")
    
    try:
        logger.info("전체 수집 기간의 실제 영업일 목록을 조회합니다...")
        all_trading_days_full_period = stock.get_index_ohlcv('20200101', end_d.strftime('%Y%m%d'), "1001").index
    except Exception as e:
        logger.error(f"영업일 목록 조회 중 오류 발생: {e}. 데이터 수집을 중단합니다.")
        return False

    trading_days_in_period = all_trading_days_full_period[
        (all_trading_days_full_period >= start_d) & (all_trading_days_full_period <= end_d)
    ]
    
    # [수정] 이 라인을 추가하여 'overwrite=True'일 때 final_days_to_collect 변수가 정의되도록 합니다.
    final_days_to_collect = trading_days_in_period

    if not overwrite:
        logger.info("="*50)
        logger.info("      <<< 월별 데이터 완결성 검사 시작 >>>")
        logger.info("="*50)
        
        months_to_check = sorted(list(set(pd.PeriodIndex(trading_days_in_period, freq='M'))))
        
        incomplete_months = []
        for ym_period in tqdm(months_to_check, desc="Checking Month Completeness"):
            is_last_month = (ym_period == months_to_check[-1]) if months_to_check else False
            if is_last_month or not _check_month_completeness(ym_period, paths["merged"], all_trading_days_full_period):
                incomplete_months.append(ym_period)
            else:
                logger.info(f"☑️  {ym_period} 데이터는 이미 완결되어 건너뜁니다.")
        
        if not incomplete_months:
            logger.info("모든 월별 데이터가 완결되었습니다. 수집할 신규 데이터가 없습니다.")
            write_meta(paths["meta"], "last_collected_date", str(end_d.date()))
            return True

        final_days_to_collect = trading_days_in_period[
            pd.to_datetime(trading_days_in_period).to_period('M').isin(incomplete_months)
        ]

    if final_days_to_collect.empty:
        logger.info("수집할 영업일이 없습니다.")
        return True

    logger.info(f"총 {len(final_days_to_collect)}일의 영업일 데이터를 수집합니다.")
    
    all_data_collected = []
    for day in tqdm(final_days_to_collect, desc="일별 데이터 통합 수집"):
        daily_merged_data = _fetch_and_merge_daily_data(day, REQUIRED_RAW_COLS)
        if daily_merged_data is not None and not daily_merged_data.empty:
            all_data_collected.append(daily_merged_data)
        time.sleep(delay_seconds)
    
    if all_data_collected:
        block_df = pd.concat(all_data_collected, ignore_index=True)
        save_parquet_partitioned_monthly(block_df, paths["merged"])
        del block_df, all_data_collected
        gc.collect()
    else:
        logger.warning("수집된 신규 데이터가 없습니다.")

    logger.info("="*50)
    logger.info("      <<< KOSPI 지수 데이터 수집 시작 >>>")
    logger.info("="*50)
    _collect_and_save_index_data(to_date('2020-01-01'), end_d, paths["raw_index"])

    try:
        df_check = load_partition_day(paths["merged"], start_d, end_d)

        if df_check.empty:
            if not trading_days_in_period.empty:
                logger.error(f"데이터 수집 실패: 영업일({len(trading_days_in_period)}일) 데이터가 없습니다.")
                return False
            else:
                logger.info("수집 기간에 영업일이 없어 신규 데이터가 없습니다. (정상)")
                write_meta(paths["meta"], "last_collected_date", str(end_d.date()))
                return True

        last_saved = df_check['date'].max()
        if pd.notna(last_saved):
             write_meta(paths["meta"], "last_collected_date", str(end_d.date()))
             logger.info(f"데이터 수집 완료. last_collected_date = {end_d.date()}")
             return True
        else:
            logger.error(f"데이터 수집 실패: 최신 데이터 누락 (마지막 저장일: {last_saved})")
            return False
    except Exception as e:
        logger.error(f"수집 성공 여부 확인 중 예외 발생: {e}", exc_info=True)
        return False