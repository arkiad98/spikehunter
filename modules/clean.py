
# modules/clean.py
"""
[SpikeHunter] 데이터 정제 및 이상치 보정 모듈
- 기본적으로 수집된 데이터(merged)에서 이상치(상한가/하한가 폭 +30% 초과)를 감지합니다.
- 감지된 종목에 대해 pykrx의 '수정주가(adjusted=True)' 데이터를 다시 받아 교체를 시도합니다.
- 교체 후에도 이상치가 지속되면 경고를 남기고, 피처 생성 시 제외하도록 마킹(metadata)합니다.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pykrx import stock
from modules.utils_io import (
    load_partition_day, save_parquet_partitioned_monthly, 
    read_yaml, ensure_dir, retry_request
)
from modules.utils_logger import logger

def detect_anomalies(df: pd.DataFrame, threshold: float = 0.31) -> pd.DataFrame:
    """
    일별 수익률이 임계값(30% + alpha)을 초과하는 구간을 찾습니다.
    (상장폐지 정리매매, 감자/합병 등 이벤트성 등락)
    """
    df = df.sort_values('date')
    df['prev_close'] = df['close'].shift(1)
    df['pct_change'] = df['close'].pct_change()
    
    # 상장이후 첫날 등은 제외
    mask = (abs(df['pct_change']) > threshold) & (df['prev_close'] > 0)
    anomalies = df[mask]
    return anomalies

def fetch_adjusted_history(code: str, start: str, end: str) -> pd.DataFrame:
    """특정 종목의 수정주가 전체 히스토리를 가져옵니다."""
    try:
        # 시작/종료일 포맷 변환 (YYYYMMDD)
        s_str = pd.Timestamp(start).strftime("%Y%m%d")
        e_str = pd.Timestamp(end).strftime("%Y%m%d")
        
        df = retry_request(stock.get_market_ohlcv_by_date, s_str, e_str, code, adjusted=True)
        if df.empty: return pd.DataFrame()
        
        df = df.reset_index()
        # 컬럼명 표준화
        rename_map = {
            "날짜": "date", "시가": "open", "고가": "high", "저가": "low", "종가": "close",
            "거래량": "volume", "거래대금": "value"
        }
        df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch adjusted history for {code}: {e}")
        return pd.DataFrame()

def run_clean_pipeline(settings_path: str):
    logger.info("=== [Data Cleaning] 데이터 정제 및 이상치 보정 시작 ===")
    cfg = read_yaml(settings_path)
    base_dir = cfg['paths']['merged']
    
    # 1. 전체 데이터 로드 (최적화를 위해 최근 3년치만 먼저 스캔하거나, 파티션 전체 스캔)
    # 여기서는 "전체"를 스캔해야 정확함.
    # 하지만 load_partition_day로 전체를 로드하면 메모리 부족 가능성.
    # -> 종목별로 처리하는 것은 파티션 구조상 어려움 (Files are by Y/M, not by Code).
    
    # 전략:
    # 1) 전체 데이터를 로드하여 'code' 별로 그룹화하는 것은 무리.
    # 2) 파티션된 데이터를 순회하며 'pct_change'를 계산하기엔 연속성이 끊김.
    
    # 해결:
    # 파티션된 데이터 전체를 로드하되, 필요한 컬럼(date, code, close)만 로드하여 
    # 이상치를 먼저 탐지한 후, 해당 Code에 대해서만 전체 재수집 및 파티션 업데이트 수행.
    
    logger.info(" >> 전체 데이터 스캔 중... (Anomaly Detection)")
    
    # 3년치 정도면 충분하다고 가정 (또는 전체 기간)
    start_date = "2020-01-01"
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    df_scan = load_partition_day(base_dir, start_date, end_date, columns=['date', 'code', 'close'])
    if df_scan.empty:
        logger.warning("스캔할 데이터가 없습니다.")
        return

    # 종목별 등락률 계산
    df_scan = df_scan.sort_values(['code', 'date'])
    df_scan['prev_close'] = df_scan.groupby('code')['close'].shift(1)
    df_scan['pct_change'] = (df_scan['close'] - df_scan['prev_close']) / df_scan['prev_close']
    
    # 이상치 탐지
    anomalous_rows = df_scan[abs(df_scan['pct_change']) > 0.31] # 30% 상한가 제한 + 1% 버퍼
    anomalous_codes = anomalous_rows['code'].unique().tolist()
    
    # [NEW] Exclude already known excluded dates to avoid infinite loop of detection
    # If the date is already in exclude_dates.yaml, we don't need to re-fetch/re-clean it.
    exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
    if os.path.exists(exclude_path):
         # from modules.utils_io import read_yaml
         ex_cfg = read_yaml(exclude_path)
         exclusions = ex_cfg.get('exclusions', [])
         
         # Build a set of (code, date) to ignore
         ignore_set = set()
         for item in exclusions:
             c = item['code']
             for d in item['dates']:
                 ignore_set.add((c, pd.Timestamp(d).date()))
         
         # Check if filtered anomalies are truly new
         real_new_codes = []
         for code in anomalous_codes:
             # Get rows for this code
             rows = anomalous_rows[anomalous_rows['code'] == code]
             is_new = False
             for _, r in rows.iterrows():
                 if (r['code'], r['date'].date()) not in ignore_set:
                     is_new = True
                     break
             if is_new:
                 real_new_codes.append(code)
         
         anomalous_codes = real_new_codes

    logger.info(f" >> 감지된 이상 종목 수: {len(anomalous_codes)}개")
    if not anomalous_codes:
        logger.info(" >> 보정이 필요한 이상치가 없습니다.")
        return

    # 2. 이상 종목에 대해 수정주가 재수집 및 교체
    # 업데이트할 데이터프레임을 모아두었다가 한 번에 저장 (파티션 구조 유지)
    
    # '해당 종목들'에 대해서만 전체 데이터를 로드해서 덮어쓰기 위해,
    # 우선 전체를 로드하지 않고, "새로 받은 데이터"를 기존 파티션에 병합(Overwrite)하는 방식 사용.
    
    fixed_dfs = []
    
    for code in tqdm(anomalous_codes, desc="Correcting Anomalies"):
        # 전체 히스토리 재수집 (수정주가 적용)
        df_adj = fetch_adjusted_history(code, start_date, end_date)
        if df_adj.empty: continue
        
        # 재수집된 데이터도 이상치 체크 (여전히 30% 넘는지?)
        df_adj['prev_close'] = df_adj['close'].shift(1)
        df_adj['pct_change'] = (df_adj['close'] - df_adj['prev_close']) / df_adj['prev_close']
        
        # 여전히 이상치가 많다면? -> 그래도 '수정주가'가 원본보단 나을 확률이 높음.
        # 하지만 010145 사례처럼 +196%가 남을 수도 있음.
        # 일단은 교체 진행. (사용자 요청: "수정주가로 변경하는 로직")
        
        df_adj['code'] = code
        
        # Check available columns
        available = set(df_adj.columns)
        required = {'date', 'code', 'open', 'high', 'low', 'close', 'volume'}
        
        if not required.issubset(available):
            logger.warning(f"Missing required columns for {code}. Available: {available}")
            continue
            
        # Ensure 'value' exists
        if 'value' not in available:
            # Approximate value if missing
            df_adj['value'] = df_adj['close'] * df_adj['volume']
            
        cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'value']
        fixed_dfs.append(df_adj[cols])
            
    if fixed_dfs:
        logger.info(" >> 데이터 업데이트 및 병합 저장 중...")
        # 기존 데이터와 병합은 save_parquet_partitioned_monthly 내부에서 처리됨 (Code/Date 키 기준 덮어쓰기 로직 필요)
        # 현재 save_parquet_partitioned_monthly는 "로드 -> concat -> drop_duplicates -> save" 방식이므로
        # 새로운 df_adj를 넘기면 기존 데이터 위에 덮어써지게 됨 (중복 날짜/코드일 경우).
        # 단, 기존 파일에 'inst_net_val' 등이 있고 새 파일에 없으면? 
        # pandas concat 시 컬럼이 다르면 NaN이 됨. -> 수급 데이터 날라감.
        
        # [중요] 수급 데이터 보전 로직
        # "기존 파일 로드 -> 해당 코드의 가격 컬럼만 업데이트 -> 저장" 해야 함.
        # 그러나 파티션 구조라 파일이 월별로 쪼개져 있음.
        # 따라서 "새 데이터"를 월별로 쪼개서, 각 파티션 파일을 열고 -> 그 안에서 해당 코드 영역을 찾아 -> 가격만 업데이트 -> 저장.
        
        # 이를 수행하기 위해 save_parquet_partitioned_monthly 함수를 개선하거나,
        # 여기서 직접 월별 루프를 돌며 처리해야 함.
        
        df_all_fixed = pd.concat(fixed_dfs, ignore_index=True)
        # 수급 컬럼 0으로 채워서 병합 시 기존 데이터 날라가는 것 방지? 
        # 아니, concat하면 기존 파티션 파일(A) + 새 데이터(B) = C
        # A에만 수급이 있고 B엔 없으면 C에는 수급이 유지되나(NaN)? 아니면 B행이 A행을 대체하나?
        # drop_duplicates(subset=['date', 'code'], keep='last')를 쓰면 B가 A를 완전히 덮어씀.
        # -> 수급 데이터 유실됨.
        
        # 해결책: 교체할 데이터(B)에 기존 데이터(A)의 수급 정보를 Merge한 뒤 덮어써야 함.
        # 이건 복잡함.
        
        # 타협안: 
        # 1. 1단계: clean.py는 "가격 데이터"만 수정주가로 다시 받음.
        # 2. 2단계: 기존 파티션 파일을 순회하며, 수정할 종목/날짜가 있으면 '가격 컬럼'만 업데이트.
        
        _update_partitioned_prices(base_dir, df_all_fixed)
    
        # [NEW] Add persistent anomalies to 'config/exclude_dates.yaml' to exclude ONLY those dates
        # Identify which dates are still anomalous in the fixed data
        exclude_map = {} # code -> list of dates
        
        # 1. 새로 받은 데이터에서 다시 이상치 감지
        df_all_fixed['prev_close'] = df_all_fixed.groupby('code')['close'].shift(1)
        df_all_fixed['pct_change'] = (df_all_fixed['close'] - df_all_fixed['prev_close']) / df_all_fixed['prev_close']
        
        still_anomalous = df_all_fixed[abs(df_all_fixed['pct_change']) > 0.31]
        
        if not still_anomalous.empty:
            logger.info(f" >> 정제 후에도 이상치 잔존: {len(still_anomalous)}건 -> 제외 목록에 추가")
            
            for code, group in still_anomalous.groupby('code'):
                dates = group['date'].dt.strftime('%Y-%m-%d').tolist()
                exclude_map[code] = dates
                
        # 2. Save to YAML
        exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
        ensure_dir(os.path.dirname(exclude_path))
        
        # 2. Save to YAML
        exclude_path = os.path.join(os.path.dirname(settings_path) if settings_path else "config", "exclude_dates.yaml")
        ensure_dir(os.path.dirname(exclude_path))
        
        # from modules.utils_io import read_yaml, update_yaml # [Removed] Shadowing fix
        from ruamel.yaml import YAML
        
        yaml = YAML()
        current_data = {}
        if os.path.exists(exclude_path):
            with open(exclude_path, 'r', encoding='utf-8') as f:
                current_data = yaml.load(f) or {}
                
        # Merge logic
        if 'exclusions' not in current_data:
            current_data['exclusions'] = []
            
        # Convert list of dicts to a dict map for easy update
        # Structure: exclusions: [{code: "...", dates: [...]}, ...]
        existing_map = {item['code']: set(item['dates']) for item in current_data['exclusions']}
        
        for code, dates in exclude_map.items():
            if code in existing_map:
                existing_map[code].update(dates)
            else:
                existing_map[code] = set(dates)
                
        # Re-construct list
        new_exclusions = []
        for code, dates in existing_map.items():
            new_exclusions.append({'code': code, 'dates': sorted(list(dates))})
            
        current_data['exclusions'] = new_exclusions
        
        with open(exclude_path, 'w', encoding='utf-8') as f:
            yaml.dump(current_data, f)
            
        logger.info(f" >> 'exclude_dates.yaml' 업데이트 완료 ({len(new_exclusions)}개 종목)")

    logger.info("=== 데이터 정제 완료 ===")

def _update_partitioned_prices(base_dir: str, new_df: pd.DataFrame):
    """
    기존 파티션 파일의 가격 정보를 새로운 데이터프레임 값으로 부분 업데이트합니다.
    (수급 데이터 보존)
    """
    if new_df.empty: return
    
    new_df['Y'] = new_df['date'].dt.year
    new_df['M'] = new_df['date'].dt.month
    
    grouped = new_df.groupby(['Y', 'M'])
    
    for (y, m), group_new in tqdm(grouped, desc="Updating Partitions"):
        part_path = os.path.join(base_dir, f"Y={y}", f"M={m:02d}", "part.parquet")
        
        if not os.path.exists(part_path):
            # 파일이 없으면 그냥 저장 (수급은 0 처리)
            # 수급 컬럼 추가
            for c in ['inst_net_val', 'foreign_net_val']:
                group_new[c] = 0
            
            ensure_dir(os.path.dirname(part_path))
            group_new.drop(columns=['Y', 'M']).to_parquet(part_path, index=False)
            continue
            
        # 기존 파일 로드
        df_old = pd.read_parquet(part_path)
        
        # 업데이트할 코드 목록
        update_codes = group_new['code'].unique()
        
        # 1. 업데이트 대상이 아닌 행들은 그대로 유지 (df_keep)
        df_keep = df_old[~df_old['code'].isin(update_codes)].copy()
        
        # 2. 업데이트 대상인 행들은... 수급 정보(inst/foreign)를 살려야 함.
        # df_old에서 수급 정보만 추출
        df_old_targets = df_old[df_old['code'].isin(update_codes)][['date', 'code', 'inst_net_val', 'foreign_net_val', 'value']]
        
        # group_new에는 가격 정보가 있음.
        # group_new와 df_old_targets를 merge (left join implied by group_new dates)
        # group_new가 'master' (가격은 이게 맞음).
        
        group_new_clean = group_new.drop(columns=['Y', 'M'])
        
        # 기존 수급 데이터가 있으면 병합, 없으면 0
        # 주의: 날짜가 키.
        if 'inst_net_val' in df_old_targets.columns:
            # 병합
            merged_targets = pd.merge(
                group_new_clean, 
                df_old_targets[['date', 'code', 'inst_net_val', 'foreign_net_val']], # value는 새 데이터(거래대금) 사용
                on=['date', 'code'], 
                how='left'
            )
            # 만약 새 데이터에 날짜가 있는데 기존 데이터에 없으면 수급은 NaN -> 0 처리
            merged_targets['inst_net_val'] = merged_targets['inst_net_val'].fillna(0)
            merged_targets['foreign_net_val'] = merged_targets['foreign_net_val'].fillna(0)
            
        else:
            merged_targets = group_new_clean.copy()
            merged_targets['inst_net_val'] = 0
            merged_targets['foreign_net_val'] = 0
            
        # 3. 합치기 (Keep + Updated Targets)
        df_final = pd.concat([df_keep, merged_targets], ignore_index=True)
        
        # 저장
        df_final.to_parquet(part_path, index=False, compression='zstd')

if __name__ == "__main__":
    run_clean_pipeline("config/settings.yaml")
