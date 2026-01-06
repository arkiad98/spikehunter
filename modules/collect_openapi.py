import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from modules.utils_logger import logger
from modules.utils_io import to_date

def collect_openapi_index(start_date, end_date, api_key):
    """
    KRX OpenAPI (Debug Env)를 사용하여 KOSPI 지수 시세 데이터를 수집합니다.
    Spec: KRX_KOSPI_시세정보_개발명세서.docx
    URL: https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd
    Param: basDd (YYYYMMDD)
    Auth: Header 'AUTH_KEY' (Assumed)
    """
    s_date = to_date(start_date)
    e_date = to_date(end_date)
    
    logger.info(f"[OpenAPI] Fetching KOSPI Data: {s_date.date()} ~ {e_date.date()}")
    
    base_url = "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "AUTH_KEY": api_key  # Try Header first
    }
    
    all_data = []
    current_date = s_date
    
    # Iterate dates
    while current_date <= e_date:
        if current_date.weekday() >= 5: # Skip weekend locally to save calls, though API handles it
            current_date += timedelta(days=1)
            continue
            
        day_str = current_date.strftime("%Y%m%d")
        
        params = {
            "basDd": day_str
        }
        
        try:
            # 1. First Try: Header Auth
            response = requests.get(base_url, params=params, headers=headers, timeout=5)
            
            # If 401/403, try Param Auth? (Just in case spec is different)
            if response.status_code in [401, 403]:
                params["AUTH_KEY"] = api_key
                response = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            
            if response.status_code != 200:
                 logger.debug(f"[OpenAPI] {day_str} Skip (Status {response.status_code})")
                 current_date += timedelta(days=1)
                 continue
                 
            data = response.json()
            
            # Extract list from response (OutBlock1 usually)
            # Response structure needs inspection. Assuming { "OutBlock1": [...] } or similar list
            # If direct list:
            if isinstance(data, list):
                df_day = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Find the list value
                found_list = False
                for k, v in data.items():
                    if isinstance(v, list):
                        df_day = pd.DataFrame(v)
                        found_list = True
                        break
                if not found_list:
                    df_day = pd.DataFrame()
            else:
                df_day = pd.DataFrame()
                
            if not df_day.empty:
                # Add Date column if missing
                if 'BAS_DD' not in df_day.columns and 'date' not in df_day.columns:
                     df_day['date'] = current_date
                
                # Filter for KOSPI (Assuming 'IDX_NM' or 'IDX_IND_CD' exists)
                # We want standard KOSPI (1001). 
                # Let's keep all for now and filter later using map.
                all_data.append(df_day)
                
        except Exception as e:
            logger.warning(f"[OpenAPI] Error on {day_str}: {e}")
            
        # Respect rate limit
        time.sleep(0.1) 
        current_date += timedelta(days=1)
        
    if not all_data:
        return pd.DataFrame()
        
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Standardize Columns (Based on Actual API Response)
    # Response: date, IDX_CLSS, CLSPRC_IDX, OPNPRC_IDX, ...
    col_map = {
        'BAS_DD': 'date',
        'CLSPRC_IDX': 'close',
        'OPNPRC_IDX': 'open',
        'HGPRC_IDX': 'high',
        'LWPRC_IDX': 'low',
        'ACC_TRDVOL': 'volume',
        'ACC_TRDVAL': 'value',
        'IDX_NM': 'name',
    }
    df_all = df_all.rename(columns=col_map)
    
    # [Fix] Convert date to datetime to match existing parquet schema (Timestamp)
    if 'date' in df_all.columns:
        df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Filter for KOSPI (1001) using Data Logic
    # 1. Convert numeric columns, coerce errors (removes header/invalid rows like Row 0)
    num_cols = ['close', 'open', 'high', 'low', 'volume', 'value', 'MKTCAP']
    for c in num_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c].astype(str).str.replace(',', ''), errors='coerce')
            
    # 2. Drop rows where Close is NaN (Invalid data)
    df_all = df_all.dropna(subset=['close'])
    
    # 3. Filter by Name (if available) or Sort by Market Cap
    # "코스피" main index has the largest Market Cap usually (or 2nd after Total?)
    # We want the Representative Index.
    if 'name' in df_all.columns:
         # Rough filter for KOSPI family first
         mask = df_all['name'].str.contains('코스피|KOSPI', na=False)
         df_all = df_all[mask]
         
    # 4. Sort by MKTCAP descending -> Top 1 is likely KOSPI Main
    if 'MKTCAP' in df_all.columns and not df_all.empty:
         df_all = df_all.sort_values(by='MKTCAP', ascending=False)
         # Pick top 1
         # But verify it's not "KOSPI 200" if Top 1 is KOSPI 200?
         # KOSPI (Index) ~ 1996T Cap. KOSPI 200 ~ 1756T.
         # So KOSPI > KOSPI 200. Correct.
         df_all = df_all.iloc[:1]
    
    # Drop usage columns
    drop_cols = ['name', 'MKTCAP', 'IDX_CLSS', 'CMPPREVDD_IDX', 'FLUC_RT']
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns])
         
    return df_all
