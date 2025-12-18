# modules/utils_io.py (ver 2.8.21)
"""프로젝트 전반에 사용되는 입출력(I/O) 및 데이터 처리 유틸리티 함수 모음.

[v2.8.21] 캐시 잠금(Lock) 메커니즘 안정성 강화
- 문제: 이전 프로세스가 비정상 종료되며 남겨진 .lock 파일로 인해,
  새로운 프로세스가 KOSPI 데이터 캐시 생성에 실패하고 TimeoutError로
  종료되는 교착 상태(Deadlock) 문제 발생.
- 해결: get_cached_kospi_data 함수에 오래된 잠금 파일을 감지하고
  자동으로 정리하는 로직을 추가. .lock 파일 발견 시, 파일 생성 시간을
  확인하여 5분이 경과했으면 이를 'stale lock'으로 간주하고 삭제한 후
  캐시 생성을 재시도하도록 수정.
- 효과: 어떤 상황에서도 캐시 생성 로직이 교착 상태에 빠지지 않고 안정적으로
  동작하도록 하여 파이프라인의 전체 안정성을 크게 향상.
"""
import io
import os
import json
import time
import random
import gc
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np # 파일 상단에 numpy 임포트 추가
from modules.utils_logger import logger
import sys  # [추가]
from ruamel.yaml import YAML
# --- 디렉터리 및 파일 관리 ---

def ensure_dir(path: str):
    """주어진 경로에 디렉터리가 없으면 생성합니다."""
    os.makedirs(path, exist_ok=True)

# [수정] read_yaml 함수를 아래 코드로 교체
def read_yaml(path: str) -> Dict[str, Any]:
    """YAML 파일을 라운드트립 모드로 읽어 딕셔너리로 반환합니다."""
    try:
        # ruamel.yaml 인스턴스를 라운드트립 모드(기본값)로 생성
        yaml = YAML()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        return data
    except Exception as e:
        from .utils_logger import logger
        logger.error(f"YAML 파일('{path}')을 읽는 중 오류 발생: {e}", exc_info=True)
        return {}

# [수정] update_yaml 함수를 아래 코드로 교체
def update_yaml(file_path: str, key1: str, key2: str, params_to_update: dict):
    """
    YAML 파일에서 특정 중첩 키 아래의 파라미터들만 선택적으로 업데이트합니다.
    (앵커/별칭 구조를 최대한 보존)
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    if data and key1 in data and key2 in data[key1]:
        # 대상 딕셔너리에 있는 키들만 업데이트
        for key, value in params_to_update.items():
            if key in data[key1][key2]:
                data[key1][key2][key] = value
            else:
                # 새로운 파라미터가 추가되는 경우 (일반적으로는 발생하지 않음)
                data[key1][key2][key] = value
    else:
        logger.error(f"'{file_path}'에서 업데이트할 경로 '{key1}.{key2}'를 찾을 수 없습니다.")
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

def write_meta(meta_dir: str, key: str, value: Any):
    """메타데이터 파일(JSON)에 키-값 쌍을 기록합니다."""
    ensure_dir(meta_dir)
    path = os.path.join(meta_dir, "metadata.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data[key] = value
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def read_meta(meta_dir: str, key: str, default: Any = None) -> Any:
    """메타데이터 파일(JSON)에서 키에 해당하는 값을 읽어옵니다."""
    path = os.path.join(meta_dir, "metadata.json")
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get(key, default)
    except (json.JSONDecodeError, FileNotFoundError):
        return default

# --- 날짜 및 시간 처리 ---

def to_date(s: Any) -> Optional[pd.Timestamp]:
    """다양한 형태의 입력을 Pandas Timestamp 객체로 변환합니다."""
    if s is None: return None
    if isinstance(s, (pd.Timestamp, datetime)): return pd.Timestamp(s).normalize()
    try:
        return pd.Timestamp(str(s)).normalize()
    except (ValueError, TypeError):
        return None

def yyyymmdd(dt: pd.Timestamp) -> str:
    """Timestamp 객체를 'YYYYMMDD' 형식의 문자열로 변환합니다."""
    return pd.Timestamp(dt).strftime("%Y%m%d")

def months_between(start: Any, end: Any) -> List[str]:
    """두 날짜 사이의 모든 월(YYYY-MM) 목록을 반환합니다."""
    s = to_date(start)
    e = to_date(end)
    if s is None or e is None:
        return []
    return [d.strftime("%Y-%m") for d in pd.date_range(s.replace(day=1), e, freq='MS')]

# --- 데이터 수집 및 캐싱 ---

def retry_request(func, *args, max_attempts=3, backoff_sec=2.0, **kwargs):
    """API 요청 실패 시 지수 백오프 방식으로 재시도합니다."""
    from .utils_logger import logger
    for i in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Request failed (attempt {i+1}/{max_attempts}): {e}. Retrying in {backoff_sec * (2**i)}s...")
            if i == max_attempts - 1: raise e
            time.sleep(backoff_sec * (2 ** i))
# [수정] get_cached_kospi_data 함수를 아래 코드로 전체 교체합니다.
# pykrx 호출, 캐싱, 잠금 로직이 모두 제거되고 파일을 직접 읽도록 변경됩니다.
def load_index_data(start_d: pd.Timestamp, end_d: pd.Timestamp, index_dir: str) -> pd.DataFrame:
    """사전에 수집된 KOSPI 지수 데이터를 파일에서 로드합니다."""
    from .utils_logger import logger
    index_file = os.path.join(index_dir, "kospi.parquet")

    if not os.path.exists(index_file):
        logger.error(f"KOSPI 지수 파일이 없습니다: {index_file}. '데이터 수집' 메뉴를 먼저 실행해주세요.")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(index_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # 날짜 필터링
        mask = (df['date'] >= start_d) & (df['date'] <= end_d)
        filtered_df = df.loc[mask].copy()

        # pykrx 원본 컬럼명과의 호환성을 위해 컬럼명 변경
        # backtest.py, predict.py 등에서 'kospi_close'를 사용
        rename_map = {
            'close': 'kospi_close', 
            'value': 'kospi_value',
            'date': 'date'
        }
        # 필요한 컬럼만 선택하고 이름 변경
        final_df = filtered_df[[col for col in rename_map.keys() if col in filtered_df.columns]].rename(columns=rename_map)
        
        return final_df
    except Exception as e:
        logger.error(f"KOSPI 지수 파일 로드 중 오류 발생: {e}", exc_info=True)
        return pd.DataFrame()


def get_stock_names(tickers: List[str]) -> Dict[str, str]:
    """티커 목록에 해당하는 종목명을 반환합니다."""
    from pykrx import stock
    names = {}
    for ticker in tickers:
        try:
            name = stock.get_market_ticker_name(ticker)
            names[ticker] = name
        except Exception:
            names[ticker] = "N/A"
    return names

# --- Parquet 파티션 데이터 처리 ---

def save_parquet_partitioned_monthly(df: pd.DataFrame, base_dir: str):
    """데이터프레임을 '연도=YYYY/월=MM' 구조의 월별 파티션으로 저장합니다."""
    if df.empty: return
    df["Y"] = df["date"].dt.year
    df["M"] = df["date"].dt.month
    for (y, m), g in df.groupby(["Y", "M"]):
        part_dir = os.path.join(base_dir, f"Y={y}", f"M={m:02d}")
        ensure_dir(part_dir)
        out_path = os.path.join(part_dir, "part.parquet")
        g = g.drop(columns=["Y", "M"])
        if os.path.exists(out_path):
            merged = pd.concat([pd.read_parquet(out_path), g]).drop_duplicates(subset=["date", "code"])
            merged.to_parquet(out_path, compression="zstd", index=False)
        else:
            g.to_parquet(out_path, compression="zstd", index=False)
    del df; gc.collect()

def list_partition_months(base_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """지정된 기간 내에 데이터가 존재하는 월(YYYY-MM) 목록을 반환합니다."""
    targets = set(months_between(start, end))
    available = []
    if not os.path.exists(base_dir): return []
    for y_dir in os.listdir(base_dir):
        if not (y_dir.startswith("Y=") and os.path.isdir(os.path.join(base_dir, y_dir))): continue
        for m_dir in os.listdir(os.path.join(base_dir, y_dir)):
            if not (m_dir.startswith("M=") and os.path.isdir(os.path.join(base_dir, y_dir, m_dir))): continue
            ym = f"{y_dir[2:]}-{int(m_dir[2:]):02d}"
            if ym in targets: available.append(ym)
    return sorted(available)

def load_partition_day(base_dir: str, start_date: Any, end_date: Any, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """월별 파티션된 Parquet 데이터를 특정 기간에 맞춰 로드합니다."""
    s_date = to_date(start_date)
    e_date = to_date(end_date)
    if s_date is None or e_date is None:
        from .utils_logger import logger
        logger.error(f"Invalid date range provided: start={start_date}, end={end_date}")
        return pd.DataFrame()

    date_col_requested = True
    cols_to_load = columns
    if columns is not None:
        if 'date' not in columns:
            date_col_requested = False
            cols_to_load = list(set(columns + ['date']))

    ym_list = months_between(s_date, e_date)
    frames = []
    for ym in ym_list:
        year, month = ym.split("-")
        path = os.path.join(base_dir, f"Y={int(year)}", f"M={int(month):02d}", "part.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path, columns=cols_to_load)
                df['date'] = pd.to_datetime(df['date'])
                filtered_df = df[(df['date'] >= s_date) & (df['date'] <= e_date)]
                if not filtered_df.empty:
                    frames.append(filtered_df)
            except Exception as e:
                from .utils_logger import logger
                logger.error(f"Failed to read or process parquet file {path}: {e}")

    if not frames:
        return pd.DataFrame()

    final_df = pd.concat(frames, ignore_index=True)

    if not date_col_requested:
        final_df = final_df.drop(columns=['date'])

    return final_df

# [추가] run_pipeline.py에서 옮겨온 get_user_input 함수
def get_user_input(prompt: str) -> str:
    """사용자 입력을 처리하고 로깅하는 헬퍼 함수."""
    # 순환 참조를 피하기 위해 함수 내에서 logger를 import
    from .utils_logger import logger
    logger.info(f"User Prompt: \"{prompt.strip()}\"")
    original_stdout = sys.stdout
    response = ""
    try:
        sys.stdout = sys.__stdout__
        response = input(prompt)
    finally:
        sys.stdout = original_stdout
    logger.info(f"User Response: \"{response}\"")
    return response

# --- 데이터 타입 최적화 ---

def downcast_numeric(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """메모리 사용량 감소를 위해 숫자형 데이터의 타입을 다운캐스팅합니다."""
    df = df.copy()
    for col_type, cols in kwargs.items():
        for c in cols:
            if c in df.columns:
                if 'price' in col_type or 'value' in col_type or 'float' in col_type:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
                elif 'vol' in col_type or 'int' in col_type:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    return df

# [추가] 데이터프레임 메모리 사용량을 최적화하는 함수
def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 각 컬럼을 메모리에 가장 효율적인 타입으로 변환합니다.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in str(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if logger:
        logger.info(f"메모리 사용량 최적화 완료: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% 감소)")
    
    return df


def update_strategies_with_anchor(file_path: str, r1_params: dict, other_regime_params: dict, anchor_name: str):
    """
    R1 전략의 max_market_vol에 앵커를 설정하고, 다른 체제들의 파라미터를 업데이트하는
    전용 YAML 업데이트 함수. (텍스트 처리 방식을 이용한 최종 안정화 버전)
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=6, offset=4) # 원본 파일의 들여쓰기 스타일에 맞게 조정

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    # 1. 메모리상의 데이터 객체(딕셔너리)에서 모든 값을 우선 업데이트합니다.
    all_params = {"SpikeHunter_R1_BullStable": r1_params, **other_regime_params}
    
    for regime_key, params in all_params.items():
        if regime_key in data['strategies']:
            for key, value in params.items():
                if key in data['strategies'][regime_key]:
                    data['strategies'][regime_key][key] = value

    # 2. 업데이트된 데이터를 파일이 아닌, 메모리상의 문자열로 변환합니다.
    string_stream = io.StringIO()
    yaml.dump(data, string_stream)
    yaml_string = string_stream.getvalue()

    # 3. 텍스트를 직접 수정하여 앵커(&)와 별칭(*)을 정확하게 설정합니다.
    lines = yaml_string.split('\n')
    final_lines = []
    r1_vol_value = r1_params.get('max_market_vol')

    # R1, R2, R3, R4 키 목록
    regime_keys = [
        "SpikeHunter_R1_BullStable", 
        "SpikeHunter_R2_BullVolatile", 
        "SpikeHunter_R3_BearStable", 
        "SpikeHunter_R4_BearStable"
    ]
    
    current_regime_key = None
    for line in lines:
        stripped_line = line.strip()

        # 현재 어느 전략 블록에 있는지 확인
        for key in regime_keys:
            if stripped_line.startswith(key + ":"):
                current_regime_key = key
                break
        
        # max_market_vol 라인 처리
        if stripped_line.startswith("max_market_vol:"):
            if current_regime_key == "SpikeHunter_R1_BullStable":
                # R1 블록이면 앵커를 추가
                indentation = line[:line.find('m')] # 들여쓰기 유지
                final_lines.append(f"{indentation}max_market_vol: &{anchor_name} {r1_vol_value}")
            elif current_regime_key in regime_keys:
                # R2, R3, R4 블록이면 별칭으로 교체
                indentation = line[:line.find('m')]
                final_lines.append(f"{indentation}max_market_vol: *{anchor_name}")
        else:
            final_lines.append(line)

    # 마지막에 추가될 수 있는 빈 줄 제거
    final_yaml_string = '\n'.join(final_lines).rstrip() + '\n'
    
    # 4. 최종적으로 완성된 문자열을 파일에 씁니다.
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_yaml_string)