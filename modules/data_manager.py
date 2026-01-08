# modules/data_manager.py
import os
import pandas as pd
import glob
from modules.utils_logger import logger
from modules.utils_io import read_yaml, yyyymmdd, read_meta, write_meta

def _delete_date_from_file(file_path: str, target_date: pd.Timestamp, date_col='date') -> bool:
    """단일 Parquet 파일에서 해당 날짜의 데이터를 삭제합니다."""
    if not os.path.exists(file_path):
        return False
        
    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            return False
            
        # date 컬럼 확인
        if date_col not in df.columns:
            return False
            
        initial_count = len(df)
        
        # 날짜 타입 변환 및 필터링
        df[date_col] = pd.to_datetime(df[date_col])
        df_filtered = df[df[date_col].dt.normalize() != target_date.normalize()]
        
        final_count = len(df_filtered)
        
        if initial_count != final_count:
            if final_count == 0:
                os.remove(file_path) # 비어있으면 파일 삭제
                logger.info(f"  [삭제] 파일 삭제됨 (데이터 없음): {file_path}")
            else:
                df_filtered.to_parquet(file_path, index=False)
                logger.info(f"  [수정] {file_path}: {initial_count}행 -> {final_count}행 ({initial_count - final_count}행 삭제)")
            return True
        return False
        
    except Exception as e:
        logger.error(f"파일 처리 중 오류 ({file_path}): {e}")
        return False

def _delete_date_from_partitioned_dir(base_dir: str, target_date: pd.Timestamp) -> int:
    """파티션된 디렉토리(Y=YYYY/M=MM)에서 해당 날짜의 데이터를 삭제합니다."""
    if not os.path.exists(base_dir):
        return 0
        
    year = target_date.year
    month = target_date.month
    
    # 파티션 경로 추정
    partition_path = os.path.join(base_dir, f"Y={year}", f"M={month:02d}", "part.parquet")
    
    # M=1, M=01 등 포맷 대응을 위해 globbing (utils_io의 방식은 M=02d)
    if not os.path.exists(partition_path):
        # 혹시 모를 다른 포맷 확인
        candidates = glob.glob(os.path.join(base_dir, f"Y={year}", f"M={month}*", "*.parquet"))
        if not candidates:
            return 0
        file_path = candidates[0]
    else:
        file_path = partition_path
        
    if _delete_date_from_file(file_path, target_date):
        return 1
    return 0

def delete_data_by_date(settings_path: str, target_date_str: str):
    """
    지정된 날짜의 데이터를 시스템 전반(Raw, Merged, ML, Predictions, DB)에서 삭제합니다.
    """
    try:
        target_date = pd.to_datetime(target_date_str)
    except Exception:
        logger.error(f"유효하지 않은 날짜 형식입니다: {target_date_str}")
        return

    logger.info(f"=== [{target_date.date()}] 데이터 일괄 삭제 시작 ===")
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    deleted_count = 0
    
    # 1. Raw Prices (Partitioned)
    raw_prices_dir = paths.get("raw_prices", "data/raw/prices")
    logger.info(f"1. Raw Prices 확인: {raw_prices_dir}")
    if _delete_date_from_partitioned_dir(raw_prices_dir, target_date):
        deleted_count += 1
        
    # 2. Raw Index (Single File)
    raw_index_dir = paths.get("raw_index", "data/raw/index")
    index_file = os.path.join(raw_index_dir, "kospi.parquet")
    logger.info(f"2. Raw Index 확인: {index_file}")
    if _delete_date_from_file(index_file, target_date):
        deleted_count += 1
        
    # 3. Merged Data (Partitioned)
    merged_dir = paths.get("merged", "data/proc/merged")
    logger.info(f"3. Merged Data 확인: {merged_dir}")
    if _delete_date_from_partitioned_dir(merged_dir, target_date):
        deleted_count += 1
        
    # 4. ML Dataset (Single Files)
    ml_dir = paths.get("ml_dataset", "data/proc/ml_dataset")
    logger.info(f"4. ML Dataset 확인: {ml_dir}")
    for file_name in ["ml_classification_dataset.parquet", "ml_regression_dataset.parquet"]:
        file_path = os.path.join(ml_dir, file_name)
        if _delete_date_from_file(file_path, target_date):
            deleted_count += 1

    # 5. Predictions (CSV File)
    pred_dir = paths.get("predictions", "data/proc/predictions")
    pred_file = os.path.join(pred_dir, f"{target_date.strftime('%Y-%m-%d')}.csv")
    logger.info(f"5. Prediction 확인: {pred_file}")
    if os.path.exists(pred_file):
        try:
            os.remove(pred_file)
            logger.info("  [삭제] 추천 결과 파일 삭제됨.")
            deleted_count += 1
        except Exception as e:
            logger.error(f"추천 파일 삭제 실패: {e}")

    # 6. DB Signals (Optional)
    # DB에서 해당 날짜의 PENDING 시그널 등을 지울 수도 있음
    # 필요하다면 추가 구현
    
    # 7. Metadata Rewind / Repair
    # 삭제한 날짜가 last_collected_date보다 아예 같거나 이전이면 (=정상 상황), last_collected_date를 삭제일 전날로 되돌림.
    # 반대로 last_collected_date가 삭제일보다 훨씬 이전이면 (=Meta가 Stale한 상황), 실제 데이터(Merged)를 확인해 Meta를 최신화(Repair).
    meta_dir = paths.get("meta", "data/meta")
    last_collected = read_meta(meta_dir, "last_collected_date")
    
    if last_collected:
        last_date_ts = pd.to_datetime(last_collected)
        
        # Case A: Normal Rewind
        if target_date <= last_date_ts:
            new_last = target_date - pd.Timedelta(days=1)
            write_meta(meta_dir, "last_collected_date", new_last.strftime("%Y-%m-%d"))
            logger.info(f"  [Meta] last_collected_date 업데이트(Rewind): {last_collected} -> {new_last.date()}")
            
        # Case B: Stale Meta Repair
        else:
            # target_date > last_collected (Meta가 실제 데이터보다 뒤처져 있음)
            # Merged 파일의 실제 마지막 날짜를 확인하여 Meta를 앞당겨줌.
            try:
                # 현재 삭제 작업이 일어난 파티션(또는 그 이전)을 확인
                # 삭제가 완료된 상태이므로, 파일에 남은 가장 최신 날짜가 '보유한 데이터의 끝'임.
                year = target_date.year
                month = target_date.month
                
                # util function logic reused inline for simplicity / or check partition
                p_path = os.path.join(merged_dir, f"Y={year}", f"M={month:02d}", "part.parquet")
                if not os.path.exists(p_path):
                     # 삭제로 인해 파일이 없어졌다면, 그 전날(target_date-1)이 유력하지만 확신 불가.
                     # 하지만 삭제를 시도했다는 건 데이터가 있었다는 뜻이므로, target_date-1로 복구하는 것이 합리적.
                     repaired_last = target_date - pd.Timedelta(days=1)
                else:
                    target_df = pd.read_parquet(p_path, columns=['date'])
                    if not target_df.empty:
                        repaired_last = pd.to_datetime(target_df['date']).max()
                    else:
                        repaired_last = target_date - pd.Timedelta(days=1)

                if repaired_last > last_date_ts:
                    write_meta(meta_dir, "last_collected_date", repaired_last.strftime("%Y-%m-%d"))
                    logger.info(f"  [Meta] last_collected_date 보정(Repair): {last_collected} -> {repaired_last.date()}")

            except Exception as e:
                logger.warning(f"  [Meta] Repair 시도 중 오류: {e}")

    logger.info(f"=== 삭제 완료 (총 {deleted_count}개 영역 수정됨) ===")
