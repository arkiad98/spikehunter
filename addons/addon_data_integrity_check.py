# addons/addon_data_integrity_check.py
"""
수집된 원본 데이터와 생성된 피처 데이터의 완결성 및 분포를
검사하여 데이터 파이프라인의 문제를 진단하는 애드온 모듈.
"""
import os
import pandas as pd
from tabulate import tabulate

from modules.utils_io import read_yaml, load_partition_day
from modules.utils_logger import logger
from modules.derive import _get_feature_cols # [추가] 피처 목록을 직접 가져옵니다.

def analyze_feature_distribution(features_df: pd.DataFrame): # [수정] cfg 인자 제거
    """모든 피처의 분포(결측치, 0의 비율 등)를 분석합니다."""
    
    # [수정] 데이터프레임의 컬럼을 직접 넘겨 피처 목록을 동적으로 가져옴
    features_to_check = _get_feature_cols(features_df.columns)
    
    results = []
    for feature in sorted(features_to_check): # 정렬하여 보기 좋게 출력
        if feature not in features_df.columns:
            results.append([feature, "N/A", "N/A", "N/A", "피처 없음"])
            continue
            
        total_count = len(features_df)
        nan_count = features_df[feature].isna().sum()
        zero_count = (features_df[feature] == 0).sum()
        
        nan_pct = (nan_count / total_count) * 100 if total_count > 0 else 0
        zero_pct = (zero_count / total_count) * 100 if total_count > 0 else 0
        
        meaningful_data = features_df.loc[features_df[feature] != 0, feature]
        if not meaningful_data.empty:
            stats = f"평균={meaningful_data.mean():.2f}, 최대={meaningful_data.max():.2f}"
        else:
            stats = "유의미 데이터 없음"

        results.append([feature, f"{nan_pct:.2f}%", f"{zero_pct:.2f}%", f"{100 - nan_pct - zero_pct:.2f}%", stats])

    headers = ["피처 이름", "결측치 비율", "0 값 비율", "유효값 비율", "유효값 통계"]
    logger.info("\n" + "="*80)
    logger.info("      <<< 전체 피처 데이터 분포 분석 결과 >>>")
    logger.info("="*80)
    logger.info(f"\n{tabulate(results, headers=headers, tablefmt='grid')}")
    logger.info("="*80)


def analyze_source_data_by_year(merged_df: pd.DataFrame):
    """모든 원본 데이터가 연도별로 얼마나 잘 수집되었는지 확인합니다."""
    
    # [수정] 검사 대상을 모든 원본 숫자형 데이터로 확대
    source_columns = ['open', 'high', 'low', 'close', 'volume', 'value', 'foreign_net_val', 'inst_net_val']
    
    merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year
    
    results = []
    grouped = merged_df.groupby('year')
    
    for year, group in grouped:
        row_data = {'year': year}
        total_count = len(group)
        for col in source_columns:
            if col not in group.columns:
                row_data[col] = "0.00%"
                continue
            # 0이 아닌 유효한 데이터의 비율을 계산
            meaningful_count = (group[col].fillna(0) != 0).sum()
            meaningful_pct = (meaningful_count / total_count) * 100 if total_count > 0 else 0
            row_data[col] = f"{meaningful_pct:.2f}%"
        results.append(row_data)

    # 결과를 표(tabulate) 형식으로 출력
    df_results = pd.DataFrame(results)
    headers = ["연도"] + [f"{col}(유효 %)" for col in source_columns]
    df_results = df_results[['year'] + source_columns] # 순서 고정
    df_results.columns = headers

    logger.info("\n" + "="*120)
    logger.info("      <<< 전체 원본 데이터 연도별 수집 현황 (0이 아닌 데이터의 비율) >>>")
    logger.info("="*120)
    logger.info(f"\n{tabulate(df_results, headers='keys', tablefmt='grid', showindex=False)}")
    logger.info("="*120)

def run_data_integrity_check(settings_path: str):
    """데이터 무결성 검사 파이프라인을 실행합니다."""
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    logger.info("전체 피처 데이터를 로드하여 분석합니다. (시간이 소요될 수 있습니다)")
    
    # [수정] 단일 파일 로드 방식으로 변경 (Derive 모듈 변경 사항 반영)
    dataset_path = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    
    if os.path.exists(dataset_path):
        all_features = pd.read_parquet(dataset_path)
    else:
        # 혹시 기존 방식(폴더)일 수도 있으니 fallback
        if os.path.exists(paths["features"]):
            all_features = load_partition_day(paths["features"], start_date="2020-01-01", end_date="2025-12-31")
        else:
            all_features = pd.DataFrame()

    if all_features.empty:
        logger.error(f"분석할 피처 데이터가 없습니다. '피처 데이터 생성'을 먼저 실행해주세요. (경로: {dataset_path})")
        return
    analyze_feature_distribution(all_features)
    
    logger.info("전체 원본 데이터를 로드하여 분석합니다. (시간이 소요될 수 있습니다)")
    all_merged = load_partition_day(paths["merged"], start_date="2020-01-01", end_date="2025-12-31")
    if all_merged.empty:
        logger.error("분석할 원본 데이터가 없습니다. '데이터 수집'을 먼저 실행해주세요.")
        return
    analyze_source_data_by_year(all_merged)
    
    
    logger.info("\n### 최종 진단 가이드 ###")
    logger.info("1. '주요 피처 데이터 분포 분석' 표에서 '유효값 비율'이 비정상적으로 낮다면(e.g., 5% 미만),")
    logger.info("   -> `derive.py`의 피처 계산 로직에 버그가 있을 가능성이 높습니다.")
    logger.info("2. '원본 수급 데이터 연도별 수집 현황' 표에서 특정 연도의 '유효 데이터 %'가 비정상적으로 낮다면,")
    logger.info("   -> `collect.py`의 데이터 수집 로직에 문제가 있거나 해당 시점의 API가 데이터를 제공하지 않았을 수 있습니다.")