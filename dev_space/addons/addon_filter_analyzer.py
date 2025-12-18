# addons/addon_filter_analyzer.py
import pandas as pd
import os
import joblib
from tqdm import tqdm
from tabulate import tabulate

from modules.utils_io import read_yaml, load_partition_day, load_index_data
from modules.utils_logger import logger
from modules.backtest import _determine_regime

def run_filter_analysis(settings_path: str):
    """
    지정된 Walk-Forward 기간 동안 일별 필터링 과정을 상세히 추적하고
    병목 현상이 발생하는 필터를 진단합니다.
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< Walk-Forward 필터 분석기 시작 >>>")
    logger.info("="*80)

    # 1. 사용자로부터 분석 대상 Walk-Forward 실행 경로 입력받기
    wf_run_dir = input("분석할 Walk-Forward 실행 폴더의 경로를 입력하세요 (예: data/proc/backtest/WF-...): ")
    if not os.path.isdir(wf_run_dir):
        logger.error("잘못된 경로입니다. 폴더가 존재하지 않습니다.")
        return
    
    period_num = int(input("분석할 기간(Period) 번호를 입력하세요 (예: 1): "))

    # 2. 분석에 필요한 설정 및 데이터 로드
    base_cfg = read_yaml(settings_path)
    temp_settings_path = os.path.join(wf_run_dir, 'temp_settings.yaml')
    if not os.path.exists(temp_settings_path):
        logger.error(f"'{temp_settings_path}' 파일이 없습니다. 경로를 다시 확인해주세요.")
        return
    
    wf_cfg = read_yaml(temp_settings_path) # 분석에는 임시 설정 파일 사용
    
    # 분석할 기간(Period)의 날짜 계산
    train_months = base_cfg['walk_forward']['train_months']
    test_months = base_cfg['walk_forward']['test_months']
    total_start_date = pd.to_datetime(base_cfg['walk_forward']['total_start_date'])
    
    start_offset = (period_num - 1) * test_months
    train_start = total_start_date + pd.DateOffset(months=start_offset)
    train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
    test_start = train_end + pd.Timedelta(days=1)
    test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

    logger.info(f"분석 대상 기간: {test_start.date()} ~ {test_end.date()}")

    # 해당 기간의 피처 데이터 로드
    features = load_partition_day(base_cfg['paths']['features'], test_start, test_end)
    kospi = load_index_data(test_start - pd.DateOffset(days=300), test_end, base_cfg['paths']['raw_index'])
    
    # 해당 기간에 학습된 ML 모델 로드
    model_path = os.path.join(wf_run_dir, 'temp_models', f'period_{period_num}_temp_clf.joblib')
    if not os.path.exists(model_path):
        # Pass 2 모델 경로도 확인
        model_path_final = os.path.join(wf_cfg['paths']['models'], 'lgbm_model.joblib')
        if not os.path.exists(model_path_final):
             logger.error(f"ML 모델 파일({model_path_final})이 없습니다.")
             return
        model_clf = joblib.load(model_path_final)
    else:
        model_clf = joblib.load(model_path)
    
    logger.info(f"'{model_path}' 모델을 사용하여 분석합니다.")

    # 3. 일별 필터링 과정 추적
    all_dates = sorted(features['date'].unique())
    analysis_log = []
    
    filter_stats = {
        'total_candidates': 0,
        'fail_dist_ma': 0,
        'fail_avg_value': 0,
        'fail_calmdown': 0,
        'fail_ml_score': 0,
        'final_recs': 0,
    }

    for date in tqdm(all_dates, desc="일별 필터 분석 중"):
        df_today = features[features['date'] == date].copy()
        if df_today.empty: continue

        # 시장 국면 결정
        kospi_past = kospi[kospi['date'] <= date]
        if len(kospi_past) < 200: continue
        
        current_kospi_close = kospi_past['kospi_close'].iloc[-1]
        current_ma200 = kospi_past['kospi_close'].rolling(200).mean().iloc[-1]
        current_kospi_vol_20d = kospi_past['kospi_close'].pct_change().rolling(20).std().iloc[-1]
        vol_threshold = wf_cfg["strategies"]["SpikeHunter_R1_BullStable"]["max_market_vol"]
        current_regime = _determine_regime(current_kospi_close > current_ma200, current_kospi_vol_20d < vol_threshold)
        
        if current_regime == 'R4_BearVolatile': continue
            
        # 해당 국면의 파라미터 로드
        params = wf_cfg['strategies'][f'SpikeHunter_{current_regime}']
        params.update({'min_avg_value': wf_cfg['min_avg_value'], 'top_n': wf_cfg['top_n']})
        
        # 필터링 시작
        candidates = df_today[df_today['signal_spike_hunter'] == 1]
        if candidates.empty: continue
            
        filter_stats['total_candidates'] += len(candidates)

        for _, row in candidates.iterrows():
            log_entry = {'date': date, 'code': row['code'], 'regime': current_regime, 'reason': 'PASS'}
            
            # 필터 1: 이격도
            if not (row['dist_from_ma20'] < params['max_dist_from_ma']):
                log_entry['reason'] = 'FAIL: dist_from_ma'
                filter_stats['fail_dist_ma'] += 1
                analysis_log.append(log_entry)
                continue
            
            # 필터 2: 평균 거래대금
            if not (row['avg_value_20'] >= params['min_avg_value']):
                log_entry['reason'] = 'FAIL: avg_value'
                filter_stats['fail_avg_value'] += 1
                analysis_log.append(log_entry)
                continue
                
            # 필터 3: 진정 필터
            if not (row['daily_ret'] < params['max_daily_ret_entry']):
                log_entry['reason'] = 'FAIL: calmdown'
                filter_stats['fail_calmdown'] += 1
                analysis_log.append(log_entry)
                continue
            
            # 필터 4: ML 스코어
            features_for_ml = row[model_clf.feature_names_in_].to_frame().T
            
            # [추가] 모델 예측 전, 데이터 타입을 float으로 강제 변환하여 오류 해결
            features_for_ml = features_for_ml.apply(pd.to_numeric, errors='coerce').fillna(0)
      
            score = model_clf.predict_proba(features_for_ml)[0, 1]
            log_entry['ml_score'] = score
            
            # [추가] 실제 필터링 직전, 사용되는 값들을 직접 출력하는 디버그 코드
            print(f"\n[DEBUG / filter_analyzer.py] Date: {date.date()} | Code: {row['code']}")
            print(f"  - Filtering with min_ml_score threshold: {params['min_ml_score']}")
            print(f"  - Candidate score: {score}")

            if not (score >= params['min_ml_score']):
                log_entry['reason'] = 'FAIL: ml_score'            
            if not (score >= params['min_ml_score']):
                log_entry['reason'] = 'FAIL: ml_score'
                filter_stats['fail_ml_score'] += 1
                analysis_log.append(log_entry)
                continue
            
            filter_stats['final_recs'] += 1
            analysis_log.append(log_entry)

    # 4. 결과 요약 및 저장
    logger.info("\n" + "="*80)
    logger.info("      <<< 필터 분석 최종 요약 >>>")
    logger.info("="*80)
    
    summary_table = [
        ["초기 후보 종목 수 (signal_spike_hunter=1)", filter_stats['total_candidates']],
        ["탈락: 이격도 필터", filter_stats['fail_dist_ma']],
        ["탈락: 거래대금 필터", filter_stats['fail_avg_value']],
        ["탈락: 진정 필터", filter_stats['fail_calmdown']],
        ["탈락: ML 스코어 필터", filter_stats['fail_ml_score']],
        ["최종 통과 종목 수", filter_stats['final_recs']]
    ]
    logger.info(tabulate(summary_table, headers=["필터 단계", "종목 수"], tablefmt="grid"))
    
    if analysis_log:
        log_df = pd.DataFrame(analysis_log)
        output_path = os.path.join(wf_run_dir, f'filter_analysis_log_period_{period_num}.csv')
        log_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n상세 분석 로그가 '{output_path}'에 저장되었습니다.")

    logger.info("="*80)