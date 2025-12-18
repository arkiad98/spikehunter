# modules/verify_daily_signals.py
"""
추천 종목(Daily Signals)에 대한 사후 성과 검증 모듈.
DB에 'PENDING' 상태로 저장된 추천 종목들을 불러와,
최신 주가 데이터를 기준으로 승리(WIN), 패배(LOSS), 또는 기간 만료(TIME_OUT) 여부를 판별합니다.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from .utils_db import get_pending_signals, update_signal_outcome, update_signal_intermediate, get_recent_signals # [수정] get_recent_signals 추가
from .utils_io import read_yaml, load_partition_day, get_stock_names, ensure_dir # [수정] get_stock_names, ensure_dir 추가
from .utils_logger import logger

def verify_signals(settings_path: str):
    """현재 Pending 상태인 모든 신호를 검증하고 DB를 업데이트합니다."""
    
    # 1. 검증 대상 신호 로드
    pending_df = get_pending_signals()
    if pending_df.empty:
        logger.info("검증할 대기 중인(Pending) 신호가 없습니다.")
        return

    logger.info(f"검증 대상 신호 수: {len(pending_df)}개")
    
    # 2. 최신 주가 데이터 로드 (모든 종목의 최근 데이터 필요)
    # 효율성을 위해 전체 머지 데이터를 로드하는 대신, 필요한 기간만큼 로드하거나
    # predict.py처럼 load_partition_day를 사용
    cfg = read_yaml(settings_path)
    prices_dir = cfg["paths"].get("merged", "data/proc/merged")
    
    # 검증에 필요한 데이터 범위 설정
    # 가장 오래된 pending 신호 날짜부터 오늘까지
    min_date_str = pending_df['date'].min()
    min_date = pd.to_datetime(min_date_str)
    today = datetime.now()
    
    logger.info(f"가격 데이터 로드 중 ({min_date.date()} ~ {today.date()})...")
    # 넉넉하게 로드
    recent_prices = load_partition_day(prices_dir, min_date, today)
    
    if recent_prices.empty:
        logger.warning("검증할 최근 가격 데이터가 없습니다.")
        return

    recent_prices['date'] = pd.to_datetime(recent_prices['date'])
    recent_prices = recent_prices.sort_values(['code', 'date'])
    
    update_count = 0
    
    # 3. 각 신호별 검증 로직 수행
    for _, row in pending_df.iterrows():
        signal_id = row['signal_id']
        code = str(row['code'])
        signal_date_str = row['date']
        signal_date = pd.to_datetime(signal_date_str)
        
        # 진입가, 목표가, 손절가, 만기일
        entry_price = float(row['entry_price'])
        target_price = float(row['target_price'])
        stop_price = float(row['stop_price'])
        max_hold_days = int(row['max_hold_days'])
        
        # 현재 기록된 최고/최저가
        curr_high = float(row['highest_price']) if row['highest_price'] else entry_price
        curr_low = float(row['lowest_price']) if row['lowest_price'] else entry_price
        curr_holding = int(row['holding_days']) if row['holding_days'] else 0
        
        # 신호 발생 다음 날부터의 데이터 조회
        stock_prices = recent_prices[
            (recent_prices['code'] == code) & 
            (recent_prices['date'] > signal_date)
        ].copy()
        
        if stock_prices.empty:
            continue
            
        # 아직 처리되지 않은 새로운 일자들만 순회
        # (이미 holding_days만큼 지났다고 가정하고, 그 이후 데이터만 볼 수도 있지만
        # 여기서는 단순하게 전체 기간 재확인 - 데이터량이 많지 않으므로)
        # 단, 성능 최적화를 위해 이미 지난 날짜는 DB에 기록된 curr_holding 등으로 스킵 가능
        # 하지만 정확성을 위해 전체 재계산이 안전함 (수정된 데이터 반영 등)
        
        is_finished = False
        final_status = "PENDING"
        exit_date = None
        exit_price = 0.0
        ret = 0.0
        
        days_passed = 0
        temp_high = entry_price
        temp_low = entry_price
        
        for idx, p_row in stock_prices.iterrows():
            d_high = float(p_row['high'])
            d_low = float(p_row['low'])
            d_close = float(p_row['close'])
            d_date = p_row['date']
            
            days_passed += 1
            
            # 최고/최저가 갱신
            temp_high = max(temp_high, d_high)
            temp_low = min(temp_low, d_low)
            
            # 1. 목표가 달성 (WIN) - 고가 기준
            # (보수적으로: 시가가 갭상승으로 목표가 넘으면 시가 체결, 아니면 목표가 체결 가정)
            # 여기서는 단순하게 High >= Target 이면 Win으로 처리
            if d_high >= target_price:
                final_status = "WIN"
                exit_price = target_price # 목표가 익절 가정
                exit_date = d_date.strftime('%Y-%m-%d')
                ret = (exit_price - entry_price) / entry_price
                is_finished = True
                break
            
            # 2. 손절가 이탈 (LOSS) - 저가 기준
            if d_low <= stop_price:
                final_status = "LOSS"
                exit_price = stop_price # 손절가 매도 가정
                exit_date = d_date.strftime('%Y-%m-%d')
                ret = (exit_price - entry_price) / entry_price
                is_finished = True
                break
                
            # 3. 만기 도달 (TIME_OUT)
            if days_passed >= max_hold_days:
                final_status = "TIME_OUT"
                exit_price = d_close # 종가 청산
                exit_date = d_date.strftime('%Y-%m-%d')
                ret = (exit_price - entry_price) / entry_price
                is_finished = True
                break
        
        if is_finished:
            update_signal_outcome(
                signal_id, final_status, exit_date, exit_price, ret, temp_high, temp_low, days_passed
            )
            logger.info(f"[검증 완료] 종목:{code}, 상태:{final_status}, 수익률:{ret*100:.2f}% ({signal_date_str} -> {exit_date})")
            update_count += 1
        else:
            # 아직 진행 중 -> 중간 상태 업데이트
            if (temp_high != curr_high) or (temp_low != curr_low) or (days_passed != curr_holding):
                update_signal_intermediate(signal_id, temp_high, temp_low, days_passed)

    if update_count > 0:
        logger.info(f"총 {update_count}건의 신호 검증이 완료되었습니다.")
    else:
        logger.info("새롭게 완료된 검증 건이 없습니다 (모두 진행 중).")

    print("="*60 + "\n")

def print_verification_summary():
    """DB에서 전체 검증 현황을 요약 출력합니다."""
    try:
        from .utils_db import get_db_connection
        with get_db_connection() as conn:
            # 상태별 카운트
            df_status = pd.read_sql("SELECT status, COUNT(*) as cnt FROM daily_signals GROUP BY status", conn)
            
            # 완료된 건들의 수익률 통계
            df_perf = pd.read_sql("SELECT AVG(return_rate) as avg_ret, SUM(CASE WHEN return_rate > 0 THEN 1 ELSE 0 END) as win_cnt, COUNT(*) as total_done FROM daily_signals WHERE status != 'PENDING'", conn)
            
    except Exception as e:
        logger.error(f"요약 출력 중 오류: {e}")
        return

    print("\n" + "="*60)
    print("      <<< 실전 검증(Field Test) 현황 요약 >>>")
    print("="*60)
    
    if not df_status.empty:
        print(" [상태별 카운트]")
        for _, r in df_status.iterrows():
            print(f"  - {r['status']}: {r['cnt']} 건")
            
    if not df_perf.empty and df_perf.iloc[0]['total_done'] > 0:
        total = df_perf.iloc[0]['total_done']
        wins = df_perf.iloc[0]['win_cnt']
        avg_ret = df_perf.iloc[0]['avg_ret']
        win_rate = (wins / total) * 100
        
        print("\n [완료된 트레이드 성과]")
        print(f"  - 총 완료: {total} 건")
        print(f"  - 승률: {win_rate:.1f}%")
        print(f"  - 평균 수익률: {avg_ret*100:.2f}%")
    else:
        print("\n 아직 완료된 트레이드가 없습니다.")
        
    print("="*60 + "\n")

def print_detailed_verification_report(settings_path: str):
    """
    최근 추천 신호들의 일별 흐름(+1~+5일)을 상세 테이블로 출력하고 파일로 저장합니다.
    """
    # [수정] 한글 출력을 위해 stdout 인코딩 설정
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # 1. 최근 신호 조회
    recent_signals = get_recent_signals(limit_days=30)
    if recent_signals.empty:
        print("최근 30일 내 추천 신호가 없습니다.")
        return

    # 2. 주가 데이터 로드
    cfg = read_yaml(settings_path)
    prices_dir = cfg["paths"].get("merged", "data/proc/merged")
    
    min_date_str = recent_signals['date'].min()
    min_date = pd.to_datetime(min_date_str)
    today = datetime.now()
    
    print(f">> 상세 리포트 작성을 위해 주가 데이터 로드 중 ({min_date.date()} ~ {today.date()})...")
    df_prices = load_partition_day(prices_dir, min_date, today)
    
    if df_prices.empty:
        print("주가 데이터가 없어 상세 리포트를 생성할 수 없습니다.")
        return
        
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    # 빠른 조회를 위해 MultiIndex 설정 (code, date) -> close
    # 중복 제거 (혹시 모를 중복 대비)
    df_prices = df_prices.drop_duplicates(subset=['code', 'date'])
    price_map = df_prices.set_index(['code', 'date'])['close'].to_dict()
    
    # 3. 테이블 데이터 구성
    report_data = []
    
    # 종목명 확보 (DB에 name이 없는 경우 대비)
    stock_codes = recent_signals['code'].unique().tolist()
    name_map = get_stock_names(stock_codes)
    
    for _, row in recent_signals.iterrows():
        signal_date = pd.to_datetime(row['date'])
        code = str(row['code'])
        entry_price = float(row['entry_price'])
        
        # 기본 정보
        item = {
            '추천일': signal_date.strftime('%Y-%m-%d'),
            '종목명': row['name'] if row['name'] else name_map.get(code, code),
            '추천가(0일)': entry_price,
            '상태': row['status']
        }
        
        # 영업일 찾기 로직: 단순 날짜 더하기가 아니라, 해당 종목의 실제 데이터가 있는 날짜를 찾아야 함
        # signal_date 이후의 날짜들을 가져옴
        # 효율성을 위해 해당 코드의 전체 가격 데이터에서 필터링
        # (위에서 price_map을 만들었지만, 날짜 순서를 알기 위해 다시 필터링 필요)
        # -> 개선: df_prices가 이미 존재하므로 활용
        
        future_prices = df_prices[
            (df_prices['code'] == code) & (df_prices['date'] > signal_date)
        ].sort_values('date')
        
        # +1일 ~ +5일 종가 매핑
        for i in range(1, 6):
            col_name = f"+{i}일"
            if len(future_prices) >= i:
                price = future_prices.iloc[i-1]['close']
                item[col_name] = price
            else:
                item[col_name] = None # 미래 데이터 없음
        
        # 수익률 (DB에 있으면 사용, 없으면 현재가 기준 계산)
        if row['status'] in ['WIN', 'LOSS', 'TIME_OUT'] and row['return_rate'] is not None:
             item['수익률'] = float(row['return_rate'])
        else:
             # 진행 중이면 가장 최근 종가 기준 (데이터가 있을 경우)
             if not future_prices.empty:
                 last_price = future_prices.iloc[-1]['close']
                 item['수익률'] = (last_price - entry_price) / entry_price
             else:
                 item['수익률'] = 0.0
        
        report_data.append(item)
        
    df_report = pd.DataFrame(report_data)
    
    # 4. 파일 저장 (Raw Data)
    save_dir = "data/proc/verification_reports"
    ensure_dir(save_dir)
    filename = f"verification_report_{datetime.now().strftime('%Y%m%d')}.csv"
    save_path = os.path.join(save_dir, filename)
    df_report.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f">> 상세 리포트가 저장되었습니다: {save_path}")
    
    # 5. 터미널 출력용 포맷팅
    # 복사본 생성하여 포맷팅 적용
    df_display = df_report.copy()
    
    # 컬럼 순서 정리
    cols_order = ['추천일', '종목명', '추천가(0일)', '+1일', '+2일', '+3일', '+4일', '+5일', '수익률', '상태']
    # +1~+5일 컬럼이 없을 수도 있으니(데이터 전무) 확인
    existing_cols = [c for c in cols_order if c in df_display.columns]
    df_display = df_display[existing_cols]

    # 숫자 포맷팅 (천단위 컴마, 소수점 제거)
    price_cols = ['추천가(0일)'] + [f"+{i}일" for i in range(1, 6)]
    for c in price_cols:
        if c in df_display.columns:
            df_display[c] = df_display[c].apply(lambda x: f"{int(x):,}" if pd.notnull(x) and x != '' else "-")
            
    # 수익률 포맷팅
    if '수익률' in df_display.columns:
        df_display['수익률'] = df_display['수익률'].apply(lambda x: f"{x*100:+.2f}%" if pd.notnull(x) else "-")
        
    print("\n" + "="*80)
    print("                      <<< 추천 종목 상세 일별 추이 >>>")
    print("="*80)
    # to_string으로 전체 출력
    print(df_display.to_string(index=False))
    print("="*80 + "\n")
