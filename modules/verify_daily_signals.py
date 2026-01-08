# modules/verify_daily_signals.py
"""
추천 종목(Daily Signals)에 대한 사후 성과 검증 모듈.
DB에 'PENDING' 상태로 저장된 추천 종목들을 불러와,
동적 전략 파라미터(설정값, 최적값 등)를 기반으로 승리/패배/만료 여부를 시뮬레이션하고 검증합니다.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

from .utils_db import get_pending_signals, update_signal_outcome, update_signal_intermediate, get_recent_signals
from .utils_io import read_yaml, load_partition_day, get_stock_names, ensure_dir
from .utils_logger import logger

def simulate_trade(
    entry_price: float,
    stock_prices: pd.DataFrame,
    target_rate: float,
    stop_rate: float,
    max_hold_days: int,
    fee_rate: float = 0.0
) -> Dict[str, Any]:
    """
    단일 종목에 대한 매매 시뮬레이션을 수행합니다.
    필드 테스트 모듈 등에서 재사용 가능하도록 독립 로직으로 분리하였습니다.
    
    Args:
        entry_price: 진입 가격
        stock_prices: 진입일 *이후*의 일별 주가 데이터 (Open/High/Low/Close, Date 오름차순)
        target_rate: 목표 수익률 (예: 0.10)
        stop_rate: 손절률 (예: -0.05)
        max_hold_days: 최대 보유일 (예: 5)
        fee_rate: 수수료율 (예: 0.0015 -> 매수/매도 각각 적용)
        
    Returns:
        Dict: 결과 리포트 (status, exit_date, exit_price, return, highest_price 등)
    """
    if stock_prices.empty:
        return {
            'status': 'PENDING',
            'exit_date': None,
            'exit_price': 0.0,
            'return_rate': 0.0,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'days_passed': 0,
            'final_close': entry_price,
            'price_flow': [] # 일별 등락률 리스트
        }

    target_price = entry_price * (1 + target_rate)
    stop_price = entry_price * (1 + stop_rate)

    final_status = 'PENDING'
    exit_date_str = None
    exit_price = 0.0
    
    # 모니터링 변수
    temp_high = entry_price
    temp_low = entry_price
    days_passed = 0
    price_flow = [] # 일별 (날짜, 종가, 등락률)

    for _, row in stock_prices.iterrows():
        d_open = float(row['open'])
        d_high = float(row['high'])
        d_low = float(row['low'])
        d_close = float(row['close'])
        d_date = row['date']
        
        days_passed += 1
        
        # 일별 통계 갱신
        temp_high = max(temp_high, d_high)
        temp_low = min(temp_low, d_low)
        
        # 흐름 기록
        daily_ret = (d_close - entry_price) / entry_price
        price_flow.append({
            'date': d_date,
            'close': d_close,
            'ret': daily_ret
        })

        # 이미 종료된 상태라면 데이터 수집만 하고 로직 스킵 (혹은 바로 break)
        if final_status != 'PENDING':
            continue

        # --- 매도 로직 (backtest.py와 유사) ---
        calculated_exit = None
        
        # 1. 갭(Gap) 체크 (시가 기준)
        if d_open >= target_price:
            calculated_exit = d_open
            final_status = 'WIN' # GapTP
        elif d_open <= stop_price:
            calculated_exit = d_open
            final_status = 'LOSS' # GapSL
        else:
            # 2. 장중(Intraday) 체크
            if d_high >= target_price:
                calculated_exit = target_price
                final_status = 'WIN'
            elif d_low <= stop_price:
                calculated_exit = stop_price
                final_status = 'LOSS'
            # 3. 만기(Time) 체크
            elif days_passed >= max_hold_days:
                calculated_exit = d_close
                final_status = 'TIME_OUT'
        
        if calculated_exit is not None:
            exit_price = calculated_exit
            exit_date_str = d_date.strftime('%Y-%m-%d')
            # 여기서 loop break를 하면 exit 이후의 가격 흐름은 기록되지 않음
            # 사용자 요청: "5일간의 흐름" -> 이미 매도했더라도 5일까지는 기록하고 싶을 수 있음.
            # 하지만 통상적으로 매도 후의 가격은 '내 꺼'가 아니므로 의미가 다름.
            # 여기서는 '트레이딩 종료' 시점이므로 break 하는 것이 깔끔함.
            break 
            
    # 최종 수익률 계산 (수수료 적용)
    # Return = ((Exit * (1 - fee)) - (Entry * (1 + fee))) / (Entry * (1 + fee))
    #        = (Exit*(1-f) / Entry*(1+f)) - 1
    
    if final_status == 'PENDING':
        # 진행 중일 때는 현재가 평가손익 (단순 등락률로 표시하거나, 가상 청산 기준)
        # 리포트 통일성을 위해 단순 등락률(Net)로 계산
        current_price = stock_prices.iloc[-1]['close']
        # PENDING 상태에서도 수수료 감안한 평가 손익을 볼 것인가? -> 보통은 단순 등락률을 봄.
        # 하지만 결과값 일관성을 위해 수수료 로직을 태움 (가상 청산)
        calc_exit = current_price
    else:
        calc_exit = exit_price

    if calc_exit > 0:
        net_return = ((calc_exit * (1 - fee_rate)) - (entry_price * (1 + fee_rate))) / (entry_price * (1 + fee_rate))
    else:
        net_return = 0.0

    return {
        'status': final_status,
        'exit_date': exit_date_str,
        'exit_price': calc_exit,
        'return_rate': net_return,
        'highest_price': temp_high,
        'lowest_price': temp_low,
        'days_passed': days_passed,
        'price_flow': price_flow
    }

def verify_signals(settings_path: str):
    """현재 Pending 상태인 모든 신호를 검증하고 DB를 업데이트합니다."""
    
    # 1. 설정 및 파라미터 로드
    cfg = read_yaml(settings_path)
    
    # [설정 로드]
    # 1순위: settings.yaml의 ml_params (최적화된 값)
    # 2순위: 전략 설정 (개별 튜닝 값)
    # 여기서는 '전역 설정'을 우선하는 것이 관리상 편하므로 settings.yaml의 ml_params를 global standard로 봄
    # 하지만 predict.py는 strategy_params를 썼음.
    # 일관성을 위해: 전략별 params를 우선하되, 없으면 ml_params 사용.
    
    # DB에 저장된 strategy_name을 활용
    # 로드를 위해 전체 전략 맵이 필요할 수 있음
    all_strategies = cfg.get("strategies", {})
    ml_params = cfg.get("ml_params", {})
    
    pending_df = get_pending_signals()
    if pending_df.empty:
        logger.info("검증할 대기 중인(Pending) 신호가 없습니다.")
        return

    logger.info(f"검증 대상 신호 수: {len(pending_df)}개")
    
    # 2. 최신 주가 데이터 로드
    prices_dir = cfg["paths"].get("merged", "data/proc/merged")
    
    min_date_str = pending_df['date'].min()
    min_date = pd.to_datetime(min_date_str)
    today = datetime.now()
    
    logger.info(f"가격 데이터 로드 중 ({min_date.date()} ~ {today.date()})...")
    recent_prices = load_partition_day(prices_dir, min_date, today)
    
    if recent_prices.empty:
        logger.warning("검증할 최근 가격 데이터가 없습니다.")
        return

    recent_prices['date'] = pd.to_datetime(recent_prices['date'])
    # 검색 최적화를 위해 sort
    recent_prices = recent_prices.sort_values(['code', 'date'])
    
    update_count = 0
    
    # 3. 각 신호별 검증
    for _, row in pending_df.iterrows():
        signal_id = row['signal_id']
        code = str(row['code'])
        signal_date = pd.to_datetime(row['date'])
        entry_price = float(row['entry_price'])
        strategy_name = row.get('strategy_name', 'SpikeHunter')
        
        # 동적 파라미터 결정
        # 동적 파라미터 결정 (DB에 저장된 당시 파라미터 우선 사용)
        # DB에 target_rate/stop_rate가 있다면 그것을 사용하고, 없다면(Old Data) 현재 설정 사용
        db_target_r = row.get('target_rate')
        db_stop_r = row.get('stop_rate')
        
        st_cfg = all_strategies.get(strategy_name, {})
        
        if pd.notna(db_target_r):
            target_rate = float(db_target_r)
        else:
            target_rate = st_cfg.get('target_r', ml_params.get('target_surge_rate', 0.10))
            
        if pd.notna(db_stop_r):
            stop_rate = float(db_stop_r)
        else:
            stop_rate = st_cfg.get('stop_r', ml_params.get('stop_loss_rate', -0.05))
            
        max_hold = st_cfg.get('max_hold', ml_params.get('target_hold_period', 5))
        fee_rate = float(cfg.get('fee_rate', 0.0))
        
        # 해당 종목의 미래 데이터 추출
        future_prices = recent_prices[
            (recent_prices['code'] == code) & 
            (recent_prices['date'] > signal_date)
        ].copy()
        
        if future_prices.empty:
            continue
            
        # 시뮬레이션 실행
        result = simulate_trade(
            entry_price=entry_price,
            stock_prices=future_prices,
            target_rate=target_rate,
            stop_rate=stop_rate,
            max_hold_days=max_hold,
            fee_rate=fee_rate
        )
        
        # 결과 DB 반영
        if result['status'] != 'PENDING':
            update_signal_outcome(
                signal_id, 
                result['status'], 
                result['exit_date'], 
                result['exit_price'], 
                result['return_rate'], 
                result['highest_price'], 
                result['lowest_price'], 
                result['days_passed']
            )
            # 심플 로그
            logger.info(f"[검증 완료] {code} -> {result['status']} ({result['return_rate']*100:.2f}%)")
            update_count += 1
        else:
            # 중간 상태 업데이트
            from .utils_db import update_signal_intermediate
            update_signal_intermediate(
                signal_id, 
                result['highest_price'], 
                result['lowest_price'], 
                result['days_passed']
            )

    if update_count > 0:
        logger.info(f"총 {update_count}건의 신호 검증 완료.")
    else:
        logger.info("신규 완료 건 없음 (진행 중).")
    
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
        wins = int(df_perf.iloc[0]['win_cnt'])
        avg_ret = df_perf.iloc[0]['avg_ret']
        win_rate = (wins / total) * 100
        
        print("\n [완료된 트레이드 성과]")
        print(f"  - 총 완료: {total} 건")
        print(f"  - 승률: {win_rate:.1f}% ({wins}/{total})")
        print(f"  - 평균 수익률: {avg_ret*100:.2f}% (수수료 반영 추정)")
    else:
        print("\n 아직 완료된 트레이드가 없습니다.")
        
    print("="*60 + "\n")

def print_detailed_verification_report(settings_path: str):
    """
    추천 신호들의 상세 흐름(5일치)을 포함한 리포트를 생성하고 출력합니다.
    """
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except: pass

    cfg = read_yaml(settings_path)
    # 파라미터 로드 (리포트 표시용)
    ml_params = cfg.get("ml_params", {})
    fee_rate = float(cfg.get('fee_rate', 0.0))
    
    # 1. 최근 신호 조회
    recent_signals = get_recent_signals(limit_days=30)
    if recent_signals.empty:
        print("최근 30일 내 추천 신호가 없습니다.")
        return

    # 2. 주가 데이터 로드
    prices_dir = cfg["paths"].get("merged", "data/proc/merged")
    min_date_str = recent_signals['date'].min()
    min_date = pd.to_datetime(min_date_str)
    today = datetime.now()
    
    print(f">> 주가 데이터 로드 중 ({min_date.date()} ~)...")
    df_prices = load_partition_day(prices_dir, min_date, today)
    
    if df_prices.empty:
        print("데이터 부족으로 리포트 실패.")
        return
        
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    # 3. 테이블 구성
    report_data = []
    
    # 종목명 확보
    stock_codes = recent_signals['code'].unique().tolist()
    name_map = get_stock_names(stock_codes)
    
    for _, row in recent_signals.iterrows():
        signal_date = pd.to_datetime(row['date'])
        code = str(row['code'])
        entry_price = float(row['entry_price'])
        
        # 동적 기준가 재계산 (DB값보다 현재 설정 우선 시각화 -> 혼동 방지를 위해 DB 저장값(히스토리)이 있다면 그것을 쓰거나, 
        # 아니면 현재 설정을 쓴다고 명시해야 함. 여기선 현재 설정값으로 통일)
        # 동적 기준가 재계산 (DB에 저장된 당시 파라미터 우선)
        strategy_name = row.get('strategy_name', 'SpikeHunter')
        st_cfg = cfg.get("strategies", {}).get(strategy_name, {})
        
        db_target_r = row.get('target_rate')
        db_stop_r = row.get('stop_rate')
        
        if pd.notna(db_target_r):
            target_rate = float(db_target_r)
        else:
            target_rate = st_cfg.get('target_r', ml_params.get('target_surge_rate', 0.10))

        if pd.notna(db_stop_r):
            stop_rate = float(db_stop_r)
        else:
            stop_rate = st_cfg.get('stop_r', ml_params.get('stop_loss_rate', -0.05))
        
        # 목표가/손절가 계산 (DB에 저장된 값이 있으면 그것을 참고해도 되지만, rate 기반 일관성 유지)
        target_price = entry_price * (1 + target_rate)
        stop_price = entry_price * (1 + stop_rate)
        
        highest_price = float(row['highest_price']) if row['highest_price'] else entry_price
        
        # 기본 정보
        item = {
            '추천일': signal_date.strftime('%Y-%m-%d'),
            '수익조건': f"{target_rate*100:+.2f}%", # [수정] 컬럼 분리
            '손절조건': f"{stop_rate*100:+.2f}%",   # [수정] 컬럼 분리
            '종목명': row['name'] if row['name'] else name_map.get(code, code),
            'ML점수': f"{float(row.get('ml_score', 0.0)):.4f}",
            '추천가': entry_price,
            '목표가': f"{int(target_price)}", 
            '손절가': f"{int(stop_price)}",
        }
        
        # 5일치 흐름
        # 해당 종목의 '추천일 이후' 데이터 조회
        future_prices = df_prices[
            (df_prices['code'] == code) & (df_prices['date'] > signal_date)
        ].sort_values('date')
        
        # 최대 5일 루프
        for i in range(1, 6):
            col_name = f"+{i}일"
            if len(future_prices) >= i:
                p_row = future_prices.iloc[i-1]
                p_close = p_row['close']
                p_ret = (p_close - entry_price) / entry_price * 100
                
                # 터미널 공간을 위해 등락률 위주 표시 -> 사용자 요청: 가격 표기
                # "10,500 (+5.0%)" 포맷 사용
                item[col_name] = f"{int(p_close):,} ({p_ret:+.1f}%)"
            else:
                item[col_name] = "-"
        
        # 최고가 (Rate)
        high_ret = (highest_price - entry_price) / entry_price * 100
        item['최고가'] = f"{int(highest_price)} ({high_ret:+.1f}%)"
        
        # 결과
        exit_price = float(row['exit_price']) if row['exit_price'] else 0
        ret_val = float(row['return_rate']) if row['return_rate'] is not None else 0.0
        
        item['판매가'] = int(exit_price) if exit_price > 0 else "-"
        item['수익률'] = f"{ret_val*100:+.2f}%"
        item['보유일'] = f"{int(row['holding_days'])}일"
        item['상태'] = row['status']
        
        report_data.append(item)

    df_report = pd.DataFrame(report_data)
    
    # 4. 파일 저장
    save_dir = "data/proc/verification_reports"
    ensure_dir(save_dir)
    filename = f"verification_report_{datetime.now().strftime('%Y%m%d')}.csv"
    save_path = os.path.join(save_dir, filename)
    df_report.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f">> 상세 리포트 저장 완료 (전체 데이터 포함): {save_path}")
    
    # 5. 터미널 출력 (Column Selection & Formatting)
    # 핵심 컬럼만 추려서 출력
    # [수정] 터미널에는 3일까지만 표시 (가독성 확보)
    # [수정] 조건 -> 수익조건, 손절조건 분리
    display_cols = ['추천일', '수익조건', '손절조건', '종목명', 'ML점수', '추천가', '목표가', '손절가', '+1일', '+2일', '+3일', '최고가', '수익률', '상태']
    
    # 추천가 천단위 포맷팅을 위해 df_display 별도 생성
    df_disp = df_report[display_cols].copy()
    df_disp['추천가'] = df_disp['추천가'].apply(lambda x: f"{int(x):,}")
    
    print("\n" + "="*150)
    print("                                      <<< 추천 종목 상세 일별 추이 (3일 단축 표시) >>>")
    print("="*150)
    print(df_disp.to_string(index=False))
    print("="*150 + "\n")

