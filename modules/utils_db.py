# modules/utils_db.py (ver 2.7.10)
"""DB 관리 유틸리티 모듈.

[v2.7.10] SHAP 분석 결과 로깅 기능 추가
- SHAP 분석 결과를 저장하기 위한 'shap_importance_log' 테이블 정의 추가.
- SHAP 분석 결과를 DB에 삽입하는 'insert_shap_results' 함수 신설.
"""
import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import hashlib
import time
import os
import typing

if typing.TYPE_CHECKING:
    import optuna

from .utils_logger import logger

DB_PATH = "data/db/spikehunter_log.db"
TABLE_DEFINITIONS = {
    "backtest_summary": """
        CREATE TABLE IF NOT EXISTS backtest_summary (
            backtest_id TEXT PRIMARY KEY,
            run_timestamp TEXT NOT NULL,
            strategy_name TEXT,
            start_date TEXT,
            end_date TEXT,
            cagr REAL,
            sharpe REAL,
            mdd REAL,
            total_trades INTEGER,
            win_rate REAL,
            params_json TEXT,
            equity_file_path TEXT
        );
    """,
    "trade_log": """
        CREATE TABLE IF NOT EXISTS trade_log (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_id TEXT,
            entry_date TEXT,
            exit_date TEXT,
            code TEXT,
            return REAL,
            reason TEXT,
            FOREIGN KEY (backtest_id) REFERENCES backtest_summary (backtest_id)
        );
    """,
    "optimization_log": """
        CREATE TABLE IF NOT EXISTS optimization_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_name TEXT NOT NULL,
            trial_number INTEGER NOT NULL,
            state TEXT,
            value REAL,
            params_json TEXT,
            run_timestamp TEXT
        );
    """,
    # 🔴 [추가] SHAP 분석 결과 저장을 위한 신규 테이블
    "shap_importance_log": """
        CREATE TABLE IF NOT EXISTS shap_importance_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            item_name TEXT NOT NULL,
            mean_abs_shap REAL,
            rank INTEGER,
            run_timestamp TEXT
        );
    """,
    # 🔴 [추가] 실전/모의 검증용 추천 종목 추적 테이블
    "daily_signals": """
        CREATE TABLE IF NOT EXISTS daily_signals (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            strategy_name TEXT,
            ml_score REAL,
            rank INTEGER,
            
            -- 진입 조건 (기록용)
            entry_price REAL, -- 추천일 종가 (기준가)
            target_price REAL,
            stop_price REAL,
            target_rate REAL, -- [추가] 목표 수익률 (0.10 등)
            stop_rate REAL,   -- [추가] 손절률 (-0.05 등)
            max_hold_days INTEGER,
            
            -- 상태 추적
            status TEXT DEFAULT 'PENDING', -- PENDING, WIN, LOSS, TIME_OUT
            exit_date TEXT,
            exit_price REAL,
            return_rate REAL,
            
            -- 모니터링
            highest_price REAL, -- 기간 내 최고가
            lowest_price REAL,  -- 기간 내 최저가
            holding_days INTEGER DEFAULT 0,
            
            created_at TEXT
        );
    """
}

# [수정] _sanitize_for_db 함수를 더 안정적인 버전으로 교체합니다.
def _sanitize_for_db(value):
    """NumPy/Pandas 타입을 DB에 저장 가능한 파이썬 기본 타입으로 변환합니다."""
    # pd.isna는 None, np.nan 등을 모두 처리할 수 있습니다.
    if pd.isna(value):
        return None
    # numpy float 또는 python float를 python float으로 명시적 변환
    if isinstance(value, (np.floating, float)):
        return float(value)
    # numpy int 또는 python int를 python int로 명시적 변환
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value # 그 외 타입(주로 str)은 그대로 반환

def get_db_connection():
    """DB 연결을 생성하고 반환합니다."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30)

def _migrate_schema(conn):
    """기존 테이블의 스키마를 최신 변경사항에 맞춰 마이그레이션합니다."""
    try:
        cursor = conn.cursor()
        
        # 1. daily_signals 테이블 컬럼 추가 (target_rate, stop_rate)
        # 2026-01-08: 컬럼 존재 여부 확인 후 추가
        cursor.execute("PRAGMA table_info(daily_signals)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'daily_signals' in columns or columns: # 테이블이 존재하는 경우에만
            if 'target_rate' not in columns:
                logger.info("마이그레이션: daily_signals에 target_rate 컬럼 추가")
                cursor.execute("ALTER TABLE daily_signals ADD COLUMN target_rate REAL")
            
            if 'stop_rate' not in columns:
                logger.info("마이그레이션: daily_signals에 stop_rate 컬럼 추가")
                cursor.execute("ALTER TABLE daily_signals ADD COLUMN stop_rate REAL")
                
    except Exception as e:
        logger.error(f"스키마 마이그레이션 중 오류: {e}")

def create_tables():
    """프로그램 시작 시 필요한 모든 테이블을 생성하거나 검증합니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for table_name, ddl_script in TABLE_DEFINITIONS.items():
                cursor.execute(ddl_script)
            
            # [추가] 스키마 마이그레이션 실행
            _migrate_schema(conn)
            
            conn.commit()
        logger.info(f"데이터베이스 테이블이 성공적으로 준비되었습니다. (경로: {DB_PATH})")
    except Exception as e:
        logger.error(f"DB 테이블 생성 중 오류 발생: {e}", exc_info=True)

def insert_backtest_results(backtest_id: str, metrics: dict, params: dict, tradelog_df: pd.DataFrame, equity_path: str, start_date: str, end_date: str, strategy_name: str):
    """백테스트 최종 결과를 DB에 삽입합니다."""
    # ... 기존 코드 ...
    params_str = json.dumps({k: _sanitize_for_db(v) for k, v in params.items()}, ensure_ascii=False)

    if len(params_str) > 5000:
        logger.warning(f"DB에 저장될 파라미터 JSON의 길이가 5000자를 초과합니다 (길이: {len(params_str)}).")

    summary_data = {
        'backtest_id': backtest_id,
        'run_timestamp': datetime.now().isoformat(),
        'strategy_name': strategy_name,
        'start_date': start_date,
        'end_date': end_date,
        'cagr': _sanitize_for_db(metrics.get('CAGR_raw', 0.0)),
        'sharpe': _sanitize_for_db(metrics.get('Sharpe_raw', 0.0)),
        'mdd': _sanitize_for_db(metrics.get('MDD_raw', 0.0)),
        'total_trades': _sanitize_for_db(metrics.get('총거래횟수', 0)),
        'win_rate': _sanitize_for_db(metrics.get('win_rate_raw', 0.0)),
        'params_json': params_str,
        'equity_file_path': equity_path
    }

    try:
        with get_db_connection() as conn:
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_sql('backtest_summary', conn, if_exists='append', index=False)

            if not tradelog_df.empty:
                tradelog_to_insert = tradelog_df.copy()
                tradelog_to_insert['backtest_id'] = backtest_id

                tradelog_to_insert['entry_date'] = pd.to_datetime(tradelog_to_insert['entry_date']).map(lambda x: x.isoformat())
                tradelog_to_insert['exit_date'] = pd.to_datetime(tradelog_to_insert['exit_date']).map(lambda x: x.isoformat())

                tradelog_to_insert = tradelog_to_insert[['backtest_id', 'entry_date', 'exit_date', 'code', 'return', 'reason']]
                tradelog_to_insert.to_sql('trade_log', conn, if_exists='append', index=False)

            logger.info(f"백테스트 결과가 DB에 성공적으로 기록되었습니다 (ID: {backtest_id}).")
    except Exception as e:
        logger.error(f"DB 기록 중 오류 발생: {e}", exc_info=True)

def insert_optimization_logs(study: 'optuna.study.Study'):
    """완료된 Optuna 스터디의 모든 Trial 결과를 DB에 일괄 삽입합니다."""
    # ... 기존 코드 ...
    if not study:
        return

    records = []
    run_ts = datetime.now().isoformat()
    for trial in study.trials:
        records.append({
            'study_name': study.study_name,
            'trial_number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'params_json': json.dumps(trial.params),
            'run_timestamp': run_ts
        })

    if not records:
        logger.warning(f"'{study.study_name}' 스터디에서 DB에 기록할 Trial이 없습니다.")
        return

    try:
        with get_db_connection() as conn:
            df = pd.DataFrame(records)
            df.to_sql('optimization_log', conn, if_exists='append', index=False)
            logger.info(f"'{study.study_name}' 스터디의 Trial {len(records)}개가 DB에 성공적으로 기록되었습니다.")
    except Exception as e:
        logger.error(f"최적화 로그 DB 기록 중 오류 발생: {e}", exc_info=True)

# 🔴 [추가] SHAP 분석 결과를 DB에 저장하는 함수
def insert_shap_results(run_id: str, analysis_type: str, shap_df: pd.DataFrame):
    """
    SHAP 분석 결과(피처 중요도 등)를 데이터베이스에 기록합니다.
    """
    if shap_df.empty:
        return
        
    records = shap_df.copy()
    records['run_id'] = run_id
    records['analysis_type'] = analysis_type
    records['run_timestamp'] = datetime.now().isoformat()
    
    # DB 테이블 컬럼 순서에 맞게 재정렬
    records = records[['run_id', 'analysis_type', 'item_name', 'mean_abs_shap', 'rank', 'run_timestamp']]
    
    try:
        with get_db_connection() as conn:
            records.to_sql('shap_importance_log', conn, if_exists='append', index=False)
        logger.info(f"SHAP 분석 결과({analysis_type}, {len(records)}개)가 DB에 성공적으로 기록되었습니다.")
    except Exception as e:
        logger.error(f"SHAP 분석 결과 DB 기록 중 오류 발생: {e}", exc_info=True)

# 🔴 [추가] 추천 종목(Signal) 관리 함수들

def insert_daily_signals(signals: pd.DataFrame, strategy_name: str, target_rate: float = 0.10, stop_rate: float = -0.05):
    """오늘의 추천 종목들을 DB에 등록합니다. (해당 일자의 기존 기록은 제거 - 최신본 유지)"""
    if signals.empty: return
    
    current_time = datetime.now().isoformat()
    signals_to_insert = []
    
    # [수정] 날짜별로 기존 데이터를 삭제하기 위해 처리 대상 날짜를 수집
    target_dates = set()

    for _, row in signals.iterrows():
        # 기본 정보
        close_price = float(row['close'])
        
        # [수정] 외부에서 주입받은 전략 파라미터 사용
        entry_price = close_price
        target_price = entry_price * (1 + target_rate)
        stop_price = entry_price * (1 + stop_rate)
        max_hold = 5
        
        date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date'])[:10]
        target_dates.add(date_str)

        record = {
            'date': date_str,
            'code': str(row['code']),
            'name': row.get('name', ''),
            'strategy_name': strategy_name,
            'ml_score': float(row.get('ml_score', 0.0)),
            'rank': int(row.get('rank', 0)),
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_price': stop_price,
            'target_rate': target_rate,
            'stop_rate': stop_rate,
            'max_hold_days': max_hold,
            'status': 'PENDING',
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'created_at': current_time
        }
        signals_to_insert.append(record)
        
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # [수정] 해당 날짜의 기존 데이터 삭제 (Overwrite)
            for d in target_dates:
                # logger.info(f"기존 추천 신호 삭제(Overwrite) - Date: {d}")
                cursor.execute("DELETE FROM daily_signals WHERE date = ?", (d,))
            
            # 신규 데이터 삽입
            for rec in signals_to_insert:
                cols = ', '.join(rec.keys())
                placeholders = ', '.join(['?'] * len(rec))
                sql = f"INSERT INTO daily_signals ({cols}) VALUES ({placeholders})"
                cursor.execute(sql, list(rec.values()))
            
            conn.commit()
            logger.info(f"{len(signals_to_insert)}개의 신규 추천 신호를 DB에 등록했습니다. (기존 {len(target_dates)}일치 데이터 덮어씀)")
            
    except Exception as e:
        logger.error(f"추천 신호 등록 중 오류: {e}", exc_info=True)

def get_pending_signals() -> pd.DataFrame:
    """검증이 완료되지 않은(PENDING) 신호들을 조회합니다."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql("SELECT * FROM daily_signals WHERE status = 'PENDING'", conn)
            return df
    except Exception as e:
        logger.error(f"Pending 신호 조회 오류: {e}")
        return pd.DataFrame()

def get_recent_signals(limit_days: int = 30) -> pd.DataFrame:
    """최근 N일 이내에 생성된 신호들을 조회합니다 (상태 무관)."""
    try:
        with get_db_connection() as conn:
            # sqlite의 date 함수 사용: date(created_at) >= date('now', '-30 days')
            # 하지만 created_at 포맷이 isoformat()이라 TEXT 비교도 가능하지만, 
            # date 컬럼을 기준으로 하는 것이 더 정확함.
            cutoff_date = (datetime.now() - timedelta(days=limit_days)).strftime('%Y-%m-%d')
            sql = f"SELECT * FROM daily_signals WHERE date >= '{cutoff_date}' ORDER BY date DESC, rank ASC"
            df = pd.read_sql(sql, conn)
            return df
    except Exception as e:
        logger.error(f"최근 신호 조회 오류: {e}")
        return pd.DataFrame()

def update_signal_outcome(signal_id: int, status: str, exit_date: str, exit_price: float, 
                          return_rate: float, high: float, low: float, holding_days: int):
    """신호의 최종 결과를 업데이트합니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            sql = """
                UPDATE daily_signals 
                SET status = ?, exit_date = ?, exit_price = ?, return_rate = ?,
                    highest_price = ?, lowest_price = ?, holding_days = ?
                WHERE signal_id = ?
            """
            cursor.execute(sql, (status, exit_date, exit_price, return_rate, high, low, holding_days, signal_id))
            conn.commit()
    except Exception as e:
        logger.error(f"신호 결과 업데이트 오류 (ID: {signal_id}): {e}")

def update_signal_intermediate(signal_id: int, high: float, low: float, holding_days: int):
    """신호의 중간 상태(최고/최저가 등)만 업데이트합니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            sql = """
                UPDATE daily_signals 
                SET highest_price = ?, lowest_price = ?, holding_days = ?
                WHERE signal_id = ?
            """
            cursor.execute(sql, (high, low, holding_days, signal_id))
            conn.commit()
    except Exception as e:
        logger.error(f"신호 중간 상태 업데이트 오류 (ID: {signal_id}): {e}")
