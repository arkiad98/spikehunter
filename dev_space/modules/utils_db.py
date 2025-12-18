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
from datetime import datetime
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

def create_tables():
    """프로그램 시작 시 필요한 모든 테이블을 생성하거나 검증합니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for table_name, ddl_script in TABLE_DEFINITIONS.items():
                cursor.execute(ddl_script)
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
