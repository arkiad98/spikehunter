import sys
import os
import sqlite3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules import utils_db

def verify():
    print(">>> [DB Schema Verification] Starting...")
    
    # 1. Trigger Migration
    utils_db.create_tables()
    
    # 2. Check Columns
    conn = utils_db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(daily_signals)")
    cols = [info[1] for info in cursor.fetchall()]
    
    print(f"Current Columns in 'daily_signals': {cols}")
    
    required = ['target_rate', 'stop_rate']
    missing = [c for c in required if c not in cols]
    
    if missing:
        print(f"FAILED: Missing columns: {missing}")
        sys.exit(1)
    else:
        print("SUCCESS: All required columns present.")
        sys.exit(0)

if __name__ == "__main__":
    verify()
