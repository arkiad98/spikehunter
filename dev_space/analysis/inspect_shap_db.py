import sqlite3
import pandas as pd
import os

db_path = 'data/db/spikehunter_log.db'
if not os.path.exists(db_path):
    print("DB file not found")
    exit()

conn = sqlite3.connect(db_path)
try:
    # Check tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    # print("Tables:", tables.values)
    
    # Get latest SHAP results
    # Assuming 'run_id' groups the results.
    query = """
    SELECT item_name, mean_abs_shap, rank 
    FROM shap_importance_log 
    WHERE run_id = (SELECT run_id FROM shap_importance_log ORDER BY run_timestamp DESC LIMIT 1)
    ORDER BY rank ASC
    """
    df = pd.read_sql(query, conn)
    print(df.to_csv(index=False))
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
