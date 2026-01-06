import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/spikehunter_log.db')
# Get top 5 runs by CAGR to find the 372% run and check its MDD
query = """
SELECT run_timestamp, strategy_name, cagr, mdd, win_rate 
FROM backtest_summary 
ORDER BY run_timestamp DESC 
LIMIT 10
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = pd.read_sql(query, conn)
print(df)
conn.close()
