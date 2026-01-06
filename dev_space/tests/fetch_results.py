import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/spikehunter_log.db')
query = "SELECT * FROM backtest_summary ORDER BY run_timestamp DESC LIMIT 1"
df = pd.read_sql(query, conn)
print(df.T)
conn.close()
