import sqlite3

db_path = r'd:\spikehunter\data\db\spikehunter_log.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking daily_signals date format:")
cursor.execute("SELECT date FROM daily_signals LIMIT 5;")
print(cursor.fetchall())

print("\nChecking feature_importance run_timestamp format:")
cursor.execute("SELECT run_timestamp FROM feature_importance LIMIT 5;")
print(cursor.fetchall())

conn.close()
