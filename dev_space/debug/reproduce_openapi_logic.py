
import pandas as pd
import datetime

# Mocking the accumulated dataframe from the API loop
data_list = []
dates = pd.date_range("2026-01-01", "2026-01-03")

for d in dates:
    # Mocking a day's data containing KOSPI and KOSPI 200
    df_day = pd.DataFrame({
        'BAS_DD': [d, d],
        'IDX_NM': ['코스피', '코스피 200'],
        'CLSPRC_IDX': [3000, 400],
        'MKTCAP': [2000000, 1800000] # KOSPI has higher market cap
    })
    data_list.append(df_day)

df_all = pd.concat(data_list, ignore_index=True)

# Logic from modules/collect_openapi.py (lines 99-148)
col_map = {
    'BAS_DD': 'date',
    'CLSPRC_IDX': 'close',
    'IDX_NM': 'name',
    'MKTCAP': 'MKTCAP'
}
df_all = df_all.rename(columns=col_map)
df_all['date'] = pd.to_datetime(df_all['date'])

print("Data before sorting/filtering:")
print(df_all)

# THE BUGGY LOGIC
if 'name' in df_all.columns:
     mask = df_all['name'].str.contains('코스피|KOSPI', na=False)
     df_all = df_all[mask]

if 'MKTCAP' in df_all.columns and not df_all.empty:
     df_all = df_all.sort_values(by='MKTCAP', ascending=False)
     # Pick top 1
     df_all = df_all.iloc[:1]

print("\nData after buggy logic:")
print(df_all)

expected_rows = len(dates)
actual_rows = len(df_all)

print(f"\nExpected Rows: {expected_rows}")
print(f"Actual Rows: {actual_rows}")

if actual_rows != expected_rows:
    print("BUG CONFIRMED: Logic incorrectly reduces data to 1 row")
else:
    print("Logic correct?")
