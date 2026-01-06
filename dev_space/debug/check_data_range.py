
import pandas as pd
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath("d:/spikehunter"))

path = r"d:\spikehunter\data\proc\ml_dataset\ml_classification_dataset.parquet"
if not os.path.exists(path):
    print("Dataset not found")
    sys.exit()

df = pd.read_parquet(path)
df['date'] = pd.to_datetime(df['date'])

print(f"Dataset Range: {df['date'].min()} ~ {df['date'].max()}")
print(f"Total Samples: {len(df)}")

print("\nSamples per Year:")
print(df['date'].dt.year.value_counts().sort_index())
