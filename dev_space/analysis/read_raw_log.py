
import os

path = r"d:\spikehunter\logs\20251230_090210_pipeline.json"

if os.path.exists(path):
    print(f"File size: {os.path.getsize(path)} bytes")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            print(f.read(2000))
    except Exception as e:
        print(f"UTF-8 failed: {e}")
        try:
            with open(path, 'r', encoding='cp949') as f:
                print(f.read(2000))
        except Exception as e2:
            print(f"CP949 failed: {e2}")
else:
    print("File not found.")
