import os
import sys

def tail(file_path, n=100):
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            
            lines_found = 0
            block_size = 1024
            blocks = []
            
            while lines_found < n and file_size > 0:
                if file_size - block_size > 0:
                    f.seek(file_size - block_size)
                    blocks.append(f.read(block_size))
                else:
                    f.seek(0)
                    blocks.append(f.read(file_size))
                
                lines_found = blocks[-1].count(b'\n')
                file_size -= block_size
                
            text = b''.join(reversed(blocks)).decode('utf-8', errors='ignore')
            print('\n'.join(text.splitlines()[-n:]))
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tail(sys.argv[1])
    else:
        print("Usage: python read_log_tail.py <file_path>")
