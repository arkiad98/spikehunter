import os
import pykrx

def find_krxwebio():
    package_dir = os.path.dirname(pykrx.__file__)
    print(f"Searching in: {package_dir}")
    
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'class KrxWebIo' in content:
                            print(f"Found 'class KrxWebIo' in: {path}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    find_krxwebio()
