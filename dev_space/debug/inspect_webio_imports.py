from pykrx.website.comm import webio
import inspect

def inspect_webio_imports():
    print("Inspecting webio global vars...")
    print(dir(webio))
    if hasattr(webio, 'requests'):
        print("webio has requests")
    if hasattr(webio, 'session'):
        print("webio has session")

if __name__ == "__main__":
    inspect_webio_imports()
