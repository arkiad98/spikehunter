from pykrx.website.krx import krxio
import inspect

def inspect_methods():
    print("Inspecting KrxWebIo methods...")
    methods = inspect.getmembers(krxio.KrxWebIo, predicate=inspect.isfunction)
    for name, func in methods:
        print(f"Method: {name}")
    
    # Check parent classes
    print(f"MRO: {krxio.KrxWebIo.mro()}")

if __name__ == "__main__":
    inspect_methods()
