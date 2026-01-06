from pykrx.website.comm import webio
import inspect

def inspect_post():
    print("Inspecting webio.Post methods...")
    methods = inspect.getmembers(webio.Post, predicate=inspect.isfunction)
    for name, func in methods:
        print(f"Method: {name}")

if __name__ == "__main__":
    inspect_post()
