from pykrx.website import comm
import inspect

def inspect_pykrx():
    print("Inspecting pykrx.website.comm...")
    print(f"Dir: {dir(comm)}")
    
    # Try to find webio
    try:
        from pykrx.website.comm import webio
        print("Found 'webio' module")
        if hasattr(webio, 'KrxWebIo'):
            print("Found KrxWebIo in webio")
    except ImportError:
        print("'webio' not found in comm")

    # Check other common places
    try:
        import pykrx.website.comm.util
        print("Found 'util'")
    except:
        pass

if __name__ == "__main__":
    inspect_pykrx()
