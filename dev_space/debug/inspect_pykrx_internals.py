import inspect
from pykrx import stock
from pykrx.website import krx

def inspect_pykrx():
    print("\nInspecting pykrx.website.krx.IndexTicker...")
    try:
        from pykrx.website.krx.market.ticker import IndexTicker
        print(f"IndexTicker class: {IndexTicker}")
        
        # Instantiate and check df
        ticker_instance = IndexTicker()
        print(f"\nIndexTicker.df (head):")
        print(ticker_instance.df.head() if hasattr(ticker_instance, 'df') else "No df attribute")
        
        # Check source of get_name
        print("\nSource of get_name:")
        print(inspect.getsource(IndexTicker.get_name))
        
    except Exception as e:
        print(f"Inspection failed: {e}")
        
if __name__ == "__main__":
    inspect_pykrx()
