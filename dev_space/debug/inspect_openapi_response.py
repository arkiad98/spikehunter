import requests

URL = "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"
KEY = "9366E92B67004AC7889ED857B0A2C589C8A69D1B"
DATE = "20250103"

def inspect():
    print(f"Requesting {URL} with basDd={DATE}...")
    
    # Try Header Auth
    headers = {"AUTH_KEY": KEY, "User-Agent": "Mozilla/5.0"}
    params = {"basDd": DATE}
    
    try:
        resp = requests.get(URL, params=params, headers=headers, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Headers: {resp.headers}")
        print(f"Text: {resp.text[:500]}...") # Print first 500 chars
        
        # Try Param Auth if fail
        if resp.status_code != 200:
            print("\nRetrying with Param Auth...")
            params["AUTH_KEY"] = KEY
            resp = requests.get(URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            print(f"Status: {resp.status_code}")
            print(f"Text: {resp.text[:500]}...")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
