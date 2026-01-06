import requests

def test_otp():
    url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201",
        "Origin": "https://data.krx.co.kr",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    
    # Params from log (Direct Mode)
    data = {
        "indIdx2": "001",
        "indIdx": "1",
        "strtDd": "20260106",
        "endDd": "20260106",
        "bld": "dbms/MDC/STAT/standard/MDCSTAT00301"
    }
    
    session = requests.Session()
    session.get(headers['Referer'], headers=headers)
    
    print(f"Sending to {url}...")
    resp = session.post(url, data=data, headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Text: {resp.text[:500]}")
    
    if len(resp.text) < 100 and "html" not in resp.text:
         print("SUCCESS! OTP received.")
    else:
         print("FAILED.")

if __name__ == "__main__":
    test_otp()
