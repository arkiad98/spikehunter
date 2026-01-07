import logging
import requests
from pykrx.website.comm import webio
from pykrx.website import krx
from pykrx import stock
import datetime

def patch_pykrx_referer():
    """
    [Pykrx Patch]
    GitHub PR #249: Fix data collection error by updating HTTP Referer header.
    Ref: https://github.com/sharebook-kr/pykrx/pull/249
    Plus: Patch stock_api to handle column mismatch/English columns for 2026 data.
    """
    logger = logging.getLogger(__name__)

    # 1. Patch Post.read to inject Referer
    if not getattr(webio.Post.read, '_is_patched', False):
        original_read = webio.Post.read
        
        def patched_read(self, **kwargs):
            if self.headers is None:
                self.headers = {}
            
            # Add Referer (PR #249 Solution)
            self.headers.update({
                "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            return original_read(self, **kwargs)
            
        patched_read._is_patched = True
        webio.Post.read = patched_read
        logger.info("pykrx patched: Post.read updated with Referer.")

    # 2. Patch stock.get_market_ohlcv_by_ticker
    def patched_get_market_ohlcv_by_ticker(date, market="KOSPI", alternative=False):
        if hasattr(date, 'strftime'):
            date = date.strftime("%Y%m%d")
        if isinstance(date, datetime.datetime):
             date = date.strftime("%Y%m%d")
        
        date = str(date).replace("-", "")
        
        # Call backend
        df = krx.get_market_ohlcv_by_ticker(date, market)
        
        # [Debug] Log columns to see what we are getting
        logger.info(f"[Patch] {date} Raw columns: {df.columns.tolist()}")

        # Check and Fix Columns if needed
        # Expected: ['시가', '고가', '저가', '종가', '거래량', '거래대금', '등락률']
        rename_map = {
            'TDD_OPNPRC': '시가', 'TDD_HGPRC': '고가', 'TDD_LWPRC': '저가', 'TDD_CLSPRC': '종가',
            'ACC_TRDVOL': '거래량', 'ACC_TRDVAL': '거래대금', 'FLUC_RT': '등락률', 
            'MKTCAP': '상장시가총액', 'LIST_SHRS': '상장주식수'
        }
        df = df.rename(columns=rename_map)
        
        # [Debug] Log columns after rename
        logger.info(f"[Patch] {date} Renamed columns: {df.columns.tolist()}")
        
        # Original logic: Check holiday (if all 0)
        # We modify this to be safe - only check if columns exist
        required = ['시가', '고가', '저가', '종가']
        if all(col in df.columns for col in required):
             holiday = (df[required] == 0).all(axis=None)
             # Logic skip for safety
        else:
             # If required columns are missing, it implies renaming failed or empty data
             if not df.empty:
                 logging.warning(f"[Patch] {date} Missing required columns. Available: {df.columns.tolist()}")
        
        return df

    # Patch both stock module and stock_api module
    import sys
    if 'pykrx.stock.stock_api' in sys.modules:
        sys.modules['pykrx.stock.stock_api'].get_market_ohlcv_by_ticker = patched_get_market_ohlcv_by_ticker
    
    stock.get_market_ohlcv_by_ticker = patched_get_market_ohlcv_by_ticker
    logger.info("pykrx patched: stock.get_market_ohlcv_by_ticker replaced (including sys.modules).")

if __name__ == "__main__":
    patch_pykrx_referer()
