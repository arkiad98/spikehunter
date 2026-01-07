# dev_space/tests/test_dynamic_verification.py
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Project root setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.verify_daily_signals import simulate_trade

def approx_equal(a, b, tolerance=1e-6):
    return abs(a - b) < tolerance

def test_simulate_trade_win():
    """목표가 도달 (WIN) 테스트"""
    entry_price = 10000
    target_rate = 0.10  # 11000
    stop_rate = -0.05   # 9500
    max_hold = 5
    
    # +1일차에 바로 목표가 도달
    prices = pd.DataFrame([
        {'date': datetime(2025, 1, 2), 'open': 10000, 'high': 11200, 'low': 9900, 'close': 11100},
    ])
    
    result = simulate_trade(entry_price, prices, target_rate, stop_rate, max_hold, fee_rate=0.0)
    
    assert result['status'] == 'WIN', f"Expected WIN, got {result['status']}"
    assert result['exit_price'] == 11000, f"Expected 11000, got {result['exit_price']}"
    assert approx_equal(result['return_rate'], 0.10), f"Expected 0.10, got {result['return_rate']}"
    print("[PASS] test_simulate_trade_win")

def test_simulate_trade_loss():
    """손절가 이탈 (LOSS) 테스트"""
    entry_price = 10000
    target_rate = 0.10
    stop_rate = -0.05 # 9500
    max_hold = 5
    
    prices = pd.DataFrame([
        {'date': datetime(2025, 1, 2), 'open': 10000, 'high': 10100, 'low': 9400, 'close': 9600},
    ])
    
    result = simulate_trade(entry_price, prices, target_rate, stop_rate, max_hold, fee_rate=0.0)
    
    assert result['status'] == 'LOSS', f"Expected LOSS, got {result['status']}"
    assert result['exit_price'] == 9500, f"Expected 9500, got {result['exit_price']}"
    assert approx_equal(result['return_rate'], -0.05), f"Expected -0.05, got {result['return_rate']}"
    print("[PASS] test_simulate_trade_loss")

def test_simulate_trade_timeout():
    """보유기간 만료 (TIME_OUT) 테스트"""
    entry_price = 10000
    target_rate = 0.10
    stop_rate = -0.05
    max_hold = 3
    
    prices = pd.DataFrame([
        {'date': datetime(2025, 1, 2), 'open': 10000, 'high': 10100, 'low': 9900, 'close': 10050},
        {'date': datetime(2025, 1, 3), 'open': 10050, 'high': 10200, 'low': 10000, 'close': 10100},
        {'date': datetime(2025, 1, 4), 'open': 10100, 'high': 10300, 'low': 10050, 'close': 10200}, # 3일차
        {'date': datetime(2025, 1, 5), 'open': 10200, 'high': 10400, 'low': 10100, 'close': 10300}, # 4일차 (무시되어야 함)
    ])
    
    result = simulate_trade(entry_price, prices, target_rate, stop_rate, max_hold, fee_rate=0.0)
    
    assert result['status'] == 'TIME_OUT', f"Expected TIME_OUT, got {result['status']}"
    assert result['days_passed'] == 3, f"Expected 3 days, got {result['days_passed']}"
    assert result['exit_price'] == 10200, f"Expected 10200, got {result['exit_price']}"
    assert approx_equal(result['return_rate'], 0.02), f"Expected 0.02, got {result['return_rate']}"
    print("[PASS] test_simulate_trade_timeout")

def test_simulate_trade_fee():
    """수수료 적용 테스트"""
    entry_price = 10000
    target_rate = 0.10 # 11000
    stop_rate = -0.05
    max_hold = 5
    fee = 0.0015 # 0.15%
    
    prices = pd.DataFrame([
        {'date': datetime(2025, 1, 2), 'open': 10000, 'high': 11500, 'low': 9900, 'close': 11100},
    ])
    
    result = simulate_trade(entry_price, prices, target_rate, stop_rate, max_hold, fee_rate=fee)
    
    # WIN but with fees
    assert result['status'] == 'WIN', f"Expected WIN, got {result['status']}"
    
    expected_ret = ((11000 * (1 - fee)) - (10000 * (1 + fee))) / (10000 * (1 + fee))
    assert approx_equal(result['return_rate'], expected_ret), f"Expected {expected_ret}, got {result['return_rate']}"
    print("[PASS] test_simulate_trade_fee")

if __name__ == "__main__":
    try:
        test_simulate_trade_win()
        test_simulate_trade_loss()
        test_simulate_trade_timeout()
        test_simulate_trade_fee()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
