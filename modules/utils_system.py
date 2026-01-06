import os
import multiprocessing

def get_optimal_cpu_count(ratio: float = 0.75) -> int:
    """
    현재 시스템의 CPU 코어 수 대비 특정 비율(기본 75%)을 계산하여 반환합니다.
    최소값은 1입니다.
    
    Args:
        ratio (float): 사용할 코어 비율 (0.0 ~ 1.0)
        
    Returns:
        int: 계산된 코어 수
    """
    try:
        total_cores = os.cpu_count() or 1
        optimal_cores = int(total_cores * ratio)
        return max(1, optimal_cores)
    except Exception:
        return 1
