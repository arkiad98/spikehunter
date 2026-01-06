# Dev Space

이 폴더는 개발 과정에서의 실험, 분석, 디버깅 코드를 모아두는 곳입니다.
루트 디렉토리의 청결을 유지하기 위해 모든 일회성 스크립트나 분석 도구는 성격에 맞는 하위 폴더에 위치해야 합니다.

## 구조 (Structure)

### `analysis/`
- 데이터 분석, 모델 비교, 통계 산출 등을 위한 스크립트.
- 예: `analyze_market_cycles.py`, `compare_models_direct.py`

### `debug/`
- 특정 모듈의 동작을 확인하거나 버그를 추적하기 위한 스크립트.
- 예: `debug_model.py`, `verify_data_slice.py`

### `tests/`
- 실험적인 기능 구현, 새로운 파라미터 최적화 기법 테스트 등.
- 예: `experiment_wfo_lookback.py`, `run_opt_global.py`

### `images/`
- 분석 과정에서 생성된 차트나 스크린샷 저장.

## 주의사항 (Caution)
이 폴더 내의 스크립트들은 루트(`d:\spikehunter`)를 기준으로 실행해야 정상적으로 `modules` 패키지를 임포트할 수 있는 경우가 많습니다.
만약 `ModuleNotFoundError`가 발생하면 다음 두 가지 방법 중 하나를 사용하세요:

1. **루트에서 모듈로 실행**:
   ```bash
   python -m dev_space.analysis.analyze_market_cycles
   ```

2. **sys.path 추가**:
   스크립트 상단에 다음 코드를 추가:
   ```python
   import sys, os
   sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # dev_space 상위(루트) 추가
   ```
