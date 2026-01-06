# Dev Space (Development & Analysis Area)

## 개요
이 폴더(`d:\spikehunter\dev_space`)는 **개발, 테스트, 디버깅, 실험적 분석**을 위한 전용 공간입니다. 메인 파이프라인(`run_pipeline.py`) 및 핵심 모듈(`modules/`)을 깔끔하게 유지하기 위해, 관련된 보조 스크립트들을 이곳에서 관리합니다.

## 폴더 구조
- **`analysis/`**: 데이터 심층 분석, 통계, 시스템 진단 및 리포트 파일
  - 예: `analyze_market_cycles.py`, `diagnose_filtering.py`, `analysis_report.md`
- **`debug/`**: 버그 추적, 로그 확인, 데이터 무결성 단순 점검 스크립트
  - 예: `debug_pykrx_2026.py`, `read_raw_log.py`
- **`tests/`**: 단위 테스트, 성능 검증, 실험적 최적화 스크립트
  - 예: `test_pykrx_connection.py`, `verify_fix.py`, `compare_models.py`
- **`images/`**: 분석 과정에서 생성된 차트 및 이미지 파일
  - 예: `analysis_rfe_result.png`, `analysis_pre_spike_pattern.png`
- **(Root)**: 기타 임시 스크립트

## 사용 규칙
1. **Core 코드 보호**: `modules/` 내의 핵심 코드를 수정하기 전, 이곳에서 `tests/` 스크립트를 통해 충분히 검증하십시오.
2. **분석 결과 정리**: 데이터 분석을 수행한 후 생성된 로그나 리포트는 `analysis/` 폴더에 정리하십시오.
3. **경로 주의**: 루트 디렉토리에서 실행되던 스크립트들이 이동되었으므로, import 경로나 파일 경로(`../config/settings.yaml` 등) 수정이 필요할 수 있습니다.
