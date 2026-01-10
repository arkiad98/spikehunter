# 프로젝트 리뉴얼 계획

## 목표
프로젝트 의존성을 최신 안정 버전으로 업데이트하고, 코드베이스가 경고나 오류 없이 정상 작동하도록 합니다.

## 현재 문제점
1.  **`pkg_resources` 지원 중단 경고**: `pykrx`에서 발생 (`setuptools` 관련).
2.  **`scikit-learn` 버전 불일치**: 이전 버전(1.7.1)에서 학습된 모델과 현재 설치된 라이브러리(1.8.0+) 간의 버전 차이로 인한 `InconsistentVersionWarning`.

## 제안된 변경 사항

### 1. 의존성 업그레이드
- `pip install --upgrade -r requirement.txt`를 실행하여 모든 핵심 라이브러리(`pandas`, `numpy`, `lightgbm`, `optuna` 등)를 업그레이드합니다.
- **영향**: `pandas` 2.x 또는 `numpy` 2.x 변경으로 인한 API 변화가 있을 수 있습니다.

### 2. 경고 해결
- **`pkg_resources`**:
    - `pykrx` 최신 버전 확인.
    - `run_pipeline.py`에서 특정 경고를 무시하도록 처리.
- **`scikit-learn`**:
    - **모델 재학습** 파이프라인(`train.run_train_pipeline`)을 실행하여 현재 라이브러리 버전으로 `lgbm_model.joblib`을 재생성.

### 3. 검증
- `run_pipeline.py` -> 전략 백테스트(Strategy Backtest)를 실행하여 엔드투엔드 흐름 확인.
