# ML 최적화 모듈 통합 및 리팩토링 완료

중복된 `optimization_ml.py`를 제거하고, `train.py`에 ML 최적화 및 고급 분석 기능을 통합하였습니다. 이제 `Train` 메뉴 내에서 일관된 데이터 로드 로직과 로깅 시스템을 통해 최적화를 수행할 수 있습니다.

## Changes

### 1. [train.py](file:///d:/spikehunter/modules/train.py) (Enhanced)
- **`logging_callback` 추가**: Optuna의 `stderr` 로그 대신 프로젝트 표준 `logger`를 사용하여 JSON 로그 파일에 Trial 정보를 기록합니다.
- **`analyze_importance` 이식**: 최적화 완료 후 하이퍼파라미터 중요도(Importance)를 분석하여 출력하는 기능을 추가했습니다.
- **`calculate_top_n_precision` 추가**: 보조 지표인 Top-N Precision을 계산하고 기록합니다.
- **데이터 로드 로직 통일**: `dataset_v4.parquet`(WF 학습 데이터)가 존재할 경우 우선 로드하도록 하여, 학습과 최적화 간의 기간 불일치 문제를 해결했습니다.

### 2. [run_pipeline.py](file:///d:/spikehunter/run_pipeline.py) (Simplified)
- **메뉴 통합**: 중복된 "3. ML 하이퍼파라미터 최적화 (Advanced)" 메뉴를 제거했습니다.
- **사용성 개선**: "2. ML 하이퍼파라미터 최적화 (Optuna)" 선택 시, `n_trials` 모드(Fast Scan vs Deep Search)를 선택할 수 있도록 UX를 개선했습니다.

### 3. [optimization_ml.py] (Deleted)
- 중복 코드를 제거하여 유지보수성을 높였습니다.

## Validation Results

### 1. 통합 최적화 파이프라인 검증
`python run_pipeline.py`를 통해 ML 최적화를 실행한 결과, 모든 기능이 정상 작동함을 확인했습니다.

```text
[Optimization Result]
 Best AP: 0.1816
 Top-N Precision (Proxy): 0.2277
 Best Params: {'learning_rate': 0.0035, 'num_leaves': 102, ...}

[Hyperparameter Importance Analysis]
 1. num_leaves          : 47.0%
 2. scale_pos_weight    : 13.4%
 3. min_child_samples   : 13.2%
 ...
```

### 2. 기간 불일치 해결
`train.py`가 사용하는 `_run_classification_training`과 `Objective` 클래스가 동일한 데이터 로드 로직을 공유하므로, 더 이상 데이터셋 불일치로 인한 기간 차이가 발생하지 않습니다.
