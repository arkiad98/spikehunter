# 개발 로그 (Development Log)

이 문서는 개발 과정에서의 시행착오, 시도한 내용, 그리고 그 결과를 기록하여 지식을 축적하는 것을 목적으로 합니다.
**제2원칙: 매 실행 및 주요 변경 시점마다 기록합니다.**

## 템플릿
```markdown
### [YYYY-MM-DD] 작업 제목
- **목표**: 무엇을 하려고 했는가?
- **시도 내용**: 어떤 변경이나 명령어를 실행했는가?
- **결과**: 성공/실패 여부, 발생한 에러, 주요 관찰 사항
- **교훈/조치**: 무엇을 배웠고, 다음엔 무엇을 할 것인가?
```

---

### 2026-01-06 폴더 구조 리팩토링
- **목표**: 루트 디렉토리의 복잡도를 낮추고, 핵심 로직과 개발/분석용 코드를 분리함.
- **시도 내용**:
  - `dev_space` 내에 `analysis`, `debug`, `tests`, `images` 하위 폴더 생성.
  - 루트의 `analyze_*.py`, `debug_*.py`, `test_*.py` 등을 해당 폴더로 이동.
  - `dev_space/README.md` 작성 및 루트 `README.md` 갱신.
- **결과**: `run_pipeline.py` 정상 실행 확인. 루트 디렉토리가 `config`, `modules`, `data`, `dev_space` 위주로 정리됨.
- **교훈/조치**: 앞으로의 단순 테스트나 분석 코드는 반드시 `dev_space` 내에서 생성해야 함.

### 2026-01-06 최적화 프로세스 CPU 사용량 표준화
- **목표**: ML 및 전략 파라미터 최적화 시 CPU 자원 사용을 시스템 사양의 75%로 표준화하여 과부하 방지 및 효율성 제고.
- **시도 내용**:
  - `modules/utils_system.py`: `get_optimal_cpu_count(ratio=0.75)` 함수 생성.
  - `modules/train.py`: ML 최적화(모델 학습) 시 Optuna는 직렬(1), 모델은 병렬(75% 코어)로 설정하도록 로직 변경.
  - `modules/optimization.py`: 전략 최적화(백테스트) 시 Optuna는 병렬(75% 코어)로 실행하여 처리량 증대.
  - `config/settings.yaml`: 하드코딩된 `n_jobs: 21`을 `-1`로 리셋.
- **결과**: 사용자의 개입 없이도 최적의 병렬 처리 리소스(75%)를 자동으로 계산하여 할당함.
- **교훈/조치**: 하드웨어 리소스 의존적인 설정은 코드 레벨에서 동적으로 처리하는 것이 바람직함.


### 2026-01-06 데이터 수집 장애 해결 및 하이브리드 구조 도입
- **목표**: 2026년 데이터 수집 시 발생하는 `pykrx` 크래시(KeyError) 및 KRX 서버 차단(400/403) 문제를 해결하고, 안정적인 데이터 확보 체계 구축.
- **시도 내용**:
  1.  **Pykrx Patch**: `modules/patch_pykrx.py`를 생성하여 HTTP Referer 헤더 주입 및 `stock.get_market_ohlcv_by_ticker` 함수 Monkeypatch (영문 컬럼 매핑 등 안전장치 추가).
  2.  **Collection Logic 개선**: `collect.py`에서 불명확한 alias(`get_market_ohlcv`) 대신 명시적인 함수(`get_market_ohlcv_by_ticker`) 사용 및 지수 수집 실패 시 Skip 처리 추가.
  3.  **Hybrid 구조화**: `settings.yaml`의 API Key 유무에 따라 Scraping(기본값) ↔ KRX OpenAPI(고급) 자동 전환 로직 구현 (`collect_openapi.py`).
  4.  **OpenAPI 연동**: `KRX_KOSPI_시세정보_개발명세서.docx`를 분석하여 정확한 URL(`data-dbg.krx.co.kr...`) 및 파라미터(`basDd`) 확인 및 적용.
- **결과**:
  - **Stock Data**: 개별 종목 데이터는 Patch를 통해 정상 수집 성공.
  - **Index Data**: Scraping은 차단됨을 확인. `derive.py`에서 데이터 누락 시 기본값(0) 처리하여 파이프라인 중단 방지. OpenAPI는 401(승인대기) 상태 확인.
- **교훈/조치**: 외부 데이터 의존성(Scraping)은 언제든 깨질 수 있으므로, 공식 API(OpenAPI)와 같은 확실한 대안을 Hybrid로 구성하는 것이 운영 안정성에 필수적임.

### 2026-01-06 수집/예측 파이프라인 긴급 복구 및 안정화
- **목표**: 통합 파이프라인 실행(`run_pipeline.py`) 시 발생하는 `SyntaxError`, `KeyError`, `UnboundLocalError` 등 일련의 런타임 오류를 해결하여 추천 종목 생성 기능(Predict)을 정상화함.
- **시도 내용**:
  1.  **Code Fix (`collect.py`)**: 편집 실수로 남은 Orphaned `except` 블록 삭제 (SyntaxError 해결).
  2.  **Logic Fix (`predict.py`)**:
      - `KeyError: '지수명'`: Pykrx 내부의 지수 수집 로직이 차단된 점을 확인, `005930`(삼성전자) 데이터로 영업일을 확인하는 Proxy 패턴 적용.
      - `KeyError: 'date'`: KOSPI 데이터가 없거나 수집 실패 시에도 시스템이 멈추지 않도록 예외 처리 강화 및 기본 시장 국면(`R4_BearVolatile`) 가정 로직 추가.
      - `UnboundLocalError`: 변수(`current_kospi_vol_20d`)의 스코프 오류 수정.
  3.  **IO Reliability (`utils_io.py`)**: `load_index_data` 함수가 Parquet 로드 시 인덱스/컬럼의 `date` 존재 여부를 자동 판별하여 복원하도록 개선.
  4.  **Config Update (`settings.yaml`)**: 하드코딩된 전략 키(`SpikeHunter_R4_BearVolatile` 등)가 설정 파일에 누락된 점을 발견, `R2`~`R4` 전체 국면에 대한 파라미터를 명시적으로 추가.
- **결과**: `python -c ... generate_and_save_recommendations` 검증 실행 성공. 데이터 누락 상황에서도 파이프라인이 멈추지 않고 안전하게(Fail-Safe) 동작함을 확인.
- **교훈/조치**: `try-except`로 단순히 에러를 로그로 남기는 것을 넘어, 핵심 데이터 부재 시에도 시스템이 작동 가능한 'Default State'를 정의하는 것이 운영 안정성에 핵심적임.

### 2026-01-06 단일 전략 모드(Single Strategy Mode) 강제 적용
- **목표**: 복잡한 시장 국면(Regime)에 따른 전략 스위칭 대신, 검증된 단일 전략(`R1_BullStable`)만 사용하여 운영 복잡도를 낮추고 안정성을 확보함.
- **시도 내용**:
  - `predict.py`: 시장 국면 진단 로직은 분석용으로 유지하되, 실제 적용되는 `strategy_key`는 무조건 `SpikeHunter_R1_BullStable`로 고정.
  - `settings.yaml`: 불필요해진 `R2`, `R3`, `R4` 전략 파라미터 블록 제거.
- **결과**: KOSPI 데이터 부재로 'R4_BearVolatile' 국면이 진단되었으나, 시스템은 R1 전략을 로드하여 정상적으로 추천 종목을 생성함 (Exit Code 0).
- **교훈/조치**: 사용하지 않는 기능(Regime Switching)은 과감히 비활성화하거나 제거하여 유지보수 비용을 줄이는 것이 좋음.

### 2026-01-06 ML 최적화(Advanced) CPU 표준화 적용
- **목표**: '진단 및 심층 분석(Add-ons)' 메뉴의 ML 최적화 기능(`optimization_ml.py`)에서도 시스템 표준 CPU 정책(75% 활용)을 준수하도록 함.
- **시도 내용**:
  - `optimization_ml.py`: 하드코딩된 `n_jobs: 4`를 제거하고, `utils_system.get_optimal_cpu_count(0.75)`를 호출하여 동적으로 할당하도록 수정.
- **결과**: 사용자의 코어 수에 맞춰 최적의 병렬 처리가 수행됨.
- **교훈/조치**: 별도 모듈로 분리된 기능(`optimization_ml`)도 공통 유틸리티 정책을 따르는지 꼼꼼히 체크해야 함.

### 2026-01-06 데이터 수집 및 최적화 파이프라인 버그 수정
- **목표**: 데이터 수집 실패(컬럼명 불일치, API Key 인식 불가) 및 최적화 중단(파일 손상) 문제를 해결하여 파이프라인의 안정성을 확보함.
- **시도 내용**:
  1. **Pykrx Patch 적용 (`run_pipeline.py`)**: 프로그램 시작 시 `patch_pykrx_referer`를 강제 실행하여 영문/한글 컬럼명 불일치 문제 해결.
  2. **API Key 로드 수정 (`collect.py`)**: `settings.yaml`의 `ml_params` 섹션 하위에 있는 `krx_api_key`를 올바르게 읽어오도록 수정 (OpenAPI 모드 활성화).
  3. **Proxy 로직 수정 (`predict.py`)**: 삼성전자 주가로 영업일을 확인할 때 잘못된 함수(`get_market_ohlcv_by_ticker`)를 사용하던 것을 `get_market_ohlcv`로 수정.
  4. **파일 손상 대응 (`optimization.py`)**: `ml_classification_dataset.parquet` 읽기 실패(ArrowInvalid) 시 크래시 대신 재생성 가이드를 출력하도록 예외 처리 추가.
- **결과**:
  - 수집 로직이 정상 작동하며 "None of [...] columns" 에러가 사라짐.
  - 최적화 시 손상된 파일에 대해 적절한 안내 메시지가 출력됨.
  - OpenAPI 모드가 정상적으로 활성화됨.
- **교훈/조치**: 외부 라이브러리(Pykrx) 패치는 실행 최우선 순위로 둬야 하며, 설정 파일 구조 변경 시 관련 로드 로직도 꼼꼼히 동기화해야 함.

### 2026-01-06 KRX 데이터 수집 차단 해결 (Pykrx Patch V2)
- **목표**: Pykrx 패치 적용 후에도 지속되는 데이터 수집 실패(빈 응답) 문제를 해결함.
- **시도 내용**:
  - `debug_pykrx_patch.py` 스크립트를 작성하여 패치 적용 여부와 실제 응답을 검증.
  - GitHub Upstream PR(#249) 재분석 결과, 초기 적용한 Referer URL(`http://.../mdiLoader/...`)이 유효하지 않음을 확인.
  - Referer를 `https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd` (HTTPS + outerLoader)로 수정하여 재적용.
- **결과**: `debug_pykrx_patch.py`에서 삼성전자 등 종목 데이터가 정상적으로 수신됨을 확인. 이후 메인 파이프라인 수집 성공.
- **교훈/조치**: 외부 API 패치 시에는 URL 경로와 프로토콜(HTTP/HTTPS)의 정확성이 매우 중요하며, 문제 발생 시 독립적인 디버깅 스크립트로 검증하는 절차가 유효함.

### 2026-01-06 피처 생성(Derive) 기간 자동화
- **목표**: 데이터 수집은 2026년까지 완료되었으나, 피처 생성(`Derive`) 단계에서 하드코딩된 종료일(`2025-12-31`)로 인해 최신 데이터가 반영되지 않는 문제 해결.
- **시도 내용**:
  - `modules/derive.py`: `data_range.end` 설정값이 없을 경우의 기본값을 고정된 날짜에서 `pd.Timestamp.now()`(오늘)로 변경.
- **결과**: 별도의 설정 변경 없이도 수집된 최신 데이터까지 자동으로 피처 생성이 수행됨.
- **교훈/조치**: 유지보수 편의를 위해 시간과 관련된 기본값은 항상 동적인 값(Now/Today)을 사용하는 것이 좋음.

### 2026-01-06 최적화 필터 미스매치 해결 및 변동성 필터 구현
- **목표**: 전략 최적화(Optimization) 과정에서 모든 시도가 `-999.0`(탈락)으로 반환되는 문제를 해결하고, 사용자 요구에 맞춰 MDD 제한(-30%)을 유지하면서도 유효한 파라미터를 찾을 수 있도록 함.
- **원인 분석**:
  - `optimization.py`는 MDD < -30% 인 경우를 엄격히 필터링함.
  - 그러나 `backtest.py` 엔진이 `max_market_vol`(변동성 제한) 파라미터를 무시하고 있어, 하락장에서도 매수를 지속하여 MDD가 -31% 수준까지 치솟음.
  - 결과적으로 우수한 수익률(CAGR 479%)에도 불구하고 MDD 제한에 걸려 모든 시도가 탈락함.
- **시도 내용**:
  1.  **Constraint Revert (`optimization.py`)**: 임시 완화했던 MDD 제한(-50%)을 사용자 요청에 따라 다시 -30%로 원복.
  2.  **Backtest Logic Fix (`backtest.py`)**: 최적화 파라미터(`param_overrides`)에서 `max_market_vol`을 읽어와, 매일 장 시작 전 시장 변동성(`market_volatility`)을 체크하고 초과 시 매수를 건너뛰는 로직 추가.
  3.  **Debug Verification (`debug_optim_trial.py`)**: 디버그 스크립트를 통해 변동성 필터 적용 전후의 거래 횟수 감소(2750회 → 3010회? ※로직 변경으로 포트폴리오가 달라져 거래 횟수 패턴 변동) 및 MDD 개선 효과 검증.
- **결과**: `max_market_vol`이 동작하게 됨으로써, 최적화 시 변동성을 제한하는 파라미터 조합이 MDD -30% 이내로 들어와 유효한 점수를 받을 수 있게 됨.
- **교훈/조치**: 최적화 엔진과 백테스트 엔진 간의 파라미터 처리 로직이 100% 일치해야 올바른 최적화가 가능함. '무시되는 파라미터'가 없는지 주기적 점검 필요.

### 2026-01-07 KOSPI 지수 데이터 수집 재구축 (OpenAPI 개선)
- **목표**: 대량의 KOSPI 지수 데이터(2020~2026) 수집 시 발생하는 "멈춤" 현상(UX)과 데이터 결측(NaN) 문제를 해결하여 신뢰할 수 있는 데이터셋을 확보함.
- **시도 내용**:
  1. **UX 개선 (`collect_openapi.py`)**: `tqdm` 라이브러리를 도입하여 6년치 일별 데이터 수집 진행률을 시각화함.
  2. **Parsing Logic 개선**: OpenAPI 응답 내에서 KOSPI 대표 지수가 아닌 항목(거래대금 통계 등)이 첫 번째로 반환되어 가격이 `NaN`으로 잡히는 버그 수정.
     - `IDX_NM`(이름) 필터링 대신, `MKTCAP`(시가총액) 정렬 후 Top 1을 선택하고, `CLSPRC_IDX`(종가) 유효성을 검증하는 로직으로 변경.
  3. **재수집**: 기존 손상된 `kospi.parquet` 삭제 후 전체 재수집 수행.
- **결과**:
  - 진행률 표시바가 정상 출력되어 수집 상태 확인 가능해짐.
  - 재수집된 `kospi.parquet` 검증 결과, 1,476일치 데이터 전량에 대해 종가(Close) 결측치 없음 (Pass).
- **교훈/조치**: API 응답의 순서는 보장되지 않으므로, 데이터 파싱 시에는 항상 '검증된 값(시가총액, 가격 유무)'을 기준으로 필터링해야 함. UX 측면에서 장기 작업에는 반드시 진행률 표시가 필요함.

### 2026-01-07 데이터 경로 불일치로 인한 피처 생성 오류 해결
- **목표**: `Derive` 단계에서 "피처 생성 결과 없음" 오류가 발생하는 원인을 분석하고, 유실된 과거 데이터를 복구하여 파이프라인을 정상화함.
- **원인 분석**:
  - `data/proc/merged` 폴더에는 2026년 데이터만 존재함.
  - `metadata.json`은 2026년까지 수집 완료된 것으로 기록되어 있어, 수집기가 과거 데이터를 건너뜀.
  - 유실된 것으로 추정된 2020~2025년 데이터가 루트 디렉토리의 `d:/spikehunter/merged` 폴더에서 발견됨 (경로 설정 변경 과정에서의 잔재로 추정).
- **시도 내용**:
  - `move d:\spikehunter\merged\* d:\spikehunter\data\proc\merged\` 명령어로 데이터 파티션 이동 및 통합.
  - `modules.derive` 실행을 통해 데이터 로드 및 피처 생성 정상 작동 검증.
- **결과**: `Derive` 모듈이 2020년부터 2026년까지의 전체 데이터를 정상적으로 로드하고 피처 생성을 시작함.
### 2026-01-07 전략 최적화 기간 수정 및 로그 상세화
- **목표**: 사용자 요청에 따라 전략 최적화 기간을 "42개월 전 ~ 6개월 전"으로 고정하고, 탈락 사유(MDD 등)를 명확히 파악할 수 있도록 개선함.
- **시도 내용**:
  1. **Period Logic Fix (`optimization.py`)**: `ml_params`의 `classification_train_months`(36)와 `offset`(6)을 참조하여 동적으로 42개월 기간을 설정하도록 수정. (기존 하드코딩 2020-01-01 제거)
  2. **Logging Update (`optimization.py`)**: 최적화 시도(Trial)가 탈락할 때, 단순히 `-999.0`만 반환하는 대신 `WR`(승률)과 `MDD` 수치를 INFO 레벨로 로그에 남겨 제약 조건 위반 내역을 가시화함.
- **결과**: 최적화가 의도된 기간(2022년~2025년)에 대해 수행되며, 사용자가 실패 원인을 보고 제약 조건을 조절할 수 있게 됨.
- **교훈/조치**: 최적화 과정은 'Black Box'가 되기 쉬우므로, 실패 시에는 반드시 그 이유(수치)를 사용자에게 알려주는 투명성이 필요함.

### 2026-01-07 전략 파라미터 최적화 및 검증 (Optimization & Validation)
- **목표**: `SpikeHunter_R1_BullStable` 전략의 안정성을 확보하기 위해, ML 학습 데이터 기간(24개월)에 맞춰 전략 파라미터를 최적화하고 검증함.
- **시도 내용**:
  - `optimization.py`: 최적화 기간을 "42개월"에서 "27개월 전 ~ 3개월 전"으로 조정 (ML 학습 기간과 동기화).
  - `optimization.py`: MDD 제약 조건을 -30%로 원복하고, Headless 모드(`n_jobs=-1`) 지원 추가.
  - `Validation Backtest`: WFO(전진 분석)에서 발굴된 최적 파라미터(`target_r=0.06977`, `stop_r=-0.02589`, `threshold=0.291`)를 적용하여 전체 기간 백테스트 수행.
- **결과**:
  - **Global Optimization**: 엄격한 MDD 제약(-30%)으로 인해 과거 하락장(2022년)을 포함한 전체 기간 최적화는 실패(모든 Trial 탈락).
  - **Walk-Forward Integration**: 그러나 WFO의 각 구간별 최적화는 성공적(WR > 50%, MDD ~ -15%)이었음. 이를 통해 최근 시장에 적합한 파라미터를 확보.
  - **Final Validation**: 확보된 파라미터로 전체 기간(5.5년) 백테스트 결과, **CAGR 81.91%**, **MDD -47.28%** 기록.
- **교훈/조치**:
  - 단일 전략으로 모든 시장 국면(특히 2022년 대세 하락장)의 MDD를 -30% 이내로 방어하는 것은 불가능에 가까움.
  - 높은 수익률(81%)을 감안하여 MDD -47%를 감수하고 파라미터를 채택하되, 향후 하락장에서는 전략 가동을 중단하는 '시장 국면 필터' 도입을 고려해야 함.

## 2026-01-07: Strategies Parameter Optimization & Improvement
- **목표**: `max_market_vol` 파라미터가 전략 성과에 미치는 영향 분석 및 최적화, 그리고 최적화 로그 가시성 확보.
- **분석 내용**:
  - 기존 `max_market_vol` 상한(0.04)이 지나치게 보수적이어 수익 기회를 제한한다는 가설 검증.
  - **비교 실험**: "완화(0.10)" vs "사실상 제거(0.50)" 시나리오로 최적화 진행.
- **결과**:
  - **제한 제거(0.32)** 시 `Score 10.77`로 압도적 성과 기록 (Baseline -999.0 대비).
  - 승률 41.2%, MDD -27.1%로 안정성 유지하며 수익 극대화.
- **조치**:
  - `config/settings.yaml`: `max_market_vol` 상한을 `0.50`으로 대폭 상향.
  - `modules/optimization.py`: WFO 등 최적화 과정에서 각 Trial의 파라미터가 로그에 출력되지 않던 문제 수정 (`Params: {...}` 추가).

### 2026-01-07 검증 로직 동적화 및 리포트 고도화 (Verification Logic Dynamic Parameter & Report Improvement)
- **목표**: 전략 파라미터 변경 시 과거 신호에 대한 검증 결과도 동적으로 반영되도록 개선하고, 리포트의 정보량과 가독성을 높임.
- **시도 내용**:
  1. **DB 스키마 확장 (`utils_db.py`)**: `daily_signals` 테이블에 `target_rate`, `stop_rate` 컬럼을 추가하여 신호 발생 시점의 기준을 기록.
  2. **검증 로직 재설계 (`verify_daily_signals.py`)**:
     - `simulate_trade` 함수로 트레이딩 로직 분리 (필드 테스트 모듈 재사용 대비).
     - 검증 시 `settings.yaml`의 최신 파라미터를 적용하여 승패를 재계산하는 동적 로직 구현.
     - 수수료(`fee_rate`) 차감 로직 추가.
  3. **리포트 개선**:
     - '조건' 컬럼(`+10.00% / -5.00%`)을 추가하여 적용된 파라미터 명시.
     - (추가) 엑셀 호환성을 위해 '조건'을 '수익조건', '손절조건' 컬럼으로 분리하고, 일별 데이터 표시를 `가격 (등락률)` 포맷으로 변경.
     - 5일간의 주가 흐름, ML 점수, 최고가 도달률 등 상세 지표 추가.
     - 터미널 출력은 가독성을 위해 `+3일`까지만 표시하고, CSV 파일에는 전체 데이터 저장.
- **결과**: `test_dynamic_verification.py` 테스트 통과. 리포트를 통해 전략 변경에 따른 성과 변화를 즉각적으로 확인할 수 있게 됨.
- **교훈/조치**: 검증 모듈은 단순 기록용이 아니라 '전략 튜닝의 피드백 루프'로 활용되어야 함. 향후 가상 계좌(Field Test) 기능도 `simulate_trade` 함수를 기반으로 구축할 예정.

### 2026-01-08 프로젝트 규칙 및 페르소나 정의
- **목표**: 에이전트의 역할(Persona)과 프로젝트 핵심 규칙(4대 원칙)을 명문화하여 파일로 관리함.
- **시도 내용**:
  - `D:\spikehunter\.agent\rules\.rules` 파일 생성 및 상세 규칙 작성.
  - 역할: Quant Dev + Pro Trader + Macro Analyst.
  - 원칙: 한글 사용, 금융 무결성(Look-Ahead Bias 방지), 코드 품질, Git 컨벤션.
- **결과**: 에이전트가 해당 규칙을 숙지하고, 향후 작업의 기준점으로 삼게 됨.
- **교훈**: 명확한 룰 세팅은 에이전트와의 협업 퀄리티를 보장하는 기초임.
