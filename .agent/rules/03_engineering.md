# 엔지니어링 및 무결성 (Engineering & Integrity)

## 금융 논리 무결성 (Financial Integrity)
- **미래 참조 금지 (Check Look-Ahead Bias)**: 학습/백테스트 시 미래 데이터(Close, High 등)가 피처로 유출되지 않도록 철저히 검증합니다.
- **거래 비용 필수 반영**: 모든 수익률 계산 시 수수료와 슬리피지를 보수적으로 적용합니다.
- **검증 우선주의**: 최적화(Optimization) 전에 논리적 타당성을 검증하고, 전진 분석(Walk-Forward) 없는 결과는 신뢰하지 않습니다.

## 코드 품질 및 구조 (Code Quality & Structure)
- **폴더 구조 준수**:
    - **핵심 로직**: `modules/` (기능 단위), `run_pipeline.py` (실행 진입점)
    - **샌드박스**: `dev_space/` (실험/테스트/임시)
    - **유틸리티**: `addons/` (보조 도구)
- **방어적 코딩**: 외부 API 및 데이터 오류 발생 시 프로그램이 중단되지 않도록 예외 처리를 강화합니다.
- **모듈화**: 복잡한 로직은 `modules/` 내의 함수나 클래스로 분리하여 재사용성을 높입니다.

## 형상 관리 (Version Control)
- **커밋 메시지 규칙 (Conventional Commits)**:
    - `feat` (기능), `fix` (수정), `refactor` (리팩토링), `docs` (문서), `chore` (기타)
- **작업 단위**: 커밋은 논리적으로 완결된, 빌드 가능한 상태의 최소 단위로 수행합니다.
