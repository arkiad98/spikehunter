# 기본 원칙 (Core Principles)

- **언어 정책 (Language Policy)**: 모든 커뮤니케이션, 주석, 그리고 에이전트가 생성하는 모든 문서(Task, Implementation Plan, Walkthrough 등)는 **반드시 한글**로 작성합니다. (UTF-8 적용하며, 변수/함수명 등 코드 식별자는 영문 사용)
- **산출물 문서화 정책 (Artifact Documentation Policy)**:
    - 에이전트가 생성한 중요 산출물(Task, Implementation Plan, Walkthrough)은 반드시 `dev_space/artifacts/` 폴더에 마크다운 파일로 저장합니다.
    - **파일명 규칙**: `YYYYMMDD_{Type}_{ShortTitle}.md` (예: `20260108_Plan_ML_Optimization_Fix.md`)
- **개발 로그 (Dev Log)**: 작업 완료 시 `dev_space/develop.md`에 로그를 의무적으로 작성하며, 저장된 산출물과 동기화할 수 있도록 기록합니다.
    - 템플릿: `### [YYYY-MM-DD] 제목`
      - **목표 (Goal)**: 작업의 배경 및 목적
      - **산출물 (Artifacts)**:
        - [Task](dev_space/artifacts/...)
        - [Plan](dev_space/artifacts/...)
        - [Walkthrough](dev_space/artifacts/...)
      - **요약 (Summary)**: 주요 변경 사항 및 의사결정 내용 요약
      - **교훈 (Lessons)**: 배운 점 및 향후 조치
