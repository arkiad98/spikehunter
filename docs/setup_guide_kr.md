# SpikeHunter 새 PC 환경 설정 가이드

이 문서는 작업을 다른 PC로 옮겨서 계속 진행할 때 필요한 설정 순서를 안내합니다.

## 1. 사전 준비물
*   **Git**: [Git 다운로드](https://git-scm.com/downloads)
*   **Python**: 3.10 이상 권장 (현재 프로젝트는 Python 기반)
*   **VS Code** (권장 에디터)

## 2. 프로젝트 내려받기 (Clone)
터미널(PowerShell 또는 CMD)을 열고, 프로젝트를 저장할 폴더에서 다음 명령어를 실행합니다.

```bash
git clone https://github.com/arkiad98/spikehunter.git
cd spikehunter
```

## 3. 파이썬 가상환경 설정 (권장)
라이브러리 충돌을 방지하기 위해 가상환경을 사용하는 것이 좋습니다.

```bash
# 가상환경 생성 (env 이름은 자유)
python -m venv venv

# 가상환경 실행 (Windows)
.\venv\Scripts\activate
```

## 4. 라이브러리 설치
프로젝트에 필요한 필수 라이브러리를 설치합니다.

```bash
pip install -r requirement.txt
```

## 5. 데이터 복원 (중요!)
이 저장소에는 용량 문제로 인해 **주식 데이터 파일들이 포함되어 있지 않습니다.**
이전 PC에서 다음 폴더들을 새 PC의 `spikehunter` 폴더 내로 **직접 복사**해야 합니다.

*   📂 `data/` (수집된 원본 데이터)
*   📂 `merged/` (병합된 데이터)
*   📂 `features/` (머신러닝용 피처 데이터)
*   📂 `logs/` (이전 로그 기록 - 선택 사항)

> **팁:** USB나 클라우드 저장소(Google Drive 등)를 이용해 옮기시는 것을 추천합니다.

## 6. API 키 설정 (필요 시)
`config/kis_keys.yaml` 파일에 한국투자증권 API 키가 저장되어 있습니다. 
만약 보안상의 이유로 이 파일이 포함되지 않았다면, 기존 PC에서 안전하게 복사해 오거나 새로 작성해야 합니다.

## 7. 실행 테스트
데이터 복사가 완료되었다면, 다음 명령어로 파이프라인이 정상 작동하는지 확인합니다.

```bash
python run_pipeline.py
```
