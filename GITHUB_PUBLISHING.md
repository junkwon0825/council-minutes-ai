# GitHub 공개 저장소 정리 가이드

## 권장 운영 방식

프로젝트별 저장소를 분리한다.

- `council-minutes-ai`: 광역 지방의회 회의록 분석 코드
- `port-policy-analysis`: 항만 프로젝트 코드

팀 작업용 저장소는 Private으로 유지하고, 외부 공개가 필요한 경우 데이터와 실행 결과를
제거한 별도의 Public 저장소를 두는 방식이 안전하다.

## 의회 회의록 공개 저장소 구조

```text
council-minutes-ai/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── configs/
│   └── pipeline.example.yaml
├── src/
│   ├── 01_inventory.py
│   ├── 02_preprocess.py
│   ├── 03_make_chunks.py
│   ├── 04_embedding.py
│   ├── 04b_issue_embedding.py
│   ├── 04c_issue_relevance.py
│   ├── 04d_embed_selected.py
│   ├── 04e_semantic_text.py
│   ├── 05_build_faiss.py
│   ├── 06_search_v2.py
│   ├── 07_clustering.py
│   ├── 08_llm_issue_labeling.py
│   ├── 09_build_issue_timeline.py
│   ├── 10_issue_query.py
│   └── 11_forecast_issue_flow.py
├── scripts/
│   ├── run_final_pipeline.sh
│   ├── run_issue_query.sh
│   ├── run_demo.sh
│   ├── run_sample_demo.sh
│   └── check_public_repo.sh
├── demo/
│   └── app.py
├── docs/
│   ├── PIPELINE.md
│   ├── PROJECT_PROGRESS.md
│   ├── MODEL_CARD.md
│   ├── TROUBLESHOOTING.md
│   ├── DECISION_LOG.md
│   └── DEMO_GUIDE.md
├── examples/
│   ├── sample_issue_timelines.json
│   └── sample_issue_forecasts.json
└── tests/
```

## 공개하면 안 되는 파일

대회 데이터는 경진대회 목적 외 사용 및 제3자 제공이 제한되므로 아래 항목을 Public
GitHub에 올리지 않는다.

- `data/raw/`, `data/processed/`, `data/index/`
- 회의록 원문·원문 구간이 포함된 CSV, JSON, Parquet
- 임베딩 `.npy`, FAISS 인덱스, KMeans 모델
- Hugging Face 모델 캐시와 Ollama 모델 파일
- 실제 회의록을 포함한 검색 결과와 LLM 프롬프트 로그
- API 키, 비밀번호, 토큰, `.env`

공개 예시는 직접 만든 합성 데이터만 사용한다. 집계 통계도 대회 규정을 먼저 확인한 후
최소한으로 공개한다.

## 최초 업로드 전 확인

```bash
git status
git check-ignore -v data/raw/2024_강원특별자치도의회.json
git ls-files | rg '^(data|models|outputs)/|\.(parquet|npy|faiss|joblib|gguf)$'
bash scripts/check_public_repo.sh
```

마지막 명령에 실제 데이터 파일이 나오면 이미 Git 추적 대상이라는 뜻이다. 아직 원격에
올리지 않았다면 `.gitignore` 적용 후 캐시에서만 제거한다.

```bash
git rm -r --cached data models outputs
git add .gitignore
git commit -m "chore: exclude local data and generated artifacts"
```

이미 공개 저장소의 과거 커밋에 데이터가 들어갔다면 단순 삭제로는 기록이 남는다.
그 경우 저장소 공개를 중단하고 새 Public 저장소를 깨끗한 코드만으로 만드는 방식을 우선한다.

## 두 원격 저장소에 올리기

```bash
git remote -v
git remote add team git@github.com:TEAM/council-minutes-ai.git
git remote add public git@github.com:PUBLIC_ORG/council-minutes-ai.git

git push team main
git push public main
```

Public 저장소에 올리기 전에는 `git ls-files`로 데이터·모델·결과물이 추적되지 않는지 다시
확인한다.

## 새 Public 저장소 업로드 예시

GitHub에서 비어 있는 `council-minutes-ai` 저장소를 만든 뒤 공개용 폴더에서 실행한다.

```bash
git init
git branch -M main
git add .
bash scripts/check_public_repo.sh
git commit -m "feat: add local AI pipeline for council minutes analysis"
git remote add origin https://github.com/kwonjun99/council-minutes-ai.git
git push -u origin main
```

저장소 이름이나 계정이 다르면 원격 URL만 수정한다. GitHub 인증은 브라우저 로그인 또는
Personal Access Token을 사용하고 토큰을 파일에 저장하지 않는다.

