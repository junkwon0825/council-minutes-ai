# Council Minutes AI — 의정 이슈 레이더

광역 지방의회 회의록을 전처리·임베딩하고, **PostgreSQL + pgvector 기반 하이브리드 검색**과 로컬 LLM을 결합해 정책 이슈의 근거를 찾는 AI 서비스 프로젝트입니다.

> 2026 DATA·AI 분석 경진대회 5인 팀 프로젝트의 공개 포트폴리오 버전입니다. 원본 회의록, DB dump, 임베딩 파일, 모델 파일은 공개하지 않습니다.

## What I Built

제가 주도적으로 담당한 영역은 다음과 같습니다.

- JSON/HTML 회의록 전처리·정제·메타데이터 구조화
- 발언 단위 청킹 및 임베딩 파이프라인
- 한국어 임베딩 모델 비교·선정
- ChromaDB / FAISS / PostgreSQL+pgvector 검색 구조 실험
- PostgreSQL+pgvector 최종 저장·검색 구조 적용
- Dense Retrieval + BM25 + RRF + Cross-Encoder reranking
- 검색 결과 기반 이슈 군집화·타임라인 분석
- 로컬 LLM 기반 근거 답변 구조 및 Streamlit 웹 연결

개인 기여 범위는 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)에 코드 위치와 함께 정리했습니다.

## Final Architecture

```text
Raw JSON / HTML
      ↓
Preprocessing / Parsing
      ↓
Chunking / Metadata
      ↓
Snowflake Arctic Korean Embedding
      ↓
┌──────────────────────────────────┐
│ Retrieval Experiments            │
│ ChromaDB · FAISS · pgvector      │
└──────────────────────────────────┘
      ↓
PostgreSQL + pgvector  ← FINAL
      ↓
Dense Search + BM25
      ↓
RRF Fusion
      ↓
BGE Cross-Encoder Reranking
      ↓
Evidence
      ↓
Local LLM Agent
      ↓
Streamlit Web
```

## Retrieval / Database Decision

초기에는 ChromaDB와 FAISS를 활용해 검색 구조를 검증했고, 최종 서비스에서는 **PostgreSQL 17 + pgvector**를 선택했습니다.

기록된 동일 조건 벤치마크:

| Query | ChromaDB p50 | pgvector p50 |
|---|---:|---:|
| Region + Year filter | 586.29 ms | **3.97 ms** |
| No filter | **1.42 ms** | 2.75 ms |

무필터 검색만 보면 ChromaDB가 더 빠른 경우가 있었지만, 최종 서비스는 지역·연도 등 메타데이터 필터, SQL 집계, 웹 서비스와의 데이터 연동이 필요했습니다. 따라서 단순 속도가 아니라 **서비스 운영 요구사항을 포함해 pgvector를 최종 선택**했습니다.

- Chroma 적재: `src/15_load_chroma.py`
- pgvector 적재: `src/16_load_pgvector.py`
- DB benchmark: `src/17_benchmark_vector_databases.py`
- 최종 retrieval: `src/18_pgvector_retrieval.py`
- reranking: `src/19_rerank_candidates.py`
- BM25 index: `src/35_build_bm25_index.py`
- RRF runtime: `src/bm25_runtime.py`
- 공통 검색 runtime: `src/council_vector_runtime.py`

상세 의사결정은 [DATABASE_AND_RETRIEVAL_DECISION.md](docs/DATABASE_AND_RETRIEVAL_DECISION.md)를 참고하세요.

## Scale

최종 실행 기록 기준:

| Item | Value |
|---|---:|
| Canonical rows | 2,057,214 |
| Child chunks | 901,358 |
| Embedding dimension | 1,024 |
| Vector DB | PostgreSQL 17 + pgvector 0.8.7 |
| Vector rows | 901,358 |
| DB size | 약 12 GB |

대용량 파일은 공개 저장소에 포함하지 않고 재현 가능한 코드와 설정만 제공합니다.

## Search Pipeline

```text
Query
 ↓
Embedding
 ↓
pgvector HNSW Dense Retrieval
 +
BM25 Lexical Retrieval
 ↓
RRF
 ↓
BGE Cross-Encoder
 ↓
Deduplication / Context Recovery
 ↓
Evidence
```

## Final Analysis Pipeline

최종 제출 원본의 후반 분석 파이프라인도 공개 저장소에 포함했습니다.

| Stage | File |
|---|---|
| 이슈 확장 / 군집 후보 | `src/20_expand_cluster_issue.py` |
| 이슈 타임라인 | `src/21_build_issue_timeline.py` |
| 예측 feature | `src/22_build_forecast_features.py` |
| 단일 이슈 예측 | `src/23_forecast_issue.py` |
| 웹 payload 생성 | `src/24_build_web_payload.py` |
| 이슈 후보 생성 | `src/25_build_issue_candidates.py` |
| 후보 상태 확인 | `src/25a_show_candidate_status.py` |
| clustering threshold sweep | `src/26_cluster_threshold_sweep.py` |
| LLM extraction 평가셋 | `src/27_make_llm_extraction_evalset.py` |
| 검수 이슈 확정 | `src/30_finalize_reviewed_issues.py` |
| 다중 이슈 panel | `src/31_build_multi_issue_panel.py` |
| 다중 이슈 forecast | `src/32_train_multi_issue_forecast.py` |
| 외부 후속 흐름 | `src/32a_external_followup_flow.py` |
| 자유 질의 분석 | `src/33_analyze_free_query.py` |
| 최종 웹 bundle | `src/34_build_final_web_bundle.py` |

> 28, 29 번호는 최종 소스에 별도 파일이 존재하지 않습니다.

## Local Agent / Web

원문 외부 전송 제한을 고려해 답변 생성은 로컬 LLM을 사용했습니다. LLM이 근거를 새로 만드는 구조가 아니라 **검색된 회의록 근거를 설명하는 계층**으로 제한했습니다.

- Agent tool: `agent/tools.py`
- Local Agent: `agent/local_issue_agent.py`
- Streamlit app: `demo/final_app.py`
- 실행 스크립트: `scripts/run_final_app.sh`

## Project Evolution

초기 파이프라인은 FAISS 및 ChromaDB 기반 검색 실험을 포함합니다.

- 전처리: `src/01_inventory.py`, `src/02_preprocess.py`
- 청킹: `src/03_make_chunks.py`
- 임베딩: `src/04_embedding.py`
- FAISS: `src/05_build_faiss.py`
- 검색: `src/06_search_v2.py`
- 군집화: `src/07_clustering.py`

초기 팀 저장소 코드도 개발 provenance 보존을 위해 `legacy/initial_team_pipeline/`에 별도로 남겼습니다.

## Run

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. PostgreSQL + pgvector

```bash
docker compose up -d pgvector
```

### 3. Prepare local artifacts

공개 저장소에는 실제 회의록/embedding/DB dump가 포함되지 않습니다. 로컬에서 다음 항목을 준비해야 합니다.

```text
data/embeddings/arctic_raw_v1/embeddings.f16.npy
data/embeddings/arctic_raw_v1/chunk_metadata.parquet
data/bm25/arctic_raw_v1/bm25.sqlite3
```

### 4. Run

```bash
bash scripts/run_final_app.sh
```

## Repository Structure

```text
agent/          Local LLM Agent / retrieval tools
demo/           Streamlit web
docs/           Architecture / decisions / troubleshooting
legacy/         Early team-pipeline snapshot
scripts/        Execution / repository validation
src/            Preprocessing / embedding / retrieval / analysis
CONTRIBUTIONS.md
docker-compose.yml
requirements.txt
```

## Presentation\n\n최종 발표자료는 [presentation/README.md](presentation/README.md)에서 확인할 수 있습니다. 대용량 PPT 원본은 Drive 링크로 제공합니다.\n\n## Public Repository Policy

다음 항목은 GitHub에 올리지 않습니다.

- 대회 원문 및 정제 원문
- PostgreSQL dump / Chroma persistent DB
- embedding NPY / Parquet
- BM25 binary index
- model weights
- API keys / .env

`scripts/check_public_repo.sh`로 공개 금지 파일을 검사할 수 있습니다.

## Notes

- 최초 관측은 전국 최초 정책 발상지가 아니라 **제공 데이터 범위 내 최초 관측**입니다.
- 지역별 관측 순서는 정책 확산의 인과관계를 의미하지 않습니다.
- 성능 수치는 기록된 실험 조건에서의 결과이며 모든 환경에 일반화하지 않습니다.
