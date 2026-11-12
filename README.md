# Council Minutes AI — 의정 이슈 레이더

2026 DATA·AI 분석 경진대회 5인 팀 프로젝트입니다.

방대한 광역 지방의회 회의록에서 주요 정책 이슈를 탐색하고, **제공 데이터 범위 내 최초 관측 시점·타 의회 후속 논의·향후 논의 전망을 원문 근거와 함께 제공하는 AI 분석 시스템**을 개발했습니다.

회의록 원문 외부 전송 제한을 고려해 상용 LLM API 대신 로컬 임베딩·로컬 LLM 중심으로 구성했습니다.

## My Role

- 한국어 임베딩 모델 비교·선정
- 대규모 회의록 임베딩
- FAISS 기반 의미 검색
- 검색·군집화 파이프라인 구축
- 로컬 LLM 후보 비교

개인 기여 범위와 관련 코드는 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)에 정리했습니다.

## Final Architecture

```text
Raw JSON / HTML
      ↓
Preprocessing / Metadata
      ↓
Parent–Child Chunking
      ↓
Snowflake Arctic Korean Embedding
      ↓
FAISS / ChromaDB / pgvector 비교
      ↓
PostgreSQL + pgvector
      ↓
Dense Retrieval + BM25
      ↓
RRF Fusion
      ↓
BGE Cross-Encoder Reranking
      ↓
Issue Clustering / Timeline / Forecast
      ↓
Evidence
      ↓
SKT A.X 4.0 Light 7B Local LLM Agent
      ↓
Streamlit
```

## Data & Scale

| Item | Value |
|---|---:|
| 대상 기간 | 2024–2025 |
| 팀 규모 | 5명 |
| Canonical rows | 2,057,214 |
| Child chunks | 901,358 |
| Embedding dimension | 1,024 |
| Vector DB | PostgreSQL 17 + pgvector 0.8.7 |
| Vector rows | 901,358 |
| DB size | 약 12 GB |

원본 회의록, 임베딩 파일, DB dump, 모델 파일은 공개 저장소에 포함하지 않습니다.

## Retrieval

초기에는 FAISS와 ChromaDB로 검색 구조를 검증하고, 최종 서비스에서는 메타데이터 필터·SQL 집계·웹 연동을 고려해 PostgreSQL + pgvector를 선택했습니다.

| Query | ChromaDB p50 | pgvector p50 |
|---|---:|---:|
| Region + Year filter | 586.29 ms | **3.97 ms** |
| No filter | **1.42 ms** | 2.75 ms |

최종 검색 구조:

```text
Query
 ↓
Arctic Korean Embedding
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

관련 코드:
- `src/15_load_chroma.py`
- `src/16_load_pgvector.py`
- `src/17_benchmark_vector_databases.py`
- `src/18_pgvector_retrieval.py`
- `src/19_rerank_candidates.py`
- `src/35_build_bm25_index.py`
- `src/bm25_runtime.py`
- `src/council_vector_runtime.py`

상세 비교는 [DATABASE_AND_RETRIEVAL_DECISION.md](docs/DATABASE_AND_RETRIEVAL_DECISION.md)에 정리했습니다.

## Issue Analysis

유사 발언을 정책 이슈 단위로 묶고, threshold sweep·중복 제거·Seed 기준·사람 검수를 통해 과병합을 제어했습니다.

후반 분석 파이프라인:

| Stage | File |
|---|---|
| 이슈 확장 / 군집 후보 | `src/20_expand_cluster_issue.py` |
| 이슈 타임라인 | `src/21_build_issue_timeline.py` |
| 예측 feature | `src/22_build_forecast_features.py` |
| 단일 이슈 예측 | `src/23_forecast_issue.py` |
| 웹 payload | `src/24_build_web_payload.py` |
| 이슈 후보 | `src/25_build_issue_candidates.py` |
| 후보 상태 확인 | `src/25a_show_candidate_status.py` |
| threshold sweep | `src/26_cluster_threshold_sweep.py` |
| LLM extraction 평가셋 | `src/27_make_llm_extraction_evalset.py` |
| 검수 이슈 확정 | `src/30_finalize_reviewed_issues.py` |
| 다중 이슈 panel | `src/31_build_multi_issue_panel.py` |
| 다중 이슈 forecast | `src/32_train_multi_issue_forecast.py` |
| 외부 후속 흐름 | `src/32a_external_followup_flow.py` |
| 자유 질의 분석 | `src/33_analyze_free_query.py` |
| 최종 웹 bundle | `src/34_build_final_web_bundle.py` |

28, 29 번호는 최종 소스에 별도 파일이 존재하지 않습니다.

## Local Agent / Web

검색·예측 결과와 회의록 근거를 **SKT A.X 4.0 Light 7B** 기반 로컬 LLM에 제공하고, LLM은 검색된 근거 범위 안에서 결과를 설명하도록 구성했습니다.

- Agent: `agent/local_issue_agent.py`
- Retrieval tools: `agent/tools.py`
- Streamlit: `demo/final_app.py`
- 실행: `scripts/run_final_app.sh`

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

docker compose up -d pgvector
bash scripts/run_final_app.sh
```

로컬 실행에는 공개 저장소에 포함하지 않은 임베딩·BM25 인덱스 등 데이터 산출물이 필요합니다.

## Repository Structure

```text
agent/          Local LLM Agent / retrieval tools
demo/           Streamlit web
docs/           Retrieval decision / troubleshooting
examples/       Public sample outputs
legacy/         Initial team pipeline snapshot
presentation/   Final presentation PDF
scripts/        Run / validation scripts
src/            Preprocessing / embedding / retrieval / analysis
CONTRIBUTIONS.md
docker-compose.yml
requirements.txt
```

## Presentation

[의정이슈레이더 최종 발표자료 PDF](presentation/의정이슈레이더_최종발표.pdf)

## Notes

- 최초 관측은 전국 최초 정책 발상지가 아니라 **제공 데이터 범위 내 최초 관측**입니다.
- 지역별 관측 순서는 정책 확산의 인과관계를 의미하지 않습니다.
- 성능 수치는 기록된 실험 조건에서의 결과입니다.
