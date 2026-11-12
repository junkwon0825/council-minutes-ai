# Council Minutes Policy Flow AI

광역 지방의회 회의록에서 정책 이슈를 검색하고, 제공 데이터 범위 내 최초 관측 시점과 지역별 후속 논의 흐름을 분석하며, 로컬 LLM 기반 근거 답변과 웹 시각화를 제공하는 AI 서비스입니다.

> 2026 DATA·AI 분석 경진대회 5인 팀 프로젝트입니다. 공개 저장소에는 원본 회의록 데이터와 실제 DB dump를 포함하지 않습니다.

## Project Summary

- **Problem**: 방대한 지방의회 회의록에서 유사 정책 이슈의 등장·확산·변화를 수작업으로 추적하기 어려움
- **Constraint**: 원문 외부 전송 제한 → 상용 LLM API 대신 로컬 embedding / local LLM 사용
- **Scale**: 최종 child chunk 약 **901,358건**, PostgreSQL DB 약 **12 GB**
- **Final Storage / Retrieval**: **PostgreSQL 17 + pgvector**
- **Retrieval**: pgvector dense search + PostgreSQL/BM25 lexical retrieval + RRF + reranking
- **Local LLM**: Ollama 기반 국내 공개 LLM 비교 후 서비스 모델 선정
- **Service**: Streamlit 웹 + Local Agent

## My Role

프로젝트에서 다음 영역을 중심으로 구현·실험했습니다.

- JSON/HTML 기반 회의록 **전처리·정제·파싱 및 metadata 구조화**
- 발언 단위 및 부모/자식 **chunking pipeline 구축·검증**
- 한국어 embedding model 비교 및 최종 모델 선정
- 대규모 embedding 실행 및 성능 측정
- **ChromaDB / PostgreSQL+pgvector / FAISS 기반 검색 구조 실험**
- ChromaDB와 pgvector의 필터 검색·적재·운영 구조 비교
- **PostgreSQL+pgvector 최종 저장/검색 구조 적용**
- dense retrieval + BM25 + RRF 기반 hybrid retrieval 실험
- clustering / issue discovery / local LLM 후보 비교
- 최종 웹·Agent 실행 경로 통합 및 결과 검증

개인 기여 범위와 코드 provenance는 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)에 별도로 정리했습니다.

## Final Architecture

```text
Raw JSON / HTML
      ↓
Preprocessing / Parsing
      ↓
Canonical Data
      ↓
Parent-Child Chunking
      ↓
Snowflake Arctic Korean Embedding
      ↓
┌────────────────────────────────────┐
│ Storage / Retrieval Experiments    │
│                                    │
│  ChromaDB                          │
│  FAISS                             │
│  PostgreSQL + pgvector  ← FINAL    │
└────────────────────────────────────┘
      ↓
pgvector Dense Retrieval
      +
BM25 / Lexical Retrieval
      ↓
RRF Fusion
      ↓
BGE Cross-Encoder Reranking
      ↓
Issue Expansion / Clustering
      ↓
Timeline / Forecast
      ↓
Local LLM Agent
      ↓
Streamlit Web
```

## Why PostgreSQL + pgvector?

ChromaDB와 PostgreSQL+pgvector를 실제 적재·검색 환경에서 비교했습니다.

| Test | ChromaDB p50 | pgvector p50 |
|---|---:|---:|
| Region + Year Filter | 586.29 ms | **3.97 ms** |
| No Filter | **1.42 ms** | 2.75 ms |

무필터 단순 벡터 검색에서는 ChromaDB가 빠른 경우가 있었지만, 최종 서비스에는 다음 요구사항이 있었습니다.

- 지역·연도·위원회 등 metadata filter
- SQL 기반 통계 집계
- 웹 서비스의 관계형 metadata 관리
- vector와 structured data의 통합 운영

따라서 모든 조건에서 더 빠르다는 의미가 아니라, **서비스 운영 요구사항과 필터 검색 성능을 함께 고려해 PostgreSQL+pgvector를 최종 선택**했습니다.

## Final Retrieval Pipeline

```text
User Query
   ↓
Query Embedding
   ↓
pgvector HNSW Dense Search
   +
BM25 / PostgreSQL Lexical Search
   ↓
Reciprocal Rank Fusion (RRF)
   ↓
BGE Cross-Encoder Reranking
   ↓
Exact-text Deduplication
   ↓
Parent Context Recovery
   ↓
Evidence
```

주요 코드:

| Stage | File |
|---|---|
| ChromaDB Load | `src/15_load_chroma.py` |
| pgvector Load | `src/16_load_pgvector.py` |
| Vector DB Benchmark | `src/17_benchmark_vector_databases.py` |
| pgvector Retrieval | `src/18_pgvector_retrieval.py` |
| Cross-Encoder Rerank | `src/19_rerank_candidates.py` |
| BM25 Index | `src/35_build_bm25_index.py` |
| Vector Runtime | `src/council_vector_runtime.py` |
| BM25 Runtime | `src/bm25_runtime.py` |

## Data / Embedding Scale

| Item | Result |
|---|---:|
| Canonical rows | 2,057,214 |
| Child chunks | 901,358 |
| Embedding dimension | 1,024 |
| DB | PostgreSQL 17 + pgvector |
| pgvector version | 0.8.7 |
| Final DB size | 약 12 GB |
| Vector table rows | 901,358 |

초기 실험 단계에서는 460,431 chunks 중 438,708건을 대상으로 별도 embedding 실험을 수행했으며, RTX 4060 Laptop 환경에서 약 39.2 chunks/s 처리량을 기록했습니다.

## Local LLM / Agent

회의록 원문을 외부 상용 LLM API로 보내지 않는 조건을 지키기 위해 로컬 Ollama 기반 모델들을 비교했습니다.

Agent는 검색 및 계산 결과를 생성하는 모델이 아니라, **검증된 retrieval 결과와 통계 결과를 설명하는 계층**으로 제한했습니다.

관련 코드:

- `agent/query_pipeline.py`
- `agent/local_issue_agent.py`
- `agent/tools.py`
- `agent/followup_tools.py`

## Web Service

최종 웹은 검색 결과, 정책 이슈 타임라인, 지역별 관측 흐름, 예측 결과, 원문 근거와 Agent 답변을 연결합니다.

관련 코드:

- `demo/final_app.py`
- `src/24_build_web_payload.py`
- `src/34_build_final_web_bundle.py`
- `scripts/run_final_app.sh`
- `docker-compose.yml`

## Forecast Evaluation

기존 최종 평가 기록:

| Metric | Result |
|---|---:|
| Discussion continuation F1 | 0.8814 |
| Mention count MAE | 7.1224 |
| Mention count RMSE | 10.0425 |
| Direction accuracy | 76% |
| MAE improvement vs previous-month baseline | 29.83% |

위 수치는 별도 forecast evaluation 산출물의 기록이며, 최종 웹의 모든 화면이 동일한 예측 모델을 실시간 재학습한다는 의미는 아닙니다.

## Repository Structure

```text
agent/          Local Agent / Tool pipeline
config/         Retrieval / experiment configuration
demo/           Streamlit web applications
docs/           Architecture / benchmark / runbook
eval/           Retrieval evaluation queries
results/        Evaluation metrics
scripts/        DB export/restore / execution scripts
src/            Final data / retrieval / analysis pipeline
legacy/         Early team pipeline snapshot
CONTRIBUTIONS.md
docker-compose.yml
requirements-*.txt
```

## Legacy Pipeline

초기 개발 단계에서는 별도 팀 저장소에서 ChromaDB 기반 적재·검색과 전처리 실험을 진행했습니다.
해당 코드는 개발 과정의 provenance를 보존하기 위해
`legacy/initial_team_pipeline/`에 정리했습니다.

현재 포트폴리오의 기준은 **최종 PostgreSQL+pgvector 기반 파이프라인**입니다.

## Data Policy

다음 파일은 공개 저장소에 포함하지 않습니다.

- 대회 원본 회의록
- canonical / chunk parquet
- embedding NPY
- PostgreSQL dump
- ChromaDB persistent files
- BM25 index binary
- model weights
- API keys / .env

## Documentation

- [개인 기여 범위](CONTRIBUTIONS.md)
- `docs/ACTIVE_PIPELINE.md`
- `docs/BENCHMARK_PROTOCOL.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/POSTGRES_DB_TEAM_SHARE.md`
- `docs/HYBRID_BM25_RUNBOOK.md`
- `docs/REPRESENTATIVE_ISSUES_AND_AGENT_RUNBOOK.md`

## Notes

- 최초 관측은 전국 최초 정책 발상지가 아니라 **제공 데이터 범위 내 최초 관측**입니다.
- 지역 간 시간 순서는 정책 전달의 인과관계를 의미하지 않습니다.
- LLM은 수치 예측 모델을 대신하지 않습니다.
- 검색·예측·Agent 출력은 가능한 경우 원문 근거와 함께 검증합니다.
