# Contribution — Kwon Jun

## Project Context

2026 DATA·AI 분석 경진대회 5인 팀 프로젝트입니다.

개발 과정에서는 팀 저장소, 개인 로컬 작업본, 최종 제출용 통합 소스를 함께 사용했습니다.
기존 팀 저장소의 commit author만으로 실제 구현 범위를 정확히 확인하기 어려웠기 때문에,
현재 포트폴리오 저장소에서는 **담당 영역·관련 코드·실험 결과를 함께 표시**합니다.

기존 Git history를 개인 커밋처럼 재작성하지 않습니다.

---

## My Main Responsibilities

### 1. Data Preprocessing / Parsing

- 지역별 JSON 내부 HTML 구조 확인 및 본문 추출
- 안건·화자·발언 단위 파싱
- 불필요한 HTML / 공백 / 중복 데이터 정리
- 날짜·의회·회의명·안건·발언자 metadata 연결
- canonical 데이터 생성을 위한 입력 검증

관련 최종 코드:

- `src/01_restart_inventory.py`
- `src/02_probe_received_metadata.py`
- `src/03_validate_row_id_join.py`
- `src/04_build_direct_canonical.py`

초기 코드:

- `legacy/initial_team_pipeline/scripts/`
- `legacy/initial_team_pipeline/src/parser.py`

### 2. Parent-Child Chunking

- 작은 child chunk는 검색 단위로 사용
- 큰 parent chunk는 검색 결과의 주변 문맥 복원에 사용
- chunk ID / parent ID / source metadata 정합성 검증

관련 코드:

- `src/05_build_parent_child_chunks.py`
- `src/05b_validate_chunks.py`

최종 기록 기준:

- canonical rows: **2,057,214**
- child chunks: **901,358**

### 3. Embedding Model Selection / Large-scale Embedding

한국어 지방의회 문서 검색에 사용할 embedding model을 비교했습니다.

검토 모델에는 Arctic Korean, KURE 등 한국어/다국어 embedding 후보가 포함됐으며,
최종 파이프라인에서는:

`dragonkue/snowflake-arctic-embed-l-v2.0-ko`

를 사용했습니다.

관련 코드:

- `src/06_make_embedding_eval_sample.py`
- `src/07_embed_eval_sample.py`
- `src/08_embed_full_corpus.py`
- `src/10_compare_embedding_variants.py`
- `src/11_build_embedding_model_pool.py`
- `src/12_compare_embedding_models.py`

초기 대규모 실험:

- 460,431 chunks
- embedding 대상 438,708 chunks
- bf16
- sequence length 2048
- batch size 8
- RTX 4060 Laptop
- 약 **39.2 chunks/s**

### 4. Vector Database Experiment — ChromaDB vs PostgreSQL+pgvector

초기 ChromaDB pipeline을 구성한 뒤 PostgreSQL+pgvector와 실제 적재·검색 환경에서 비교했습니다.

관련 코드:

- `src/15_load_chroma.py`
- `src/16_load_pgvector.py`
- `src/17_benchmark_vector_databases.py`

기록된 p50 latency:

| Condition | ChromaDB | pgvector |
|---|---:|---:|
| Region + Year filter | 586.29 ms | **3.97 ms** |
| No filter | **1.42 ms** | 2.75 ms |

단순 무필터 벡터 검색 성능만으로 DB를 선택하지 않았습니다.

최종 서비스에는:

- 지역 / 연도 / 위원회 metadata filter
- SQL 기반 통계 집계
- 관계형 metadata 관리
- 웹 서비스 연동

이 필요했기 때문에 **PostgreSQL + pgvector**를 최종 저장·검색 구조로 선택했습니다.

최종 DB 기록:

- PostgreSQL 17
- pgvector 0.8.7
- table: `council_arctic_raw_v1`
- rows: **901,358**
- DB size: 약 **12 GB**

### 5. Retrieval Pipeline

최종 retrieval은 단순 vector similarity search가 아니라
dense + lexical 검색을 결합하는 구조로 확장했습니다.

```text
Query
 ↓
Embedding
 ↓
pgvector HNSW dense retrieval
 +
BM25 / lexical retrieval
 ↓
RRF
 ↓
BGE Cross-Encoder reranking
 ↓
Deduplication
 ↓
Parent context recovery
 ↓
Evidence
```

관련 코드:

- `src/18_pgvector_retrieval.py`
- `src/19_rerank_candidates.py`
- `src/35_build_bm25_index.py`
- `src/bm25_runtime.py`
- `src/council_vector_runtime.py`

### 6. Issue Discovery / Clustering

검색한 유사 발언을 seed로 이슈 후보를 확장하고,
cosine similarity graph와 Union-Find를 이용해 연결 요소를 구성했습니다.

관련 코드:

- `src/20_expand_cluster_issue.py`
- `src/25_build_issue_candidates.py`
- `src/26_cluster_threshold_sweep.py`
- `src/30_finalize_reviewed_issues.py`

### 7. Timeline / Forecast

이슈 구성원에 대해:

- 최초 관측
- 의회별 후속 관측
- 월별 활동 의회 수
- 월별 언급량

을 생성하고 다음 달 흐름을 평가했습니다.

관련 코드:

- `src/21_build_issue_timeline.py`
- `src/22_build_forecast_features.py`
- `src/31_build_multi_issue_panel.py`
- `src/32_train_multi_issue_forecast.py`

기록된 평가 결과:

- discussion continuation F1: **0.8814**
- mention count MAE: **7.1224**
- RMSE: **10.0425**
- direction accuracy: **76%**

### 8. Local LLM / Agent

원문 외부 전송 제한을 고려해 local LLM을 비교·적용했습니다.

검색 결과와 계산 결과를 LLM이 새로 만들어내는 구조가 아니라,
검증된 결과를 설명하고 원문 근거를 연결하는 역할로 제한했습니다.

관련 코드:

- `agent/query_pipeline.py`
- `agent/local_issue_agent.py`
- `agent/tools.py`
- `agent/followup_tools.py`

### 9. Web / Final Integration

검색·타임라인·전망·원문 근거·Agent 답변을 하나의 웹 실행 경로로 연결하고
실행 결과를 점검했습니다.

관련 코드:

- `demo/final_app.py`
- `src/24_build_web_payload.py`
- `src/34_build_final_web_bundle.py`
- `scripts/run_final_app.sh`
- `docker-compose.yml`

---

## Development Provenance

### Final source

최종 제출 원본 소스에는 다음 구조가 존재합니다.

```text
agent/
config/
demo/
docs/
eval/
results/
scripts/
src/
docker-compose.yml
requirements-*.txt
```

### Legacy source

초기 ChromaDB / preprocessing 실험 코드는
`legacy/initial_team_pipeline/`에 별도로 보존했습니다.

### Why the original team commits look different

기존 팀 저장소에서는 한 팀원이 통합하여 commit / push한 코드가 많아
commit author만으로 실제 작성자를 판단하기 어렵습니다.

따라서 이 저장소에서는 과거 commit author를 변경하지 않고,
실제 담당 영역과 관련 코드·실험 결과를 명시하는 방식으로 기여도를 설명합니다.

앞으로의 팀 프로젝트에서는 개인 branch, 기능 단위 commit, Pull Request를 사용합니다.
