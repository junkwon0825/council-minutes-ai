# Vector Database & Retrieval Decision

## Goal

지방의회 회의록 검색 서비스에 적합한 vector storage / retrieval 구조를 비교했습니다.

평가 기준은 단순 벡터 검색 latency 하나가 아니라 다음을 포함했습니다.

- 대규모 vector 적재
- metadata filter
- SQL 통계 집계
- structured metadata 관리
- 웹 서비스 연동
- 운영 및 재현 가능성

## Compared Approaches

### ChromaDB

초기 prototype 및 검색 실험에 사용했습니다.

관련 코드:

- `src/15_load_chroma.py`
- `legacy/initial_team_pipeline/src/indexer.py`

### PostgreSQL + pgvector

최종 서비스 저장 및 dense retrieval backend로 선택했습니다.

관련 코드:

- `src/16_load_pgvector.py`
- `src/18_pgvector_retrieval.py`

환경 기록:

- PostgreSQL 17
- pgvector 0.8.7
- Docker image: `pgvector/pgvector:0.8.7-pg17`
- table: `council_arctic_raw_v1`
- rows: 901,358
- database size: 약 12 GB

### FAISS

Embedding 검색 및 검색 구조 실험 단계에서 사용했습니다.

FAISS는 빠른 in-memory vector search에 유용했지만,
최종 웹 서비스에서는 관계형 metadata와 SQL 집계가 함께 필요했습니다.

## Benchmark Record

기존 실험 기록의 p50 latency:

| Query Condition | ChromaDB | PostgreSQL + pgvector |
|---|---:|---:|
| Region + Year filter | 586.29 ms | **3.97 ms** |
| No metadata filter | **1.42 ms** | 2.75 ms |

이 결과를 모든 하드웨어·모든 쿼리에 일반화하지 않습니다.

무필터 검색에서는 ChromaDB가 더 빠른 기록도 있었지만,
metadata filter가 포함된 서비스 쿼리와 SQL 집계·운영 요구사항을 함께 고려하여
PostgreSQL+pgvector를 최종 선택했습니다.

## Final Retrieval

```text
User Query
    ↓
Snowflake Arctic Query Embedding
    ↓
PostgreSQL + pgvector HNSW
    +
BM25 / lexical retrieval
    ↓
RRF
    ↓
BGE Cross-Encoder
    ↓
Deduplication / Parent Diversity
    ↓
Parent Context Recovery
    ↓
Evidence
```

관련 코드:

- `src/18_pgvector_retrieval.py`
- `src/19_rerank_candidates.py`
- `src/35_build_bm25_index.py`
- `src/bm25_runtime.py`

## Decision

최종 선택은:

> **PostgreSQL + pgvector as the primary service database**

이유:

1. vector와 relational metadata를 한 DB에서 관리
2. 지역·연도 등 metadata filter
3. SQL 기반 월별/지역별 통계 집계
4. 웹 서비스와 DB 연결 구조 단순화
5. pgvector HNSW를 이용한 dense retrieval
6. lexical retrieval / hybrid retrieval과 통합 가능

이는 “pgvector가 모든 조건에서 ChromaDB나 FAISS보다 빠르다”는 주장이 아니라,
**이 프로젝트의 실제 서비스 요구사항에 가장 적합한 구조를 선택했다는 의미**입니다.
