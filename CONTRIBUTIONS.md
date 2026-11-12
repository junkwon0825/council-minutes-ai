# Contribution — Kwon Jun

이 문서는 5인 팀 프로젝트에서 제가 담당한 기술 범위를 공개 코드와 연결해 설명하기 위한 문서입니다.

## 1. Data Preprocessing

**주요 역할**
- JSON 내부 HTML 구조 확인 및 본문 정제
- 회의일·의회·안건·발언자 metadata 구조화
- 중복/공백/HTML 제거 및 데이터 품질 점검

**관련 코드**
- `src/01_inventory.py`
- `src/02_preprocess.py`
- `legacy/initial_team_pipeline/scripts/`
- `legacy/initial_team_pipeline/src/parser.py`

## 2. Chunking / Embedding

**주요 역할**
- 검색 단위 청킹 설계 및 실험
- 한국어 임베딩 모델 후보 비교
- 대규모 embedding 실행 및 실행조건 검증

**관련 코드**
- `src/03_make_chunks.py`
- `src/04_embedding.py`
- `src/04b_issue_embedding.py`
- `src/04d_embed_selected.py`

초기 대규모 실험 기록:
- 전체 chunks: 460,431
- embedding 대상: 438,708
- bf16 / sequence length 2048 / batch 8
- RTX 4060 Laptop
- 약 39.2 chunks/s

## 3. Vector Search / Database Experiment

**주요 역할**
- FAISS 기반 semantic search 구축
- ChromaDB 적재 및 검색 실험
- PostgreSQL+pgvector 적재/검색 구조 구현
- 동일 query/vector 조건의 DB benchmark
- 서비스 요구사항을 고려해 pgvector 최종 선택

**관련 코드**
- `src/05_build_faiss.py`
- `src/06_search_v2.py`
- `src/15_load_chroma.py`
- `src/16_load_pgvector.py`
- `src/17_benchmark_vector_databases.py`

최종 PostgreSQL 기록:
- PostgreSQL 17
- pgvector 0.8.7
- vector rows: 901,358
- DB size: 약 12 GB

## 4. Hybrid Retrieval

**주요 역할**
- pgvector dense retrieval
- BM25 lexical retrieval
- RRF rank fusion
- Cross-Encoder reranking
- metadata filter 및 검색 결과 중복 제어

**관련 코드**
- `src/18_pgvector_retrieval.py`
- `src/19_rerank_candidates.py`
- `src/35_build_bm25_index.py`
- `src/bm25_runtime.py`
- `src/search_aliases.py`
- `src/council_vector_runtime.py`

## 5. Issue Discovery

**주요 역할**
- 검색 결과 기반 유사 발언 군집화
- 정책 이슈 후보 탐색
- 검색 → 군집화 → LLM 이슈명 제안 흐름 연결

**관련 코드**
- `src/07_clustering.py`
- `src/08_llm_issue_labeling.py`
- `src/09_build_issue_timeline.py`

## 6. Local LLM / Web Integration

**주요 역할**
- 외부 상용 LLM 대신 로컬 모델 기반 답변 구조 검토
- retrieval 결과를 근거로 답변하도록 Agent 구조 연결
- 최종 공개 포트폴리오용 Streamlit 실행 경로 구성

**관련 코드**
- `agent/tools.py`
- `agent/local_issue_agent.py`
- `demo/final_app.py`
- `scripts/run_final_app.sh`

## Provenance

개발 당시 팀 저장소에서 일부 코드는 한 번에 통합되어 Git commit author만으로 개인별 작업 범위를 확인하기 어려웠습니다. 과거 commit을 개인 커밋처럼 수정하지 않고, 현재 저장소에서는 **담당 영역과 관련 코드 위치를 명시**하는 방식으로 기여도를 설명합니다.

초기 팀 파이프라인은 `legacy/initial_team_pipeline/`에 보존되어 있으며, 현재 포트폴리오의 대표 구현은 PostgreSQL+pgvector 기반 retrieval 구조입니다.
