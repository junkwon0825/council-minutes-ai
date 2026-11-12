# Contribution — Kwon Jun

2026 DATA·AI 분석 경진대회 5인 팀 프로젝트에서 제가 담당한 기술 범위와 공개 코드의 연결을 정리한 문서입니다.

## 1. Embedding

**담당**
- 한국어 임베딩 모델 후보 비교 및 선정
- 대규모 회의록 임베딩 실행
- 검색·군집화에 사용할 벡터 표현 검증

**관련 코드**
- `src/04_embedding.py`
- `src/04b_issue_embedding.py`
- `src/04d_embed_selected.py`
- `src/04e_semantic_text.py`

**실행 기록**
- 초기 chunks: 460,431
- embedding 대상: 438,708
- embedding dimension: 1,024
- RTX 4060 Laptop
- 약 39.2 chunks/s

## 2. Semantic Search

**담당**
- FAISS 기반 의미 검색 구축
- query embedding과 검색 결과 검증
- 이후 최종 검색 구조로 확장 가능한 retrieval 흐름 구성

**관련 코드**
- `src/05_build_faiss.py`
- `src/06_search_v2.py`

최종 팀 시스템은 ChromaDB와 PostgreSQL+pgvector를 추가 비교한 뒤 PostgreSQL+pgvector 기반 구조를 적용했습니다.

## 3. Search / Clustering Pipeline

**담당**
- 검색 결과 기반 유사 발언 탐색
- 임베딩 유사도를 이용한 이슈 후보 군집화
- threshold에 따른 과병합 문제 분석
- Seed 기준·중복 제거·사람 검수 흐름 적용

**관련 코드**
- `src/07_clustering.py`
- `src/20_expand_cluster_issue.py`
- `src/25_build_issue_candidates.py`
- `src/25a_show_candidate_status.py`
- `src/26_cluster_threshold_sweep.py`
- `src/30_finalize_reviewed_issues.py`

## 4. Local LLM Comparison

**담당**
- 외부 상용 LLM API를 사용하지 않는 로컬 LLM 후보 비교
- 검색·군집 결과를 근거로 설명하는 LLM 역할 검토
- LLM이 이슈를 임의 생성하기보다 근거를 설명하도록 역할 제한

**관련 코드**
- `src/08_llm_issue_labeling.py`
- `src/27_make_llm_extraction_evalset.py`
- `agent/local_issue_agent.py`

최종 공개 실행 구조는 Ollama 기반 **SKT A.X 4.0 Light 7B**를 기본 모델로 사용합니다.

## 5. Team Final System

팀 최종 시스템에는 다음 구성도 포함됩니다.

- JSON/HTML 정제 및 메타데이터 표준화
- 부모·자식 청킹
- PostgreSQL + pgvector Dense Retrieval
- BM25 + RRF
- BGE Cross-Encoder reranking
- 이슈 타임라인 및 향후 논의 예측
- Streamlit 웹 화면

관련 구현은 `src/`, `agent/`, `demo/`에 공개되어 있습니다.

## Provenance

개발 당시 팀 저장소의 일부 코드는 통합 커밋 형태로 관리되어 Git commit author만으로 개인별 작업 범위를 구분하기 어렵습니다. 과거 커밋을 개인 커밋처럼 수정하지 않고, 이 문서에서 실제 담당 범위와 관련 코드 위치를 명시했습니다.

초기 팀 파이프라인은 `legacy/initial_team_pipeline/`에 보존되어 있습니다.
