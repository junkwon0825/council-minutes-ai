# Contribution — Kwon Jun

## Why this document exists

이 프로젝트는 5인 팀 프로젝트이며, 개발 과정에서 팀 저장소와 개인 로컬 작업본을 함께 사용했습니다.
현재 공개 저장소의 최종 파이프라인은 대회 종료 후 **단일 snapshot commit**으로 정리되어 있어,
Git history만으로 개인별 구현 범위를 정확히 확인하기 어렵습니다.

이를 보완하기 위해 초기 팀 저장소의 코드를
`legacy/initial_team_pipeline/`에 보존하고, 제가 실제로 담당한 범위를 아래와 같이 정리합니다.

> 주의: 초기 팀 저장소의 commit author와 실제 코드 작성자가 항상 일치하지 않았습니다.
> 따라서 기존 commit history를 개인 기여 증거로 재작성하지 않고, 원본 이력을 유지한 상태에서 담당 범위를 문서화합니다.

## My Primary Responsibilities

### 1. Data preprocessing / parsing

- 지방의회 원본 JSON 내부 HTML에서 안건·화자·발언을 추출하는 전처리 흐름 구현 및 실험
- CSS selector 기반 파싱 규칙 생성·정리
- 텍스트 정규화, 의전성 문장 필터링, TSV 변환
- 회의록별 메타데이터 구조화

Legacy related modules:
- `legacy/initial_team_pipeline/scripts/1_split_sample.py`
- `legacy/initial_team_pipeline/scripts/2_rules_make.py`
- `legacy/initial_team_pipeline/scripts/3_rules_summary.py`
- `legacy/initial_team_pipeline/scripts/4_to_csv.py`
- `legacy/initial_team_pipeline/scripts/4_to_csv_2.py`
- `legacy/initial_team_pipeline/src/parser.py`

Final pipeline related modules:
- `src/01_inventory.py`
- `src/02_preprocess.py`
- `src/03_make_chunks.py`

### 2. Chunking / metadata structure

- 발언 단위 문맥을 유지하면서 긴 발언을 분리하는 청킹 구조 실험
- date, assembly, speaker, agenda, source row 등 검색용 metadata 구조 정리
- 초기 ChromaDB 적재를 위한 chunk / metadata 연결

Legacy related modules:
- `legacy/initial_team_pipeline/src/chunker.py`
- `legacy/initial_team_pipeline/run_ingest.py`

### 3. ChromaDB ingestion / search experiment

- LangChain + HuggingFaceEmbeddings 기반 ChromaDB 적재 파이프라인 실험
- batch 단위 vector store 적재
- embedding model preset 및 collection 분리 구조 구성
- natural language semantic search + metadata filter 확인
- embedding dimension mismatch 등 실행 오류 처리

Legacy related modules:
- `legacy/initial_team_pipeline/src/config.py`
- `legacy/initial_team_pipeline/src/indexer.py`
- `legacy/initial_team_pipeline/run_ingest.py`
- `legacy/initial_team_pipeline/query_search.py`

> PostgreSQL/pgvector 비교 실험은 로컬 작업 파일을 추가 확인한 뒤 별도 문서에 반영할 예정입니다.

### 4. Embedding model selection

- 한국어 지방의회 회의록에 사용할 임베딩 모델 후보 비교
- KURE, Ko-SBERT, KoSimCSE, BGE 계열 등 후보를 검토
- 최종 파이프라인에서 `dragonkue/snowflake-arctic-embed-l-v2.0-ko` 사용

Related modules:
- `src/04_embedding.py`
- `src/04b_issue_embedding.py`
- `src/04d_embed_selected.py`
- `src/04e_semantic_text.py`

### 5. Large-scale embedding experiment

- 전체 chunks: **460,431**
- embedding target: **438,708**
- precision: **bf16**
- sequence length: **2048**
- batch size: **8**
- environment: **RTX 4060 Laptop**
- throughput: **약 39.2 chunks/s**
- output embedding file: **약 898 MB**

실행 과정에서 batch size, precision, sequence length, GPU memory를 함께 고려했습니다.

### 6. FAISS semantic search

- embedding vector normalization 및 similarity search 구조 확인
- FAISS `IndexFlatIP` 기반 의미 검색 파이프라인 구축/검토
- 검색 결과와 metadata를 연결해 후속 군집화 입력으로 구성
- Dense retrieval만으로 부족한 고유명사·정책명을 보완하기 위해 BM25/RRF 검색 구조 실험

Related modules:
- `src/05_build_faiss.py`
- `src/06_search_v2.py`
- `src/10_issue_query.py`

### 7. Clustering / issue discovery

- 유사 발언 검색 결과에서 정책 이슈 후보를 묶는 흐름 실험
- 검색 → 군집화 → 이슈 라벨링 단계 연결
- 임베딩/검색 결과를 기반으로 정책 이슈 후보를 검토

Related module:
- `src/07_clustering.py`

### 8. Local LLM experiment

- 원문 외부 전송 금지 조건을 고려해 상용 API 대신 로컬 LLM 후보 비교
- 초기 단계에서 로컬 LLM을 이용한 정책 category / core issue / keyword / summary 추출 실험
- 최종 단계에서는 LLM이 검색·예측 결과를 임의 생성하지 않고 제공된 근거를 설명하는 역할로 제한

Legacy related modules:
- `legacy/initial_team_pipeline/extract_llm_metadata.py`
- `legacy/initial_team_pipeline/scripts/5_create_master_csv.py`

Final pipeline related module:
- `src/08_llm_issue_labeling.py`

## Team Pipeline vs. My Contribution

| Pipeline | Team Project | My Main Contribution |
|---|:---:|:---:|
| Raw data inspection | ✓ | **Primary** |
| HTML / text preprocessing | ✓ | **Primary** |
| Parsing / metadata normalization | ✓ | **Primary** |
| Chunking | ✓ | **Primary / experiment** |
| ChromaDB ingestion / search | 초기 실험 | **Primary** |
| Embedding model selection | ✓ | **Primary** |
| Large-scale embedding | ✓ | **Primary** |
| FAISS retrieval | ✓ | **Primary** |
| BM25 / RRF retrieval | ✓ | **Primary / experiment** |
| Clustering | ✓ | **Primary / experiment** |
| Local LLM experiment | ✓ | **Primary / candidate comparison** |
| Timeline construction | ✓ | 일부 참여 / 최종 통합 |
| Forecasting | ✓ | 팀 파이프라인 |
| Streamlit demo | ✓ | 별도 로컬 구현 경험, 최종 공개본은 팀 통합 |

## Evidence / provenance

- `legacy/initial_team_pipeline/`은 초기 팀 저장소의 공개 코드 스냅샷입니다.
- `src/`는 대회 종료 후 정리된 최종 통합 파이프라인입니다.
- 기존 팀 저장소의 commit author가 개인별 실제 작성 범위를 완전히 반영하지 않아 commit history를 수정하지 않았습니다.
- 로컬에 남아 있는 개인 작업 파일, 발표 자료, 실험 결과를 추가 확인해 증거 자료를 계속 보강할 예정입니다.

## Going forward

이후 팀 프로젝트에서는 다음 원칙을 적용합니다.

- 개인 branch 사용
- 기능 단위 commit
- Pull Request 기반 통합
- README에서 담당 모듈 명시
- 실험 결과 및 재현 방법 기록
