# Contribution — Kwon Jun

## Why this document exists

이 프로젝트는 5인 팀 프로젝트이며, 현재 공개 저장소는 대회 종료 후 전체 코드를 정리해
**단일 snapshot commit**으로 공개한 형태입니다.

따라서 현재 Git history만으로 개인별 구현 범위를 정확히 추적하기 어렵습니다.
이 문서는 이력서·포트폴리오에서 언급한 제 담당 범위와 관련 코드 영역을 명확히 구분하기 위해 작성했습니다.

## My Primary Responsibilities

### 1. Embedding model selection

- 한국어 지방의회 회의록에 사용할 임베딩 모델 후보 비교
- 한국어 검색 품질과 실행 가능성을 기준으로 모델 검토
- 최종 파이프라인에서 `dragonkue/snowflake-arctic-embed-l-v2.0-ko` 사용

Related modules:
- `src/04_embedding.py`
- `src/04b_issue_embedding.py`
- `src/04d_embed_selected.py`
- `src/04e_semantic_text.py`

> 위 파일들은 최종 팀 통합 코드 기준 관련 영역이며, 현재 Git history는 파일별 개인 저작권을 증명하는 구조가 아닙니다.

### 2. Large-scale embedding experiment

- 전체 chunks: **460,431**
- embedding target: **438,708**
- precision: **bf16**
- sequence length: **2048**
- batch size: **8**
- environment: **RTX 4060 Laptop**
- throughput: **약 39.2 chunks/s**
- output embedding file: **약 898 MB**

실행 과정에서 batch size, precision, sequence length, GPU memory를 함께 고려했습니다.

### 3. FAISS semantic search

- embedding vector normalization 및 similarity search 구조 확인
- FAISS `IndexFlatIP` 기반 의미 검색 파이프라인 구축/검토
- 검색 결과와 metadata를 연결해 후속 군집화 입력으로 사용할 수 있도록 구성

Related modules:
- `src/05_build_faiss.py`
- `src/06_search_v2.py`
- `src/10_issue_query.py`

### 4. Clustering / retrieval flow

- 유사 발언 검색 결과에서 정책 이슈 후보를 묶는 흐름 검토
- 검색 → 군집화 → 이슈 라벨링 단계 연결
- Dense retrieval만으로 부족한 고유명사·정책명 검색을 보완하기 위한 BM25 / RRF 방식 검토

Related module:
- `src/07_clustering.py`

### 5. Local LLM candidate comparison

- 회의록 원문 외부 전송 금지 조건을 고려해 상용 API 대신 로컬 LLM 후보 비교
- LLM이 검색/예측 결과를 임의 생성하지 않고, 제공된 근거를 설명하는 역할로 제한
- 최종 시스템에서 Qwen 계열 로컬 모델을 활용한 설명/이슈명 제안 구조 검토

Related module:
- `src/08_llm_issue_labeling.py`

## Team Pipeline vs. My Contribution

| Pipeline | Team Project | My Main Contribution |
|---|:---:|:---:|
| Inventory / preprocessing | ✓ |  |
| Parent-child chunking | ✓ | 실험 검토 |
| Embedding model selection | ✓ | **Primary** |
| Large-scale embedding | ✓ | **Primary** |
| FAISS retrieval | ✓ | **Primary** |
| BM25 / RRF retrieval | ✓ | **Primary / experiment** |
| Clustering | ✓ | **Primary / experiment** |
| LLM issue labeling | ✓ | 후보 비교 및 구조 검토 |
| Timeline construction | ✓ |  |
| Forecasting | ✓ |  |
| Streamlit demo | ✓ |  |

## Evidence Policy

이 문서는 현재 공개 저장소의 commit history가 개인별 기여를 보여주지 못하는 한계를 보완하기 위한 설명 문서입니다.

앞으로의 팀 프로젝트에서는 다음을 지키는 것을 목표로 합니다.

- 개인 branch 사용
- 기능 단위 commit
- Pull Request 기반 통합
- README에서 담당 모듈 명시
- 실험 결과 및 재현 방법 기록
