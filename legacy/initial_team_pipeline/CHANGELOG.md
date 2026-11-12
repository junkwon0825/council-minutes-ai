# 📜 프로젝트 개발 및 변경 이력 (Changelog)

본 문서는 `develop_kb` 브랜치에서 진행된 지방의회 회의록 데이터 처리, 벡터 색인, 시맨틱 검색 파이프라인의 전체 구축 및 개선 내역을 기록합니다.

---

## [2026-10-19] `develop_kb` 추가 기능 및 고도화 내역

### 1. 로컬 LLM 정책 메타데이터 추출기 (`extract_llm_metadata.py`) 신설
- **Ollama 기반 구조화 메타데이터 추출**:
  - `gemma4:e2b` (기본) 및 `gemma4:e4b` 등 로컬 경량 모델을 활용하여 발언 본문에서 **정책 분류(`policy_category`), 핵심 쟁점(`core_issue`), 복합 정책 키워드(`keywords`), 요약(`summary`)**을 순수 JSON 형식으로 추출.
  - 엄격한 Negative Constraints 시스템 프롬프트를 적용하여 포괄적 1단어(예: '사업', '예산', '조례')를 배제하고 실질적인 정책 의제 복합키워드(예: '농지제도 혁신', '미래 인구정책')만 추출하도록 강제.
- **체크포인트(이어하기) & 실시간 Append 지원**:
  - 이미 처리된 행 번호(`row_id`)를 자동 감지하여 중복 호출을 방지하고 건너뜀(Skip).
  - 1건 처리 시마다 실시간으로 파일에 `write` 및 `flush`하여 중단되어도 데이터 유실 방지.
- **입력 파일명 기반 자동 출력 경로 결정**:
  - `--output` 옵션 미지정 시 입력 텍스트 파일명에 맞춰 `data/{입력파일명}.jsonl`로 파일별 자동 분리 저장.
- **멀티스레드 병렬 가속 및 `Ctrl + C` 즉시 안전 중단 핸들러**:
  - `--workers` 파라미터(기본 2, 최대 4~6)로 동시 추론 병렬 처리 지원.
  - `KeyboardInterrupt` 감지 시 큐에 대기 중인 작업을 즉시 취소(`cancel_futures=True`)하고 스레드 풀을 대기 없이 종료(`wait=False`)하여 즉각적인 안전 종료 보장.

### 2. ChromaDB 적재 파이프라인 메타데이터 자동 병합 고도화 (`run_ingest.py`)
- **다중 메타데이터 파일 일괄 감지 및 병합**:
  - `data/*.jsonl`에 존재하는 모든 의회별 정책 메타데이터 파일을 자동으로 감지하여 `(assembly, row_id)` 기반으로 ChromaDB 벡터 청크에 통합 색인.

### 3. 고속 핵심 키워드 추출 및 워드클라우드 생성 도구 (`make_wordcloud.py`)
- `Kiwi` 형태소 분석기 기반 정책 명사 추출 및 불용어 제거를 통한 고해상도 워드클라우드 생성 도구 구현.

---

## [2026-10-16] `develop_kb` 주요 작업 내역 요약

### 1. 프로젝트 아키텍처 모듈화 (`src/`)
- **설정 및 환경 관리 ([`src/config.py`](file:///c:/study/Projects/team-ggeodeureok/src/config.py))**:
  - 프로젝트 기준 경로 및 TSV 데이터 경로(`text/` 또는 `data/text`) 자동 감지 로직 구현.
  - 장문 발언 서브청킹 크기(`CHUNK_SIZE=800`), 오버랩(`CHUNK_OVERLAP=100`), 초단문 의전 필터링(`MIN_CHAR_LEN=40`) 하이퍼파라미터 정의.
  - 추천 임베딩 모델 프리셋 딕셔너리(`EMBEDDING_PRESETS`) 및 별칭 자동 해석 헬퍼(`resolve_embedding_model`) 구현.
- **회의록 TSV 데이터 파서 ([`src/parser.py`](file:///c:/study/Projects/team-ggeodeureok/src/parser.py))**:
  - `clean_text`: 비표준 공백(`\xa0`), 중복 공백, 불필요한 제어 문자 정제.
  - `parse_speaker`: 직함(`PREFIX_POSITIONS`, `SUFFIX_POSITIONS`)과 순수 성명을 규칙 기반으로 자동 분리 (예: `'위원장이상정'` → `('위원장', '이상정')`).
  - `parse_date`: `YYYYMMDD` 문자열을 정수형 `year`, `month`, `day` 및 ISO 형식 날짜로 분해하여 메타데이터 필터링 지원.
  - `parse_tsv_row`: 7개 TSV 컬럼을 구조화된 딕셔너리로 변환.
- **발언 단위 청킹 모듈 ([`src/chunker.py`](file:///c:/study/Projects/team-ggeodeureok/src/chunker.py))**:
  - 회의록 문맥을 유지하기 위해 1차적으로 발언 단위(Utterance)를 기본 청크로 보존.
  - 800자 초과 장문에 한해 `RecursiveCharacterTextSplitter`를 적용하여 문맥 유실 방지.
  - 40자 미만의 단순 진행/의전성 멘트 자동 필터링.
  - 고유 청크 식별자(`{region}_{date}_r{row_id}_c{chunk_idx}`) 부여.
- **ChromaDB 및 임베딩 관리자 ([`src/indexer.py`](file:///c:/study/Projects/team-ggeodeureok/src/indexer.py))**:
  - `langchain_community.vectorstores.Chroma` 및 `HuggingFaceEmbeddings` 래퍼 구현.
  - CUDA GPU 가속 자동 감지 및 로드 (`torch.cuda.is_available()`).
  - 모델별 벡터 차원 불일치(Dimension mismatch) 발생 시 원인과 해결책(컬렉션 분리 또는 `--reset`)을 명시하는 친절한 예외 처리 적용.

---

### 2. 고성능 임베딩 모델 연동 및 기본 모델 설정
- **기본(Default) 임베딩 모델 변경**:
  - **`dragonkue/snowflake-arctic-embed-l-v2.0-ko`** (별칭: `arctic-ko`, `snowflake-arctic-ko`): 행정문서가 훈련에 활용된 1024차원 고성능 모델로, **공공 행정 및 조례/회의록 데이터 검색에 최적화**되어 디폴트 모델로 채택.
- **긴 문서 특화 프리셋 지원**:
  - **`nlpai-lab/KURE-v1`** (별칭: `kure`, `kure-v1`): BAAI/bge-m3 기반의 최신 한국어 검색 특화 모델로, 긴 문서 및 복잡한 의회 회의록 검색용 프리셋 지원.
- **다양한 프리셋 지원**:
  - `ko-sbert` (`jhgan/ko-sbert-nli` - 768차원, 경량/빠름)
  - `kosimcse` (`BM-K/KoSimCSE-roberta-multitask` - 768차원, 의미 유사도 특화)
  - `bge-m3` (`BAAI/bge-m3` - 1024차원)
  - `bge-m3-ko` (`dragonkyle/bge-m3-ko` - 1024차원)
  - `e5-base` (`intfloat/multilingual-e5-base` - 768차원)
  - `kr-sbert` (`snunlp/KR-SBERT-V40K-klue-subword` - 768차원)
- **손쉬운 모델 교체 인터페이스 3가지 지원**:
  1. CLI 옵션: `-m / --model <별칭_또는_경로>`, `-c / --collection <컬렉션명>`
  2. 환경 변수: `$env:EMBEDDING_MODEL_NAME = "kure"`
  3. 설정 파일: `src/config.py`의 `EMBEDDING_MODEL_NAME` 기본값 변경

---

### 3. CLI 실행 스크립트 구축 및 개선
- **대규모 데이터 적재 도구 ([`run_ingest.py`](file:///c:/study/Projects/team-ggeodeureok/run_ingest.py))**:
  - 배치 단위(`BATCH_SIZE=500`) 스트리밍 색인으로 메모리 점유 최소화.
  - `tqdm` 이중 프로그레스 바 적용: 전체 데이터셋에 대한 남은 예상 소요 시간(ETA) 및 실시간 처리 속도(it/s) 표시.
  - 테스트 적재용 `--sample-lines <N>`, 대상 컬렉션 초기화 `--reset`, 모델 지정 `-m`, 컬렉션 지정 `-c` 옵션 지원.
- **시맨틱 검색 및 메타데이터 필터링 도구 ([`query_search.py`](file:///c:/study/Projects/team-ggeodeureok/query_search.py))**:
  - 자연어 질문과 ChromaDB `$and` 메타데이터 조건식 결합.
  - 의회명(`--assembly`), 회의연도(`--year`), 발언자(`--speaker`), 출력개수(`-k`), 모델(`-m`), 컬렉션(`-c`) 파라미터 지원.
  - 코사인 유사도 거리, 일시/의회, 안건, 발언자, 발언 요약 프리뷰 출력.

---

### 4. 종합 문서화 및 Git 형상 관리
- **전체 사용법 문서 최신화 ([`README.md`](file:///c:/study/Projects/team-ggeodeureok/README.md))**:
  - 최신 디렉토리 구조 및 파일 목록 상세 기술.
  - 전처리 파이프라인(`scripts/`의 1~5단계 스크립트) 설명 추가.
  - 데이터 스키마(TSV 7개 컬럼 및 ChromaDB 메타데이터 필드 상세표) 정리.
  - 임베딩 모델 손쉬운 변경 가이드(별칭 목록 및 3가지 변경 방법, 차원 불일치 주의사항) 수록.
  - 트러블슈팅(가상환경, Windows PowerShell UTF-8 인코딩, CUDA 가속 확인) 가이드 추가.
- **Git 브랜치 관리**:
  - `develop`에서 신규 작업 브랜치 `develop_kb` 분기 및 전환.
  - 불필요한 캐시/임시파일 제외 처리(`.gitignore` 정리).
  - 작업 단위별 체계적 커밋 및 원격 저장소(`origin/develop_kb`) 푸시 완료.
