# Council Minutes Policy Flow AI

광역 지방의회 회의록에서 정책 이슈를 발견하고, 제공 데이터 범위 내 최초 관측 시점,
다른 의회의 후속 논의 순서, 월별 변화와 다음 달 흐름을 원문 근거와 함께 제시하는
로컬 AI 분석 파이프라인입니다.

> 2026 DATA·AI 분석 경진대회 팀 프로젝트입니다. 공개 저장소에는 경진대회 원문 데이터와 실제 분석 결과를 포함하지 않으며, 데모는 합성 데이터로 실행됩니다.

## Project Summary

- **Problem**: 방대한 지방의회 회의록에서 유사 정책 이슈의 등장·확산·변화를 수작업으로 추적하기 어려움
- **Constraint**: 회의록 원문 외부 전송 제한 → 상용 LLM API 대신 로컬 임베딩·로컬 LLM 중심 설계
- **Output**: 정책 키워드, 최초 관측 시점, 의회별 후속 흐름, 월별 변화, 향후 논의 전망, 근거 회의록
- **Team**: 5명
- **My focus**: 임베딩 모델 비교·선정, 대규모 임베딩 실행, FAISS 의미 검색, 군집화/검색 실험, 로컬 LLM 후보 비교

## My Contribution

이 저장소는 대회 종료 후 팀 전체 코드를 **하나의 공개용 스냅샷 커밋**으로 정리한 저장소입니다.
따라서 Git commit history만으로 개인별 기여를 분리해서 확인하기 어렵습니다.

제가 주로 담당한 범위는 다음과 같습니다.

1. **한국어 임베딩 모델 비교 및 선정**
   - 한국어 회의록 의미 검색에 사용할 임베딩 모델 후보 비교
   - 최종적으로 `dragonkue/snowflake-arctic-embed-l-v2.0-ko` 기반 파이프라인 사용

2. **대규모 임베딩 파이프라인 실행 및 검증**
   - 전체 460,431 chunks 중 품질 기준을 적용한 438,708 chunks를 임베딩 대상으로 구성
   - bf16, sequence length 2048, batch size 8 조건으로 실행
   - RTX 4060 Laptop 환경에서 약 **39.2 chunks/s** 처리량 측정
   - 약 **898 MB** 규모의 `embeddings.f16.npy` 생성

3. **FAISS 기반 의미 검색 및 검색 구조 검토**
   - 임베딩 벡터 기반 FAISS `IndexFlatIP` 검색
   - Dense 검색의 약점을 보완하기 위해 BM25 / RRF 기반 하이브리드 검색 구조 검토

4. **군집화 및 이슈 후보 탐색**
   - 유사 발언 검색 결과를 기반으로 정책 이슈 후보를 묶는 군집화 흐름 실험
   - 검색 → 군집화 → 이슈명 제안으로 이어지는 파이프라인 연결

5. **로컬 LLM 후보 비교**
   - 원문 외부 전송 제한을 지키기 위해 로컬 LLM 후보를 비교
   - LLM이 검색/군집 결과를 임의 생성하지 않고, 제공 근거를 바탕으로 이슈명·설명을 제안하도록 역할을 제한

관련 코드 영역은 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)에 별도로 정리했습니다.

## Architecture

```text
JSON/HTML
  ↓
정제 · 중복 제거
  ↓
부모/자식 청킹
  ↓
정책 후보 필터
  ↓
한국어 임베딩
  ↓
FAISS / BM25 / RRF 검색
  ↓
군집화
  ↓
로컬 LLM 이슈명 제안
  ↓
최초 관측 · 후속 흐름 · 월별 변화
  ↓
예측
  ↓
Streamlit 지도 · 타임라인 · 근거 답변
```

## Main Modules

| 단계 | 파일 | 역할 |
|---|---|---|
| 데이터 확인 | `src/01_inventory.py` | 원본 파일 구조·인코딩·메타데이터 점검 |
| 전처리 | `src/02_preprocess.py` | HTML 제거, 정규화, 중복 제거 |
| 청킹 | `src/03_make_chunks.py` | 부모/자식 청킹 |
| 임베딩 | `src/04_embedding.py`, `04d_embed_selected.py` | 임베딩 생성 및 선택 데이터 처리 |
| 검색 | `src/05_build_faiss.py`, `06_search_v2.py` | FAISS 인덱스 및 검색 |
| 군집화 | `src/07_clustering.py` | 정책 이슈 후보 군집화 |
| LLM 라벨링 | `src/08_llm_issue_labeling.py` | 로컬 LLM 기반 이슈명 제안 |
| 타임라인 | `src/09_build_issue_timeline.py` | 최초·후속 관측 및 시계열 구성 |
| 예측 | `src/11_forecast_issue_flow.py` | 다음 달 흐름 예측 |
| 데모 | `demo/app.py` | Streamlit 지도·타임라인·근거 화면 |

> 위 표는 공개 저장소의 전체 팀 파이프라인 구조입니다. 개인 기여 범위는 별도의 CONTRIBUTIONS 문서에서 구분합니다.

## Key Experiment

| 항목 | 값 |
|---|---:|
| 전체 chunks | 460,431 |
| 임베딩 대상 chunks | 438,708 |
| Embedding model | `snowflake-arctic-embed-l-v2.0-ko` |
| Precision | bf16 |
| Sequence length | 2048 |
| Batch size | 8 |
| Throughput | 약 39.2 chunks/s |
| Embedding output | 약 898 MB |

## Tech Stack

| 구분 | 기술 |
|---|---|
| Language | Python |
| Embedding | Hugging Face, `snowflake-arctic-embed-l-v2.0-ko` |
| Vector Search | FAISS |
| Hybrid Retrieval | BM25, RRF |
| Clustering | MiniBatchKMeans |
| Local LLM | Ollama, Qwen3 8B |
| Prediction | Logistic Regression, LightGBM |
| Demo | Streamlit, Plotly |
| Environment | Linux / WSL, PyTorch |

## Synthetic Demo

Ubuntu 또는 WSL2와 Python 3.12 환경을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run demo/app.py
```

브라우저에서 `http://localhost:8501`을 엽니다.

실제 결과 파일이 없으면 `examples/sample_issue_timelines.json`과
`examples/sample_issue_forecasts.json`을 사용합니다.
화면의 이슈·발언·예측은 모두 합성 예시입니다.

## Repository Structure

```text
src/           데이터 처리·검색·군집화·예측 코드
demo/          Streamlit 합성 데모
scripts/       실행·검증 스크립트
docs/          기술 설계·모델 카드·오류 해결 기록
examples/      합성 타임라인·예측 예시
CONTRIBUTIONS.md
requirements.txt
```

## Interpretation & Limitations

- 최초 관측은 전국 최초가 아니라 **제공 데이터 범위 내 최초 관측**입니다.
- 의회별 후속 순서는 실제 정책 전달의 인과관계를 증명하지 않습니다.
- LLM은 군집을 생성하지 않고 이슈명과 설명을 제안하는 보조 역할입니다.
- 예측 모델은 시간순 검증 기준을 통과한 경우에만 사용합니다.
- 성능이 부족하면 설명 가능한 추세 규칙으로 대체하고 낮은 신뢰도를 표시합니다.

## Data Policy

다음 항목은 저장소에 포함하지 않습니다.

- 경진대회 제공 회의록 원문 및 정제 데이터
- 실제 회의록 구간이 포함된 검색·타임라인 결과
- Parquet, embedding NPY, FAISS index, model weights
- 실제 LLM prompt log와 개인 환경 파일

## Documentation

- [개인 기여 범위](CONTRIBUTIONS.md)
- [전체 파이프라인](docs/PIPELINE.md)
- [모델 카드](docs/MODEL_CARD.md)
- [오류 및 수정 기록](docs/TROUBLESHOOTING.md)
- [주요 의사결정](docs/DECISION_LOG.md)
- [데모 가이드](docs/DEMO_GUIDE.md)
