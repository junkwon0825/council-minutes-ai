# Council Minutes Policy Flow AI

광역 지방의회 회의록에서 정책 이슈를 발견하고, 제공 데이터 범위 내 최초 관측 시점,
다른 의회의 후속 논의 순서, 월별 변화와 다음 달 흐름을 원문 근거와 함께 제시하는
로컬 AI 분석 파이프라인입니다.

이 공개 저장소에는 경진대회 제공 데이터나 실제 분석 결과가 포함되어 있지 않습니다.
데모는 직접 만든 합성 데이터로 실행됩니다.

## 주요 기능

- JSON 내부 HTML 회의록 정제와 완전 중복 제거
- 안건·발화·문단 경계를 고려한 부모–자식 청킹
- 한국어 임베딩과 FAISS 의미 검색
- 정책 후보 필터와 MiniBatchKMeans 군집화
- 로컬 LLM을 이용한 이슈명 제안과 근거 기반 품질 게이트
- 이슈별 최초 관측일과 의회별 후속 관측 순서 계산
- 월별 논의량과 참여 의회 변화 생성
- 시간순 검증 기반 다음 달 흐름 예측
- Streamlit 지도·타임라인·근거 화면

## 아키텍처

```text
JSON/HTML → 정제·중복 제거 → 부모 청킹 → 정책 후보 필터
→ 의미 자식 청킹 → 한국어 임베딩 → FAISS 검색·KMeans 군집화
→ 로컬 LLM 이슈명 제안·품질 검수 → issue_id 고정
→ 최초 관측·후속 관측·월별 변화 → 성능 게이트 예측 → 데모
```

## 사용 기술

| 구분 | 기술 |
|---|---|
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 벡터 검색 | FAISS `IndexFlatIP` |
| 군집화 | MiniBatchKMeans |
| 로컬 LLM | Ollama `qwen3:8b` |
| 예측 | Logistic Regression, LightGBM, trend fallback |
| 데모 | Streamlit, Plotly |

회의록 텍스트는 외부 상용 LLM API로 전송하지 않습니다. 모델 파일만 내려받고 모든
분석과 추론을 로컬에서 수행하도록 설계했습니다.

## 합성 데모 실행

Ubuntu 또는 WSL2와 Python 3.12 환경을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run demo/app.py
```

브라우저에서 `http://localhost:8501`을 엽니다. 실제 결과 파일이 없으면
`examples/sample_issue_timelines.json`과 `sample_issue_forecasts.json`을 자동으로
사용합니다. 화면에 표시되는 이슈·발언·예측은 모두 합성 예시입니다.

## VS Code

1. 저장소 폴더를 VS Code로 엽니다.
2. Python 인터프리터를 `.venv/bin/python`으로 선택합니다.
3. 통합 터미널에서 `source .venv/bin/activate`를 실행합니다.
4. `python -m streamlit run demo/app.py`를 실행합니다.

`Run Python File` 버튼으로 `app.py`를 직접 실행하는 방식이 아니라 Streamlit 명령을
사용해야 합니다.

## 저장소 구조

```text
src/           데이터 처리·검색·군집화·예측 코드
demo/          Streamlit 데모
scripts/       실행·검증·공개 안전 검사
docs/          기술 설계·모델 카드·오류 해결 기록
examples/      합성 타임라인·예측 예시
requirements.txt
```

## 분석 결과 해석

- 최초 관측은 전국 최초가 아니라 제공 데이터 범위 내 최초 관측입니다.
- 의회별 후속 순서는 실제 정책 전달의 인과관계를 증명하지 않습니다.
- LLM은 군집을 생성하지 않고 이슈명과 설명을 제안하는 보조 역할입니다.
- 예측 모델은 시간순 검증 기준을 통과한 경우에만 사용합니다.
- 성능이 부족하면 설명 가능한 추세 규칙으로 대체하고 낮은 신뢰도를 표시합니다.

## 공개 데이터 정책

다음 항목은 이 저장소에 포함하지 않습니다.

- 경진대회 제공 회의록 원문과 정제 데이터
- 실제 회의록 구간이 포함된 검색·타임라인 결과
- Parquet, 임베딩 NPY, FAISS 인덱스, 모델 가중치
- 실제 LLM 프롬프트 로그와 개인 환경 파일

데이터 이용자는 주최 측이 허용한 공식 경로와 이용 조건을 따라야 합니다.

## 문서

- [전체 파이프라인](docs/PIPELINE.md)
- [모델 카드](docs/MODEL_CARD.md)
- [오류 및 수정 기록](docs/TROUBLESHOOTING.md)
- [주요 의사결정](docs/DECISION_LOG.md)
- [데모 가이드](docs/DEMO_GUIDE.md)
- [GitHub 공개 가이드](GITHUB_PUBLISHING.md)

