# 오류 및 수정 기록

## 1. UTF-8 파일이 지원하지 않는 인코딩으로 판정됨

### 증상

`2024_경기도의회.json`이 일반 UTF-8인데 `지원하지 않는 인코딩` 오류가 발생했다.

### 원인과 수정

파일 일부의 비정상 바이트 때문에 앞부분 전체를 엄격하게 디코딩하는 검사가 실패했다.
JSON 시작 구조를 먼저 확인하고 UTF-8 스트리밍 처리에서 대체 문자를 허용하도록
감지 규칙을 보완했다.

## 2. 정책 후보가 0개로 나옴

### 원인

본문 추출 규칙과 필터 조건이 실제 청크 컬럼 구조에 맞지 않았다.

### 수정

선택·제외 사유별 개수와 대표 샘플을 출력하게 만들어 어느 조건에서 제거되는지 먼저
검증했다. 필터는 한 번에 강하게 적용하지 않고 청크 유형, 우선순위, 정책 신호,
진행성 문장 순으로 점검했다.

## 3. 전체 46만 청크 군집에 잡음이 많음

회의 진행, 인사, 명단, 단순 보고까지 함께 군집화한 것이 원인이었다. 원문 검색용 전체
인덱스는 유지하고, 정책 분석용 부분 집합을 별도로 임베딩·군집화했다.

## 4. 긴 청크에 여러 정책이 혼합됨

부모 청크를 바로 임베딩하면 복지·예산·교통 등이 한 벡터에 들어갈 수 있었다. 부모
청크는 근거 문맥으로 보존하고 600자 목표, 최대 900자의 의미 자식 청크를 추가했다.

## 5. K=100 군집이 너무 넓음

K=100과 K=160의 평균 중심 유사도, 군집 크기와 대표 발언 일관성을 비교했다. K=160이
더 구체적인 정책 단위로 나뉘어 최종 후보로 선택되었다.

## 6. LLM이 모든 군집을 coherent로 판정함

대표 발언 일부만 보고 넓은 이름을 붙이는 문제가 있었다. 발언별 주제, 지지 근거 번호,
이상치 번호를 강제로 출력하고 지지율 75% 미만을 코드가 `mixed`로 덮어쓰도록 수정했다.

## 7. 09번 근거 원문이 비어 있음

기존 군집 배정 Parquet에는 용량 절감을 위해 `semantic_text`가 빠져 있었다. 09번이
`embedding_row`를 키로 의미 청크 메타데이터를 결합해 정제 원문 구간을 복원하도록
보완했다. 군집화와 임베딩을 다시 할 필요는 없다.

## 8. 실행 스크립트 옵션을 코드가 인식하지 못함

### 증상

```text
unrecognized arguments: --evidence-metadata
unrecognized arguments: --forecasts
```

### 원인

새 `scripts/`와 예전 `src/09`, `src/10`이 섞인 버전 불일치다.

### 확인

```bash
python src/09_build_issue_timeline.py --help | grep evidence-metadata
python src/10_issue_query.py --help | grep forecasts
```

출력되지 않으면 최신 `src/` 파일로 교체한다.

## 9. `streamlit: command not found`

```bash
source .venv/bin/activate
python -m pip install streamlit==1.40.2 plotly==5.24.1
python -m streamlit run demo/app.py
```

`python -m streamlit` 방식은 다른 Python 환경의 실행 파일이 섞이는 문제를 줄인다.

## 10. VS Code 실행 버튼으로 데모가 열리지 않음

Streamlit 앱은 일반 Python 프로그램과 실행 방식이 다르다. VS Code 통합 터미널에서
다음 명령을 사용한다.

```bash
cd ~/projects/council-minutes
source .venv/bin/activate
python -m streamlit run demo/app.py
```

## 11. 노트북과 데스크톱에서 결과 공유

GitHub에는 코드만 공유한다. 대용량 데이터와 인덱스는 외장 SSD 또는 허용된 내부
스토리지로 옮기며, 다음 파일의 해시를 함께 비교한다.

- `embedding_state.json`
- `clustering_summary.json`
- 라벨링 요약 JSON
- 최종 타임라인·예측 요약 JSON

경로가 달라질 수 있으므로 코드에는 사용자 홈의 절대 경로를 고정하지 않는다.

