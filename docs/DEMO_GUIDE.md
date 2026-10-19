# 데모 실행 및 화면 설명

## VS Code 실행

1. VS Code에서 `~/projects/council-minutes` 폴더를 연다.
2. Python 인터프리터를 `.venv/bin/python`으로 선택한다.
3. `Terminal > New Terminal`을 연다.
4. 아래 명령을 실행한다.

```bash
source .venv/bin/activate
python -m streamlit run demo/app.py
```

브라우저가 자동으로 열리지 않으면 `http://localhost:8501`로 접속한다. 종료는 터미널에서
`Ctrl+C`다.

## 실제 데이터 데모

필요 파일:

```text
outputs/issue_timeline_final/issue_timelines.json
outputs/issue_forecast_final/issue_forecasts.json
data/index/arctic_ko_semantic_test_v2/clustering_k160/cluster_centers.f32.npy
models/huggingface/
```

검색어를 입력하면 로컬 임베딩으로 가까운 이슈 군집을 찾고, 선택된 이슈의 최초 관측,
후속 의회 순서, 월별 변화, 다음 달 예측과 근거 원문을 보여준다.

## 공개 합성 데모

실제 결과 파일이 없으면 다음 파일을 자동 사용한다.

```text
examples/sample_issue_timelines.json
examples/sample_issue_forecasts.json
```

이 화면은 UI 구조를 설명하기 위한 합성 예시이며 실제 분석 결과가 아니다. 군집 중심
벡터가 없을 때 검색 버튼을 누르면 키워드 검색으로 자동 대체된다.

## 화면 발표 순서

1. `출산율` 또는 `저출생` 검색
2. 가장 가까운 검수 이슈 선택
3. 최초 관측일과 최초 의회 설명
4. 지도에서 의회별 최초 관측 순서와 시차 설명
5. 월별 언급량과 다음 달 예측점 설명
6. 예측 신뢰도와 모델 한계 설명
7. 회의록명·일자·원문 근거 구간을 펼쳐 결과의 추적 가능성 제시

지도 화살표는 시간순 관측을 시각화한 것이며 실제 정책 전달의 인과관계를 단정하지
않는다고 반드시 설명한다.

