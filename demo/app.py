#!/usr/bin/env python3
"""광역 지방의회 정책 이슈 검색·전파·예측 Streamlit 데모."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
MODEL_REVISION = "55ec6e9"
DEFAULT_TIMELINE_PATH = (
    "outputs/issue_timeline_final/issue_timelines.json"
    if Path("outputs/issue_timeline_final/issue_timelines.json").exists()
    else "examples/sample_issue_timelines.json"
)
DEFAULT_FORECAST_PATH = (
    "outputs/issue_forecast_final/issue_forecasts.json"
    if Path("outputs/issue_forecast_final/issue_forecasts.json").exists()
    else "examples/sample_issue_forecasts.json"
)
COUNCIL_COORDS = {
    "서울특별시의회": (37.5665, 126.9780),
    "부산광역시의회": (35.1796, 129.0756),
    "대구광역시의회": (35.8714, 128.6014),
    "인천광역시의회": (37.4563, 126.7052),
    "광주광역시의회": (35.1595, 126.8526),
    "대전광역시의회": (36.3504, 127.3845),
    "울산광역시의회": (35.5384, 129.3114),
    "세종특별자치시의회": (36.4800, 127.2890),
    "경기도의회": (37.2897, 127.0535),
    "강원특별자치도의회": (37.8854, 127.7298),
    "충청북도의회": (36.6357, 127.4915),
    "충청남도의회": (36.6588, 126.6728),
    "전북특별자치도의회": (35.8203, 127.1088),
    "전라북도의회": (35.8203, 127.1088),
    "전라남도의회": (34.8161, 126.4629),
    "경상북도의회": (36.5760, 128.5058),
    "경상남도의회": (35.2383, 128.6924),
    "제주특별자치도의회": (33.4996, 126.5312),
}
# 네트워크 지도 타일 없이도 실행되도록 단순화한 대한민국 외곽선(화면 배치 전용).
KOREA_OUTLINE = [
    (38.55, 128.36), (38.22, 128.60), (37.75, 128.88), (37.10, 129.25),
    (36.35, 129.48), (35.65, 129.38), (35.10, 128.85), (34.72, 128.48),
    (34.55, 127.75), (34.35, 126.55), (34.75, 126.10), (35.25, 126.25),
    (35.85, 126.05), (36.55, 126.45), (37.15, 126.55), (37.75, 126.25),
    (38.10, 126.65), (38.55, 128.36),
]


st.set_page_config(page_title="광역 지방의회 정책 흐름", page_icon="📍", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; max-width: 1450px;}
    [data-testid="stMetric"] {background:#f5f7fb; border:1px solid #e2e8f0; padding:14px; border-radius:12px;}
    .issue-box {background:#0f2747; color:white; padding:20px 24px; border-radius:14px; margin:8px 0 18px 0;}
    .issue-box h2 {margin:0 0 6px 0; font-size:1.45rem;}
    .issue-box p {margin:0; color:#dbeafe;}
    .note {color:#64748b; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_centers(path: str) -> np.ndarray:
    centers = np.asarray(np.load(path), dtype=np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return centers / np.maximum(norms, 1e-12)


@st.cache_resource(show_spinner=False)
def load_model(cache_dir: str, device: str):
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_folder=str(Path(cache_dir).resolve()),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.max_seq_length = 2048
    return model


def lexical_bonus(query: str, timeline: dict) -> float:
    query_key = re.sub(r"\s+", "", query.lower())
    text = " ".join(
        str(timeline.get(key, ""))
        for key in ("issue_name", "issue_summary", "keywords", "policy_domain")
    )
    return 0.12 if query_key and query_key in re.sub(r"\s+", "", text.lower()) else 0.0


def lexical_score(query: str, timeline: dict) -> float:
    tokens = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", query.lower()))
    text = " ".join(str(timeline.get(key, "")) for key in ("issue_name", "issue_summary", "keywords")).lower()
    return sum(token in text for token in tokens) / max(1, len(tokens))


def semantic_rank(query: str, timelines: list[dict], centers_path: str, cache_dir: str, device: str):
    centers = load_centers(centers_path)
    model = load_model(cache_dir, device)
    kwargs = {"convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": False}
    if "query" in (getattr(model, "prompts", {}) or {}):
        kwargs["prompt_name"] = "query"
    vector = np.asarray(model.encode([query], **kwargs)[0], dtype=np.float32)
    ranked = []
    for timeline in timelines:
        cluster_id = int(timeline["cluster_id"])
        if 0 <= cluster_id < len(centers):
            semantic = float(np.dot(vector, centers[cluster_id]))
            ranked.append((semantic + lexical_bonus(query, timeline), timeline))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def council_coordinate(name: str):
    if name in COUNCIL_COORDS:
        return COUNCIL_COORDS[name]
    simplified = name.replace("의회", "")
    for council, coordinate in COUNCIL_COORDS.items():
        if simplified and (simplified in council or council.replace("의회", "") in name):
            return coordinate
    return None


def propagation_map(events: list[dict]) -> go.Figure:
    located = []
    for event in events:
        coordinate = council_coordinate(str(event.get("council", "")))
        if coordinate:
            located.append((event, coordinate))
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[point[1] for point in KOREA_OUTLINE],
        y=[point[0] for point in KOREA_OUTLINE],
        mode="lines", fill="toself", fillcolor="#f8fafc",
        line={"width": 1.5, "color": "#94a3b8"},
        hoverinfo="skip", showlegend=False,
    ))
    # 제주도는 외곽선 지도와 분리된 작은 타원으로 표시한다.
    figure.add_shape(
        type="circle", x0=126.18, x1=126.86, y0=33.28, y1=33.68,
        fillcolor="#f8fafc", line={"color": "#94a3b8", "width": 1.5},
    )
    for index in range(len(located) - 1):
        (_, start), (_, end) = located[index], located[index + 1]
        figure.add_trace(go.Scatter(
            x=[start[1], end[1]], y=[start[0], end[0]], mode="lines",
            line={"width": 2, "color": "#60a5fa"}, hoverinfo="skip", showlegend=False,
        ))
    if located:
        figure.add_trace(go.Scatter(
            x=[item[1][1] for item in located],
            y=[item[1][0] for item in located],
            text=[
                f"{index + 1}. {item[0].get('council')}<br>{item[0].get('first_date')}<br>시차 {item[0].get('lag_days', 0)}일"
                for index, item in enumerate(located)
            ],
            marker={
                "size": [18 if index == 0 else 12 for index in range(len(located))],
                "color": list(range(1, len(located) + 1)),
                "colorscale": "Blues", "line": {"width": 1, "color": "white"},
            },
            mode="markers+text",
            textposition="top center",
            customdata=[index + 1 for index in range(len(located))],
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))
    figure.update_layout(
        height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#ffffff", plot_bgcolor="#e0f2fe",
        xaxis={"range": [125.7, 129.8], "visible": False, "scaleanchor": "y"},
        yaxis={"range": [33.0, 39.0], "visible": False},
    )
    return figure


def timeline_chart(series: list[dict], forecast: dict | None) -> go.Figure:
    months = [row.get("month") for row in series]
    counts = [row.get("mention_count", 0) for row in series]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=months, y=counts, mode="lines+markers", name="관측 언급량",
        line={"color": "#2563eb", "width": 3}, marker={"size": 7},
    ))
    if forecast:
        figure.add_trace(go.Scatter(
            x=[forecast.get("forecast_month")], y=[forecast.get("predicted_mention_count", 0)],
            mode="markers", name="다음 달 예측", marker={"size": 14, "color": "#f97316", "symbol": "diamond"},
        ))
    figure.update_layout(
        height=360, margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="월", yaxis_title="근거 발언 수", hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def korean_direction(value: str) -> str:
    return {
        "expansion": "증가 예상", "persistence": "유지 예상",
        "attenuation": "감소 예상", "dormant": "비활성 예상",
    }.get(value, value or "예측 없음")


st.title("광역 지방의회 정책 이슈 흐름")
st.caption("2024–2025년 제공 회의록만 이용한 최초 관측, 의회 간 후속 관측, 다음 달 흐름 예측")

with st.sidebar:
    st.header("데이터 설정")
    timeline_path = st.text_input("타임라인 JSON", DEFAULT_TIMELINE_PATH)
    forecast_path = st.text_input("예측 JSON", DEFAULT_FORECAST_PATH)
    centers_path = st.text_input(
        "군집 중심 벡터",
        "data/index/arctic_ko_semantic_test_v2/clustering_k160/cluster_centers.f32.npy",
    )
    cache_dir = st.text_input("모델 캐시", "models/huggingface")
    device = st.selectbox("검색 장치", ["cuda", "cpu"], index=0)
    st.caption("지도 좌표는 화면 배치 전용이며 분석·예측 변수로 사용하지 않습니다.")

try:
    timelines = load_json(timeline_path)
except Exception as error:
    st.error(f"타임라인 파일을 열 수 없습니다: {error}")
    st.stop()

try:
    forecasts = load_json(forecast_path) if Path(forecast_path).exists() else []
except Exception as error:
    st.error(f"예측 파일을 열 수 없습니다: {error}")
    st.stop()
forecast_by_issue = {str(row.get("issue_id")): row for row in forecasts}

if timeline_path.startswith("examples/"):
    st.info("현재 화면은 공개 저장소용 합성 예시입니다. 실제 경진대회 분석 결과가 아닙니다.")

query_col, button_col = st.columns([5, 1])
query = query_col.text_input("정책 이슈 검색", placeholder="예: 저출생, 버스노선 교통약자, 수돗물 안전")
search_clicked = button_col.button("검색", use_container_width=True)

ranked = []
if query and search_clicked:
    try:
        with st.spinner("로컬 임베딩으로 검색 중입니다."):
            ranked = semantic_rank(query, timelines, centers_path, cache_dir, device)
    except Exception as error:
        st.warning(f"임베딩 검색을 사용할 수 없어 키워드 검색으로 대체합니다: {error}")
        ranked = sorted(
            ((lexical_score(query, row), row) for row in timelines),
            key=lambda item: item[0], reverse=True,
        )
    st.session_state["ranked_issue_ids"] = [row[1]["issue_id"] for row in ranked[:5]]

ranked_ids = st.session_state.get("ranked_issue_ids", [])
if ranked_ids:
    candidate_timelines = [row for issue_id in ranked_ids for row in timelines if row.get("issue_id") == issue_id]
    st.caption("검색 상위 5개 이슈")
else:
    candidate_timelines = sorted(timelines, key=lambda row: row.get("issue_name", ""))

selected_name = st.selectbox(
    "분석할 이슈",
    [f"{row.get('issue_name')} ({row.get('issue_id')})" for row in candidate_timelines],
)
selected_index = [f"{row.get('issue_name')} ({row.get('issue_id')})" for row in candidate_timelines].index(selected_name)
timeline = candidate_timelines[selected_index]
forecast = forecast_by_issue.get(str(timeline.get("issue_id")))

st.markdown(
    f"<div class='issue-box'><h2>{timeline.get('issue_name')}</h2>"
    f"<p>{timeline.get('issue_summary', '')}</p></div>", unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("최초 관측", timeline.get("first_observed_date", "-"))
metric_columns[1].metric("최초 의회", ", ".join(timeline.get("origin_councils", [])) or "-")
metric_columns[2].metric("관측 의회", f"{timeline.get('observed_council_count', 0)}개")
metric_columns[3].metric("관측 근거", f"{timeline.get('observation_count', 0)}건")
metric_columns[4].metric("다음 달", korean_direction((forecast or {}).get("predicted_direction", "")))

map_column, flow_column = st.columns([1.25, 1])
with map_column:
    st.subheader("의회별 최초 관측 순서")
    st.plotly_chart(propagation_map(timeline.get("propagation", [])), use_container_width=True)
with flow_column:
    st.subheader("전파 순서")
    propagation_rows = [
        {
            "순서": event.get("propagation_order"), "의회": event.get("council"),
            "최초 관측일": event.get("first_date"), "시차(일)": event.get("lag_days"),
            "근거 수": event.get("mention_count"), "평균 유사도": event.get("mean_similarity"),
        }
        for event in timeline.get("propagation", [])
    ]
    st.dataframe(pd.DataFrame(propagation_rows), hide_index=True, use_container_width=True, height=520)

st.subheader("월별 논의 변화와 다음 달 예측")
st.plotly_chart(timeline_chart(timeline.get("monthly_series", []), forecast), use_container_width=True)

if forecast:
    forecast_columns = st.columns(4)
    forecast_columns[0].metric("논의 증가 확률", f"{float(forecast.get('growth_probability', 0)):.1%}")
    forecast_columns[1].metric("예측 언급량", f"{forecast.get('predicted_mention_count', 0)}건")
    forecast_columns[2].metric("신규 의회 확산", f"{float(forecast.get('spread_probability', 0)):.1%}")
    forecast_columns[3].metric("예측 신뢰도", forecast.get("forecast_confidence", "-"))
    if forecast.get("forecast_confidence") == "low":
        st.warning("검증 성능 또는 표본 수가 제한되어 보수적으로 해석해야 하는 예측입니다.")
    st.info(forecast.get("forecast_narrative", ""))

st.subheader("근거 회의록 원문 구간")
evidence_count = 0
for event in timeline.get("propagation", []):
    for evidence in event.get("evidence", []):
        evidence_count += 1
        title = f"{event.get('council')} · {evidence.get('meeting_date')} · {evidence.get('meeting_name') or '회의록'}"
        with st.expander(title, expanded=evidence_count <= 2):
            st.write(f"안건: {evidence.get('agenda_title') or '-'}")
            st.write(f"발화자: {evidence.get('speaker_label') or '-'}")
            st.write(evidence.get("original_excerpt") or "원문 구간이 없습니다. 09번 v2를 다시 실행하세요.")
            st.caption(
                f"근거 참조: {evidence.get('evidence_ref') or '-'} · "
                f"문자 범위: {evidence.get('char_start', -1)}–{evidence.get('char_end', -1)} · "
                f"군집 유사도: {evidence.get('similarity', 0)}"
            )

st.markdown(
    "<p class='note'>전파는 제공 데이터 안에서 관측된 시간적 순서이며 실제 정책의 인과적 전파를 단정하지 않습니다. "
    "예측에는 2025-12-31 이후 외부 정보를 사용하지 않았습니다.</p>",
    unsafe_allow_html=True,
)
