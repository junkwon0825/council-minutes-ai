#!/usr/bin/env python3
"""의정 이슈 레이더 공개 포트폴리오용 Streamlit 실행 화면."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from agent.local_issue_agent import LocalIssueAgent
from agent.tools import CouncilAgentTools
from src.council_vector_runtime import CouncilSearchService


ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def build_agent() -> LocalIssueAgent:
    embedding_dir = Path(os.environ.get(
        "COUNCIL_EMBEDDING_DIR",
        ROOT / "data" / "embeddings" / "arctic_raw_v1",
    ))
    bm25 = Path(os.environ.get(
        "COUNCIL_BM25_INDEX",
        ROOT / "data" / "bm25" / "arctic_raw_v1" / "bm25.sqlite3",
    ))
    service = CouncilSearchService(
        embedding_dir=embedding_dir,
        pg_dsn=os.environ.get("PGVECTOR_DSN"),
        pg_table=os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1"),
        retrieval_mode=os.environ.get("COUNCIL_RETRIEVAL_MODE", "hybrid"),
        bm25_index=bm25 if bm25.exists() else None,
    )
    return LocalIssueAgent(CouncilAgentTools(service))


st.set_page_config(page_title="의정 이슈 레이더", layout="wide")
st.title("의정 이슈 레이더")
st.caption("PostgreSQL + pgvector · BM25/RRF · Local LLM")

question = st.text_input("정책 또는 현안을 질문하세요", "청년 주거 지원 정책은 어디에서 논의됐나요?")
if st.button("검색"):
    try:
        result = build_agent().ask(question)
        st.subheader("답변")
        st.write(result["answer"])
        st.subheader("근거")
        for row in result["evidence"].get("results", []):
            meta = row.get("metadata") or {}
            with st.expander(
                f"#{row['rank']} {meta.get('council','')} · {meta.get('meeting_date','')} · score={row.get('similarity',0):.4f}"
            ):
                st.write(row.get("text") or "")
                st.json(meta)
    except Exception as error:
        st.error(f"실행 환경을 확인하세요: {error}")
