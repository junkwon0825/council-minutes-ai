#!/usr/bin/env python3
"""SQLite FTS5 BM25 검색과 Dense/BM25 RRF 결합 공통 모듈."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search_aliases import (
    COMMITTEE_ALIAS_GROUPS, COUNCIL_ALIAS_GROUPS,
    committee_alias_values, council_alias_values, expand_query_aliases,
)

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
KOREAN_PARTICLES = (
    "이었습니다", "하였습니다", "했습니다", "되었습니다", "됐습니다",
    "입니다", "합니다", "됩니다", "이었다", "하였다", "했다", "되었다",
    "으로부터", "에서부터", "에게서", "으로써", "으로서", "에게로",
    "까지", "부터", "에서", "에게", "한테", "으로", "와의", "과의",
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "로", "도", "만",
)
KNOWN_ALIAS_TERMS = {alias for group in (*COUNCIL_ALIAS_GROUPS, *COMMITTEE_ALIAS_GROUPS) for alias in group}

def lexical_tokens(text: str, max_tokens: int | None = None) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    result: list[str] = []
    seen: set[str] = set()
    for match in TOKEN_PATTERN.finditer(normalized):
        token = match.group(0)
        candidates = [token]
        if token not in KNOWN_ALIAS_TERMS and len(token) >= 4 and any("가" <= char <= "힣" for char in token):
            for suffix in KOREAN_PARTICLES:
                if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                    candidates.append(token[:-len(suffix)])
                    break
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
                if max_tokens is not None and len(result) >= max_tokens:
                    return result
    return result

def fts_query(query: str) -> tuple[str, list[str]]:
    expanded = expand_query_aliases(query)
    tokens = lexical_tokens(expanded, max_tokens=48)
    if not tokens:
        raise ValueError("BM25로 검색할 수 있는 토큰이 없습니다")
    expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
    return expression, tokens

@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float
    rank: int

class BM25Index:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"BM25 인덱스가 없습니다: {self.path}")

    def search(self, query: str, top_k: int, council: str | None = None, year: int | None = None,
               committee: str | None = None, date_from: str | None = None, date_to: str | None = None,
               priorities: list[str] | None = None, policy_category: str | None = None) -> tuple[list[BM25Hit], list[str]]:
        expression, tokens = fts_query(query)
        conditions = ["bm25_chunks MATCH ?"]
        parameters: list[Any] = [expression]
        council_values = council_alias_values(council)
        if council_values:
            placeholders = ",".join("?" for _ in council_values)
            conditions.append(f"council IN ({placeholders})")
            parameters.extend(council_values)
        if year is not None:
            conditions.append("year = ?")
            parameters.append(int(year))
        committee_values = committee_alias_values(committee)
        if committee_values:
            conditions.append("(" + " OR ".join("committee LIKE ?" for _ in committee_values) + ")")
            parameters.extend(f"%{value}%" for value in committee_values)
        if date_from:
            conditions.append("meeting_date >= ?")
            parameters.append(str(date_from)[:10])
        if date_to:
            conditions.append("meeting_date <= ?")
            parameters.append(str(date_to)[:10])
        if priorities:
            placeholders = ",".join("?" for _ in priorities)
            conditions.append(f"priority IN ({placeholders})")
            parameters.extend(priorities)
        if policy_category:
            conditions.append("policy_categories LIKE ?")
            parameters.append(f"%{policy_category}%")
        parameters.append(int(top_k))
        statement = f"""
            SELECT chunk_id, bm25(bm25_chunks) AS raw_score
            FROM bm25_chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY raw_score ASC
            LIMIT ?
        """
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            rows = connection.execute(statement, parameters).fetchall()
        finally:
            connection.close()
        hits = [BM25Hit(str(chunk_id), -float(raw_score), rank) for rank, (chunk_id, raw_score) in enumerate(rows, start=1)]
        return hits, tokens

def reciprocal_rank_fusion(dense_rows: list[dict[str, Any]], bm25_rows: list[dict[str, Any]],
                           limit: int, rrf_k: int = 60, dense_weight: float = 1.0,
                           bm25_weight: float = 1.0) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source, rows, weight in (("dense", dense_rows, dense_weight), ("bm25", bm25_rows, bm25_weight)):
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row["chunk_id"])
            current = merged.setdefault(chunk_id, dict(row))
            current.setdefault("rrf_score", 0.0)
            current["rrf_score"] += float(weight) / (rrf_k + rank)
            current[f"{source}_rank"] = rank
            if source == "dense":
                current["dense_similarity"] = float(row.get("similarity") or 0.0)
            else:
                current["bm25_score"] = float(row.get("bm25_score") or 0.0)
            sources = set(current.get("retrieval_sources") or [])
            sources.add(source)
            current["retrieval_sources"] = sorted(sources)
    ranked = sorted(
        merged.values(),
        key=lambda row: (
            -float(row.get("rrf_score") or 0.0),
            int(row.get("dense_rank") or 10**9),
            int(row.get("bm25_rank") or 10**9),
            str(row.get("chunk_id") or ""),
        ),
    )[:limit]
    for rank, row in enumerate(ranked, start=1):
        row["rrf_rank"] = rank
        row["rrf_score"] = round(float(row["rrf_score"]), 9)
        row.setdefault("similarity", float(row.get("dense_similarity") or 0.0))
    return ranked
