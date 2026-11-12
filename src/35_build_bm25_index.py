#!/usr/bin/env python3
"""확정 청크 metadata로 재사용 가능한 SQLite FTS5 BM25 인덱스를 만든다."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from bm25_runtime import lexical_tokens
from council_vector_runtime import atomic_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "chunk_id", "text", "council", "year", "committee", "meeting_date",
    "priority", "meeting_name", "agenda_title", "policy_categories_json",
    "core_issues_json", "keywords_json",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1" / "chunk_metadata.parquet")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "bm25" / "arctic_raw_v1" / "bm25.sqlite3")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()

def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten(item) for item in value.values())
    text = str(value).strip()
    if text[:1] in {"[", "{"}:
        try:
            return flatten(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            pass
    return text

def weighted_search_text(row: dict[str, Any]) -> str:
    weighted_fields = (
        (row.get("text"), 1), (row.get("council"), 2), (row.get("committee"), 1),
        (row.get("meeting_name"), 1), (row.get("agenda_title"), 2),
        (row.get("policy_categories_json"), 1), (row.get("core_issues_json"), 2),
        (row.get("keywords_json"), 2),
    )
    tokens: list[str] = []
    for value, repeat in weighted_fields:
        current = lexical_tokens(flatten(value))
        for _ in range(repeat):
            tokens.extend(current)
    return " ".join(tokens)

def iter_rows(parquet: pq.ParquetFile, columns: list[str], batch_size: int, limit: int | None) -> Iterator[list[tuple[Any, ...]]]:
    emitted = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        if limit is not None:
            rows = rows[:max(0, limit - emitted)]
        if not rows:
            break
        values = []
        for row in rows:
            values.append((
                str(row["chunk_id"]), str(row.get("council") or ""),
                int(row.get("year") or 0), str(row.get("committee") or ""),
                str(row.get("meeting_date") or "")[:10], str(row.get("priority") or ""),
                flatten(row.get("policy_categories_json")), weighted_search_text(row),
            ))
        yield values
        emitted += len(values)
        if limit is not None and emitted >= limit:
            break

def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        if not args.rebuild:
            raise FileExistsError("BM25 인덱스가 이미 있습니다. 교체하려면 --rebuild를 사용하세요")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(source)
    available = set(parquet.schema_arrow.names)
    required = {"chunk_id", "text"}
    if required - available:
        raise ValueError(f"필수 열 누락: {sorted(required - available)}")
    columns = [name for name in FIELDS if name in available]
    target = min(args.limit, parquet.metadata.num_rows) if args.limit else parquet.metadata.num_rows
    started = time.perf_counter()
    connection = sqlite3.connect(output)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("""
            CREATE VIRTUAL TABLE bm25_chunks USING fts5(
              chunk_id UNINDEXED, council UNINDEXED, year UNINDEXED,
              committee UNINDEXED, meeting_date UNINDEXED, priority UNINDEXED,
              policy_categories UNINDEXED, search_text,
              tokenize='unicode61 remove_diacritics 0'
            )
        """)
        insert = """
            INSERT INTO bm25_chunks(
              chunk_id, council, year, committee, meeting_date,
              priority, policy_categories, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        completed = 0
        for values in iter_rows(parquet, columns, args.batch_size, args.limit):
            with connection:
                connection.executemany(insert, values)
            completed += len(values)
        connection.execute("INSERT INTO bm25_chunks(bm25_chunks) VALUES('optimize')")
        connection.commit()
        row_count = int(connection.execute("SELECT count(*) FROM bm25_chunks").fetchone()[0])
        unique_count = int(connection.execute("SELECT count(DISTINCT chunk_id) FROM bm25_chunks").fetchone()[0])
    finally:
        connection.close()
    if row_count != target or unique_count != target:
        raise RuntimeError(f"BM25 인덱스 정합성 실패: rows={row_count:,}, unique={unique_count:,}, target={target:,}")
    elapsed = time.perf_counter() - started
    atomic_json(output.with_name("bm25_state.json"), {
        "status": "complete", "engine": "sqlite_fts5_bm25",
        "source": str(source), "index": str(output), "row_count": row_count,
        "unique_chunk_ids": unique_count, "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(row_count / elapsed, 3) if elapsed else None,
        "file_size_bytes": output.stat().st_size,
    })

if __name__ == "__main__":
    main()
