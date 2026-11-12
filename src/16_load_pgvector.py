#!/usr/bin/env python3
"""검증된 전체 임베딩을 PostgreSQL+pgvector에 적재하고 정합성을 검사한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from council_vector_runtime import EmbeddingArtifact, atomic_json, date_string, json_safe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_COLUMNS = {
    "parent_chunk_id", "document_id", "source_key", "council", "meeting_date",
    "year", "committee", "meeting_name", "agenda_title", "policy_categories_json",
    "core_issues_json", "keywords_json", "substantive_ratio", "chunk_type", "priority",
    "source_row_start", "source_row_end", "char_length", "canonical_version",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1",
    )
    parser.add_argument("--dsn", default=os.environ.get("PGVECTOR_DSN"))
    parser.add_argument("--table", default="council_arctic_raw_v1")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--state", type=Path,
        default=PROJECT_ROOT / "outputs" / "vector_db" / "pgvector_arctic_raw_v1.json",
    )
    return parser.parse_args()


def iter_batches(
    artifact: EmbeddingArtifact,
    batch_size: int,
    skip_rows: int,
    target_rows: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    seen = 0
    wanted = ["chunk_id", "text", *sorted(METADATA_COLUMNS)]
    columns = [name for name in wanted if name in artifact.columns]
    for batch in artifact.parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        batch_end = seen + len(rows)
        if batch_end <= skip_rows:
            seen = batch_end
            continue
        start_in_batch = max(0, skip_rows - seen)
        absolute_start = seen + start_in_batch
        remaining = target_rows - absolute_start
        if remaining <= 0:
            break
        selected = rows[start_in_batch:start_in_batch + remaining]
        if selected:
            yield absolute_start, selected
        seen = batch_end


def metadata_json(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json_safe(row[key])
        for key in METADATA_COLUMNS
        if key in row and row[key] is not None
    }


def validate_probe(connection: Any, table: Any, artifact: EmbeddingArtifact) -> dict[str, Any]:
    from psycopg import sql

    first_group = artifact.parquet.read_row_group(0, columns=["chunk_id"])
    probe_id = str(first_group["chunk_id"][0].as_py())
    source = np.asarray(artifact.vectors[0], dtype=np.float32)
    stored_row = connection.execute(
        sql.SQL("SELECT embedding FROM {} WHERE chunk_id = %s").format(table),
        (probe_id,),
    ).fetchone()
    if stored_row is None:
        raise RuntimeError(f"pgvector에 검증 ID가 없습니다: {probe_id}")
    stored_value = stored_row[0]
    if hasattr(stored_value, "to_numpy"):
        stored_value = stored_value.to_numpy()
    elif hasattr(stored_value, "to_list"):
        stored_value = stored_value.to_list()
    stored = np.asarray(stored_value, dtype=np.float32)
    max_abs_error = float(np.max(np.abs(source - stored)))
    query_row = connection.execute(
        sql.SQL(
            "SELECT chunk_id, embedding <=> %s AS distance "
            "FROM {} ORDER BY embedding <=> %s LIMIT 1"
        ).format(table),
        (source, source),
    ).fetchone()
    top1_id = str(query_row[0])
    top1_distance = float(query_row[1])
    result = {
        "probe_id": probe_id,
        "stored_vector_max_abs_error": max_abs_error,
        "self_query_top1_id": top1_id,
        "self_query_top1_distance": top1_distance,
        "self_query_top1_matches": top1_id == probe_id,
    }
    if max_abs_error > 1e-6 or top1_id != probe_id:
        raise RuntimeError(f"pgvector 적재 검증 실패: {result}")
    return result


def main() -> None:
    args = parse_args()
    if not args.dsn:
        raise ValueError("--dsn 또는 PGVECTOR_DSN 환경변수가 필요합니다")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.table):
        raise ValueError("table 이름은 영문자·숫자·밑줄만 허용됩니다")
    if min(args.batch_size, args.m, args.ef_construction) < 1:
        raise ValueError("batch-size와 HNSW 설정은 1 이상이어야 합니다")

    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql
    from psycopg.types.json import Jsonb

    artifact = EmbeddingArtifact(args.embedding_dir)
    target_rows = min(args.limit, artifact.row_count) if args.limit else artifact.row_count
    if target_rows < 1:
        raise ValueError("적재 대상 행이 없습니다")
    table = sql.Identifier(args.table)
    state_path = args.state.expanduser().resolve()
    started = time.perf_counter()

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        extension_version = str(connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0])
        if args.rebuild:
            connection.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(table))
            if state_path.exists():
                state_path.unlink()
        connection.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
              chunk_id text PRIMARY KEY,
              parent_chunk_id text,
              document_id text,
              council text,
              meeting_date date,
              year integer,
              committee text,
              meeting_name text,
              agenda_title text,
              priority text,
              document text NOT NULL,
              metadata jsonb NOT NULL,
              embedding vector({}) NOT NULL
            )
        """).format(table, sql.Literal(artifact.dimension)))

        existing = int(connection.execute(
            sql.SQL("SELECT count(*) FROM {}").format(table)
        ).fetchone()[0])
        if existing > target_rows:
            raise RuntimeError(f"기존 테이블이 목표보다 큽니다: {existing:,} > {target_rows:,}")
        if existing and not state_path.is_file():
            raise RuntimeError("기존 테이블은 있지만 상태 파일이 없습니다. --rebuild를 사용하세요")
        if existing:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if previous.get("artifact_identity") != artifact.identity:
                raise RuntimeError("기존 테이블과 현재 임베딩이 다릅니다. --rebuild가 필요합니다")

        insert = sql.SQL("""
            INSERT INTO {} (
              chunk_id, parent_chunk_id, document_id, council, meeting_date, year,
              committee, meeting_name, agenda_title, priority, document, metadata, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
              metadata = EXCLUDED.metadata,
              document = EXCLUDED.document,
              embedding = EXCLUDED.embedding
        """).format(table)
        completed = existing
        initial_count = existing
        for start, rows in iter_batches(artifact, args.batch_size, completed, target_rows):
            values = []
            for offset, row in enumerate(rows):
                values.append((
                    str(row["chunk_id"]), str(row.get("parent_chunk_id") or "") or None,
                    str(row.get("document_id") or "") or None, str(row.get("council") or ""),
                    date_string(row.get("meeting_date")), int(row.get("year") or 0),
                    str(row.get("committee") or ""), str(row.get("meeting_name") or ""),
                    str(row.get("agenda_title") or ""), str(row.get("priority") or ""),
                    str(row.get("text") or ""), Jsonb(metadata_json(row)),
                    np.asarray(artifact.vectors[start + offset], dtype=np.float32),
                ))
            with connection.cursor() as cursor:
                cursor.executemany(insert, values)
            completed = start + len(rows)
            atomic_json(state_path, {
                "status": "running", "database": "postgresql+pgvector",
                "table": args.table, "target_rows": target_rows,
                "completed_rows": completed, "dimension": artifact.dimension,
                "artifact_identity": artifact.identity,
            })
            print(f"pgvector 적재: {completed:,}/{target_rows:,}", flush=True)

        index_seconds = None
        if not args.skip_index:
            index_started = time.perf_counter()
            hnsw_name = sql.Identifier(f"{args.table}_embedding_hnsw_idx")
            connection.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw "
                "(embedding vector_cosine_ops) WITH (m = {}, ef_construction = {})"
            ).format(hnsw_name, table, sql.Literal(args.m), sql.Literal(args.ef_construction)))
            for suffix, column in (
                ("council", "council"), ("year", "year"), ("committee", "committee"),
                ("date", "meeting_date"), ("priority", "priority"),
            ):
                connection.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} ({})").format(
                    sql.Identifier(f"{args.table}_{suffix}_idx"), table, sql.Identifier(column)
                ))
            connection.execute(sql.SQL("ANALYZE {}").format(table))
            index_seconds = time.perf_counter() - index_started

        final_count = int(connection.execute(
            sql.SQL("SELECT count(*) FROM {}").format(table)
        ).fetchone()[0])
        if final_count != target_rows:
            raise RuntimeError(f"pgvector count 불일치: {final_count:,} != {target_rows:,}")
        probe = validate_probe(connection, table, artifact)
        relation_size = int(connection.execute(
            "SELECT pg_total_relation_size(%s::regclass)", (args.table,)
        ).fetchone()[0])

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete", "database": "postgresql+pgvector",
        "table": args.table, "row_count": final_count, "dimension": artifact.dimension,
        "distance": "cosine", "index": None if args.skip_index else "hnsw",
        "artifact_identity": artifact.identity, "probe_validation": probe,
        "pgvector_version": extension_version, "relation_size_bytes": relation_size,
        "hnsw_m": args.m, "hnsw_ef_construction": args.ef_construction,
        "elapsed_seconds_current_session": round(elapsed, 3),
        "index_elapsed_seconds": round(index_seconds, 3) if index_seconds is not None else None,
        "inserted_rows_current_session": final_count - initial_count,
        "rows_per_second_current_session": round((final_count - initial_count) / elapsed, 3)
        if elapsed and final_count > initial_count else None,
        "dsn_redacted": True,
    }
    atomic_json(state_path, summary)
    print("
=== PostgreSQL+pgvector 적재 완료 ===")
    print(f"테이블: {args.table} | 저장: {final_count:,}")
    print(f"벡터 최대 오차: {probe['stored_vector_max_abs_error']:.10f}")
    print(f"자기검색 Top-1: {probe['self_query_top1_matches']}")
    print(f"관계 크기: {relation_size / 1024 / 1024:.1f} MiB")
    print(f"상태: {state_path}")


if __name__ == "__main__":
    main()
