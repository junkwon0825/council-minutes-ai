#!/usr/bin/env python3
"""같은 10만 벡터로 ChromaDB와 PostgreSQL+pgvector를 공정하게 비교한다.

측정 항목
- exact cosine Top-K 대비 Recall@K / Top-1 일치율
- 검색 지연시간 p50 / p95 / p99
- 두 DB 결과의 overlap@K
- 지역(council)+연도(year) 필터를 적용한 검색 성능

질의 임베딩 시간은 DB 검색 지연시간에서 제외한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from council_vector_runtime import EmbeddingArtifact, QueryEncoder, atomic_json, exact_topk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "eval" / "queries.csv")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--exact-limit", type=int, default=100_000)
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1",
    )
    parser.add_argument(
        "--chroma-dir", type=Path,
        default=PROJECT_ROOT / "data" / "chroma" / "arctic_raw_v1",
    )
    parser.add_argument("--collection", default="council_arctic_raw_v1")
    parser.add_argument("--pg-dsn", default=os.environ.get("PGVECTOR_DSN"))
    parser.add_argument("--pg-table", default="council_arctic_raw_v1")
    parser.add_argument("--pg-ef-search", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--model-cache", type=Path,
        default=PROJECT_ROOT / "models" / "huggingface",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "vector_db" / "chroma_vs_pgvector_100k.json",
    )
    return parser.parse_args()


def load_queries(path: Path, limit: int) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"query_id", "query"} <= set(rows[0]):
        raise ValueError("질의 CSV에는 query_id,query 열이 필요합니다")
    return [
        {"query_id": str(row["query_id"]), "query": str(row["query"]).strip()}
        for row in rows[:limit] if str(row.get("query") or "").strip()
    ]


def load_scope_metadata(
    artifact: EmbeddingArtifact, scope_rows: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    required = {"chunk_id", "council", "year"}
    missing = sorted(required - set(artifact.columns))
    if missing:
        raise ValueError(f"metadata 필수 열이 없습니다: {missing}")

    chunk_ids: list[str] = []
    councils: list[str] = []
    years: list[int] = []
    for batch in artifact.parquet.iter_batches(
        batch_size=20_000, columns=["chunk_id", "council", "year"],
    ):
        for row in batch.to_pylist():
            if len(chunk_ids) >= scope_rows:
                break
            chunk_ids.append(str(row["chunk_id"]))
            councils.append(str(row.get("council") or ""))
            years.append(int(row.get("year") or 0))
        if len(chunk_ids) >= scope_rows:
            break
    if len(chunk_ids) != scope_rows:
        raise RuntimeError(f"metadata 범위 불일치: {len(chunk_ids):,} != {scope_rows:,}")
    return chunk_ids, np.asarray(councils, dtype=object), np.asarray(years, dtype=np.int32)


def exact_filtered_ids(
    artifact: EmbeddingArtifact,
    query_vector: np.ndarray,
    chunk_ids: list[str],
    councils: np.ndarray,
    years: np.ndarray,
    council: str,
    year: int,
    top_k: int,
) -> list[str]:
    candidate_indices = np.flatnonzero((councils == council) & (years == year))
    if len(candidate_indices) == 0:
        return []
    scores = np.asarray(
        np.asarray(artifact.vectors[candidate_indices], dtype=np.float32) @ query_vector,
        dtype=np.float32,
    )
    count = min(top_k, len(scores))
    positions = np.argpartition(scores, -count)[-count:]
    ordered = positions[np.argsort(scores[positions])[::-1]]
    return [chunk_ids[int(candidate_indices[position])] for position in ordered]


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values), q)), 3) if values else 0.0


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in runs]
    recalls = [float(row["recall_at_k"]) for row in runs]
    top1s = [float(row["top1_matches"]) for row in runs]
    return {
        "requests": len(runs),
        "latency_ms_mean": round(float(np.mean(latencies)), 3),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_p99": percentile(latencies, 99),
        "recall_at_k_mean": round(float(np.mean(recalls)), 6),
        "top1_accuracy": round(float(np.mean(top1s)), 6),
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def score_result(found: list[str], truth: list[str]) -> tuple[float, bool]:
    denominator = max(1, len(truth))
    recall = len(set(found) & set(truth)) / denominator
    top1 = bool(found and truth and found[0] == truth[0])
    return recall, top1


def main() -> None:
    args = parse_args()
    if min(args.max_queries, args.top_k, args.repeat, args.exact_limit) < 1:
        raise ValueError("질의 수, top-k, 반복 수, exact-limit는 1 이상이어야 합니다")
    if args.warmup < 0:
        raise ValueError("warmup은 0 이상이어야 합니다")
    if not args.pg_dsn:
        raise ValueError("--pg-dsn 또는 PGVECTOR_DSN 환경변수가 필요합니다")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.pg_table):
        raise ValueError("pg-table 이름은 영문자·숫자·밑줄만 허용됩니다")

    artifact = EmbeddingArtifact(args.embedding_dir)
    scope_rows = min(args.exact_limit, artifact.row_count)
    queries = load_queries(args.queries, args.max_queries)
    if not queries:
        raise ValueError("평가할 질의가 없습니다")
    chunk_ids, councils, years = load_scope_metadata(artifact, scope_rows)

    encoder = QueryEncoder(
        artifact,
        device=args.device,
        model_cache=args.model_cache,
        local_files_only=not args.allow_download,
    )
    encode_started = time.perf_counter()
    query_vectors = encoder.encode([row["query"] for row in queries])
    query_embedding_ms = (time.perf_counter() - encode_started) * 1000

    exact_indices, _, exact_ms = exact_topk(
        artifact.vectors, query_vectors, args.top_k, limit=scope_rows,
    )
    exact_ids = [
        [chunk_ids[int(index)] for index in row if index >= 0]
        for row in exact_indices
    ]

    filters: list[dict[str, Any]] = []
    filtered_truth: list[list[str]] = []
    id_to_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    for query_index, truth in enumerate(exact_ids):
        top_index = id_to_index[truth[0]]
        current = {"council": str(councils[top_index]), "year": int(years[top_index])}
        filters.append(current)
        filtered_truth.append(exact_filtered_ids(
            artifact, query_vectors[query_index], chunk_ids, councils, years,
            current["council"], current["year"], args.top_k,
        ))

    import chromadb
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql

    chroma_dir = args.chroma_dir.expanduser().resolve()
    collection = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(
        args.collection, embedding_function=None,
    )
    chroma_count = int(collection.count())
    if chroma_count != scope_rows:
        raise RuntimeError(
            f"Chroma 행이 exact 범위와 다릅니다: {chroma_count:,} != {scope_rows:,}. "
            "두 DB를 같은 --limit로 적재했는지 확인하세요."
        )

    connection = psycopg.connect(args.pg_dsn, autocommit=True)
    register_vector(connection)
    table = sql.Identifier(args.pg_table)
    pg_count = int(connection.execute(
        sql.SQL("SELECT count(*) FROM {}").format(table)
    ).fetchone()[0])
    if pg_count != scope_rows:
        connection.close()
        raise RuntimeError(
            f"pgvector 행이 exact 범위와 다릅니다: {pg_count:,} != {scope_rows:,}"
        )
    connection.execute(
        "SELECT set_config('hnsw.ef_search', %s, false)", (str(args.pg_ef_search),),
    )
    connection.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', false)")
    pg_unfiltered = sql.SQL(
        "SELECT chunk_id FROM {} ORDER BY embedding <=> %s LIMIT %s"
    ).format(table)
    pg_filtered = sql.SQL(
        "SELECT chunk_id FROM {} WHERE council = %s AND year = %s "
        "ORDER BY embedding <=> %s LIMIT %s"
    ).format(table)

    def chroma_query(vector: np.ndarray, top_k: int, current_filter: dict[str, Any] | None) -> list[str]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector.tolist()],
            "n_results": top_k,
            "include": ["distances"],
        }
        if current_filter:
            kwargs["where"] = {"$and": [
                {"council": current_filter["council"]},
                {"year": current_filter["year"]},
            ]}
        result = collection.query(**kwargs)
        return [str(value) for value in result["ids"][0]]

    def pg_query(vector: np.ndarray, top_k: int, current_filter: dict[str, Any] | None) -> list[str]:
        if current_filter:
            rows = connection.execute(
                pg_filtered,
                (current_filter["council"], current_filter["year"], vector, top_k),
            ).fetchall()
        else:
            rows = connection.execute(pg_unfiltered, (vector, top_k)).fetchall()
        return [str(row[0]) for row in rows]

    query_functions: dict[str, Callable[[np.ndarray, int, dict[str, Any] | None], list[str]]] = {
        "chroma": chroma_query,
        "pgvector": pg_query,
    }
    mode_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "unfiltered": defaultdict(list),
        "filtered_council_year": defaultdict(list),
    }
    overlaps: dict[str, list[float]] = defaultdict(list)

    try:
        for mode_name, truth_rows, use_filter in (
            ("unfiltered", exact_ids, False),
            ("filtered_council_year", filtered_truth, True),
        ):
            for query_index, (query_row, vector, truth) in enumerate(
                zip(queries, query_vectors, truth_rows)
            ):
                current_filter = filters[query_index] if use_filter else None
                for _ in range(args.warmup):
                    chroma_query(vector, args.top_k, current_filter)
                    pg_query(vector, args.top_k, current_filter)

                for repeat_index in range(args.repeat):
                    order = ["chroma", "pgvector"]
                    if (query_index + repeat_index) % 2:
                        order.reverse()
                    found_by_backend: dict[str, list[str]] = {}
                    for backend in order:
                        started = time.perf_counter()
                        found = query_functions[backend](vector, args.top_k, current_filter)
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        recall, top1 = score_result(found, truth)
                        found_by_backend[backend] = found
                        mode_runs[mode_name][backend].append({
                            "query_id": query_row["query_id"],
                            "repeat": repeat_index + 1,
                            "latency_ms": round(elapsed_ms, 3),
                            "recall_at_k": round(recall, 6),
                            "top1_matches": top1,
                            "result_count": len(found),
                        })
                    overlap = len(
                        set(found_by_backend["chroma"]) & set(found_by_backend["pgvector"])
                    ) / max(1, args.top_k)
                    overlaps[mode_name].append(overlap)
                print(
                    f"{mode_name:23s} | {query_row['query_id']} | "
                    f"filter={current_filter or '-'}",
                    flush=True,
                )
    finally:
        connection.close()

    with psycopg.connect(args.pg_dsn, autocommit=True) as size_connection:
        pg_relation_size = int(size_connection.execute(
            "SELECT pg_total_relation_size(%s::regclass)", (args.pg_table,)
        ).fetchone()[0])

    mode_reports: dict[str, Any] = {}
    for mode_name, backend_runs in mode_runs.items():
        mode_reports[mode_name] = {
            "chroma": summarize(backend_runs["chroma"]),
            "pgvector": summarize(backend_runs["pgvector"]),
            "backend_overlap_at_k_mean": round(float(np.mean(overlaps[mode_name])), 6),
            "runs": dict(backend_runs),
        }

    report = {
        "status": "complete",
        "protocol": {
            "same_vectors": True,
            "same_queries": True,
            "same_top_k": True,
            "query_embedding_excluded_from_db_latency": True,
            "alternating_backend_order": True,
            "warm_cache_benchmark": True,
        },
        "artifact": artifact.identity,
        "scope_rows": scope_rows,
        "dimension": artifact.dimension,
        "query_count": len(queries),
        "top_k": args.top_k,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "query_embedding_total_ms": round(query_embedding_ms, 3),
        "exact_search_total_ms": round(exact_ms, 3),
        "pg_hnsw_ef_search": args.pg_ef_search,
        "storage": {
            "chroma_directory_bytes": directory_size(chroma_dir),
            "pgvector_relation_bytes": pg_relation_size,
        },
        "queries": [
            {**query, "filter": current_filter}
            for query, current_filter in zip(queries, filters)
        ],
        "modes": mode_reports,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, report)

    print("
=== ChromaDB vs pgvector 동일 조건 비교 완료 ===")
    print(f"대상: {scope_rows:,}행 | 질의: {len(queries)} | Top-K: {args.top_k}")
    for mode_name, mode in mode_reports.items():
        print(f"
[{mode_name}]")
        for backend in ("chroma", "pgvector"):
            value = mode[backend]
            print(
                f"{backend:9s} | p50={value['latency_ms_p50']:.2f}ms "
                f"| p95={value['latency_ms_p95']:.2f}ms "
                f"| Recall@{args.top_k}={value['recall_at_k_mean']:.4f} "
                f"| Top1={value['top1_accuracy']:.4f}"
            )
        print(f"두 DB overlap@{args.top_k}: {mode['backend_overlap_at_k_mean']:.4f}")
    print(f"
결과: {output}")


if __name__ == "__main__":
    main()
