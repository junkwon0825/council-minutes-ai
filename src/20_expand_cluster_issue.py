#!/usr/bin/env python3
"""재순위 근거를 seed로 관련 발언을 확장하고 임베딩 군집을 만든다.

외부 LLM을 사용하지 않는다. pgvector에 저장된 동일 Arctic 벡터를 이용해
후보를 확장하고, cosine similarity graph의 연결요소를 이슈 군집으로 사용한다.
생성된 issue_id는 검수 전 provisional 상태이며 이후 registry에서 고정한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from council_vector_runtime import atomic_json, json_safe


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("PGVECTOR_DSN"))
    parser.add_argument("--table", default="council_arctic_raw_v1")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--expansion-k", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=500)
    parser.add_argument("--min-centroid-similarity", type=float, default=0.45)
    parser.add_argument("--cluster-similarity", type=float, default=0.72)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=128)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dsn:
        raise ValueError("--dsn 또는 PGVECTOR_DSN 환경변수가 필요합니다")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.table):
        raise ValueError("table 이름은 영문자·숫자·밑줄만 허용됩니다")
    if min(
        args.seed_count, args.expansion_k, args.ef_search,
        args.min_cluster_size, args.block_size,
    ) < 1:
        raise ValueError("개수 관련 설정은 1 이상이어야 합니다")
    if args.ef_search < args.expansion_k:
        raise ValueError("ef-search는 expansion-k 이상이어야 합니다")
    for name, value in (
        ("min-centroid-similarity", args.min_centroid_similarity),
        ("cluster-similarity", args.cluster_similarity),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}는 0~1 범위여야 합니다")


def vector_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "to_list"):
        value = value.to_list()
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("DB에서 읽은 벡터가 올바르지 않습니다")
    return vector


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def graph_clusters(
    vectors: np.ndarray, threshold: float, block_size: int,
) -> tuple[np.ndarray, int]:
    count = len(vectors)
    union_find = UnionFind(count)
    edge_count = 0
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        similarities = vectors[start:end] @ vectors.T
        for local_index in range(end - start):
            global_index = start + local_index
            linked = np.flatnonzero(similarities[local_index] >= threshold)
            linked = linked[linked > global_index]
            for other in linked.tolist():
                union_find.union(global_index, int(other))
            edge_count += len(linked)
    roots = [union_find.find(index) for index in range(count)]
    root_to_label: dict[int, int] = {}
    labels = []
    for root in roots:
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])
    return np.asarray(labels, dtype=np.int32), edge_count


def build_issue_id(member_ids: list[str]) -> str:
    stable = "\n".join(sorted(member_ids)).encode("utf-8")
    return f"ISSUE_{hashlib.sha256(stable).hexdigest()[:12].upper()}"


def expansion_row(row: tuple[Any, ...]) -> tuple[dict[str, Any], np.ndarray]:
    metadata = dict(row[10] or {})
    explicit = {
        "parent_chunk_id": row[1],
        "council": row[2],
        "meeting_date": row[3],
        "year": row[4],
        "committee": row[5],
        "meeting_name": row[6],
        "agenda_title": row[7],
        "priority": row[8],
    }
    for key, value in explicit.items():
        if value is not None:
            metadata[key] = json_safe(value)
    item = {
        "chunk_id": str(row[0]),
        "centroid_similarity": round(float(row[12]), 6),
        "text": str(row[9] or ""),
        "metadata": {key: json_safe(value) for key, value in metadata.items()},
    }
    return item, vector_to_numpy(row[11])


def cluster_summary(
    label: int,
    indices: list[int],
    items: list[dict[str, Any]],
    seed_ids: set[str],
) -> dict[str, Any]:
    councils = Counter(
        str(items[index]["metadata"].get("council") or "") for index in indices
    )
    dates = sorted(
        str(items[index]["metadata"].get("meeting_date") or "")[:10]
        for index in indices if items[index]["metadata"].get("meeting_date")
    )
    similarities = [items[index]["centroid_similarity"] for index in indices]
    return {
        "cluster_label": label,
        "size": len(indices),
        "seed_count": sum(items[index]["chunk_id"] in seed_ids for index in indices),
        "council_count": len([key for key in councils if key]),
        "top_councils": councils.most_common(5),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "centroid_similarity_mean": round(float(np.mean(similarities)), 6),
        "centroid_similarity_min": round(float(np.min(similarities)), 6),
        "centroid_similarity_max": round(float(np.max(similarities)), 6),
    }


def item_group_key(item: dict[str, Any]) -> str:
    parent_id = str(item["metadata"].get("parent_chunk_id") or "")
    return f"parent:{parent_id}" if parent_id else f"chunk:{item['chunk_id']}"


def source_seed_item(
    row: dict[str, Any], vector: np.ndarray, centroid: np.ndarray,
) -> dict[str, Any]:
    return {
        "chunk_id": str(row["chunk_id"]),
        "centroid_similarity": round(float(vector @ centroid), 6),
        "text": str(row.get("text") or ""),
        "metadata": {
            key: json_safe(value)
            for key, value in dict(row.get("metadata") or {}).items()
        },
    }


def deduplicate_by_parent(
    parsed: list[tuple[dict[str, Any], np.ndarray]],
    seed_ids: set[str],
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], int]:
    selected: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for item, vector in parsed:
        key = item_group_key(item)
        previous = selected.get(key)
        if previous is None:
            selected[key] = (item, vector)
            continue
        current_is_seed = item["chunk_id"] in seed_ids
        previous_is_seed = previous[0]["chunk_id"] in seed_ids
        if current_is_seed and not previous_is_seed:
            selected[key] = (item, vector)
        elif current_is_seed == previous_is_seed and (
            item["centroid_similarity"] > previous[0]["centroid_similarity"]
        ):
            selected[key] = (item, vector)
    result = sorted(
        selected.values(),
        key=lambda pair: pair[0]["centroid_similarity"],
        reverse=True,
    )
    return result, len(parsed) - len(result)


def normalized_text_key(item: dict[str, Any]) -> str:
    """공백 차이만 있는 동일 원문을 하나의 근거 단위로 묶는다."""
    normalized = re.sub(r"\s+", "", str(item.get("text") or "")).casefold()
    if not normalized:
        return f"chunk:{item['chunk_id']}"
    return "text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_by_text(
    parsed: list[tuple[dict[str, Any], np.ndarray]],
    seed_ids: set[str],
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], int]:
    """동일 원문을 제거하되 검수 seed 정보는 대표 청크에 보존한다."""
    selected: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for item, vector in parsed:
        item.setdefault("duplicate_chunk_ids", [])
        item.setdefault("duplicate_seed_chunk_ids", [])
        key = normalized_text_key(item)
        previous = selected.get(key)
        if previous is None:
            selected[key] = (item, vector)
            continue

        previous_item, previous_vector = previous
        current_ids = [item["chunk_id"], *item["duplicate_chunk_ids"]]
        previous_ids = [
            previous_item["chunk_id"], *previous_item["duplicate_chunk_ids"]
        ]
        all_ids = list(dict.fromkeys([*previous_ids, *current_ids]))
        seed_aliases = sorted(set(all_ids) & seed_ids)

        current_is_seed = item["chunk_id"] in seed_ids
        previous_is_seed = previous_item["chunk_id"] in seed_ids
        replace = current_is_seed and not previous_is_seed
        if current_is_seed == previous_is_seed:
            replace = (
                item["centroid_similarity"]
                > previous_item["centroid_similarity"]
            )

        if replace:
            item["duplicate_chunk_ids"] = [
                value for value in all_ids if value != item["chunk_id"]
            ]
            item["duplicate_seed_chunk_ids"] = [
                value for value in seed_aliases if value != item["chunk_id"]
            ]
            selected[key] = (item, vector)
        else:
            previous_item["duplicate_chunk_ids"] = [
                value for value in all_ids if value != previous_item["chunk_id"]
            ]
            previous_item["duplicate_seed_chunk_ids"] = [
                value for value in seed_aliases
                if value != previous_item["chunk_id"]
            ]
            selected[key] = (previous_item, previous_vector)

    result = sorted(
        selected.values(),
        key=lambda pair: pair[0]["centroid_similarity"],
        reverse=True,
    )
    return result, len(parsed) - len(result)


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    query = str(payload.get("query") or "").strip()
    source_results = payload.get("results")
    if not query or not isinstance(source_results, list) or not source_results:
        raise ValueError("입력 JSON에 query 또는 results가 없습니다")
    seed_ids_ordered = [
        str(row["chunk_id"]) for row in source_results[:args.seed_count]
    ]
    seed_ids = set(seed_ids_ordered)

    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql

    table = sql.Identifier(args.table)
    started = time.perf_counter()
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        register_vector(connection)
        seed_rows = connection.execute(
            sql.SQL("SELECT chunk_id, embedding FROM {} WHERE chunk_id = ANY(%s)").format(table),
            (seed_ids_ordered,),
        ).fetchall()
        vectors_by_id = {str(row[0]): vector_to_numpy(row[1]) for row in seed_rows}
        missing_seeds = [value for value in seed_ids_ordered if value not in vectors_by_id]
        if missing_seeds:
            raise RuntimeError(f"pgvector에 seed가 없습니다: {missing_seeds[:5]}")

        seed_matrix = np.stack([vectors_by_id[value] for value in seed_ids_ordered])
        centroid = seed_matrix.mean(axis=0, dtype=np.float32)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 0 or not np.isfinite(centroid_norm):
            raise RuntimeError("seed centroid를 정규화할 수 없습니다")
        centroid = centroid / centroid_norm

        connection.execute(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(args.ef_search),),
        )
        connection.execute(
            "SELECT set_config('hnsw.iterative_scan', 'strict_order', false)"
        )
        statement = sql.SQL("""
            SELECT
              chunk_id, parent_chunk_id, council, meeting_date, year, committee,
              meeting_name, agenda_title, priority, document, metadata, embedding,
              1 - (embedding <=> %s) AS similarity
            FROM {}
            ORDER BY embedding <=> %s
            LIMIT %s
        """).format(table)
        expansion_started = time.perf_counter()
        rows = connection.execute(
            statement, (centroid, centroid, args.expansion_k),
        ).fetchall()
        expansion_ms = (time.perf_counter() - expansion_started) * 1000

    parsed = [expansion_row(row) for row in rows]
    parsed = [
        (item, vector) for item, vector in parsed
        if item["centroid_similarity"] >= args.min_centroid_similarity
    ]
    expansion_ids = {item["chunk_id"] for item, _ in parsed}
    source_by_id = {
        str(row["chunk_id"]): row for row in source_results[:args.seed_count]
    }
    forced_seed_count = 0
    for seed_id in seed_ids_ordered:
        if seed_id in expansion_ids:
            continue
        seed_vector = vectors_by_id[seed_id]
        parsed.append((
            source_seed_item(source_by_id[seed_id], seed_vector, centroid),
            seed_vector,
        ))
        forced_seed_count += 1

    before_parent_dedup = len(parsed)
    parsed, parent_duplicate_count = deduplicate_by_parent(parsed, seed_ids)
    parsed, text_duplicate_count = deduplicate_by_text(parsed, seed_ids)
    if len(parsed) < args.min_cluster_size:
        raise RuntimeError(
            "centroid similarity 조건을 통과한 후보가 너무 적습니다: "
            f"{len(parsed)}개"
        )
    items = [item for item, _ in parsed]
    vectors = np.stack([vector for _, vector in parsed]).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    clustering_started = time.perf_counter()
    labels, edge_count = graph_clusters(
        vectors, args.cluster_similarity, args.block_size,
    )
    clustering_ms = (time.perf_counter() - clustering_started) * 1000
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        indices_by_label[int(label)].append(index)
    summaries = [
        cluster_summary(label, indices, items, seed_ids)
        for label, indices in indices_by_label.items()
    ]
    summaries.sort(
        key=lambda row: (row["seed_count"], row["size"], row["centroid_similarity_mean"]),
        reverse=True,
    )
    # 하위 군집은 분리해 보존하되, 검수된 seed가 하나라도 포함된 군집은
    # 동일한 상위 정책 이슈의 구성원으로 결합한다.
    selected_labels = {
        int(row["cluster_label"]) for row in summaries if row["seed_count"] > 0
    }
    if not selected_labels:
        selected_labels = {int(summaries[0]["cluster_label"])}
    member_indices = sorted(
        index
        for label in selected_labels
        for index in indices_by_label[label]
    )
    members = [items[index] for index in member_indices]
    members.sort(key=lambda row: row["centroid_similarity"], reverse=True)
    selected_seed_ids: set[str] = set()
    for row in members:
        represented_ids = {
            row["chunk_id"], *row.get("duplicate_seed_chunk_ids", [])
        }
        selected_seed_ids.update(represented_ids & seed_ids)
    selected_councils = {
        str(row["metadata"].get("council") or "") for row in members
        if row["metadata"].get("council")
    }
    selected_dates = sorted(
        str(row["metadata"].get("meeting_date") or "")[:10]
        for row in members if row["metadata"].get("meeting_date")
    )
    selected_summary = {
        "selected_cluster_labels": sorted(selected_labels),
        "selected_subcluster_count": len(selected_labels),
        "size": len(members),
        "seed_count": len(selected_seed_ids),
        "seed_coverage": round(len(selected_seed_ids) / max(1, len(seed_ids)), 6),
        "missing_seed_chunk_ids": sorted(seed_ids - selected_seed_ids),
        "council_count": len(selected_councils),
        "first_date": selected_dates[0] if selected_dates else None,
        "last_date": selected_dates[-1] if selected_dates else None,
    }
    issue_id = build_issue_id([row["chunk_id"] for row in members])
    for rank, row in enumerate(members, start=1):
        row["issue_rank"] = rank
        row["is_seed"] = bool(
            {row["chunk_id"], *row.get("duplicate_seed_chunk_ids", [])}
            & seed_ids
        )

    report = {
        "status": "complete",
        "issue": {
            "issue_id": issue_id,
            "issue_name": query,
            "id_status": "provisional_until_review",
            "member_count": len(members),
        },
        "source": {
            "input": str(input_path),
            "seed_count_requested": args.seed_count,
            "seed_count_used": len(seed_ids_ordered),
            "seed_chunk_ids": seed_ids_ordered,
        },
        "expansion": {
            "backend": "postgresql+pgvector",
            "table": args.table,
            "expansion_k": args.expansion_k,
            "ef_search": args.ef_search,
            "retrieved_count": len(rows),
            "retained_before_parent_dedup": before_parent_dedup,
            "parent_duplicates_removed": parent_duplicate_count,
            "exact_text_duplicates_removed": text_duplicate_count,
            "forced_seed_count": forced_seed_count,
            "retained_count": len(items),
            "min_centroid_similarity": args.min_centroid_similarity,
        },
        "clustering": {
            "method": "cosine_similarity_graph_connected_components",
            "cluster_similarity": args.cluster_similarity,
            "min_cluster_size": args.min_cluster_size,
            "edge_count": edge_count,
            "cluster_count": len(summaries),
            "selection_method": "union_of_all_seed_bearing_subclusters",
            "selected_issue": selected_summary,
            "cluster_summaries": summaries,
        },
        "timing_ms": {
            "expansion_search": round(expansion_ms, 3),
            "clustering": round(clustering_ms, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
        "members": members,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, report)

    print("\n=== 이슈 확장·군집화 완료 ===")
    print(f"질의/임시 이슈명: {query}")
    print(f"issue_id: {issue_id} ({report['issue']['id_status']})")
    print(
        f"확장: {len(rows)} → 기준 통과: {len(items)} | "
        f"군집: {len(summaries)} | 선택 구성원: {len(members)}"
    )
    print(
        f"선택 이슈 seed: {selected_summary['seed_count']}/{len(seed_ids)} "
        f"({selected_summary['seed_coverage']:.1%}) | "
        f"의회: {selected_summary['council_count']} | "
        f"기간: {selected_summary['first_date']} ~ {selected_summary['last_date']}"
    )
    print(
        f"부모 중복 제거: {parent_duplicate_count} | "
        f"동일 원문 제거: {text_duplicate_count} | "
        f"강제 포함 seed: {forced_seed_count}"
    )
    print(f"검색: {expansion_ms:.1f}ms | 군집화: {clustering_ms:.1f}ms")
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
