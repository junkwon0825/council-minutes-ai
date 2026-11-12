#!/usr/bin/env python3
"""검증된 전체 임베딩을 로컬 ChromaDB에 배치 적재하고 정합성을 검사한다."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import chromadb
import numpy as np

from council_vector_runtime import EmbeddingArtifact, atomic_json, json_safe


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
    parser.add_argument(
        "--persist-dir", type=Path,
        default=PROJECT_ROOT / "data" / "chroma" / "arctic_raw_v1",
    )
    parser.add_argument("--collection", default="council_arctic_raw_v1")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
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


def make_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in METADATA_COLUMNS:
        if key not in row:
            continue
        value = json_safe(row[key])
        if value is not None:
            metadata[key] = value
    meeting_date = str(metadata.get("meeting_date") or "").replace("-", "")[:8]
    if meeting_date.isdigit():
        metadata["meeting_date_int"] = int(meeting_date)
    return metadata or {"metadata_available": False}


def create_collection(client: Any, name: str, artifact: EmbeddingArtifact) -> Any:
    common = {"name": name, "embedding_function": None}
    try:
        return client.create_collection(
            **common,
            metadata={"artifact": artifact.paths.root.name},
            configuration={"hnsw": {"space": "cosine"}},
        )
    except TypeError:
        return client.create_collection(
            **common,
            metadata={"artifact": artifact.paths.root.name, "hnsw:space": "cosine"},
        )


def validate_probe(collection: Any, artifact: EmbeddingArtifact) -> dict[str, Any]:
    first_group = artifact.parquet.read_row_group(0, columns=["chunk_id"])
    probe_id = str(first_group["chunk_id"][0].as_py())
    source = np.asarray(artifact.vectors[0], dtype=np.float32)
    stored_result = collection.get(ids=[probe_id], include=["embeddings", "metadatas"])
    if not stored_result.get("ids"):
        raise RuntimeError(f"Chroma에 검증 ID가 없습니다: {probe_id}")
    stored = np.asarray(stored_result["embeddings"][0], dtype=np.float32)
    max_abs_error = float(np.max(np.abs(source - stored)))
    query = collection.query(
        query_embeddings=[source.tolist()], n_results=1, include=["distances"],
    )
    top1_id = str(query["ids"][0][0])
    top1_distance = float(query["distances"][0][0])
    result = {
        "probe_id": probe_id,
        "stored_vector_max_abs_error": max_abs_error,
        "self_query_top1_id": top1_id,
        "self_query_top1_distance": top1_distance,
        "self_query_top1_matches": top1_id == probe_id,
    }
    if max_abs_error > 1e-6 or top1_id != probe_id:
        raise RuntimeError(f"Chroma 적재 검증 실패: {result}")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다")
    artifact = EmbeddingArtifact(args.embedding_dir)
    target_rows = min(args.limit, artifact.row_count) if args.limit else artifact.row_count
    if target_rows < 1:
        raise ValueError("적재 대상 행이 없습니다")

    persist_dir = args.persist_dir.expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)
    state_path = persist_dir / "chroma_load_state.json"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_names = {
        item if isinstance(item, str) else item.name for item in client.list_collections()
    }

    if args.rebuild and args.collection in collection_names:
        client.delete_collection(args.collection)
        collection_names.remove(args.collection)
        if state_path.exists():
            state_path.unlink()

    if args.collection in collection_names:
        collection = client.get_collection(args.collection, embedding_function=None)
    else:
        collection = create_collection(client, args.collection, artifact)
        if state_path.exists():
            state_path.unlink()

    completed = collection.count()
    initial_count = completed
    if completed and not state_path.is_file():
        raise RuntimeError("기존 컬렉션은 있지만 상태 파일이 없습니다. 확인 후 --rebuild를 사용하세요")
    if completed:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("artifact_identity") != artifact.identity:
            raise RuntimeError("기존 컬렉션과 현재 임베딩이 다릅니다. --rebuild가 필요합니다")
    if completed > target_rows:
        raise RuntimeError(f"기존 저장 수가 목표보다 큽니다: {completed:,} > {target_rows:,}")

    started = time.perf_counter()
    for start, rows in iter_batches(artifact, args.batch_size, completed, target_rows):
        end = start + len(rows)
        collection.upsert(
            ids=[str(row["chunk_id"]) for row in rows],
            embeddings=np.asarray(artifact.vectors[start:end], dtype=np.float32),
            documents=[str(row.get("text") or "") for row in rows],
            metadatas=[make_metadata(row) for row in rows],
        )
        completed = end
        elapsed = time.perf_counter() - started
        atomic_json(state_path, {
            "status": "running", "database": "chromadb", "collection": args.collection,
            "persist_dir": str(persist_dir), "target_rows": target_rows,
            "completed_rows": completed, "dimension": artifact.dimension,
            "artifact_identity": artifact.identity,
            "elapsed_seconds_current_session": round(elapsed, 3),
        })
        print(f"Chroma 적재: {completed:,}/{target_rows:,}", flush=True)

    final_count = collection.count()
    if final_count != target_rows:
        raise RuntimeError(f"Chroma count 불일치: {final_count:,} != {target_rows:,}")
    probe = validate_probe(collection, artifact)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete", "database": "chromadb", "collection": args.collection,
        "persist_dir": str(persist_dir), "row_count": final_count,
        "dimension": artifact.dimension, "distance": "cosine",
        "artifact_identity": artifact.identity, "probe_validation": probe,
        "elapsed_seconds_current_session": round(elapsed, 3),
        "inserted_rows_current_session": final_count - initial_count,
        "rows_per_second_current_session": round((final_count - initial_count) / elapsed, 3)
        if elapsed and final_count > initial_count else None,
        "chromadb_version": chromadb.__version__,
    }
    atomic_json(state_path, summary)

    print("\n=== ChromaDB 적재 완료 ===")
    print(f"컬렉션: {args.collection} | 저장: {final_count:,}")
    print(f"벡터 최대 오차: {probe['stored_vector_max_abs_error']:.10f}")
    print(f"자기검색 Top-1: {probe['self_query_top1_matches']}")
    print(f"상태: {state_path}")


if __name__ == "__main__":
    main()
