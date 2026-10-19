#!/usr/bin/env python3
"""04_embedding.py의 정규화 벡터로 정확 검색용 FAISS 인덱스를 만든다.

기본 입력
    data/index/arctic_ko/embeddings.f16.npy
    data/index/arctic_ko/chunk_metadata.parquet

기본 출력
    data/index/arctic_ko/faiss_flat_ip.index
    outputs/faiss/faiss_summary.json

IndexFlatIP는 근사 검색이 아닌 정확 검색이다. 벡터를 L2 정규화한 뒤
Inner Product를 사용하므로 검색 점수는 cosine similarity와 같다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import faiss
except ImportError as error:
    raise SystemExit("faiss가 필요합니다. 실행 전: pip install faiss-cpu") from error

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("numpy가 필요합니다. 실행 전: pip install numpy") from error

try:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정규화 임베딩으로 FAISS FlatIP 인덱스 생성")
    parser.add_argument("--embedding", default="data/index/arctic_ko/embeddings.f16.npy")
    parser.add_argument("--metadata", default="data/index/arctic_ko/chunk_metadata.parquet")
    parser.add_argument("--output", default="data/index/arctic_ko/faiss_flat_ip.index")
    parser.add_argument("--report-dir", default="outputs/faiss")
    parser.add_argument("--batch-size", type=int, default=8192, help="한 번에 FAISS에 추가할 벡터 수")
    parser.add_argument("--self-test-count", type=int, default=5, help="완료 후 자기 자신 검색 테스트 수")
    return parser.parse_args()


def human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_metadata(metadata_path: Path, vector_count: int) -> None:
    source = pq.ParquetFile(metadata_path)
    if source.metadata.num_rows != vector_count:
        raise SystemExit(
            f"벡터/메타데이터 행 수 불일치: {vector_count:,} != {source.metadata.num_rows:,}"
        )
    if "embedding_row" not in source.schema_arrow.names:
        raise SystemExit("메타데이터에 embedding_row 컬럼이 없습니다.")

    first = pq.read_table(metadata_path, columns=["embedding_row"]).slice(0, 1)
    last = pq.read_table(metadata_path, columns=["embedding_row"]).slice(vector_count - 1, 1)
    if not first.num_rows or not last.num_rows:
        raise SystemExit("메타데이터가 비어 있습니다.")
    if first["embedding_row"][0].as_py() != 0 or last["embedding_row"][0].as_py() != vector_count - 1:
        raise SystemExit("embedding_row가 0부터 연속적으로 정렬되어 있지 않습니다.")


def validate_existing_index(index_path: Path, vector_count: int, dimension: int) -> bool:
    if not index_path.exists():
        return False
    index = faiss.read_index(str(index_path))
    if index.ntotal != vector_count or index.d != dimension:
        raise SystemExit(
            f"기존 FAISS 인덱스가 현재 벡터와 다릅니다: "
            f"ntotal={index.ntotal:,}, d={index.d} / expected={vector_count:,}, {dimension}\n"
            "기존 파일을 백업한 뒤 다시 실행하세요."
        )
    print(f"이미 정상 인덱스가 있습니다: {index_path}")
    print(f"벡터: {index.ntotal:,}개 | 차원: {index.d:,}")
    return True


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")
    if args.self_test_count < 1:
        raise SystemExit("--self-test-count는 1 이상이어야 합니다.")

    embedding_path = Path(args.embedding).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    index_path = Path(args.output).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    for path in (embedding_path, metadata_path):
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise SystemExit(f"임베딩 배열은 2차원이어야 합니다: {embeddings.shape}")
    vector_count, dimension = map(int, embeddings.shape)
    if vector_count == 0 or dimension == 0:
        raise SystemExit("임베딩 배열이 비어 있습니다.")
    validate_metadata(metadata_path, vector_count)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "faiss_summary.json"
    if validate_existing_index(index_path, vector_count, dimension):
        return

    print(f"입력 벡터: {vector_count:,} x {dimension:,} ({embeddings.dtype})", flush=True)
    print("인덱스: IndexFlatIP (정확 cosine 검색)", flush=True)
    print(f"예상 인덱스 크기: 약 {human_bytes(vector_count * dimension * 4)}", flush=True)

    index = faiss.IndexFlatIP(dimension)
    started_at = time.time()
    for start in range(0, vector_count, args.batch_size):
        end = min(start + args.batch_size, vector_count)
        vectors = np.asarray(embeddings[start:end], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise RuntimeError(f"NaN 또는 inf 벡터 발견: rows {start}:{end}")
        faiss.normalize_L2(vectors)
        index.add(vectors)

        done = end
        elapsed = time.time() - started_at
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = vector_count - done
        eta = remaining / rate if rate > 0 else 0.0
        if done == vector_count or done % 100000 < args.batch_size:
            print(
                f"FAISS {done:,}/{vector_count:,} ({done / vector_count * 100:.2f}%) "
                f"| {rate:,.0f} vectors/s | ETA {format_duration(eta)}",
                flush=True,
            )

    if index.ntotal != vector_count:
        raise RuntimeError(f"FAISS 추가 개수 불일치: {index.ntotal:,} != {vector_count:,}")

    sample_rows = np.linspace(0, vector_count - 1, min(args.self_test_count, vector_count), dtype=np.int64)
    sample_vectors = np.asarray(embeddings[sample_rows], dtype=np.float32)
    faiss.normalize_L2(sample_vectors)
    scores, indices = index.search(sample_vectors, min(5, vector_count))
    self_tests = []
    for query_row, found_rows, found_scores in zip(sample_rows, indices, scores):
        query_row = int(query_row)
        found_list = [int(value) for value in found_rows]
        self_tests.append({
            "query_row": query_row,
            "top_rows": found_list,
            "top_scores": [round(float(value), 7) for value in found_scores],
            "query_row_in_top5": query_row in found_list,
        })
    if not all(item["query_row_in_top5"] for item in self_tests):
        raise RuntimeError("FAISS 자기 자신 검색 검증에 실패했습니다.")

    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    faiss.write_index(index, str(temporary_path))
    os.replace(temporary_path, index_path)

    elapsed = time.time() - started_at
    summary = {
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "index_path": str(index_path),
        "index_type": "IndexFlatIP",
        "similarity": "cosine (L2 normalized vectors + inner product)",
        "vector_count": vector_count,
        "dimension": dimension,
        "source_dtype": str(embeddings.dtype),
        "faiss_storage_dtype": "float32",
        "index_size_bytes": index_path.stat().st_size,
        "elapsed_seconds": round(elapsed, 3),
        "self_tests": self_tests,
    }
    atomic_write_json(summary_path, summary)

    print("\n완료")
    print(f"FAISS 인덱스: {index_path} ({human_bytes(index_path.stat().st_size)})")
    print(f"요약: {summary_path}")


if __name__ == "__main__":
    main()
