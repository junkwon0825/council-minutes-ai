#!/usr/bin/env python3
"""회의록 청크 임베딩을 이슈 후보 군집으로 묶는다.

04d_embed_selected.py가 만든 정책 전용 정규화 임베딩과 메타데이터를 입력으로 사용한다.
회의록 원문은 외부로 전송하지 않으며, CPU MiniBatchKMeans로 군집화한다.

기본 입력
    data/index/arctic_ko/embeddings.f16.npy
    data/index/arctic_ko/chunk_metadata.parquet

기본 출력
    data/index/arctic_ko/clustering/minibatch_kmeans.joblib
    data/index/arctic_ko/clustering/cluster_centers.f32.npy
    data/index/arctic_ko/clustering/cluster_assignments.parquet
    outputs/clustering/cluster_summary.csv
    outputs/clustering/cluster_representatives.csv
    outputs/clustering/clustering_summary.json

주의
    여기서 만드는 군집명은 TF-IDF 키워드 기반 '임시 이름'이다. 다음 단계에서
    대표 발언과 키워드를 로컬 LLM에 넣어 이슈명을 제안받고 사람이 검수한 뒤
    issue_id/issue_name을 고정한다.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import joblib
except ImportError as error:
    raise SystemExit("joblib이 필요합니다. 실행 전: pip install joblib") from error

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("numpy가 필요합니다. 실행 전: pip install numpy") from error

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error

try:
    import sklearn
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as error:
    raise SystemExit("scikit-learn이 필요합니다. 실행 전: pip install scikit-learn") from error


SCRIPT_VERSION = "2026-10-19-semantic-clustering-v3"

REQUIRED_METADATA_COLUMNS = {
    "embedding_row", "chunk_id", "doc_id", "council", "meeting_name",
    "meeting_date",
}

OUTPUT_METADATA_COLUMNS = [
    "embedding_row", "chunk_id", "parent_chunk_id", "semantic_source_embedding_row",
    "semantic_unit_index", "semantic_unit_count", "doc_id", "council", "committee",
    "meeting_name", "meeting_date", "meeting_year", "chunk_type",
    "agenda_title", "speaker_label", "char_start", "char_end", "evidence_ref",
    "relevance_score", "relevance_reason",
    "semantic_text", "issue_text", "chunk_text",
]

# 형태소 분석기 없이도 회의 진행 상투어가 키워드 상위를 점령하지 않게 하는
# 최소 불용어 목록이다. 프로젝트 후반에는 팀 검수 결과에 따라 확장한다.
STOPWORDS = {
    "가겠습니다", "가지고", "감사합니다", "같습니다", "그것", "그렇게", "그리고",
    "그러니까", "그런데", "그래서", "금년도", "금번", "답변", "대해서", "대한",
    "되겠습니다", "되어", "됩니다", "드리겠습니다", "말씀", "말씀드리겠습니다",
    "말씀해", "말씀을", "먼저", "문제", "바랍니다", "부분", "사실", "생각합니다",
    "설명", "수고하셨습니다", "아니고", "안건", "업무", "없습니다", "예산",
    "오늘", "위원", "위원님", "위원장", "의견", "의원", "의원님", "이것", "이번",
    "있겠습니다", "있고", "있습니다", "저희", "조금", "질문", "질의", "추진",
    "통해서", "하도록", "하겠습니다", "하는", "하여", "하고", "합니다", "현재",
    "회의", "관련", "경우", "정도", "내용", "필요", "검토", "진행", "자료",
}

TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="임베딩 기반 지방의회 이슈 후보 군집화")
    parser.add_argument("--embedding", default="data/index/arctic_ko/embeddings.f16.npy")
    parser.add_argument("--metadata", default="data/index/arctic_ko/chunk_metadata.parquet")
    parser.add_argument("--model-dir", default="data/index/arctic_ko/clustering")
    parser.add_argument("--output-dir", default="outputs/clustering")
    parser.add_argument("--clusters", type=int, default=100, help="생성할 군집 수")
    parser.add_argument(
        "--train-sample", type=int, default=80000,
        help="중심 학습에 사용할 무작위 청크 수. 0이면 전체",
    )
    parser.add_argument("--epochs", type=int, default=2, help="학습 표본 반복 횟수")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--representatives", type=int, default=5, help="군집별 대표 발언 수")
    parser.add_argument(
        "--keyword-samples", type=int, default=200,
        help="군집별 키워드 계산용 reservoir 표본 수",
    )
    parser.add_argument("--keywords", type=int, default=12, help="군집별 출력 키워드 수")
    parser.add_argument(
        "--low-similarity-threshold", type=float, default=0.30,
        help="품질 보고서의 저유사도 기준",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.clusters < 2:
        raise SystemExit("--clusters는 2 이상이어야 합니다.")
    if args.train_sample < 0:
        raise SystemExit("--train-sample은 0 이상이어야 합니다.")
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs와 --batch-size는 1 이상이어야 합니다.")
    if args.representatives < 1 or args.keyword_samples < 1 or args.keywords < 1:
        raise SystemExit("대표 발언/키워드 관련 값은 1 이상이어야 합니다.")
    if not -1.0 <= args.low_similarity_threshold <= 1.0:
        raise SystemExit("--low-similarity-threshold는 -1 이상 1 이하여야 합니다.")


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """float32 변환 후 행별 L2 정규화한다."""
    result = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    np.divide(result, np.maximum(norms, 1e-12), out=result)
    return result


def validate_inputs(embedding_path: Path, metadata_path: Path) -> tuple[np.ndarray, pq.ParquetFile]:
    for path in (embedding_path, metadata_path):
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise SystemExit(f"임베딩 배열 형식이 올바르지 않습니다: {embeddings.shape}")
    metadata = pq.ParquetFile(metadata_path)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.schema_arrow.names)
    if missing:
        raise SystemExit("메타데이터 필수 컬럼 누락: " + ", ".join(sorted(missing)))
    if not ({"semantic_text", "issue_text", "chunk_text"} & set(metadata.schema_arrow.names)):
        raise SystemExit("메타데이터에 군집화 본문 컬럼이 없습니다.")
    if metadata.metadata.num_rows != embeddings.shape[0]:
        raise SystemExit(
            f"벡터/메타데이터 행 수 불일치: {embeddings.shape[0]:,} != "
            f"{metadata.metadata.num_rows:,}"
        )
    return embeddings, metadata


def make_training_rows(row_count: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count == 0 or sample_count >= row_count:
        rows = np.arange(row_count, dtype=np.int64)
    else:
        rows = np.random.default_rng(seed).choice(row_count, size=sample_count, replace=False)
    return np.asarray(rows, dtype=np.int64)


def train_model(
    embeddings: np.ndarray,
    training_rows: np.ndarray,
    clusters: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> MiniBatchKMeans:
    if len(training_rows) < clusters:
        raise SystemExit(
            f"학습 표본({len(training_rows):,})이 군집 수({clusters:,})보다 작습니다."
        )

    model = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        n_init=3,
        batch_size=batch_size,
        random_state=seed,
        reassignment_ratio=0.01,
    )
    rng = np.random.default_rng(seed)
    fitted_batches = 0
    started = time.time()
    for epoch in range(epochs):
        order = training_rows.copy()
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            # memmap 파일의 무작위 디스크 접근을 줄이기 위해 배치 안에서는 정렬한다.
            # KMeans는 배치 내부 순서에 영향을 받지 않는다.
            rows = np.sort(order[start:start + batch_size])
            if fitted_batches == 0 and len(rows) < clusters:
                raise SystemExit("첫 학습 배치가 군집 수보다 작습니다. --batch-size를 늘리세요.")
            model.partial_fit(normalize_rows(embeddings[rows]))
            fitted_batches += 1
        print(
            f"중심 학습 {epoch + 1}/{epochs} 완료 | 표본 {len(order):,}개 | "
            f"누적 {format_duration(time.time() - started)}",
            flush=True,
        )
    return model


def keyword_analyzer(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(str(text or "")):
        token = token.lower()
        if token in STOPWORDS or token.isdigit():
            continue
        tokens.append(token)
    # 연속 두 단어도 후보로 추가한다. 지나치게 긴 문서를 막기 위해 unigram 수를 제한한다.
    tokens = tokens[:30000]
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def extract_keywords(texts_by_cluster: list[list[str]], top_n: int) -> list[list[str]]:
    documents = ["\n".join(texts) if texts else "빈군집" for texts in texts_by_cluster]
    vectorizer = TfidfVectorizer(
        analyzer=keyword_analyzer,
        lowercase=False,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return [[] for _ in documents]
    terms = np.asarray(vectorizer.get_feature_names_out())
    results = []
    for row_index in range(matrix.shape[0]):
        row = matrix.getrow(row_index)
        if row.nnz == 0:
            results.append([])
            continue
        order = np.argsort(row.data)[::-1][:top_n]
        results.append([str(terms[row.indices[index]]) for index in order])
    return results


def safe_text(value) -> str:
    return "" if value is None else str(value)


def shorten(text: str, limit: int = 800) -> str:
    cleaned = re.sub(r"\s+", " ", safe_text(text)).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"


def counter_text(counter: Counter, limit: int = 5) -> str:
    return " | ".join(f"{key}:{value}" for key, value in counter.most_common(limit) if key)


def assign_all(
    embeddings: np.ndarray,
    metadata: pq.ParquetFile,
    model: MiniBatchKMeans,
    assignment_path: Path,
    clusters: int,
    batch_size: int,
    representative_count: int,
    keyword_sample_count: int,
    low_similarity_threshold: float,
    seed: int,
) -> dict:
    available = set(metadata.schema_arrow.names)
    read_columns = [column for column in OUTPUT_METADATA_COLUMNS if column in available]
    for text_column in ("semantic_text", "issue_text", "chunk_text"):
        if text_column in available and text_column not in read_columns:
            read_columns.append(text_column)

    centers = normalize_rows(model.cluster_centers_)
    cluster_sizes = np.zeros(clusters, dtype=np.int64)
    similarity_sums = np.zeros(clusters, dtype=np.float64)
    low_similarity_counts = np.zeros(clusters, dtype=np.int64)
    date_min = [None] * clusters
    date_max = [None] * clusters
    council_counts = [Counter() for _ in range(clusters)]
    year_counts = [Counter() for _ in range(clusters)]
    representative_heaps: list[list[tuple]] = [[] for _ in range(clusters)]
    keyword_texts: list[list[str]] = [[] for _ in range(clusters)]
    keyword_seen = np.zeros(clusters, dtype=np.int64)
    rng = np.random.default_rng(seed + 1)

    temporary = assignment_path.with_suffix(assignment_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = None
    processed = 0
    started = time.time()
    try:
        for record_batch in metadata.iter_batches(batch_size=batch_size, columns=read_columns):
            table = pa.Table.from_batches([record_batch])
            count = table.num_rows
            end = processed + count
            stored_rows = np.asarray(table["embedding_row"].to_numpy(), dtype=np.int64)
            expected_rows = np.arange(processed, end, dtype=np.int64)
            if not np.array_equal(stored_rows, expected_rows):
                raise RuntimeError(f"embedding_row 순서 불일치: {processed:,}:{end:,}")

            vectors = normalize_rows(embeddings[processed:end])
            labels = np.asarray(model.predict(vectors), dtype=np.int32)
            similarities = np.einsum("ij,ij->i", vectors, centers[labels]).astype(np.float32)

            assignment = table.select([column for column in OUTPUT_METADATA_COLUMNS if column in table.column_names])
            assignment = assignment.append_column("cluster_id", pa.array(labels, type=pa.int32()))
            assignment = assignment.append_column(
                "cluster_similarity", pa.array(similarities, type=pa.float32())
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    assignment.schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(assignment)

            rows = table.to_pylist()
            for local_index, (item, cluster_id, similarity) in enumerate(
                zip(rows, labels.tolist(), similarities.tolist())
            ):
                global_row = processed + local_index
                cluster_sizes[cluster_id] += 1
                similarity_sums[cluster_id] += similarity
                if similarity < low_similarity_threshold:
                    low_similarity_counts[cluster_id] += 1

                council = safe_text(item.get("council"))
                if council:
                    council_counts[cluster_id][council] += 1
                date = re.sub(r"\D", "", safe_text(item.get("meeting_date")))[:8]
                if date:
                    date_min[cluster_id] = date if date_min[cluster_id] is None else min(date_min[cluster_id], date)
                    date_max[cluster_id] = date if date_max[cluster_id] is None else max(date_max[cluster_id], date)
                year = safe_text(item.get("meeting_year")) or date[:4]
                if year:
                    year_counts[cluster_id][year] += 1

                # 실제 임베딩 입력인 semantic_text를 키워드·대표문장 선정에 우선 사용한다.
                text = safe_text(
                    item.get("semantic_text") or item.get("issue_text") or item.get("chunk_text")
                )
                rep_item = {
                    "embedding_row": global_row,
                    "similarity": float(similarity),
                    "chunk_id": safe_text(item.get("chunk_id")),
                    "parent_chunk_id": safe_text(item.get("parent_chunk_id")),
                    "semantic_unit_index": item.get("semantic_unit_index"),
                    "semantic_unit_count": item.get("semantic_unit_count"),
                    "doc_id": safe_text(item.get("doc_id")),
                    "council": council,
                    "committee": safe_text(item.get("committee")),
                    "meeting_name": safe_text(item.get("meeting_name")),
                    "meeting_date": safe_text(item.get("meeting_date")),
                    "agenda_title": safe_text(item.get("agenda_title")),
                    "speaker_label": safe_text(item.get("speaker_label")),
                    "char_start": item.get("char_start"),
                    "char_end": item.get("char_end"),
                    "evidence_ref": safe_text(item.get("evidence_ref")),
                    "relevance_score": item.get("relevance_score"),
                    "relevance_reason": safe_text(item.get("relevance_reason")),
                    "semantic_text": shorten(safe_text(item.get("semantic_text"))),
                    "issue_text": shorten(safe_text(item.get("issue_text"))),
                    "chunk_text": shorten(safe_text(item.get("chunk_text"))),
                }
                heap = representative_heaps[cluster_id]
                heap_value = (float(similarity), global_row, rep_item)
                if len(heap) < representative_count:
                    heapq.heappush(heap, heap_value)
                elif heap_value[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, heap_value)

                if text:
                    keyword_seen[cluster_id] += 1
                    seen = int(keyword_seen[cluster_id])
                    samples = keyword_texts[cluster_id]
                    if len(samples) < keyword_sample_count:
                        samples.append(text)
                    else:
                        replacement = int(rng.integers(0, seen))
                        if replacement < keyword_sample_count:
                            samples[replacement] = text

            processed = end
            elapsed = time.time() - started
            rate = processed / elapsed if elapsed else 0.0
            eta = (len(embeddings) - processed) / rate if rate else 0.0
            if processed == len(embeddings) or processed % 50000 < count:
                print(
                    f"전체 배정 {processed:,}/{len(embeddings):,} "
                    f"({processed / len(embeddings) * 100:.1f}%) | "
                    f"{rate:,.0f} chunks/s | ETA {format_duration(eta)}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    if processed != len(embeddings):
        raise RuntimeError(f"전체 배정 행 수 불일치: {processed:,} != {len(embeddings):,}")
    os.replace(temporary, assignment_path)

    return {
        "cluster_sizes": cluster_sizes,
        "similarity_sums": similarity_sums,
        "low_similarity_counts": low_similarity_counts,
        "date_min": date_min,
        "date_max": date_max,
        "council_counts": council_counts,
        "year_counts": year_counts,
        "representative_heaps": representative_heaps,
        "keyword_texts": keyword_texts,
        "elapsed_seconds": time.time() - started,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    embedding_path = Path(args.embedding).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = validate_inputs(embedding_path, metadata_path)
    row_count, dimension = map(int, embeddings.shape)
    training_rows = make_training_rows(row_count, args.train_sample, args.random_state)
    print(f"스크립트: {SCRIPT_VERSION}")
    print(f"입력: {row_count:,}개 x {dimension:,}차원 ({embeddings.dtype})")
    print(
        f"설정: K={args.clusters} | 학습 표본={len(training_rows):,} | "
        f"epochs={args.epochs} | batch={args.batch_size}"
    )
    print("군집화 장치: CPU (기존 임베딩은 재생성하지 않음)", flush=True)

    total_started = time.time()
    training_started = time.time()
    model = train_model(
        embeddings=embeddings,
        training_rows=training_rows,
        clusters=args.clusters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.random_state,
    )
    training_seconds = time.time() - training_started

    model_path = model_dir / "minibatch_kmeans.joblib"
    centers_path = model_dir / "cluster_centers.f32.npy"
    assignment_path = model_dir / "cluster_assignments.parquet"
    joblib.dump(model, model_path)
    np.save(centers_path, np.asarray(model.cluster_centers_, dtype=np.float32))

    stats = assign_all(
        embeddings=embeddings,
        metadata=metadata,
        model=model,
        assignment_path=assignment_path,
        clusters=args.clusters,
        batch_size=args.batch_size,
        representative_count=args.representatives,
        keyword_sample_count=args.keyword_samples,
        low_similarity_threshold=args.low_similarity_threshold,
        seed=args.random_state,
    )

    print("키워드 계산 중...", flush=True)
    keywords = extract_keywords(stats["keyword_texts"], args.keywords)
    cluster_sizes = stats["cluster_sizes"]
    cluster_rows = []
    representative_rows = []
    for cluster_id in range(args.clusters):
        size = int(cluster_sizes[cluster_id])
        mean_similarity = (
            float(stats["similarity_sums"][cluster_id] / size) if size else 0.0
        )
        cluster_keywords = keywords[cluster_id]
        provisional_label = " · ".join(cluster_keywords[:3]) if cluster_keywords else "검수 필요"
        cluster_rows.append({
            "cluster_id": cluster_id,
            "provisional_label": provisional_label,
            "size": size,
            "share_pct": round(size / row_count * 100, 4),
            "mean_similarity": round(mean_similarity, 6),
            "low_similarity_count": int(stats["low_similarity_counts"][cluster_id]),
            "low_similarity_pct": round(
                int(stats["low_similarity_counts"][cluster_id]) / size * 100, 4
            ) if size else 0.0,
            "first_date": stats["date_min"][cluster_id] or "",
            "last_date": stats["date_max"][cluster_id] or "",
            "council_count": len(stats["council_counts"][cluster_id]),
            "top_councils": counter_text(stats["council_counts"][cluster_id]),
            "year_distribution": counter_text(stats["year_counts"][cluster_id], limit=10),
            "keywords": " | ".join(cluster_keywords),
            "keyword_sample_count": len(stats["keyword_texts"][cluster_id]),
        })

        representatives = sorted(
            stats["representative_heaps"][cluster_id],
            key=lambda value: (value[0], value[1]),
            reverse=True,
        )
        for rank, (_, _, item) in enumerate(representatives, start=1):
            representative_rows.append({
                "cluster_id": cluster_id,
                "provisional_label": provisional_label,
                "representative_rank": rank,
                **item,
            })

    cluster_summary_path = output_dir / "cluster_summary.csv"
    representatives_path = output_dir / "cluster_representatives.csv"
    write_csv(cluster_summary_path, list(cluster_rows[0].keys()), cluster_rows)
    rep_fields = list(representative_rows[0].keys()) if representative_rows else [
        "cluster_id", "provisional_label", "representative_rank"
    ]
    write_csv(representatives_path, rep_fields, representative_rows)

    nonempty_sizes = cluster_sizes[cluster_sizes > 0]
    total_seconds = time.time() - total_started
    summary = {
        "script_version": SCRIPT_VERSION,
        "security": "기존 로컬 임베딩만 사용하며 회의록 데이터 외부 전송 없음",
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "row_count": row_count,
        "dimension": dimension,
        "source_dtype": str(embeddings.dtype),
        "algorithm": "MiniBatchKMeans",
        "cluster_count": args.clusters,
        "training_sample_count": int(len(training_rows)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "random_state": args.random_state,
        "representatives_per_cluster": args.representatives,
        "keyword_samples_per_cluster": args.keyword_samples,
        "keyword_method": "cluster reservoir sample + TF-IDF unigram/bigram",
        "provisional_label_warning": "키워드 기반 임시 이름이며 로컬 LLM 제안과 사람 검수 후 고정해야 함",
        "low_similarity_threshold": args.low_similarity_threshold,
        "low_similarity_count": int(stats["low_similarity_counts"].sum()),
        "empty_cluster_count": int((cluster_sizes == 0).sum()),
        "cluster_size_min": int(nonempty_sizes.min()) if len(nonempty_sizes) else 0,
        "cluster_size_median": float(np.median(nonempty_sizes)) if len(nonempty_sizes) else 0.0,
        "cluster_size_max": int(nonempty_sizes.max()) if len(nonempty_sizes) else 0,
        "mean_assigned_similarity": round(
            float(stats["similarity_sums"].sum() / row_count), 6
        ),
        "training_elapsed_seconds": round(training_seconds, 3),
        "assignment_elapsed_seconds": round(stats["elapsed_seconds"], 3),
        "total_elapsed_seconds": round(total_seconds, 3),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "scikit_learn_version": sklearn.__version__,
        "generated_files": {
            "model": str(model_path),
            "centers": str(centers_path),
            "assignments": str(assignment_path),
            "cluster_summary": str(cluster_summary_path),
            "representatives": str(representatives_path),
        },
    }
    summary_path = output_dir / "clustering_summary.json"
    atomic_write_json(summary_path, summary)

    print("\n완료")
    print(f"군집 모델: {model_path}")
    print(f"청크별 군집: {assignment_path}")
    print(f"군집 요약: {cluster_summary_path}")
    print(f"대표 발언: {representatives_path}")
    print(f"실행 요약: {summary_path}")
    print(f"총 소요: {format_duration(total_seconds)}")


if __name__ == "__main__":
    main()
