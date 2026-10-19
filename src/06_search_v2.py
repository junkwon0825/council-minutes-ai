#!/usr/bin/env python3
"""FAISS 인덱스에서 한국어 질의와 유사한 지방의회 정책 이슈를 검색한다.

04d_embed_selected.py와 동일한 모델을 로컬에서 불러오며 회의록 원문을
외부로 전송하지 않는다. 검색과 중복 제거에는 실제 임베딩 입력인
semantic_text를 우선 사용하고, 근거 확인용 원문도 함께 출력한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
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
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit("sentence-transformers가 필요합니다. 실행 전: pip install sentence-transformers") from error


DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9"
SCRIPT_VERSION = "2026-10-19-semantic-search-v4"
RESULT_COLUMNS = [
    "embedding_row", "chunk_id", "parent_chunk_id", "semantic_source_embedding_row",
    "semantic_unit_index", "semantic_unit_count", "doc_id", "source_file", "source_record_index",
    "council", "committee", "meeting_name", "meeting_date", "meeting_type",
    "chunk_type", "agenda_title", "speaker_label", "speaker_role", "speaker_name",
    "char_start", "char_end", "chunk_char_count", "embedding_priority",
    "evidence_ref", "relevance_score", "relevance_reason", "semantic_topic_hint",
    "semantic_text", "issue_text", "chunk_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAISS 지방의회 발언 검색")
    parser.add_argument("--query", action="append", default=[], help="검색 질의. 여러 번 지정 가능")
    parser.add_argument("--query-file", default="", help="한 줄에 질의 하나인 UTF-8 파일")
    parser.add_argument("--index", default="data/index/arctic_ko/faiss_flat_ip.index")
    parser.add_argument("--metadata", default="data/index/arctic_ko/chunk_metadata.parquet")
    parser.add_argument("--output-dir", default="outputs/search")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100, help="중복 제거 전 FAISS 후보 수")
    parser.add_argument("--max-per-doc", type=int, default=2, help="동일 회의록 최대 결과 수, 0이면 제한 없음")
    parser.add_argument("--max-per-council", type=int, default=0, help="동일 의회 최대 결과 수, 0이면 제한 없음")
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.97,
        help="정규화 본문 유사도가 이 값 이상이면 중복으로 제외, 0이면 사용 안 함",
    )
    parser.add_argument("--no-query-prompt", action="store_true", help="모델의 query 프롬프트를 사용하지 않음")
    return parser.parse_args()


def read_queries(args: argparse.Namespace) -> list[str]:
    queries = [value.strip() for value in args.query if value.strip()]
    if args.query_file:
        query_path = Path(args.query_file).expanduser().resolve()
        if not query_path.exists():
            raise SystemExit(f"질의 파일이 없습니다: {query_path}")
        queries.extend(line.strip() for line in query_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    unique = []
    for query in queries:
        if query not in unique:
            unique.append(query)
    if not unique:
        raise SystemExit("--query 또는 --query-file로 검색어를 입력하세요.")
    return unique


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise SystemExit("CUDA를 사용할 수 없습니다. PyTorch CUDA 설치와 nvidia-smi를 확인하세요.")
    return requested


def load_model(args: argparse.Namespace, device: str) -> SentenceTransformer:
    cache = Path(args.model_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"질의 모델 로딩: {args.model_name} @ {args.revision} | {device}", flush=True)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        cache_folder=str(cache),
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model.max_seq_length = 2048
    model.eval()
    return model


def encode_queries(model: SentenceTransformer, queries: list[str], use_prompt: bool) -> tuple[np.ndarray, str | None]:
    kwargs = {
        "batch_size": min(16, len(queries)),
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    prompt_name = None
    prompts = getattr(model, "prompts", {}) or {}
    if use_prompt and "query" in prompts:
        kwargs["prompt_name"] = "query"
        prompt_name = "query"
    elif use_prompt:
        print("경고: 모델 설정에 query 프롬프트가 없어 프롬프트 없이 검색합니다.", file=sys.stderr)
    vectors = model.encode(queries, **kwargs)
    return np.asarray(vectors, dtype=np.float32), prompt_name


class ParquetRowAccessor:
    """요청한 전역 행 번호가 포함된 row group만 읽는다."""

    def __init__(self, path: Path):
        self.source = pq.ParquetFile(path)
        self.starts = []
        position = 0
        for index in range(self.source.num_row_groups):
            self.starts.append(position)
            position += self.source.metadata.row_group(index).num_rows
        self.row_count = position
        available = set(self.source.schema_arrow.names)
        missing = {"embedding_row", "chunk_id", "doc_id", "council", "meeting_name", "meeting_date"} - available
        if missing:
            raise SystemExit("메타데이터 필수 컬럼 누락: " + ", ".join(sorted(missing)))
        if not ({"semantic_text", "issue_text", "chunk_text"} & available):
            raise SystemExit("메타데이터에 검색 본문 컬럼이 없습니다.")
        self.columns = [column for column in RESULT_COLUMNS if column in available]

    def _row_group_for(self, row: int) -> int:
        import bisect
        return bisect.bisect_right(self.starts, row) - 1

    def read_rows(self, rows: list[int]) -> dict[int, dict]:
        requested = sorted(set(int(row) for row in rows if 0 <= int(row) < self.row_count))
        grouped = defaultdict(list)
        for row in requested:
            grouped[self._row_group_for(row)].append(row)

        result = {}
        for group_index, global_rows in grouped.items():
            table = self.source.read_row_group(group_index, columns=self.columns)
            group_start = self.starts[group_index]
            local_rows = [row - group_start for row in global_rows]
            selected = table.take(pa.array(local_rows, type=pa.int64())).to_pylist()
            for requested_row, item in zip(global_rows, selected):
                stored_row = int(item.get("embedding_row", requested_row))
                if stored_row != requested_row:
                    raise RuntimeError(f"embedding_row 순서 불일치: {stored_row} != {requested_row}")
                result[requested_row] = item
        return result


def shorten(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def canonical_text(text: str) -> str:
    """공백·문장부호 차이만 있는 복제 회의록을 같은 본문으로 비교한다."""
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def select_results(
    candidate_rows: list[int],
    candidate_scores: list[float],
    metadata: dict[int, dict],
    top_k: int,
    max_per_doc: int,
    max_per_council: int,
    near_duplicate_threshold: float,
) -> tuple[list[dict], dict]:
    selected = []
    filter_counts = Counter()
    doc_counts = Counter()
    council_counts = Counter()
    seen_chunk_ids = set()
    selected_canonical_texts = []

    for row, score in zip(candidate_rows, candidate_scores):
        if row < 0 or row not in metadata:
            filter_counts["missing_metadata"] += 1
            continue
        item = metadata[row]
        chunk_id = str(item.get("chunk_id") or row)
        doc_id = str(item.get("doc_id") or "")
        council = str(item.get("council") or "")
        if chunk_id in seen_chunk_ids:
            filter_counts["duplicate_chunk_id"] += 1
            continue
        if max_per_doc and doc_counts[doc_id] >= max_per_doc:
            filter_counts["document_limit"] += 1
            continue
        if max_per_council and council_counts[council] >= max_per_council:
            filter_counts["council_limit"] += 1
            continue

        normalized_text = canonical_text(
            item.get("semantic_text") or item.get("issue_text") or item.get("chunk_text", "")
        )
        if normalized_text and any(normalized_text == previous for previous in selected_canonical_texts):
            filter_counts["duplicate_exact_text"] += 1
            continue
        if near_duplicate_threshold and normalized_text and any(
            SequenceMatcher(None, normalized_text, previous).ratio() >= near_duplicate_threshold
            for previous in selected_canonical_texts
        ):
            filter_counts["duplicate_near_text"] += 1
            continue

        selected.append({**item, "embedding_row": row, "similarity": round(float(score), 7)})
        seen_chunk_ids.add(chunk_id)
        if normalized_text:
            selected_canonical_texts.append(normalized_text)
        doc_counts[doc_id] += 1
        council_counts[council] += 1
        if len(selected) >= top_k:
            break
    filter_counts["selected"] = len(selected)
    return selected, dict(filter_counts)


def write_outputs(output_dir: Path, payload: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "search_results.json"
    csv_path = output_dir / "search_results.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "script_version", "query", "rank", "similarity", "embedding_row", "chunk_id", "doc_id", "council",
        "committee", "meeting_name", "meeting_date", "chunk_type", "agenda_title",
        "speaker_label", "char_start", "char_end", "evidence_ref",
        "parent_chunk_id", "semantic_unit_index", "semantic_unit_count",
        "relevance_score", "relevance_reason", "semantic_topic_hint",
        "semantic_text", "issue_text", "chunk_text",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for query_result in payload["queries"]:
            for rank, item in enumerate(query_result["results"], start=1):
                writer.writerow({
                    "script_version": payload["script_version"],
                    "query": query_result["query"],
                    "rank": rank,
                    **item,
                })
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    queries = read_queries(args)
    if args.top_k < 1 or args.candidate_k < args.top_k:
        raise SystemExit("--top-k는 1 이상이고 --candidate-k는 --top-k 이상이어야 합니다.")
    if args.max_per_doc < 0 or args.max_per_council < 0:
        raise SystemExit("결과 제한 값은 0 이상이어야 합니다.")
    if not 0 <= args.near_duplicate_threshold <= 1:
        raise SystemExit("--near-duplicate-threshold는 0 이상 1 이하여야 합니다.")

    index_path = Path(args.index).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (index_path, metadata_path):
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")

    print(f"FAISS 로딩: {index_path}", flush=True)
    print(f"검색 스크립트 버전: {SCRIPT_VERSION}", flush=True)
    index = faiss.read_index(str(index_path))
    accessor = ParquetRowAccessor(metadata_path)
    if index.ntotal != accessor.row_count:
        raise SystemExit(f"인덱스/메타데이터 행 수 불일치: {index.ntotal:,} != {accessor.row_count:,}")

    device = resolve_device(args.device)
    model = load_model(args, device)
    query_vectors, prompt_name = encode_queries(model, queries, not args.no_query_prompt)
    if query_vectors.shape[1] != index.d:
        raise SystemExit(f"질의/인덱스 차원 불일치: {query_vectors.shape[1]} != {index.d}")

    candidate_k = min(args.candidate_k, index.ntotal)
    scores, indices = index.search(query_vectors, candidate_k)
    payload = {
        "script_version": SCRIPT_VERSION,
        "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "model_name": args.model_name,
        "model_revision": args.revision,
        "query_prompt": prompt_name,
        "index_vector_count": int(index.ntotal),
        "dimension": int(index.d),
        "top_k": args.top_k,
        "candidate_k": candidate_k,
        "max_per_doc": args.max_per_doc,
        "max_per_council": args.max_per_council,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "queries": [],
    }

    for query, query_rows, query_scores in zip(queries, indices, scores):
        row_list = [int(value) for value in query_rows if int(value) >= 0]
        score_list = [float(value) for value, row in zip(query_scores, query_rows) if int(row) >= 0]
        metadata = accessor.read_rows(row_list)
        results, filter_counts = select_results(
            candidate_rows=row_list,
            candidate_scores=score_list,
            metadata=metadata,
            top_k=args.top_k,
            max_per_doc=args.max_per_doc,
            max_per_council=args.max_per_council,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
        payload["queries"].append({
            "query": query,
            "query_id": hashlib.sha1(query.encode()).hexdigest()[:12],
            "filter_counts": filter_counts,
            "results": results,
        })

        print(f"\n=== {query} ===")
        print(
            "후처리: "
            f"exact 중복 {filter_counts.get('duplicate_exact_text', 0)}개 | "
            f"근접 중복 {filter_counts.get('duplicate_near_text', 0)}개 | "
            f"문서 제한 {filter_counts.get('document_limit', 0)}개 | "
            f"의회 제한 {filter_counts.get('council_limit', 0)}개"
        )
        for rank, item in enumerate(results, start=1):
            print(
                f"[{rank:02d}] score={item['similarity']:.4f} | "
                f"{item.get('council', '')} | {item.get('meeting_date', '')} | "
                f"{item.get('meeting_name', '')}"
            )
            context = " / ".join(value for value in (item.get("agenda_title"), item.get("speaker_label")) if value)
            if context:
                print(f"     {context}")
            print(
                f"     relevance={item.get('relevance_score', '')} | "
                f"{shorten(item.get('semantic_text') or item.get('issue_text') or item.get('chunk_text', ''))}"
            )

    json_path, csv_path = write_outputs(output_dir, payload)
    print("\n완료")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
