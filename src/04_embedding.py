#!/usr/bin/env python3
"""회의록 청크를 로컬 한국어 임베딩 벡터로 변환한다.

기본 모델
    dragonkue/snowflake-arctic-embed-l-v2.0-ko

보안 원칙
    Hugging Face에서 모델 파일만 내려받는다. 회의록 텍스트는 외부 API로
    전송하지 않으며 로컬 SentenceTransformer 추론만 수행한다.

주요 출력
    data/index/arctic_ko/embeddings.f16.npy
    data/index/arctic_ko/chunk_metadata.parquet
    data/index/arctic_ko/embedding_state.json
    outputs/embedding/embedding_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("numpy가 필요합니다. 실행 전: pip install numpy") from error

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9"

EMBED_INPUT_COLUMNS = ["chunk_id", "embedding_priority", "embedding_text"]
METADATA_COLUMNS = [
    "chunk_id", "doc_id", "chunk_index", "source_file", "source_record_index",
    "council", "committee", "meeting_name", "meeting_date", "meeting_year",
    "meeting_month", "meeting_type", "chunk_type", "agenda_title", "speaker_label",
    "speaker_role", "speaker_name", "char_start", "char_end", "chunk_char_count",
    "estimated_tokens", "embedding_priority", "evidence_ref", "chunk_sha256", "chunk_text",
]
REQUIRED_COLUMNS = sorted(set(EMBED_INPUT_COLUMNS + METADATA_COLUMNS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="로컬 한국어 회의록 임베딩 생성")
    parser.add_argument("--input", default="data/processed/chunks.parquet", help="03번 청크 Parquet")
    parser.add_argument("--output-dir", default="data/index/arctic_ko", help="벡터/메타데이터 저장 폴더")
    parser.add_argument("--report-dir", default="outputs/embedding", help="실행 보고서 저장 폴더")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Hugging Face 모델명 또는 로컬 경로")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="모델 커밋/태그. 재현성을 위해 고정")
    parser.add_argument("--model-cache", default="models/huggingface", help="모델 다운로드 캐시")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU 추론 배치. OOM 시 자동 감소")
    parser.add_argument("--encode-buffer-size", type=int, default=512, help="한 번에 encode로 전달할 문서 수")
    parser.add_argument("--read-batch-size", type=int, default=4096, help="Parquet 읽기 배치")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="토크나이저 최대 길이")
    parser.add_argument(
        "--priorities",
        default="high,medium",
        help="임베딩할 우선순위. 기본값은 개회식/UI 중심 low 제외",
    )
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체, 양수면 앞 N개만 처리")
    parser.add_argument("--prepare-only", action="store_true", help="메타데이터 매니페스트만 생성")
    parser.add_argument("--local-files-only", action="store_true", help="이미 받은 모델만 사용하고 네트워크 차단")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")
    if args.encode_buffer_size < args.batch_size:
        raise SystemExit("--encode-buffer-size는 --batch-size 이상이어야 합니다.")
    if args.read_batch_size < 1:
        raise SystemExit("--read-batch-size는 1 이상이어야 합니다.")
    if args.max_seq_length < 128:
        raise SystemExit("--max-seq-length는 128 이상이어야 합니다.")
    if args.limit < 0:
        raise SystemExit("--limit은 0 이상이어야 합니다.")


def parse_priorities(value: str) -> list[str]:
    allowed = {"high", "medium", "low"}
    priorities = []
    for item in value.split(","):
        item = item.strip().lower()
        if item and item not in priorities:
            priorities.append(item)
    unknown = set(priorities) - allowed
    if unknown:
        raise SystemExit(f"알 수 없는 priority: {', '.join(sorted(unknown))}")
    if not priorities:
        raise SystemExit("--priorities에 하나 이상 지정해야 합니다.")
    return priorities


def atomic_write_json(path: Path, data: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def source_signature(input_path: Path, source: pq.ParquetFile) -> dict:
    stat = input_path.stat()
    return {
        "input_path": str(input_path),
        "input_size_bytes": stat.st_size,
        "input_row_count": source.metadata.num_rows,
        "input_schema": str(source.schema_arrow),
    }


def check_required_columns(source: pq.ParquetFile) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in source.schema_arrow.names]
    if missing:
        raise SystemExit(f"입력 Parquet 필수 컬럼 누락: {', '.join(missing)}")


def filter_priorities(table: pa.Table, priorities: list[str]) -> pa.Table:
    mask = pc.is_in(table["embedding_priority"], value_set=pa.array(priorities, type=pa.string()))
    return table.filter(mask)


def prepare_manifest(
    source: pq.ParquetFile,
    metadata_path: Path,
    priorities: list[str],
    read_batch_size: int,
    limit: int,
) -> tuple[int, str]:
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    writer = None
    selected_count = 0
    digest = hashlib.sha256()
    try:
        for batch in source.iter_batches(batch_size=read_batch_size, columns=METADATA_COLUMNS):
            table = filter_priorities(pa.Table.from_batches([batch]), priorities)
            if table.num_rows == 0:
                continue
            if limit:
                remaining = limit - selected_count
                if remaining <= 0:
                    break
                table = table.slice(0, remaining)

            chunk_ids = table["chunk_id"].to_pylist()
            chunk_hashes = table["chunk_sha256"].to_pylist()
            for chunk_id, chunk_hash in zip(chunk_ids, chunk_hashes):
                digest.update(str(chunk_id).encode("utf-8"))
                digest.update(b"\x1f")
                digest.update(str(chunk_hash).encode("utf-8"))
                digest.update(b"\n")

            rows = pa.array(range(selected_count, selected_count + table.num_rows), type=pa.int64())
            table = table.add_column(0, "embedding_row", rows)
            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table)
            selected_count += table.num_rows
            if selected_count % 50000 < table.num_rows:
                print(f"매니페스트 준비: {selected_count:,}개", flush=True)
            if limit and selected_count >= limit:
                break
    finally:
        if writer is not None:
            writer.close()

    if selected_count == 0 or not temp_path.exists():
        raise SystemExit("조건에 맞는 임베딩 대상 청크가 없습니다.")
    temp_path.replace(metadata_path)
    return selected_count, digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다. PyTorch CUDA 설치와 nvidia-smi를 확인하세요.")
    return requested


def resolve_precision(requested: str, device: str) -> tuple[str, torch.dtype]:
    if requested == "float32":
        return "float32", torch.float32
    if requested == "bf16":
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise SystemExit("현재 장치에서 BF16을 지원하지 않습니다. --precision float32를 사용하세요.")
        return "bf16", torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "float32", torch.float32


def load_model(args: argparse.Namespace, device: str, torch_dtype: torch.dtype) -> SentenceTransformer:
    model_cache = Path(args.model_cache).expanduser().resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    print(f"모델 로딩: {args.model_name} @ {args.revision}", flush=True)
    print(f"장치: {device} | 연산 dtype: {torch_dtype}", flush=True)

    common_kwargs = {
        "device": device,
        "cache_folder": str(model_cache),
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    try:
        model = SentenceTransformer(
            args.model_name,
            model_kwargs={"torch_dtype": torch_dtype},
            **common_kwargs,
        )
    except TypeError:
        # 구버전 sentence-transformers 호환 경로
        model = SentenceTransformer(args.model_name, **common_kwargs)
        model.to(dtype=torch_dtype)

    model.max_seq_length = args.max_seq_length
    model.eval()
    return model


def is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def encode_with_backoff(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, int]:
    current_batch_size = batch_size
    while True:
        try:
            vectors = model.encode(
                texts,
                batch_size=current_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32), current_batch_size
        except RuntimeError as error:
            if device != "cuda" or not is_oom_error(error) or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
            print(f"GPU 메모리 부족: 배치를 {current_batch_size}로 낮춰 재시도합니다.", flush=True)


def iter_selected_texts(
    source: pq.ParquetFile,
    priorities: list[str],
    read_batch_size: int,
    limit: int,
    skip: int,
):
    selected_position = 0
    yielded = 0
    for batch in source.iter_batches(batch_size=read_batch_size, columns=EMBED_INPUT_COLUMNS):
        table = filter_priorities(pa.Table.from_batches([batch]), priorities)
        if table.num_rows == 0:
            continue
        if limit:
            remaining_total = limit - selected_position
            if remaining_total <= 0:
                break
            table = table.slice(0, remaining_total)

        batch_start = selected_position
        batch_end = selected_position + table.num_rows
        selected_position = batch_end
        if batch_end <= skip:
            continue

        local_start = max(0, skip - batch_start)
        if local_start:
            table = table.slice(local_start)
        chunk_ids = table["chunk_id"].to_pylist()
        texts = table["embedding_text"].to_pylist()
        for chunk_id, text in zip(chunk_ids, texts):
            yield str(chunk_id), "" if text is None else str(text)
            yielded += 1
            if limit and skip + yielded >= limit:
                return


def state_compatible(state: dict, expected: dict) -> tuple[bool, list[str]]:
    keys = [
        "input_path", "input_size_bytes", "input_row_count", "model_name", "revision",
        "priorities", "limit", "output_dtype", "max_seq_length",
    ]
    differences = [key for key in keys if state.get(key) != expected.get(key)]
    return not differences, differences


def write_samples(
    metadata_path: Path,
    embeddings: np.ndarray,
    output_path: Path,
    sample_count: int = 20,
) -> None:
    table = pq.read_table(
        metadata_path,
        columns=["embedding_row", "chunk_id", "council", "meeting_name", "meeting_date", "embedding_priority"],
    ).slice(0, sample_count)
    rows = []
    for item in table.to_pylist():
        row_index = int(item["embedding_row"])
        vector = np.asarray(embeddings[row_index], dtype=np.float32)
        rows.append({
            **item,
            "vector_norm": round(float(np.linalg.norm(vector)), 6),
            "first_8_values": " ".join(f"{value:.5f}" for value in vector[:8]),
        })
    fieldnames = [
        "embedding_row", "chunk_id", "council", "meeting_name", "meeting_date",
        "embedding_priority", "vector_norm", "first_8_values",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    priorities = parse_priorities(args.priorities)

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "chunk_metadata.parquet"
    embeddings_path = output_dir / ("embeddings.f16.npy" if args.output_dtype == "float16" else "embeddings.f32.npy")
    state_path = output_dir / "embedding_state.json"
    summary_path = report_dir / "embedding_summary.json"
    samples_path = report_dir / "embedding_samples.csv"

    source = pq.ParquetFile(input_path)
    check_required_columns(source)
    signature = source_signature(input_path, source)
    expected_state = {
        **signature,
        "model_name": args.model_name,
        "revision": args.revision,
        "priorities": priorities,
        "limit": args.limit,
        "output_dtype": args.output_dtype,
        "max_seq_length": args.max_seq_length,
    }

    state = load_json(state_path) if state_path.exists() else None
    if state:
        compatible, differences = state_compatible(state, expected_state)
        if not compatible:
            raise SystemExit(
                "기존 실행 상태와 현재 옵션이 다릅니다: " + ", ".join(differences)
                + "\n다른 --output-dir을 사용하거나 기존 결과를 별도로 이동한 뒤 다시 실행하세요."
            )
        if not metadata_path.exists():
            raise SystemExit("상태 파일은 있지만 chunk_metadata.parquet가 없습니다.")
        selected_count = int(state["selected_count"])
        print(f"기존 매니페스트 사용: {selected_count:,}개", flush=True)
    else:
        if metadata_path.exists() or embeddings_path.exists():
            raise SystemExit(
                "기존 출력 파일은 있지만 embedding_state.json이 없습니다. "
                "다른 --output-dir을 사용하세요."
            )
        print(f"임베딩 대상 priority: {', '.join(priorities)}", flush=True)
        selected_count, selected_digest = prepare_manifest(
            source=source,
            metadata_path=metadata_path,
            priorities=priorities,
            read_batch_size=args.read_batch_size,
            limit=args.limit,
        )
        state = {
            **expected_state,
            "phase": "manifest_ready",
            "selected_count": selected_count,
            "selected_chunk_digest_sha256": selected_digest,
            "processed_count": 0,
            "dimension": None,
            "effective_batch_size": args.batch_size,
            "created_at_unix": time.time(),
            "updated_at_unix": time.time(),
        }
        atomic_write_json(state_path, state)
        print(f"매니페스트 완료: {selected_count:,}개", flush=True)

    if args.prepare_only:
        print("--prepare-only 완료. 모델 추론은 실행하지 않았습니다.")
        print(f"메타데이터: {metadata_path}")
        return

    if torch is None:
        raise SystemExit("PyTorch가 필요합니다. CUDA 버전에 맞는 torch를 설치하세요.")
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers가 필요합니다. 실행 전: pip install -U sentence-transformers")

    if state.get("phase") == "complete" and embeddings_path.exists() and summary_path.exists():
        print("이미 임베딩이 완료된 상태입니다.")
        print(f"벡터: {embeddings_path}")
        print(f"요약: {summary_path}")
        return

    device = resolve_device(args.device)
    precision_name, torch_dtype = resolve_precision(args.precision, device)
    if device == "cpu":
        print("경고: CPU 임베딩은 매우 오래 걸립니다. 가능하면 CUDA GPU를 사용하세요.", file=sys.stderr)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = load_model(args, device, torch_dtype)
    dimension = int(model.get_sentence_embedding_dimension())
    if state.get("dimension") not in (None, dimension):
        raise SystemExit(f"기존 dimension={state['dimension']}과 모델 dimension={dimension}이 다릅니다.")

    output_numpy_dtype = np.float16 if args.output_dtype == "float16" else np.float32
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if embeddings.shape != (selected_count, dimension):
            raise SystemExit(f"기존 벡터 shape 불일치: {embeddings.shape} != {(selected_count, dimension)}")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=output_numpy_dtype,
            shape=(selected_count, dimension),
        )

    processed_count = int(state.get("processed_count", 0))
    if processed_count < 0 or processed_count > selected_count:
        raise SystemExit(f"잘못된 processed_count: {processed_count}")
    effective_batch_size = int(state.get("effective_batch_size") or args.batch_size)
    state.update({
        "phase": "embedding",
        "device": device,
        "precision": precision_name,
        "dimension": dimension,
        "processed_count": processed_count,
        "effective_batch_size": effective_batch_size,
        "updated_at_unix": time.time(),
    })
    atomic_write_json(state_path, state)

    print(f"임베딩 시작/재개: {processed_count:,}/{selected_count:,}", flush=True)
    print(f"벡터 예상 크기: {human_bytes(selected_count * dimension * np.dtype(output_numpy_dtype).itemsize)}", flush=True)
    start_time = time.time()
    session_start_count = processed_count
    norm_sum = 0.0
    norm_min = float("inf")
    norm_max = 0.0
    checked_vectors = 0
    text_buffer: list[str] = []

    def process_buffer() -> None:
        nonlocal processed_count, effective_batch_size
        nonlocal norm_sum, norm_min, norm_max, checked_vectors, text_buffer
        if not text_buffer:
            return
        vectors, used_batch_size = encode_with_backoff(
            model=model,
            texts=text_buffer,
            batch_size=effective_batch_size,
            device=device,
        )
        effective_batch_size = min(effective_batch_size, used_batch_size)
        if vectors.shape != (len(text_buffer), dimension):
            raise RuntimeError(f"예상하지 못한 임베딩 shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError("임베딩에 NaN 또는 inf가 포함되었습니다.")

        norms = np.linalg.norm(vectors, axis=1)
        norm_sum += float(norms.sum())
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        checked_vectors += len(norms)

        end = processed_count + len(text_buffer)
        embeddings[processed_count:end] = vectors.astype(output_numpy_dtype, copy=False)
        embeddings.flush()
        processed_count = end
        state.update({
            "processed_count": processed_count,
            "effective_batch_size": effective_batch_size,
            "updated_at_unix": time.time(),
        })
        atomic_write_json(state_path, state)

        elapsed = time.time() - start_time
        session_done = processed_count - session_start_count
        rate = session_done / elapsed if elapsed > 0 else 0.0
        remaining = selected_count - processed_count
        eta = remaining / rate if rate > 0 else 0.0
        print(
            f"임베딩 {processed_count:,}/{selected_count:,} "
            f"({processed_count / selected_count * 100:.2f}%) | "
            f"{rate:.2f} chunks/s | ETA {format_duration(eta)} | batch {effective_batch_size}",
            flush=True,
        )
        text_buffer = []

    try:
        for _, text in iter_selected_texts(
            source=source,
            priorities=priorities,
            read_batch_size=args.read_batch_size,
            limit=args.limit,
            skip=processed_count,
        ):
            text_buffer.append(text)
            if len(text_buffer) >= args.encode_buffer_size:
                process_buffer()
        process_buffer()
    except KeyboardInterrupt:
        print("\n사용자 중단: 현재 진행 위치를 저장했습니다. 같은 명령으로 재개할 수 있습니다.", file=sys.stderr)
        raise SystemExit(130)

    if processed_count != selected_count:
        raise RuntimeError(f"처리 개수 불일치: {processed_count:,} != {selected_count:,}")

    total_elapsed = time.time() - start_time
    state.update({
        "phase": "complete",
        "processed_count": processed_count,
        "effective_batch_size": effective_batch_size,
        "completed_at_unix": time.time(),
        "updated_at_unix": time.time(),
    })
    atomic_write_json(state_path, state)

    write_samples(metadata_path, embeddings, samples_path)
    stored_sample = np.asarray(embeddings[: min(1000, selected_count)], dtype=np.float32)
    stored_norms = np.linalg.norm(stored_sample, axis=1) if len(stored_sample) else np.array([])
    summary = {
        "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
        "input_path": str(input_path),
        "input_chunk_count": source.metadata.num_rows,
        "selected_priorities": priorities,
        "selected_chunk_count": selected_count,
        "excluded_chunk_count": source.metadata.num_rows - selected_count,
        "model_name": args.model_name,
        "model_revision": args.revision,
        "model_cache": str(Path(args.model_cache).expanduser().resolve()),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "precision": precision_name,
        "max_seq_length": args.max_seq_length,
        "dimension": dimension,
        "normalized": True,
        "document_prompt": None,
        "query_prompt": "prompt_name=query (검색 코드에서 적용)",
        "output_dtype": args.output_dtype,
        "effective_batch_size": effective_batch_size,
        "embedding_file": str(embeddings_path),
        "embedding_file_size_bytes": embeddings_path.stat().st_size,
        "metadata_file": str(metadata_path),
        "elapsed_seconds_this_session": round(total_elapsed, 3),
        "throughput_chunks_per_second_this_session": round(
            (processed_count - session_start_count) / total_elapsed, 4
        ) if total_elapsed > 0 else None,
        "fresh_vector_norms_before_storage": {
            "count": checked_vectors,
            "mean": round(norm_sum / checked_vectors, 6) if checked_vectors else None,
            "min": round(norm_min, 6) if checked_vectors else None,
            "max": round(norm_max, 6) if checked_vectors else None,
        },
        "stored_vector_norms_first_1000": {
            "mean": round(float(stored_norms.mean()), 6) if len(stored_norms) else None,
            "min": round(float(stored_norms.min()), 6) if len(stored_norms) else None,
            "max": round(float(stored_norms.max()), 6) if len(stored_norms) else None,
        },
        "selected_chunk_digest_sha256": state["selected_chunk_digest_sha256"],
        "generated_files": [
            embeddings_path.name,
            metadata_path.name,
            state_path.name,
            summary_path.name,
            samples_path.name,
        ],
    }
    atomic_write_json(summary_path, summary)

    print("\n완료")
    print(f"임베딩: {processed_count:,} x {dimension:,}")
    print(f"벡터 파일: {embeddings_path} ({human_bytes(embeddings_path.stat().st_size)})")
    print(f"메타데이터: {metadata_path}")
    print(f"실행 요약: {summary_path}")


if __name__ == "__main__":
    main()
