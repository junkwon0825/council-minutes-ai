#!/usr/bin/env python3
"""04e가 만든 정책 의미 자식 청크를 로컬 임베딩한다.

04b의 모델 로딩/인코딩 함수를 재사용하고, 04e가 만든
chunk_metadata.parquet의 행 순서대로 embeddings.f16.npy를 생성한다.
중단되면 같은 명령으로 이어서 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("numpy와 pyarrow가 필요합니다.") from error


SCRIPT_VERSION = "2026-10-19-embed-semantic-v2"
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9"
REQUIRED_COLUMNS = ["embedding_row", "chunk_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="04e 정책 의미 자식 청크 로컬 임베딩")
    parser.add_argument(
        "--input",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index/arctic_ko_semantic_test_v2",
        help="기본값은 입력 매니페스트 폴더와 동일",
    )
    parser.add_argument("--report-dir", default="outputs/embedding_semantic_test_v2")
    parser.add_argument(
        "--text-column", default="semantic_text",
        help="임베딩에 사용할 텍스트 컬럼. 04e 출력은 semantic_text",
    )
    parser.add_argument("--helper", default="src/04b_issue_embedding.py")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-buffer-size", type=int, default=512)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper(path: Path):
    if not path.is_file():
        raise SystemExit(f"04b 보조 파일이 없습니다: {path}")
    spec = importlib.util.spec_from_file_location("issue_embedding_helper", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"04b 보조 파일을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.read_batch_size < 1:
        raise SystemExit("배치 크기는 1 이상이어야 합니다.")
    if args.encode_buffer_size < args.batch_size:
        raise SystemExit("--encode-buffer-size는 --batch-size 이상이어야 합니다.")
    if args.max_seq_length < 128:
        raise SystemExit("--max-seq-length는 128 이상이어야 합니다.")


def validate_manifest(path: Path, read_batch_size: int, text_column: str) -> tuple[int, str]:
    source = pq.ParquetFile(path)
    required = [*REQUIRED_COLUMNS, text_column]
    missing = [column for column in required if column not in source.schema_arrow.names]
    if missing:
        raise SystemExit("입력 매니페스트 필수 컬럼 누락: " + ", ".join(missing))

    digest = hashlib.sha256()
    expected_row = 0
    for batch in source.iter_batches(
        batch_size=read_batch_size,
        columns=["embedding_row", "chunk_id", text_column],
    ):
        table = pa.Table.from_batches([batch])
        for row_index, chunk_id, input_text in zip(
            table["embedding_row"].to_pylist(),
            table["chunk_id"].to_pylist(),
            table[text_column].to_pylist(),
        ):
            if int(row_index) != expected_row:
                raise SystemExit(
                    f"embedding_row가 연속적이지 않습니다: expected={expected_row}, actual={row_index}"
                )
            text = "" if input_text is None else str(input_text).strip()
            if not text:
                raise SystemExit(f"빈 {text_column}가 있습니다: embedding_row={expected_row}")
            digest.update(str(chunk_id or "").encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(hashlib.sha256(text.encode("utf-8")).digest())
            digest.update(b"\n")
            expected_row += 1

    if expected_row != source.metadata.num_rows:
        raise SystemExit(
            f"매니페스트 행 수 불일치: scanned={expected_row}, metadata={source.metadata.num_rows}"
        )
    return expected_row, digest.hexdigest()


def iter_input_texts(path: Path, read_batch_size: int, skip: int, text_column: str):
    source = pq.ParquetFile(path)
    position = 0
    for batch in source.iter_batches(batch_size=read_batch_size, columns=[text_column]):
        table = pa.Table.from_batches([batch])
        end = position + table.num_rows
        if end <= skip:
            position = end
            continue
        local_start = max(0, skip - position)
        if local_start:
            table = table.slice(local_start)
        for text in table[text_column].to_pylist():
            yield str(text)
        position = end


def state_compatible(state: dict, expected: dict) -> tuple[bool, list[str]]:
    keys = [
        "script_version", "input_path", "input_size_bytes", "input_row_count",
        "input_digest_sha256", "model_name", "revision", "output_dtype",
        "max_seq_length", "text_column",
    ]
    differences = [key for key in keys if state.get(key) != expected.get(key)]
    return not differences, differences


def write_samples(
    metadata_path: Path,
    embeddings,
    output_path: Path,
    text_column: str,
    count: int = 20,
) -> None:
    source = pq.ParquetFile(metadata_path)
    available = [
        column for column in [
            "embedding_row", "chunk_id", "council", "meeting_name", "meeting_date",
            "parent_chunk_id", "semantic_unit_index", "semantic_unit_count",
            "chunk_type", "relevance_score", "relevance_reason", text_column,
        ]
        if column in source.schema_arrow.names
    ]
    rows = pq.read_table(metadata_path, columns=available).slice(0, count).to_pylist()
    output_rows = []
    for item in rows:
        row_index = int(item["embedding_row"])
        vector = np.asarray(embeddings[row_index], dtype=np.float32)
        text = str(item.pop(text_column, "") or "")
        output_rows.append({
            **item,
            f"{text_column}_preview": " ".join(text.split())[:300],
            "vector_norm": round(float(np.linalg.norm(vector)), 6),
            "first_8_values": " ".join(f"{value:.5f}" for value in vector[:8]),
        })
    fields = list(output_rows[0]) if output_rows else ["embedding_row"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    helper_path = Path(args.helper).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"입력 매니페스트가 없습니다: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_metadata = output_dir / "chunk_metadata.parquet"
    if input_path != output_metadata and output_metadata.exists():
        raise SystemExit(
            "출력 폴더에 다른 chunk_metadata.parquet가 있습니다. "
            "입력 파일의 부모 폴더를 --output-dir로 지정하세요."
        )
    if input_path != output_metadata:
        raise SystemExit(
            "안전한 행 정렬 보존을 위해 --output-dir은 입력 chunk_metadata.parquet의 부모 폴더여야 합니다."
        )

    helper = load_helper(helper_path)
    if helper.torch is None:
        raise SystemExit("PyTorch가 필요합니다. CUDA 버전에 맞는 torch를 설치하세요.")
    if helper.SentenceTransformer is None:
        raise SystemExit("sentence-transformers가 필요합니다.")

    print("선택 매니페스트 검증", flush=True)
    selected_count, input_digest = validate_manifest(
        input_path, args.read_batch_size, args.text_column
    )
    stat = input_path.stat()
    expected = {
        "script_version": SCRIPT_VERSION,
        "input_path": str(input_path),
        "input_size_bytes": stat.st_size,
        "input_row_count": selected_count,
        "input_digest_sha256": input_digest,
        "model_name": args.model_name,
        "revision": args.revision,
        "output_dtype": args.output_dtype,
        "max_seq_length": args.max_seq_length,
        "text_column": args.text_column,
    }

    embeddings_path = output_dir / (
        "embeddings.f16.npy" if args.output_dtype == "float16" else "embeddings.f32.npy"
    )
    state_path = output_dir / "embedding_state.json"
    summary_path = report_dir / "embedding_summary.json"
    samples_path = report_dir / "embedding_samples.csv"

    state = load_json(state_path) if state_path.exists() else None
    if state:
        compatible, differences = state_compatible(state, expected)
        if not compatible:
            raise SystemExit(
                "기존 임베딩 상태와 현재 입력/옵션이 다릅니다: " + ", ".join(differences)
            )
    else:
        if embeddings_path.exists():
            raise SystemExit("벡터 파일은 있지만 embedding_state.json이 없습니다.")
        state = {
            **expected,
            "phase": "manifest_ready",
            "processed_count": 0,
            "dimension": None,
            "effective_batch_size": args.batch_size,
            "created_at_unix": time.time(),
            "updated_at_unix": time.time(),
        }
        atomic_write_json(state_path, state)

    if state.get("phase") == "complete" and embeddings_path.exists() and summary_path.exists():
        print("이미 선택 후보 임베딩이 완료되었습니다.")
        print(f"벡터: {embeddings_path}")
        return

    device = helper.resolve_device(args.device)
    precision_name, torch_dtype = helper.resolve_precision(args.precision, device)
    if device == "cuda":
        helper.torch.set_float32_matmul_precision("high")
        print(f"GPU: {helper.torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("경고: CPU 임베딩은 오래 걸립니다.", flush=True)

    model = helper.load_model(args, device, torch_dtype)
    dimension = int(model.get_sentence_embedding_dimension())
    if state.get("dimension") not in (None, dimension):
        raise SystemExit(f"기존 dimension={state['dimension']}과 모델 dimension={dimension}이 다릅니다.")

    output_numpy_dtype = np.float16 if args.output_dtype == "float16" else np.float32
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if embeddings.shape != (selected_count, dimension):
            raise SystemExit(f"기존 벡터 shape 불일치: {embeddings.shape}")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=output_numpy_dtype,
            shape=(selected_count, dimension),
        )

    processed_count = int(state.get("processed_count", 0))
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

    print(f"선택 후보 임베딩 시작/재개: {processed_count:,}/{selected_count:,}", flush=True)
    expected_bytes = selected_count * dimension * np.dtype(output_numpy_dtype).itemsize
    print(f"벡터 예상 크기: {helper.human_bytes(expected_bytes)}", flush=True)
    started = time.time()
    session_start_count = processed_count
    norm_sum = 0.0
    norm_min = float("inf")
    norm_max = 0.0
    checked_vectors = 0
    text_buffer: list[str] = []

    def process_buffer() -> None:
        nonlocal processed_count, effective_batch_size, text_buffer
        nonlocal norm_sum, norm_min, norm_max, checked_vectors
        if not text_buffer:
            return
        vectors, used_batch = helper.encode_with_backoff(
            model, text_buffer, effective_batch_size, device
        )
        effective_batch_size = min(effective_batch_size, used_batch)
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
        elapsed = time.time() - started
        session_done = processed_count - session_start_count
        rate = session_done / elapsed if elapsed else 0.0
        eta = (selected_count - processed_count) / rate if rate else 0.0
        print(
            f"임베딩 {processed_count:,}/{selected_count:,} "
            f"({processed_count / selected_count * 100:.2f}%) | "
            f"{rate:.2f} chunks/s | ETA {helper.format_duration(eta)} | "
            f"batch {effective_batch_size}",
            flush=True,
        )
        text_buffer = []

    try:
        for text in iter_input_texts(
            input_path, args.read_batch_size, processed_count, args.text_column
        ):
            text_buffer.append(text)
            if len(text_buffer) >= args.encode_buffer_size:
                process_buffer()
        process_buffer()
    except KeyboardInterrupt:
        print("\n사용자 중단: 같은 명령으로 재개할 수 있습니다.")
        raise SystemExit(130)

    if processed_count != selected_count:
        raise RuntimeError(f"처리 개수 불일치: {processed_count:,} != {selected_count:,}")

    total_elapsed = time.time() - started
    state.update({
        "phase": "complete",
        "processed_count": processed_count,
        "effective_batch_size": effective_batch_size,
        "completed_at_unix": time.time(),
        "updated_at_unix": time.time(),
    })
    atomic_write_json(state_path, state)

    write_samples(input_path, embeddings, samples_path, args.text_column)
    stored = np.asarray(embeddings[: min(1000, selected_count)], dtype=np.float32)
    stored_norms = np.linalg.norm(stored, axis=1) if len(stored) else np.array([])
    summary = {
        "script_version": SCRIPT_VERSION,
        "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
        "purpose": "04e 정책 의미 자식 청크의 검색 및 군집화 전용 임베딩",
        "input_path": str(input_path),
        "selected_chunk_count": selected_count,
        "input_digest_sha256": input_digest,
        "model_name": args.model_name,
        "model_revision": args.revision,
        "device": device,
        "gpu_name": helper.torch.cuda.get_device_name(0) if device == "cuda" else None,
        "precision": precision_name,
        "max_seq_length": args.max_seq_length,
        "text_column": args.text_column,
        "dimension": dimension,
        "normalized": True,
        "output_dtype": args.output_dtype,
        "effective_batch_size": effective_batch_size,
        "embedding_file": str(embeddings_path),
        "embedding_file_size_bytes": embeddings_path.stat().st_size,
        "metadata_file": str(input_path),
        "elapsed_seconds_this_session": round(total_elapsed, 3),
        "throughput_chunks_per_second_this_session": round(
            (processed_count - session_start_count) / total_elapsed, 4
        ) if total_elapsed else None,
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
        "generated_files": [
            str(embeddings_path), str(input_path), str(state_path),
            str(summary_path), str(samples_path),
        ],
    }
    atomic_write_json(summary_path, summary)

    print("\n완료")
    print(f"벡터: {embeddings_path}")
    print(f"메타데이터: {input_path}")
    print(f"요약: {summary_path}")
    print(f"샘플: {samples_path}")


if __name__ == "__main__":
    main()
