#!/usr/bin/env python3
"""회의록 청크에서 정책 분석 전용 텍스트를 만들고 로컬 임베딩한다.

기존 04_embedding.py는 의회명·위원회·회의록명·발언자까지 포함하므로
근거 검색에는 유리하지만 전체 군집화에서는 기관/인물 중심으로 묶일 수 있다.
이 스크립트는 기존 chunks.parquet를 다시 청킹하지 않고 다음을 수행한다.

1. speech/dialogue/mixed 청크만 정책 후보로 사용
2. 발언자 표식과 반복적인 회의 진행 문구를 제거
3. [안건] + [정책본문]만 임베딩
4. 원문 및 모든 근거 메타데이터는 별도 Parquet에 그대로 보존
5. 중단 시 같은 명령으로 이어서 실행

기본 출력
    data/index/arctic_ko_issue/embeddings.f16.npy
    data/index/arctic_ko_issue/chunk_metadata.parquet
    data/index/arctic_ko_issue/embedding_state.json
    outputs/embedding_issue/issue_filter_summary.json
    outputs/embedding_issue/issue_text_samples.csv
    outputs/embedding_issue/embedding_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

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
except ImportError:
    SentenceTransformer = None


SCRIPT_VERSION = "2026-10-16-issue-embedding-v2.1"
ISSUE_TEXT_VERSION = "agenda-body-no-identity-v2"
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9"

BASE_METADATA_COLUMNS = [
    "chunk_id", "doc_id", "chunk_index", "source_file", "source_record_index",
    "council", "committee", "meeting_name", "meeting_date", "meeting_year",
    "meeting_month", "meeting_type", "chunk_type", "agenda_title", "speaker_label",
    "speaker_role", "speaker_name", "char_start", "char_end", "chunk_char_count",
    "estimated_tokens", "embedding_priority", "evidence_ref", "chunk_sha256", "chunk_text",
]
REQUIRED_COLUMNS = BASE_METADATA_COLUMNS

ROLE_SUFFIX = (
    r"위원장직무대리|위원장대리|부위원장|위원장|부의장|의장|위원|의원|"
    r"도지사|시장|교육감|부시장|부교육감|본부장|국장|과장|실장|소장|"
    r"원장|처장|차장|팀장|담당관|전문위원|의사관|사무처장|사무국장|"
    r"교육장|주무관|공무원|증인|참고인"
)
ROLE_FIRST_RE = re.compile(
    rf"(?m)^[ \t]*[○◯●◎]\s*[가-힣A-Za-z·]{{0,22}}(?:{ROLE_SUFFIX})"
    rf"(?:\s+[가-힣·]{{2,7}})?\s*"
)
NAME_FIRST_RE = re.compile(
    rf"(?m)^[ \t]*[○◯●◎]\s*[가-힣·]{{2,7}}\s+[가-힣A-Za-z·]{{0,22}}"
    rf"(?:{ROLE_SUFFIX})\s*"
)
BARE_MARK_RE = re.compile(r"(?m)^[ \t]*[○◯●◎]\s*")
SPACE_RE = re.compile(r"[ \t]+")
MANY_NEWLINES_RE = re.compile(r"\n{3,}")
SPEAKER_MARK_ANY_RE = re.compile(r"(?m)^[ \t]*[○◯●◎]")
DOCUMENT_HEADER_RE = re.compile(
    r"(?:제\s*\d+회.{0,80}(?:본회의|위원회)?\s*회의록|"
    r"회의록\s*제\s*\d+호|강원특별자치도의회사무처|부의된\s*안건)"
)
ADMIN_REPORT_RE = re.compile(
    r"오늘\s+본회의\s+보고사항|이번\s+(?:임시회|정례회)\s+집회\s+경위|"
    r"의안\s+접수\s+및\s+회부사항|오늘\s+본회의에\s+부의된\s+안건"
)
GENERIC_AGENDA_RE = re.compile(
    r"^(?:의사일정(?:\s*\([^)]*\))?|부의된\s*안건|상정된\s*안건|"
    r"보고사항|휴회의\s*건(?:\([^)]*\))?|회의록\s*서명의원.*|"
    r"제?\s*\d+회.*(?:회기|의사일정)\s*결정의\s*건.*|"
    r"\d+분\s*자유발언.*)$"
)

# 정책 내용과 무관한 반복 회의 진행 표현. 혼합 청크에서는 해당 표현만 지우고
# 본문을 살리며, 진행 표현만 남는 청크는 최소 길이 검사에서 제외된다.
PROCEDURAL_FRAGMENT_RES = [
    re.compile(r"[^.\n]{0,80}의석을\s+정돈하여\s+주시기\s+바랍니다[^.\n]{0,80}[.]?"),
    re.compile(r"[^.\n]{0,80}(?:개의|개회|속개|산회|감사개시|감사종료)를?\s+(?:선포|선언)합니다[^.\n]{0,50}[.]?"),
    re.compile(r"[^.\n]{0,100}더\s+이상\s+질의하실\s+위원님이?\s+(?:안|없)[^.\n]{0,100}[.]?"),
    re.compile(r"[^.\n]{0,80}질의하실\s+위원님께서는\s+질의해\s+주시기\s+바랍니다[^.\n]{0,60}[.]?"),
    re.compile(r"[^.\n]{0,100}이의가\s+(?:없으므로|없습니까)[^.\n]{0,100}[.]?"),
    re.compile(r"[^.\n]{0,120}(?:원안대로|수정안대로)\s+가결[^.\n]{0,100}[.]?"),
    re.compile(r"[^.\n]{0,100}질의[·ㆍ\s]*답변\s+종결을\s+(?:선포|선언)합니다[^.\n]{0,60}[.]?"),
    re.compile(r"[^.\n]{0,100}(?:감사|회의)\s+종료를\s+(?:선포|선언)합니다[^.\n]{0,60}[.]?"),
    re.compile(r"[^.\n]{0,70}(?:수고하셨습니다|감사합니다)[^.\n]{0,30}[.]?"),
    re.compile(r"[^.\n]{0,100}원활한\s+회의\s+진행[^.\n]{0,100}(?:정회|협조)[^.\n]{0,50}[.]?"),
    re.compile(r"[^.\n]{0,100}(?:좌석|의석)을?\s+정돈[^.\n]{0,100}[.]?"),
    re.compile(r"[^.\n]{0,100}성원이\s+되었으므로[^.\n]{0,100}(?:개의|속개)[^.\n]{0,40}[.]?"),
    re.compile(r"[^.\n]{0,80}다음\s+질의하실\s+위원님[^.\n]{0,100}[.]?"),
    re.compile(r"[^.\n]{0,100}(?:발언대|단상)로\s+나오셔서[^.\n]{0,100}[.]?"),
    re.compile(r"[^.\n]{0,100}(?:방청|이석)\s+사항을\s+말씀드리겠습니다[^.\n]{0,80}[.]?"),
    re.compile(r"현재\s+방청석에는[^.\n]{0,220}[.]?"),
    re.compile(r"의원\s+여러분께서는\s+이\s+점을\s+양지[^.\n]{0,80}[.]?"),
    re.compile(r"[^.\n]{0,100}인사발령에\s+따른\s+신임\s+간부[^.\n]{0,120}[.]?"),
    re.compile(r"(?:존경하는|사랑하는)\s+[^!\n]{0,160}(?:여러분|의원님들?)[!,]?"),
    re.compile(r"\([^)]{0,80}\s인사\)"),
    re.compile(r"\(\s*\d{1,2}시\s*\d{1,2}분\s*(?:개의|개회|속개|산회|회의중지|계속개의)\s*\)"),
    re.compile(r"\(\s*[「『][^」』]{0,100}[」』]\s*하는\s+[^)]{0,40}\)"),
]

PROCEDURAL_DETECT_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"의석을\s+정돈", r"행정사무감사\s+실시를\s+(?:선언|선포)",
        r"증언하겠다는\s+서약", r"서약서에\s+서명", r"질의[·ㆍ\s]*답변\s+종결",
        r"이의가\s+없으므로", r"원안대로\s+가결", r"산회를\s+선포",
        r"회의록\s+서명의원", r"의사일정\s+제\s*\d+항을\s+상정",
    )
]

POLICY_SIGNAL_RE = re.compile(
    r"정책|사업|지원|예산|조례|계획|대책|개선|확대|축소|현황|필요|부족|문제|"
    r"지역|도민|시민|안전|교육|복지|의료|교통|청년|인구|저출생|출산|난임|"
    r"환경|기후|산업|경제|주거|돌봄|일자리|재난|농업|시설|운영|인력|공공"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정책 분석 전용 로컬 임베딩 생성")
    parser.add_argument("--input", default="data/processed/chunks.parquet")
    parser.add_argument("--output-dir", default="data/index/arctic_ko_issue")
    parser.add_argument("--report-dir", default="outputs/embedding_issue")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-buffer-size", type=int, default=512)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--priorities", default="high,medium")
    parser.add_argument("--chunk-types", default="speech,dialogue,mixed")
    parser.add_argument("--min-issue-chars", type=int, default=120)
    parser.add_argument("--agenda-max-chars", type=int, default=300)
    parser.add_argument(
        "--max-per-council-year", type=int, default=0,
        help="테스트용 의회·연도별 최대 선택 수. 0이면 제한 없음",
    )
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=0, help="필터 통과 후 앞 N개. 0이면 전체")
    parser.add_argument("--prepare-only", action="store_true", help="필터링/매니페스트만 실행")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.read_batch_size < 1:
        raise SystemExit("배치 크기는 1 이상이어야 합니다.")
    if args.encode_buffer_size < args.batch_size:
        raise SystemExit("--encode-buffer-size는 --batch-size 이상이어야 합니다.")
    if args.max_seq_length < 128:
        raise SystemExit("--max-seq-length는 128 이상이어야 합니다.")
    if args.min_issue_chars < 20:
        raise SystemExit("--min-issue-chars는 20 이상이어야 합니다.")
    if args.agenda_max_chars < 0 or args.limit < 0 or args.max_per_council_year < 0:
        raise SystemExit("안건 길이, limit, 의회·연도별 제한은 0 이상이어야 합니다.")


def parse_choices(value: str, allowed: set[str], label: str) -> list[str]:
    values = []
    for item in value.split(","):
        item = item.strip().lower()
        if item and item not in values:
            values.append(item)
    unknown = set(values) - allowed
    if unknown:
        raise SystemExit(f"알 수 없는 {label}: {', '.join(sorted(unknown))}")
    if not values:
        raise SystemExit(f"--{label}에 하나 이상 지정해야 합니다.")
    return values


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def safe_text(value) -> str:
    return "" if value is None else str(value)


def clean_body(text: str) -> tuple[str, int, int]:
    """문서 머리말·발언자 표식·반복 회의 진행 표현을 제거한다."""
    result = safe_text(text).replace("\r\n", "\n").replace("\r", "\n")
    header_removed = 0
    first_speaker = SPEAKER_MARK_ANY_RE.search(result)
    if first_speaker:
        prefix = result[:first_speaker.start()]
        if DOCUMENT_HEADER_RE.search(prefix):
            result = result[first_speaker.start():]
            header_removed = 1
    result = ROLE_FIRST_RE.sub("", result)
    result = NAME_FIRST_RE.sub("", result)
    result = BARE_MARK_RE.sub("", result)
    removed_count = 0
    for pattern in PROCEDURAL_FRAGMENT_RES:
        result, count = pattern.subn(" ", result)
        removed_count += count
    result = "\n".join(SPACE_RE.sub(" ", line).strip() for line in result.splitlines())
    result = MANY_NEWLINES_RE.sub("\n\n", result).strip()
    return result, removed_count, header_removed


def clean_agenda(agenda_text: str, agenda_max_chars: int) -> tuple[str, int]:
    """정책명이 아닌 공통 회의 안건을 제거하고 남은 안건만 반환한다."""
    raw = safe_text(agenda_text).replace("\r\n", "\n").replace("\r", "\n")
    segments = re.split(r"[;\n]+", raw)
    kept = []
    removed_count = 0
    for segment in segments:
        normalized = re.sub(r"\s+", " ", segment).strip(" ·ㆍ-|")
        if not normalized:
            continue
        comparison = re.sub(
            r"^(?:(?:제\s*)?\d+\s*(?:항|번)?[.)]?\s*|[가-힣][.)]\s*)",
            "",
            normalized,
        ).strip()
        if GENERIC_AGENDA_RE.fullmatch(comparison):
            removed_count += 1
            continue
        kept.append(normalized)
    agenda = "; ".join(kept)
    if agenda_max_chars:
        agenda = agenda[:agenda_max_chars].rstrip()
    return agenda, removed_count


def count_procedural_patterns(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in PROCEDURAL_DETECT_PATTERNS)


def make_issue_text(row: dict, agenda_max_chars: int) -> tuple[str, dict]:
    raw_text = safe_text(row.get("chunk_text"))
    raw_procedural_count = count_procedural_patterns(raw_text)
    body, removed_count, header_removed = clean_body(raw_text)
    procedural_count = count_procedural_patterns(body)
    agenda, generic_agenda_removed = clean_agenda(row.get("agenda_title"), agenda_max_chars)
    policy_signal_count = len(POLICY_SIGNAL_RE.findall(f"{agenda} {body}"))
    parts = []
    if agenda:
        parts.append(f"[안건] {agenda}")
    parts.append(f"[정책본문] {body}")
    issue_text = "\n".join(parts).strip()
    diagnostics = {
        "issue_body_char_count": len(body),
        "policy_signal_count": policy_signal_count,
        "raw_procedural_pattern_count": raw_procedural_count,
        "procedural_pattern_count": procedural_count,
        "removed_procedural_count": removed_count,
        "header_removed": header_removed,
        "generic_agenda_removed": generic_agenda_removed,
        "admin_report_detected": int(bool(ADMIN_REPORT_RE.search(raw_text[:400]))),
        "header_only_detected": int(
            bool(DOCUMENT_HEADER_RE.search(raw_text[:700]))
            and not bool(SPEAKER_MARK_ANY_RE.search(raw_text))
        ),
    }
    return issue_text, diagnostics


def select_reason(
    row: dict,
    priorities: set[str],
    chunk_types: set[str],
    min_issue_chars: int,
    diagnostics: dict,
) -> str:
    if safe_text(row.get("embedding_priority")).lower() not in priorities:
        return "excluded_priority"
    if safe_text(row.get("chunk_type")).lower() not in chunk_types:
        return "excluded_chunk_type"
    if diagnostics["admin_report_detected"]:
        return "excluded_admin_report"
    if diagnostics["header_only_detected"]:
        return "excluded_header_only"
    if diagnostics["issue_body_char_count"] < min_issue_chars:
        return "excluded_short_after_cleaning"
    if diagnostics["policy_signal_count"] == 0:
        return "excluded_no_policy_signal"
    if diagnostics["procedural_pattern_count"] >= 2 and diagnostics["policy_signal_count"] <= 1:
        return "excluded_procedural_only"
    return "selected"


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
        raise SystemExit("입력 Parquet 필수 컬럼 누락: " + ", ".join(missing))


def output_schema(source: pq.ParquetFile) -> pa.Schema:
    fields = [pa.field("embedding_row", pa.int64())]
    fields.extend(source.schema_arrow.field(column) for column in BASE_METADATA_COLUMNS)
    fields.extend([
        pa.field("issue_text", pa.string()),
        pa.field("issue_text_sha256", pa.string()),
        pa.field("issue_body_char_count", pa.int32()),
        pa.field("policy_signal_count", pa.int32()),
        pa.field("raw_procedural_pattern_count", pa.int32()),
        pa.field("procedural_pattern_count", pa.int32()),
        pa.field("removed_procedural_count", pa.int32()),
        pa.field("header_removed", pa.int8()),
        pa.field("generic_agenda_removed", pa.int32()),
        pa.field("admin_report_detected", pa.int8()),
        pa.field("header_only_detected", pa.int8()),
    ])
    return pa.schema(fields)


def write_filter_samples(path: Path, rows: list[dict]) -> None:
    fields = [
        "selection", "chunk_id", "council", "meeting_date", "chunk_type",
        "agenda_title", "speaker_label", "issue_body_char_count",
        "policy_signal_count", "raw_procedural_pattern_count",
        "procedural_pattern_count", "removed_procedural_count", "header_removed",
        "generic_agenda_removed", "admin_report_detected", "header_only_detected",
        "original_preview", "issue_text_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_manifest(
    source: pq.ParquetFile,
    metadata_path: Path,
    report_path: Path,
    samples_path: Path,
    priorities: list[str],
    chunk_types: list[str],
    min_issue_chars: int,
    agenda_max_chars: int,
    read_batch_size: int,
    limit: int,
    max_per_council_year: int,
) -> tuple[int, str, dict]:
    temporary = metadata_path.with_name(metadata_path.stem + ".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    schema = output_schema(source)
    writer = None
    selected_count = 0
    counters = Counter()
    type_selected = Counter()
    council_selected = Counter()
    council_year_selected = Counter()
    digest = hashlib.sha256()
    samples = []
    sample_reason_counts = Counter()
    priority_set = set(priorities)
    chunk_type_set = set(chunk_types)
    stop = False
    started = time.time()

    try:
        for batch in source.iter_batches(batch_size=read_batch_size, columns=BASE_METADATA_COLUMNS):
            output_rows = []
            for row in batch.to_pylist():
                counters["input"] += 1
                if counters["input"] % 50000 == 0:
                    print(
                        f"정책 후보 필터링: {counters['input']:,}/{source.metadata.num_rows:,} "
                        f"| 선택 {selected_count:,}",
                        flush=True,
                    )
                issue_text, diagnostics = make_issue_text(row, agenda_max_chars)
                reason = select_reason(row, priority_set, chunk_type_set, min_issue_chars, diagnostics)
                council = safe_text(row.get("council")) or "unknown"
                meeting_year = safe_text(row.get("meeting_year"))
                if not meeting_year:
                    meeting_year = safe_text(row.get("meeting_date"))[:4] or "unknown"
                council_year_key = (council, meeting_year)
                if (
                    reason == "selected"
                    and max_per_council_year
                    and council_year_selected[council_year_key] >= max_per_council_year
                ):
                    reason = "excluded_test_cap"
                if reason == "selected" and limit and selected_count >= limit:
                    stop = True
                    break
                counters[reason] += 1

                sample_key = (
                    f"selected:{council}:{meeting_year}"
                    if reason == "selected" and max_per_council_year
                    else reason
                )
                sample_limit = 2 if reason == "selected" and max_per_council_year else 10
                if sample_reason_counts[sample_key] < sample_limit:
                    samples.append({
                        "selection": reason,
                        "chunk_id": safe_text(row.get("chunk_id")),
                        "council": safe_text(row.get("council")),
                        "meeting_date": safe_text(row.get("meeting_date")),
                        "chunk_type": safe_text(row.get("chunk_type")),
                        "agenda_title": safe_text(row.get("agenda_title"))[:200],
                        "speaker_label": safe_text(row.get("speaker_label"))[:100],
                        **diagnostics,
                        "original_preview": re.sub(r"\s+", " ", safe_text(row.get("chunk_text")))[:350],
                        "issue_text_preview": re.sub(r"\s+", " ", issue_text)[:350],
                    })
                    sample_reason_counts[sample_key] += 1

                if reason != "selected":
                    continue

                issue_hash = hashlib.sha256(issue_text.encode("utf-8")).hexdigest()
                output_row = {column: row.get(column) for column in BASE_METADATA_COLUMNS}
                output_row.update({
                    "embedding_row": selected_count,
                    "issue_text": issue_text,
                    "issue_text_sha256": issue_hash,
                    **diagnostics,
                })
                output_rows.append(output_row)
                chunk_id = safe_text(row.get("chunk_id"))
                digest.update(chunk_id.encode("utf-8"))
                digest.update(b"\x1f")
                digest.update(issue_hash.encode("ascii"))
                digest.update(b"\n")
                selected_count += 1
                type_selected[safe_text(row.get("chunk_type"))] += 1
                council_selected[council] += 1
                council_year_selected[council_year_key] += 1

            if output_rows:
                table = pa.Table.from_pylist(output_rows, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        schema,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(table)
            if selected_count and selected_count % 50000 < len(output_rows):
                print(f"정책 후보 매니페스트: {selected_count:,}개", flush=True)
            if stop:
                break
    finally:
        if writer is not None:
            writer.close()

    report = {
        "script_version": SCRIPT_VERSION,
        "issue_text_version": ISSUE_TEXT_VERSION,
        "input_row_count": source.metadata.num_rows,
        "scanned_row_count": counters["input"],
        "selected_count": selected_count,
        "limit": limit,
        "max_per_council_year": max_per_council_year,
        "priorities": priorities,
        "chunk_types": chunk_types,
        "min_issue_chars": min_issue_chars,
        "agenda_max_chars": agenda_max_chars,
        "filter_counts": dict(counters),
        "selected_chunk_types": dict(type_selected.most_common()),
        "selected_councils": dict(council_selected.most_common()),
        "selected_council_years": {
            f"{council}|{year}": count
            for (council, year), count in sorted(council_year_selected.items())
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_json(report_path, report)
    write_filter_samples(samples_path, samples)
    if selected_count == 0:
        raise SystemExit(
            "필터 조건을 통과한 정책 후보 청크가 없습니다. "
            f"제외 사유는 {report_path}에서 확인하세요."
        )
    if not temporary.is_file():
        raise SystemExit(
            f"정책 후보 {selected_count:,}개를 선택했지만 임시 Parquet 파일이 없습니다: {temporary}"
        )
    os.replace(temporary, metadata_path)
    return selected_count, digest.hexdigest(), report


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise SystemExit("CUDA를 사용할 수 없습니다. PyTorch CUDA 설치와 nvidia-smi를 확인하세요.")
    return requested


def resolve_precision(requested: str, device: str):
    if requested == "float32":
        return "float32", torch.float32
    if requested == "bf16":
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise SystemExit("현재 장치에서 BF16을 지원하지 않습니다. --precision float32를 사용하세요.")
        return "bf16", torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "float32", torch.float32


def load_model(args: argparse.Namespace, device: str, torch_dtype) -> SentenceTransformer:
    cache = Path(args.model_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"모델 로딩: {args.model_name} @ {args.revision}", flush=True)
    print(f"장치: {device} | 연산 dtype: {torch_dtype}", flush=True)
    common = {
        "device": device,
        "cache_folder": str(cache),
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    try:
        model = SentenceTransformer(args.model_name, model_kwargs={"torch_dtype": torch_dtype}, **common)
    except TypeError:
        model = SentenceTransformer(args.model_name, **common)
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
    current = batch_size
    while True:
        try:
            vectors = model.encode(
                texts,
                batch_size=current,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32), current
        except RuntimeError as error:
            if device != "cuda" or not is_oom_error(error) or current == 1:
                raise
            current = max(1, current // 2)
            torch.cuda.empty_cache()
            print(f"GPU 메모리 부족: 배치를 {current}로 낮춰 재시도합니다.", flush=True)


def iter_issue_texts(metadata_path: Path, read_batch_size: int, skip: int):
    source = pq.ParquetFile(metadata_path)
    position = 0
    for batch in source.iter_batches(batch_size=read_batch_size, columns=["chunk_id", "issue_text"]):
        table = pa.Table.from_batches([batch])
        end = position + table.num_rows
        if end <= skip:
            position = end
            continue
        local_start = max(0, skip - position)
        if local_start:
            table = table.slice(local_start)
        for chunk_id, text in zip(table["chunk_id"].to_pylist(), table["issue_text"].to_pylist()):
            yield safe_text(chunk_id), safe_text(text)
        position = end


def state_compatible(state: dict, expected: dict) -> tuple[bool, list[str]]:
    keys = [
        "input_path", "input_size_bytes", "input_row_count", "script_version",
        "model_name", "revision",
        "priorities", "chunk_types", "min_issue_chars", "agenda_max_chars", "limit",
        "max_per_council_year", "output_dtype", "max_seq_length", "issue_text_version",
    ]
    differences = [key for key in keys if state.get(key) != expected.get(key)]
    return not differences, differences


def write_embedding_samples(
    metadata_path: Path,
    embeddings: np.ndarray,
    output_path: Path,
    sample_count: int = 20,
) -> None:
    table = pq.read_table(
        metadata_path,
        columns=[
            "embedding_row", "chunk_id", "council", "meeting_name", "meeting_date",
            "chunk_type", "issue_body_char_count", "policy_signal_count", "issue_text",
        ],
    ).slice(0, sample_count)
    rows = []
    for item in table.to_pylist():
        row_index = int(item["embedding_row"])
        vector = np.asarray(embeddings[row_index], dtype=np.float32)
        rows.append({
            **{key: value for key, value in item.items() if key != "issue_text"},
            "issue_text_preview": re.sub(r"\s+", " ", safe_text(item.get("issue_text")))[:300],
            "vector_norm": round(float(np.linalg.norm(vector)), 6),
            "first_8_values": " ".join(f"{value:.5f}" for value in vector[:8]),
        })
    fields = list(rows[0].keys()) if rows else ["embedding_row"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    priorities = parse_choices(args.priorities, {"high", "medium", "low"}, "priorities")
    chunk_types = parse_choices(
        args.chunk_types,
        {"speech", "dialogue", "mixed", "header", "agenda", "proceeding"},
        "chunk-types",
    )

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "chunk_metadata.parquet"
    embeddings_path = output_dir / (
        "embeddings.f16.npy" if args.output_dtype == "float16" else "embeddings.f32.npy"
    )
    state_path = output_dir / "embedding_state.json"
    filter_report_path = report_dir / "issue_filter_summary.json"
    issue_samples_path = report_dir / "issue_text_samples.csv"
    summary_path = report_dir / "embedding_summary.json"
    embedding_samples_path = report_dir / "embedding_samples.csv"

    source = pq.ParquetFile(input_path)
    check_required_columns(source)
    expected_state = {
        **source_signature(input_path, source),
        "script_version": SCRIPT_VERSION,
        "issue_text_version": ISSUE_TEXT_VERSION,
        "model_name": args.model_name,
        "revision": args.revision,
        "priorities": priorities,
        "chunk_types": chunk_types,
        "min_issue_chars": args.min_issue_chars,
        "agenda_max_chars": args.agenda_max_chars,
        "limit": args.limit,
        "max_per_council_year": args.max_per_council_year,
        "output_dtype": args.output_dtype,
        "max_seq_length": args.max_seq_length,
    }

    state = load_json(state_path) if state_path.exists() else None
    if state:
        compatible, differences = state_compatible(state, expected_state)
        if not compatible:
            raise SystemExit(
                "기존 실행 상태와 현재 옵션이 다릅니다: " + ", ".join(differences)
                + "\n다른 --output-dir을 사용하거나 기존 결과를 별도로 이동하세요."
            )
        if not metadata_path.exists():
            raise SystemExit("상태 파일은 있지만 chunk_metadata.parquet가 없습니다.")
        selected_count = int(state["selected_count"])
        print(f"기존 정책 후보 매니페스트 사용: {selected_count:,}개", flush=True)
    else:
        if metadata_path.exists() or embeddings_path.exists():
            raise SystemExit(
                "기존 출력 파일은 있지만 embedding_state.json이 없습니다. 다른 --output-dir을 사용하세요."
            )
        print("정책 후보 필터링 시작", flush=True)
        selected_count, selected_digest, filter_report = prepare_manifest(
            source=source,
            metadata_path=metadata_path,
            report_path=filter_report_path,
            samples_path=issue_samples_path,
            priorities=priorities,
            chunk_types=chunk_types,
            min_issue_chars=args.min_issue_chars,
            agenda_max_chars=args.agenda_max_chars,
            read_batch_size=args.read_batch_size,
            limit=args.limit,
            max_per_council_year=args.max_per_council_year,
        )
        state = {
            **expected_state,
            "phase": "manifest_ready",
            "selected_count": selected_count,
            "selected_chunk_digest_sha256": selected_digest,
            "filter_counts": filter_report["filter_counts"],
            "processed_count": 0,
            "dimension": None,
            "effective_batch_size": args.batch_size,
            "created_at_unix": time.time(),
            "updated_at_unix": time.time(),
        }
        atomic_write_json(state_path, state)
        print(f"정책 후보 매니페스트 완료: {selected_count:,}개", flush=True)
        print(f"필터 요약: {filter_report_path}", flush=True)
        print(f"정제 샘플: {issue_samples_path}", flush=True)

    if args.prepare_only:
        print("--prepare-only 완료. 모델 추론은 실행하지 않았습니다.")
        print(f"메타데이터: {metadata_path}")
        return

    if torch is None:
        raise SystemExit("PyTorch가 필요합니다. CUDA 버전에 맞는 torch를 설치하세요.")
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers가 필요합니다. 실행 전: pip install -U sentence-transformers")
    if state.get("phase") == "complete" and embeddings_path.exists() and summary_path.exists():
        print("이미 정책 전용 임베딩이 완료된 상태입니다.")
        print(f"벡터: {embeddings_path}")
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

    print(f"정책 전용 임베딩 시작/재개: {processed_count:,}/{selected_count:,}", flush=True)
    print(
        "벡터 예상 크기: "
        + human_bytes(selected_count * dimension * np.dtype(output_numpy_dtype).itemsize),
        flush=True,
    )
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
        vectors, used_batch = encode_with_backoff(model, text_buffer, effective_batch_size, device)
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
            f"{rate:.2f} chunks/s | ETA {format_duration(eta)} | batch {effective_batch_size}",
            flush=True,
        )
        text_buffer = []

    try:
        for _, text in iter_issue_texts(metadata_path, args.read_batch_size, processed_count):
            text_buffer.append(text)
            if len(text_buffer) >= args.encode_buffer_size:
                process_buffer()
        process_buffer()
    except KeyboardInterrupt:
        print("\n사용자 중단: 같은 명령으로 재개할 수 있습니다.", file=sys.stderr)
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

    write_embedding_samples(metadata_path, embeddings, embedding_samples_path)
    stored = np.asarray(embeddings[: min(1000, selected_count)], dtype=np.float32)
    stored_norms = np.linalg.norm(stored, axis=1) if len(stored) else np.array([])
    summary = {
        "script_version": SCRIPT_VERSION,
        "issue_text_version": ISSUE_TEXT_VERSION,
        "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
        "purpose": "정책 이슈 검색 및 군집화 전용 임베딩",
        "input_path": str(input_path),
        "input_chunk_count": source.metadata.num_rows,
        "selected_chunk_count": selected_count,
        "excluded_chunk_count": source.metadata.num_rows - selected_count,
        "filter_counts": state.get("filter_counts", {}),
        "selected_priorities": priorities,
        "selected_chunk_types": chunk_types,
        "min_issue_chars": args.min_issue_chars,
        "agenda_max_chars": args.agenda_max_chars,
        "max_per_council_year": args.max_per_council_year,
        "model_name": args.model_name,
        "model_revision": args.revision,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "precision": precision_name,
        "max_seq_length": args.max_seq_length,
        "dimension": dimension,
        "normalized": True,
        "output_dtype": args.output_dtype,
        "effective_batch_size": effective_batch_size,
        "embedding_file": str(embeddings_path),
        "embedding_file_size_bytes": embeddings_path.stat().st_size,
        "metadata_file": str(metadata_path),
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
        "selected_chunk_digest_sha256": state["selected_chunk_digest_sha256"],
        "generated_files": [
            embeddings_path.name,
            metadata_path.name,
            state_path.name,
            summary_path.name,
            embedding_samples_path.name,
            filter_report_path.name,
            issue_samples_path.name,
        ],
    }
    atomic_write_json(summary_path, summary)

    print("\n완료")
    print(f"정책 임베딩: {processed_count:,} x {dimension:,}")
    print(f"벡터 파일: {embeddings_path} ({human_bytes(embeddings_path.stat().st_size)})")
    print(f"메타데이터: {metadata_path}")
    print(f"실행 요약: {summary_path}")


if __name__ == "__main__":
    main()
