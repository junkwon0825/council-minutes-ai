#!/usr/bin/env python3
"""정제 회의록을 임베딩/RAG용 근거 청크로 변환한다.

입력
    data/processed/documents.parquet (02_preprocess.py 결과)

출력
    data/processed/chunks.parquet
    outputs/chunks/chunking_summary.json
    outputs/chunks/council_chunk_summary.csv
    outputs/chunks/chunk_samples.csv
    outputs/chunks/chunk_quality_issues.csv

핵심 원칙
1. 발언자와 안건 경계를 가능한 한 보존한다.
2. 긴 발언만 문장/줄 경계에서 겹침을 두어 분할한다.
3. chunk_text는 clean_text의 연속 구간 그대로이며 char_start/end로 복원할 수 있다.
4. 임베딩에는 검색 문맥을 보강한 embedding_text를 사용하고, 인용에는 chunk_text를 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error


REQUIRED_COLUMNS = [
    "doc_id", "source_file", "source_record_index", "council", "committee",
    "meeting_name", "meeting_date", "meeting_year", "meeting_month",
    "meeting_type_derived", "clean_text",
]

SPEAKER_MARK_RE = re.compile(r"^[○◯●◎]\s*(?P<body>.+)$")
ROLE_SUFFIX = (
    r"(?:위원장대리|위원장|부위원장|의장|부의장|위원|의원|도지사|시장|교육감|"
    r"부시장|부교육감|본부장|국장|과장|실장|소장|원장|처장|차장|팀장|"
    r"담당관|전문위원|의사관|사무처장|사무국장|교육장|주무관|공무원|"
    r"증인|참고인)"
)
# 교육국장, 도시주택실장처럼 부서명이 직책 앞에 붙는 표기도 포함한다.
ROLE_WORD = rf"(?:[가-힣A-Za-z·]{{0,15}}{ROLE_SUFFIX}|관계공무원)"
ROLE_FIRST_RE = re.compile(rf"^(?P<role>{ROLE_WORD})(?:\s+(?P<name>[가-힣·]{{2,7}}))?(?=\s|$)")
NAME_FIRST_RE = re.compile(rf"^(?P<name>[가-힣·]{{2,7}})\s+(?P<role>{ROLE_WORD})(?=\s|$)")
NAME_ROLE_COMBINED_RE = re.compile(
    rf"^(?P<name>[가-힣·]{{2,7}})(?P<role>{ROLE_SUFFIX})(?=\s|$)"
)

AGENDA_ITEM_RE = re.compile(
    r"^(?:제\s*)?\d{1,3}\s*[.)]\s+.+|"
    r"^[가-힣]\s*[.)]\s+.+|"
    r"^(?:안건|의안)\s*제?\s*\d+\s*[호.]?.+"
)
AGENDA_HEADER_RE = re.compile(
    r"^(?:의사일정|부의된\s*안건|부의안건|심사된\s*안건|상정된\s*안건|"
    r"감사일정|보고사항|처리안건|오늘의\s*의사일정)(?:\s*\(.*\))?$"
)
SESSION_RE = re.compile(
    r"^\([^\n]{0,50}(?:개의|개회|속개|정회|산회|감사개시|감사중지|감사종료)[^\n]{0,20}\)$"
)
SPACE_RE = re.compile(r"[ \t]+")

CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("chunk_index", pa.int32()),
    ("source_file", pa.string()),
    ("source_record_index", pa.int64()),
    ("council", pa.string()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("meeting_date", pa.string()),
    ("meeting_year", pa.int32()),
    ("meeting_month", pa.string()),
    ("meeting_type", pa.string()),
    ("chunk_type", pa.string()),
    ("agenda_title", pa.string()),
    ("speaker_label", pa.string()),
    ("speaker_role", pa.string()),
    ("speaker_name", pa.string()),
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
    ("chunk_char_count", pa.int32()),
    ("estimated_tokens", pa.int32()),
    ("embedding_priority", pa.string()),
    ("evidence_ref", pa.string()),
    ("chunk_sha256", pa.string()),
    ("chunk_text", pa.string()),
    ("embedding_text", pa.string()),
])


@dataclass
class Segment:
    start: int
    end: int
    chunk_type: str
    agenda_title: str = ""
    speaker_label: str = ""
    speaker_role: str = ""
    speaker_name: str = ""


@dataclass
class ChunkRange:
    start: int
    end: int
    chunk_type: str
    agenda_title: str = ""
    speaker_label: str = ""
    speaker_role: str = ""
    speaker_name: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="발언자/안건 기반 회의록 청크 생성")
    parser.add_argument("--input", default="data/processed/documents.parquet", help="정제 문서 Parquet")
    parser.add_argument("--output", default="data/processed/chunks.parquet", help="청크 Parquet")
    parser.add_argument("--report-dir", default="outputs/chunks", help="청크 분석 보고서 폴더")
    parser.add_argument("--target-chars", type=int, default=1400, help="권장 청크 길이")
    parser.add_argument("--max-chars", type=int, default=1800, help="최대 청크 길이")
    parser.add_argument("--overlap-chars", type=int, default=180, help="긴 발언 분할 시 겹침")
    parser.add_argument("--min-chars", type=int, default=200, help="짧은 구간 병합 기준")
    parser.add_argument("--read-batch-size", type=int, default=16, help="문서 Parquet 읽기 배치")
    parser.add_argument("--write-batch-size", type=int, default=2000, help="청크 Parquet 쓰기 배치")
    parser.add_argument("--sample-per-council", type=int, default=3, help="의회별 샘플 청크 수")
    parser.add_argument(
        "--cutoff-date",
        default="",
        help="선택: YYYY-MM-DD 또는 YYYYMMDD. 이후 문서는 제외",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_chars < 200:
        raise SystemExit("--target-chars는 200 이상이어야 합니다.")
    if args.max_chars < args.target_chars:
        raise SystemExit("--max-chars는 --target-chars 이상이어야 합니다.")
    if args.min_chars < 1 or args.min_chars >= args.max_chars:
        raise SystemExit("--min-chars는 1 이상, --max-chars 미만이어야 합니다.")
    if args.overlap_chars < 0 or args.overlap_chars >= args.target_chars:
        raise SystemExit("--overlap-chars는 0 이상, --target-chars 미만이어야 합니다.")
    if args.read_batch_size < 1 or args.write_batch_size < 1:
        raise SystemExit("배치 크기는 1 이상이어야 합니다.")


def normalize_cutoff_date(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        raise SystemExit("--cutoff-date는 YYYY-MM-DD 또는 YYYYMMDD 형식이어야 합니다.")
    year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
    try:
        import datetime as dt
        return dt.date(year, month, day).isoformat()
    except ValueError as error:
        raise SystemExit(f"유효하지 않은 --cutoff-date: {value}") from error


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_label(value: str, limit: int = 300) -> str:
    value = SPACE_RE.sub(" ", safe_text(value))
    return value[:limit].strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def line_offsets(text: str):
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        end = offset + len(raw_line)
        line = raw_line.rstrip("\r\n")
        yield offset, end, line
        offset = end
    if offset < len(text):
        yield offset, len(text), text[offset:]


def extract_speaker(line: str) -> tuple[str, str, str]:
    match = SPEAKER_MARK_RE.match(line.strip())
    if not match:
        return "", "", ""

    body = SPACE_RE.sub(" ", match.group("body").strip())
    role_match = ROLE_FIRST_RE.match(body)
    if role_match:
        role = safe_text(role_match.group("role"))
        name = safe_text(role_match.group("name"))
        label = " ".join(value for value in (role, name) if value)
        return label, role, name

    name_match = NAME_FIRST_RE.match(body)
    if name_match:
        name = safe_text(name_match.group("name"))
        role = safe_text(name_match.group("role"))
        return f"{name} {role}", role, name

    combined_match = NAME_ROLE_COMBINED_RE.match(body)
    if combined_match:
        name = safe_text(combined_match.group("name"))
        role = safe_text(combined_match.group("role"))
        return f"{name} {role}", role, name

    # ○ 간부공무원 소개, ○ 5분 자유발언 같은 안건 표식을 발언자로 오인하지 않는다.
    return "", "", ""


def is_agenda_item(line: str, seen_speaker: bool) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 350:
        return False
    if AGENDA_HEADER_RE.fullmatch(stripped):
        return True
    # 머리말의 목차 번호를 각각 별도 청크로 만들지 않는다. 실제 발언이 시작된 뒤의
    # 번호 안건만 새로운 안건 경계로 간주한다.
    return seen_speaker and bool(AGENDA_ITEM_RE.match(stripped))


def trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_segments(text: str) -> list[Segment]:
    if not text.strip():
        return []

    segments: list[Segment] = []
    current_start = 0
    current_type = "header"
    current_agenda = ""
    current_speaker = ""
    current_role = ""
    current_name = ""
    seen_speaker = False

    def close_segment(end: int) -> None:
        nonlocal current_start
        start, clean_end = trim_range(text, current_start, end)
        if start < clean_end:
            segments.append(Segment(
                start=start,
                end=clean_end,
                chunk_type=current_type,
                agenda_title=current_agenda,
                speaker_label=current_speaker,
                speaker_role=current_role,
                speaker_name=current_name,
            ))
        current_start = end

    for start, _, line in line_offsets(text):
        stripped = line.strip()
        if not stripped:
            continue

        speaker_label, speaker_role, speaker_name = extract_speaker(stripped)
        if speaker_label:
            if start > current_start:
                close_segment(start)
            current_start = start
            current_type = "speech"
            current_speaker = speaker_label
            current_role = speaker_role
            current_name = speaker_name
            seen_speaker = True
            continue

        if is_agenda_item(stripped, seen_speaker):
            if start > current_start:
                close_segment(start)
            current_start = start
            current_type = "agenda"
            current_agenda = clean_label(stripped)
            current_speaker = ""
            current_role = ""
            current_name = ""
            continue

        if SESSION_RE.fullmatch(stripped):
            if start > current_start:
                close_segment(start)
            current_start = start
            current_type = "proceeding"
            current_speaker = ""
            current_role = ""
            current_name = ""

    close_segment(len(text))
    return merge_context_segments(text, segments)


def merge_context_segments(text: str, segments: list[Segment]) -> list[Segment]:
    """짧은 안건명/진행문을 바로 다음 발언에 붙여 검색 문맥을 보존한다."""
    if not segments:
        return []
    merged: list[Segment] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        current_length = current.end - current.start
        if (
            current.chunk_type in {"header", "agenda", "proceeding"}
            and current_length < 200
            and index + 1 < len(segments)
        ):
            nxt = segments[index + 1]
            agenda = current.agenda_title or nxt.agenda_title
            merged.append(Segment(
                start=current.start,
                end=nxt.end,
                chunk_type=nxt.chunk_type if nxt.chunk_type == "speech" else current.chunk_type,
                agenda_title=agenda,
                speaker_label=nxt.speaker_label,
                speaker_role=nxt.speaker_role,
                speaker_name=nxt.speaker_name,
            ))
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def choose_split_end(text: str, start: int, hard_end: int, max_chars: int) -> int:
    minimum = min(hard_end, start + max(100, int(max_chars * 0.62)))
    candidates: list[int] = []
    for token in ("\n", "다. ", "니다. ", "요. ", "? ", "! ", ". "):
        position = text.rfind(token, minimum, hard_end)
        if position >= minimum:
            candidates.append(position + len(token))
    return max(candidates) if candidates else hard_end


def choose_overlap_start(text: str, previous_start: int, split_end: int, overlap_chars: int) -> int:
    if overlap_chars <= 0:
        return split_end
    floor = max(previous_start + 1, split_end - overlap_chars)
    newline = text.find("\n", floor, split_end)
    if newline >= 0:
        return newline + 1
    whitespace = re.search(r"\s", text[floor:split_end])
    if whitespace:
        return floor + whitespace.end()
    return floor


def split_segment(text: str, segment: Segment, max_chars: int, overlap_chars: int) -> list[ChunkRange]:
    ranges: list[ChunkRange] = []
    position = segment.start
    while position < segment.end:
        hard_end = min(position + max_chars, segment.end)
        split_end = segment.end if hard_end >= segment.end else choose_split_end(text, position, hard_end, max_chars)
        start, end = trim_range(text, position, split_end)
        if start < end:
            ranges.append(ChunkRange(
                start=start,
                end=end,
                chunk_type=segment.chunk_type,
                agenda_title=segment.agenda_title,
                speaker_label=segment.speaker_label,
                speaker_role=segment.speaker_role,
                speaker_name=segment.speaker_name,
            ))
        if split_end >= segment.end:
            break
        next_position = choose_overlap_start(text, position, split_end, overlap_chars)
        position = split_end if next_position <= position else next_position
    return ranges


def same_context(left: ChunkRange, right: ChunkRange) -> bool:
    return (
        left.chunk_type == right.chunk_type
        and left.agenda_title == right.agenda_title
        and left.speaker_label == right.speaker_label
    )


def combine_labels(left: str, right: str, limit: int = 300) -> str:
    values: list[str] = []
    for source in (left, right):
        for value in safe_text(source).split("; "):
            value = value.strip()
            if value and value not in values:
                values.append(value)
    combined = "; ".join(values)
    return combined if len(combined) <= limit else combined[:limit].rstrip("; ")


def combine_ranges(left: ChunkRange, right: ChunkRange) -> ChunkRange:
    speakers = combine_labels(left.speaker_label, right.speaker_label)
    agendas = combine_labels(left.agenda_title, right.agenda_title)
    speaker_changed = bool(left.speaker_label and right.speaker_label and left.speaker_label != right.speaker_label)

    if speaker_changed or {left.chunk_type, right.chunk_type} <= {"speech", "dialogue"}:
        chunk_type = "dialogue" if speaker_changed else "speech"
    elif "speech" in {left.chunk_type, right.chunk_type} or "dialogue" in {left.chunk_type, right.chunk_type}:
        chunk_type = "dialogue" if speakers else "mixed"
    elif left.chunk_type == right.chunk_type:
        chunk_type = left.chunk_type
    else:
        chunk_type = "mixed"

    same_single_speaker = bool(
        left.speaker_label
        and left.speaker_label == right.speaker_label
        and "; " not in speakers
    )
    return ChunkRange(
        start=left.start,
        end=right.end,
        chunk_type=chunk_type,
        agenda_title=agendas,
        speaker_label=speakers,
        speaker_role=left.speaker_role if same_single_speaker else "",
        speaker_name=left.speaker_name if same_single_speaker else "",
    )


def coalesce_short_ranges(
    ranges: list[ChunkRange],
    target_chars: int,
    max_chars: int,
    min_chars: int,
) -> list[ChunkRange]:
    """짧은 발언 교환을 검색에 적당한 대화 청크로 묶는다.

    서로 다른 발언자가 포함되어도 원문은 연속 구간으로 유지하며 speaker_label에
    발언자 목록을 남긴다. 긴 발언의 overlap 구간은 current.start < previous.end이므로
    병합하지 않는다.
    """
    if not ranges:
        return []

    merged: list[ChunkRange] = []
    for current in ranges:
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        previous_length = previous.end - previous.start
        current_length = current.end - current.start
        combined_length = current.end - previous.start
        contiguous = current.start >= previous.end and current.start - previous.end <= 3
        conversational = (
            previous.chunk_type in {"speech", "dialogue"}
            and current.chunk_type in {"speech", "dialogue"}
        )
        should_merge = contiguous and combined_length <= max_chars and (
            previous_length < min_chars
            or current_length < min_chars
            or (conversational and combined_length <= target_chars)
        )
        if should_merge:
            merged[-1] = combine_ranges(previous, current)
        else:
            merged.append(current)

    # 첫 번째 순방향 병합 뒤에도 남은 아주 짧은 꼬리 청크를 앞 청크에 붙인다.
    final_ranges: list[ChunkRange] = []
    for current in merged:
        current_length = current.end - current.start
        if final_ranges and current_length < min_chars:
            previous = final_ranges[-1]
            combined_length = current.end - previous.start
            contiguous = current.start >= previous.end and current.start - previous.end <= 3
            if contiguous and combined_length <= max_chars:
                final_ranges[-1] = combine_ranges(previous, current)
                continue
        final_ranges.append(current)
    return final_ranges


def make_chunk_ranges(
    text: str,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[ChunkRange]:
    segments = build_segments(text)
    raw_ranges: list[ChunkRange] = []
    for segment in segments:
        if segment.end - segment.start > max_chars:
            raw_ranges.extend(split_segment(text, segment, max_chars, overlap_chars))
        else:
            raw_ranges.append(ChunkRange(**segment.__dict__))

    # 같은 발언자/안건의 짧은 연속 구간만 합친다. 서로 다른 발언자는 섞지 않는다.
    packed: list[ChunkRange] = []
    for current in raw_ranges:
        if packed:
            previous = packed[-1]
            combined_length = current.end - previous.start
            previous_length = previous.end - previous.start
            if (
                same_context(previous, current)
                and current.start >= previous.end
                and combined_length <= max_chars
                and (previous_length < min_chars or combined_length <= target_chars)
            ):
                previous.end = current.end
                continue
        packed.append(current)
    return coalesce_short_ranges(
        ranges=packed,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
    )


def make_embedding_text(document: dict, chunk: ChunkRange, chunk_text: str) -> str:
    fields = [
        f"[의회] {safe_text(document.get('council'))}",
        f"[회의일자] {safe_text(document.get('meeting_date'))}",
        f"[위원회] {safe_text(document.get('committee'))}",
        f"[회의록] {safe_text(document.get('meeting_name'))}",
    ]
    if chunk.agenda_title:
        fields.append(f"[안건] {chunk.agenda_title}")
    if chunk.speaker_label:
        fields.append(f"[발언자] {chunk.speaker_label}")
    fields.extend(("[본문]", chunk_text))
    return "\n".join(fields)


def estimate_tokens(text: str) -> int:
    # 한국어/숫자/기호가 섞인 회의록의 보수적 근사치. 실제 모델 토크나이저 값은 아니다.
    return max(1, math.ceil(len(text) / 1.8))


def embedding_priority(meeting_type: str, chunk_type: str, text_length: int) -> str:
    if meeting_type == "개회식" or chunk_type == "header":
        return "low"
    if chunk_type in {"speech", "dialogue"} and text_length >= 300:
        return "high"
    return "medium"


def make_chunk_id(doc_id: str, chunk_index: int, start: int, end: int, chunk_hash: str) -> str:
    value = f"{doc_id}\x1f{chunk_index}\x1f{start}\x1f{end}\x1f{chunk_hash}"
    return "chk_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flush_rows(writer: pq.ParquetWriter, rows: list[dict]) -> int:
    if not rows:
        return 0
    writer.write_table(pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA))
    count = len(rows)
    rows.clear()
    return count


def percentile(sorted_values: list[int], fraction: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * fraction)))
    return sorted_values[index]


def main() -> None:
    args = parse_args()
    validate_args(args)
    cutoff_date = normalize_cutoff_date(args.cutoff_date)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")
    if input_path == output_path:
        raise SystemExit("입력과 출력 경로는 달라야 합니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    source = pq.ParquetFile(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source.schema_arrow.names]
    if missing_columns:
        raise SystemExit(f"입력 Parquet 필수 컬럼 누락: {', '.join(missing_columns)}")

    rows_buffer: list[dict] = []
    samples: list[dict] = []
    quality_issues: list[dict] = []
    sample_counts: Counter = Counter()
    chunk_type_counts: Counter = Counter()
    priority_counts: Counter = Counter()
    council_chunk_counts: Counter = Counter()
    council_doc_ids: defaultdict[str, set[str]] = defaultdict(set)
    council_lengths: defaultdict[str, list[int]] = defaultdict(list)
    all_lengths: list[int] = []

    input_document_count = 0
    processed_document_count = 0
    excluded_by_cutoff = 0
    empty_document_count = 0
    total_chunks = 0

    writer = pq.ParquetWriter(
        output_path,
        CHUNK_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    try:
        batches = source.iter_batches(batch_size=args.read_batch_size, columns=REQUIRED_COLUMNS)
        for batch in batches:
            for document in batch.to_pylist():
                input_document_count += 1
                if input_document_count == 1 or input_document_count % 100 == 0:
                    print(
                        f"문서 {input_document_count:,}/{source.metadata.num_rows:,} | 청크 {total_chunks:,}",
                        flush=True,
                    )

                meeting_date = safe_text(document.get("meeting_date"))
                if cutoff_date and meeting_date and meeting_date > cutoff_date:
                    excluded_by_cutoff += 1
                    continue

                text = safe_text(document.get("clean_text"))
                if not text:
                    empty_document_count += 1
                    quality_issues.append({
                        "doc_id": safe_text(document.get("doc_id")),
                        "chunk_id": "",
                        "issue": "empty_document",
                        "detail": "clean_text가 비어 있음",
                    })
                    continue

                processed_document_count += 1
                ranges = make_chunk_ranges(
                    text=text,
                    target_chars=args.target_chars,
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                    min_chars=args.min_chars,
                )
                if not ranges:
                    quality_issues.append({
                        "doc_id": safe_text(document.get("doc_id")),
                        "chunk_id": "",
                        "issue": "no_chunks",
                        "detail": f"clean_text_chars={len(text)}",
                    })
                    continue

                doc_id = safe_text(document.get("doc_id"))
                council = safe_text(document.get("council"))
                meeting_type = safe_text(document.get("meeting_type_derived"))
                for chunk_index, chunk in enumerate(ranges):
                    chunk_text = text[chunk.start:chunk.end]
                    chunk_length = len(chunk_text)
                    chunk_hash = sha256_text(chunk_text)
                    chunk_id = make_chunk_id(
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        start=chunk.start,
                        end=chunk.end,
                        chunk_hash=chunk_hash,
                    )
                    priority = embedding_priority(meeting_type, chunk.chunk_type, chunk_length)
                    evidence_ref = (
                        f"{safe_text(document.get('meeting_name'))} | {meeting_date} | "
                        f"정제본문 {chunk.start}:{chunk.end}"
                    )
                    output_row = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "source_file": safe_text(document.get("source_file")),
                        "source_record_index": int(document.get("source_record_index") or 0),
                        "council": council,
                        "committee": safe_text(document.get("committee")),
                        "meeting_name": safe_text(document.get("meeting_name")),
                        "meeting_date": meeting_date,
                        "meeting_year": document.get("meeting_year"),
                        "meeting_month": safe_text(document.get("meeting_month")),
                        "meeting_type": meeting_type,
                        "chunk_type": chunk.chunk_type,
                        "agenda_title": chunk.agenda_title,
                        "speaker_label": chunk.speaker_label,
                        "speaker_role": chunk.speaker_role,
                        "speaker_name": chunk.speaker_name,
                        "char_start": chunk.start,
                        "char_end": chunk.end,
                        "chunk_char_count": chunk_length,
                        "estimated_tokens": estimate_tokens(chunk_text),
                        "embedding_priority": priority,
                        "evidence_ref": evidence_ref,
                        "chunk_sha256": chunk_hash,
                        "chunk_text": chunk_text,
                        "embedding_text": make_embedding_text(document, chunk, chunk_text),
                    }
                    rows_buffer.append(output_row)
                    total_chunks += 1
                    all_lengths.append(chunk_length)
                    chunk_type_counts[chunk.chunk_type] += 1
                    priority_counts[priority] += 1
                    council_chunk_counts[council] += 1
                    council_doc_ids[council].add(doc_id)
                    council_lengths[council].append(chunk_length)

                    if chunk_length > args.max_chars:
                        quality_issues.append({
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "issue": "exceeds_max_chars",
                            "detail": str(chunk_length),
                        })
                    elif chunk_length < 80:
                        quality_issues.append({
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "issue": "very_short_chunk",
                            "detail": str(chunk_length),
                        })

                    if sample_counts[council] < args.sample_per_council:
                        samples.append({
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "council": council,
                            "meeting_name": safe_text(document.get("meeting_name")),
                            "meeting_date": meeting_date,
                            "chunk_type": chunk.chunk_type,
                            "agenda_title": chunk.agenda_title,
                            "speaker_label": chunk.speaker_label,
                            "char_start": chunk.start,
                            "char_end": chunk.end,
                            "chunk_char_count": chunk_length,
                            "embedding_priority": priority,
                            "chunk_preview": chunk_text[:1000].replace("\n", " | "),
                        })
                        sample_counts[council] += 1

                    if len(rows_buffer) >= args.write_batch_size:
                        flush_rows(writer, rows_buffer)
    except Exception as error:
        print(f"오류: {type(error).__name__}: {error}", file=sys.stderr)
        raise
    finally:
        flush_rows(writer, rows_buffer)
        writer.close()

    sorted_lengths = sorted(all_lengths)
    council_rows: list[dict] = []
    for council in sorted(council_chunk_counts):
        lengths = sorted(council_lengths[council])
        council_rows.append({
            "council": council,
            "document_count": len(council_doc_ids[council]),
            "chunk_count": council_chunk_counts[council],
            "avg_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "p50_chunk_chars": percentile(lengths, 0.50),
            "p90_chunk_chars": percentile(lengths, 0.90),
            "max_chunk_chars": max(lengths) if lengths else 0,
        })

    sample_fields = [
        "chunk_id", "doc_id", "council", "meeting_name", "meeting_date", "chunk_type",
        "agenda_title", "speaker_label", "char_start", "char_end", "chunk_char_count",
        "embedding_priority", "chunk_preview",
    ]
    write_csv(report_dir / "chunk_samples.csv", samples, sample_fields)
    write_csv(
        report_dir / "chunk_quality_issues.csv",
        quality_issues,
        ["doc_id", "chunk_id", "issue", "detail"],
    )
    write_csv(
        report_dir / "council_chunk_summary.csv",
        council_rows,
        [
            "council", "document_count", "chunk_count", "avg_chunk_chars",
            "p50_chunk_chars", "p90_chunk_chars", "max_chunk_chars",
        ],
    )

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "cutoff_date": cutoff_date or None,
        "parameters": {
            "target_chars": args.target_chars,
            "max_chars": args.max_chars,
            "overlap_chars": args.overlap_chars,
            "min_chars": args.min_chars,
        },
        "input_document_count": input_document_count,
        "processed_document_count": processed_document_count,
        "excluded_by_cutoff": excluded_by_cutoff,
        "empty_document_count": empty_document_count,
        "total_chunk_count": total_chunks,
        "average_chunks_per_document": round(total_chunks / processed_document_count, 3)
        if processed_document_count else None,
        "chunk_length_chars": {
            "min": min(sorted_lengths) if sorted_lengths else None,
            "p10": percentile(sorted_lengths, 0.10),
            "p50": percentile(sorted_lengths, 0.50),
            "p90": percentile(sorted_lengths, 0.90),
            "p95": percentile(sorted_lengths, 0.95),
            "max": max(sorted_lengths) if sorted_lengths else None,
            "mean": round(sum(sorted_lengths) / len(sorted_lengths), 1) if sorted_lengths else None,
        },
        "chunk_type_counts": dict(chunk_type_counts.most_common()),
        "embedding_priority_counts": dict(priority_counts.most_common()),
        "quality_issue_count": len(quality_issues),
        "generated_files": [
            output_path.name,
            "chunking_summary.json",
            "council_chunk_summary.csv",
            "chunk_samples.csv",
            "chunk_quality_issues.csv",
        ],
    }
    (report_dir / "chunking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n완료")
    print(f"입력 문서: {input_document_count:,}")
    print(f"처리 문서: {processed_document_count:,}")
    print(f"생성 청크: {total_chunks:,}")
    if sorted_lengths:
        print(
            f"청크 길이: 평균 {summary['chunk_length_chars']['mean']:,}자 | "
            f"중앙값 {summary['chunk_length_chars']['p50']:,}자 | "
            f"최대 {summary['chunk_length_chars']['max']:,}자"
        )
    print(f"청크 데이터: {output_path}")
    print(f"보고서: {report_dir}")


if __name__ == "__main__":
    main()
