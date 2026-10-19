#!/usr/bin/env python3
"""광역 지방의회 회의록 전처리.

원본 JSON 배열을 스트리밍으로 읽고 HTML/웹페이지 잡음을 제거한 뒤,
문서 단위 Parquet과 품질 점검 보고서를 만든다. 원본 파일은 수정하지 않는다.

중복 처리 원칙
1. 메타데이터(의회/위원회/회의록명/날짜)만 같은 레코드는 삭제하지 않는다.
2. 메타데이터와 정제 본문이 모두 같은 경우만 exact duplicate로 제외한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import ijson
except ImportError as error:
    raise SystemExit("ijson이 필요합니다. 실행 전: pip install ijson") from error

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error

try:
    from bs4 import BeautifulSoup, Comment
except ImportError as error:
    raise SystemExit("beautifulsoup4가 필요합니다. 실행 전: pip install beautifulsoup4 lxml") from error


EXPECTED_COLUMNS = ["RASMBLY_NM", "MTGNM", "MINTS_NM", "MTG_DE", "MINTS_HTML"]
DATE_RE = re.compile(r"^\d{8}$")
SPACE_RE = re.compile(r"[ \t\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
SPEAKER_RE = re.compile(r"^[○◯●◎]\s*[^\s]{1,20}(?:\s+[^\s]{1,12})?")

# UI에서 생기는 단독 텍스트 노드. 실제 회의 내용일 가능성이 낮은 패턴만 제거한다.
NOISE_LINE_PATTERNS = [
    re.compile(r"^Toggle navigation$", re.IGNORECASE),
    re.compile(r"^(굴림체|돋움체|바탕체|궁서체)$"),
    re.compile(r"^\d{1,3}pt$", re.IGNORECASE),
    re.compile(r"^\d{2,3}%$"),
    re.compile(r"^글자체(?:를)?\s*(?:변경|변경합니다\.?|선택)$"),
    re.compile(r"^글자크기(?:를)?\s*(?:변경|변경합니다\.?|선택)$"),
    re.compile(r"^줄간격(?:을|를)?\s*(?:변경|변경합니다\.?)$"),
    re.compile(r"^(인쇄|프린트|목록|처음으로|맨위로|TOP|SNS)$", re.IGNORECASE),
    re.compile(r"^(페이스북|트위터|블로그|카카오톡|카카오스토리)(?:공유)?$"),
    re.compile(r"^(단축URL|URL)\s*(?:복사)?$", re.IGNORECASE),
    re.compile(r"^조회된\s+(?:관련\s+)?(?:안건|부록|회의록|영상|의안).*없습니다\.?$"),
    re.compile(r"^(?:관련\s+)?(?:부록|회의록|영상회의록|의안)이?\s*없습니다\.?$"),
]

UI_RESIDUE_RE = re.compile(
    r"Toggle navigation|글자크기(?:를)?\s*변경|줄간격(?:을|를)?\s*변경|"
    r"(?:10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|26|28|30|32|34|36)pt"
)

DOCUMENT_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("source_file", pa.string()),
    ("source_record_index", pa.int64()),
    ("council", pa.string()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("meeting_date", pa.string()),
    ("meeting_year", pa.int32()),
    ("meeting_month", pa.string()),
    ("meeting_type_derived", pa.string()),
    ("raw_html_chars", pa.int64()),
    ("extracted_text_chars", pa.int64()),
    ("clean_text_chars", pa.int64()),
    ("removed_text_chars", pa.int64()),
    ("replacement_char_count", pa.int32()),
    ("speaker_marker_count", pa.int32()),
    ("raw_html_sha256", pa.string()),
    ("content_sha256", pa.string()),
    ("clean_text", pa.string()),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="회의록 HTML 정제 및 문서 단위 Parquet 생성")
    parser.add_argument("--input-dir", default="data/raw", help="원본 JSON 폴더")
    parser.add_argument("--output-dir", default="data/processed", help="정제 데이터 저장 폴더")
    parser.add_argument("--report-dir", default="outputs/preprocess", help="품질 보고서 저장 폴더")
    parser.add_argument("--batch-size", type=int, default=100, help="Parquet 기록 배치 크기")
    parser.add_argument("--sample-per-council", type=int, default=2, help="의회별 정제 샘플 수")
    parser.add_argument(
        "--cutoff-date",
        default="",
        help="선택: YYYYMMDD. 이 날짜 이후 문서는 제외. 최종 제출 기준일이 정해진 경우 사용",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="메타데이터와 정제 본문이 모두 같은 정확 중복도 보존",
    )
    return parser.parse_args()


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def find_json_files(input_dir: Path) -> list[Path]:
    extensions = {".json", ".jsonl", ".ndjson"}
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def detect_encoding(path: Path) -> str:
    with path.open("rb") as file:
        raw = file.read(65536)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    if raw.count(b"\x00") > len(raw) * 0.2:
        for encoding in ("utf-16-le", "utf-16-be"):
            try:
                decoded = raw.decode(encoding).lstrip()
                if decoded.startswith("[") or decoded.startswith("{"):
                    return encoding
            except UnicodeDecodeError:
                continue
    decoded = raw.decode("utf-8", errors="replace").lstrip()
    if decoded.startswith("[") or decoded.startswith("{"):
        return "utf-8"
    raise UnicodeError(f"지원하지 않는 인코딩: {path}")


def first_non_whitespace_char(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace") as file:
        while True:
            char = file.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def iter_records(path: Path, encoding: str):
    first_char = first_non_whitespace_char(path, encoding)
    if first_char == "[":
        with path.open("r", encoding=encoding, errors="replace") as file:
            yield from ijson.items(file, "item")
        return
    if first_char == "{":
        raise ValueError("JSON 객체 형식입니다. 최상위 배열 형식이 아닙니다.")
    if first_char == "":
        return
    with path.open("r", encoding=encoding, errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON Lines 파싱 오류: {line_number}번째 줄") from error


def parse_meeting_date(value: object) -> tuple[str, int | None, str]:
    raw = safe_text(value).replace("-", "").replace(".", "")
    if not DATE_RE.fullmatch(raw):
        return "", None, ""
    try:
        parsed = datetime.strptime(raw, "%Y%m%d")
    except ValueError:
        return "", None, ""
    return parsed.strftime("%Y-%m-%d"), parsed.year, parsed.strftime("%Y-%m")


def is_noise_line(line: str) -> bool:
    return any(pattern.fullmatch(line) for pattern in NOISE_LINE_PATTERNS)


def normalize_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", raw_text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    clean_lines: list[str] = []
    for line in text.split("\n"):
        line = SPACE_RE.sub(" ", line).strip()
        if not line or is_noise_line(line):
            continue
        clean_lines.append(line)

    # 일부 사이트는 UI 전체를 한 텍스트 노드로 제공한다. 회의 본문 시작 표식 앞의
    # Toggle navigation 블록만 잘라내고 실제 안건/발언 내용은 보존한다.
    if clean_lines and "Toggle navigation" in clean_lines[0]:
        content_markers = ("부의된 안건", "부의안건", "의사일정", "심사된 안건", "개의")
        positions = [clean_lines[0].find(marker) for marker in content_markers]
        positions = [position for position in positions if position >= 0]
        if positions:
            clean_lines[0] = clean_lines[0][min(positions):].strip()

    text = "\n".join(line for line in clean_lines if line)
    return BLANK_LINES_RE.sub("\n\n", text).strip()


def html_to_clean_text(raw_html: str) -> tuple[str, int]:
    parser = "lxml"
    try:
        soup = BeautifulSoup(raw_html, parser)
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")

    for node in soup(["script", "style", "noscript", "svg", "canvas", "form", "button"]):
        node.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    extracted = soup.get_text("\n")
    return normalize_text(extracted), len(normalize_text_basic(extracted))


def normalize_text_basic(text: str) -> str:
    """잡음 제거 전 비교용 길이를 계산하는 가벼운 정규화."""
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub("", text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def derive_meeting_type(committee: str, meeting_name: str, clean_text: str) -> str:
    value = f"{committee} {meeting_name} {clean_text[:500]}"
    if "개회식" in value:
        return "개회식"
    if "본회의" in value:
        return "본회의"
    if "예산결산특별위원회" in value or "예결특위" in value:
        return "예산결산특별위원회"
    if "특별위원회" in value or "특위" in value:
        return "특별위원회"
    if "행정사무감사" in value:
        return "행정사무감사"
    if "위원회" in value:
        return "위원회"
    return "기타"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def metadata_key(council: str, committee: str, meeting_name: str, meeting_date: str) -> str:
    return "\x1f".join((council, committee, meeting_name, meeting_date))


def make_doc_id(
    council: str,
    committee: str,
    meeting_name: str,
    meeting_date: str,
    content_sha256: str,
) -> str:
    value = metadata_key(council, committee, meeting_name, meeting_date) + "\x1f" + content_sha256
    return "doc_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def count_speaker_markers(text: str) -> int:
    return sum(1 for line in text.splitlines() if SPEAKER_RE.match(line))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flush_batch(writer: pq.ParquetWriter, rows: list[dict]) -> int:
    if not rows:
        return 0
    table = pa.Table.from_pylist(rows, schema=DOCUMENT_SCHEMA)
    writer.write_table(table)
    count = len(rows)
    rows.clear()
    return count


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"입력 폴더가 없습니다: {input_dir}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")

    cutoff_date = ""
    if args.cutoff_date:
        parsed, _, _ = parse_meeting_date(args.cutoff_date)
        if not parsed:
            raise SystemExit("--cutoff-date는 YYYYMMDD 형식의 유효한 날짜여야 합니다.")
        cutoff_date = parsed

    json_files = find_json_files(input_dir)
    if not json_files:
        raise SystemExit(f"JSON 파일을 찾지 못했습니다: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "documents.parquet"

    batch: list[dict] = []
    exact_duplicates: list[dict] = []
    metadata_collisions: list[dict] = []
    quality_issues: list[dict] = []
    cleaning_samples: list[dict] = []
    file_errors: list[dict] = []
    sample_counts: Counter = Counter()
    meeting_type_counts: Counter = Counter()
    council_counts: Counter = Counter()
    metadata_seen: dict[str, tuple[str, str]] = {}
    exact_seen: dict[str, tuple[str, str, int]] = {}
    metadata_occurrence: defaultdict[str, int] = defaultdict(int)

    total_input_records = 0
    total_output_records = 0
    total_cutoff_excluded = 0
    total_exact_duplicates = 0
    total_raw_html_chars = 0
    total_clean_text_chars = 0

    writer = pq.ParquetWriter(
        parquet_path,
        DOCUMENT_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    try:
        for file_index, path in enumerate(json_files, start=1):
            relative_path = path.relative_to(input_dir).as_posix()
            print(f"[{file_index}/{len(json_files)}] {relative_path}", flush=True)
            try:
                encoding = detect_encoding(path)
                for source_record_index, record in enumerate(iter_records(path, encoding), start=1):
                    total_input_records += 1
                    if not isinstance(record, dict):
                        quality_issues.append({
                            "source_file": relative_path,
                            "source_record_index": source_record_index,
                            "doc_id": "",
                            "issue": "record_not_object",
                            "detail": type(record).__name__,
                        })
                        continue

                    council = safe_text(record.get("RASMBLY_NM"))
                    committee = safe_text(record.get("MTGNM"))
                    meeting_name = safe_text(record.get("MINTS_NM"))
                    meeting_date, meeting_year, meeting_month = parse_meeting_date(record.get("MTG_DE"))

                    if cutoff_date and meeting_date and meeting_date > cutoff_date:
                        total_cutoff_excluded += 1
                        continue

                    raw_html = safe_text(record.get("MINTS_HTML"))
                    clean_text, extracted_text_chars = html_to_clean_text(raw_html)
                    raw_html_sha256 = sha256_text(raw_html)
                    content_sha256 = sha256_text(clean_text)
                    doc_id = make_doc_id(council, committee, meeting_name, meeting_date, content_sha256)
                    meta_key = metadata_key(council, committee, meeting_name, meeting_date)
                    exact_key = meta_key + "\x1f" + content_sha256

                    metadata_occurrence[meta_key] += 1
                    if meta_key in metadata_seen:
                        first_doc_id, first_source = metadata_seen[meta_key]
                        metadata_collisions.append({
                            "metadata_key_sha256": sha256_text(meta_key),
                            "occurrence": metadata_occurrence[meta_key],
                            "first_doc_id": first_doc_id,
                            "first_source": first_source,
                            "current_doc_id": doc_id,
                            "current_source": f"{relative_path}:{source_record_index}",
                            "same_clean_text": exact_key in exact_seen,
                            "council": council,
                            "committee": committee,
                            "meeting_name": meeting_name,
                            "meeting_date": meeting_date,
                        })
                    else:
                        metadata_seen[meta_key] = (doc_id, f"{relative_path}:{source_record_index}")

                    if exact_key in exact_seen:
                        total_exact_duplicates += 1
                        first_doc_id, first_source, first_index = exact_seen[exact_key]
                        exact_duplicates.append({
                            "duplicate_doc_id": doc_id,
                            "duplicate_source_file": relative_path,
                            "duplicate_source_record_index": source_record_index,
                            "kept_doc_id": first_doc_id,
                            "kept_source_file": first_source,
                            "kept_source_record_index": first_index,
                            "council": council,
                            "committee": committee,
                            "meeting_name": meeting_name,
                            "meeting_date": meeting_date,
                            "content_sha256": content_sha256,
                        })
                        if not args.keep_duplicates:
                            continue
                    else:
                        exact_seen[exact_key] = (doc_id, relative_path, source_record_index)

                    replacement_char_count = clean_text.count("�")
                    speaker_marker_count = count_speaker_markers(clean_text)
                    clean_text_chars = len(clean_text)
                    removed_text_chars = max(0, extracted_text_chars - clean_text_chars)
                    meeting_type = derive_meeting_type(committee, meeting_name, clean_text)

                    issues: list[tuple[str, str]] = []
                    if not council:
                        issues.append(("missing_council", "RASMBLY_NM이 비어 있음"))
                    if not meeting_date:
                        issues.append(("invalid_date", safe_text(record.get("MTG_DE"))))
                    if not clean_text:
                        issues.append(("empty_clean_text", "정제 후 본문이 비어 있음"))
                    elif clean_text_chars < 500:
                        issues.append(("very_short_text", str(clean_text_chars)))
                    if replacement_char_count:
                        issues.append(("replacement_characters", str(replacement_char_count)))
                    residue = UI_RESIDUE_RE.search(clean_text[:5000])
                    if residue:
                        issues.append(("ui_residue", residue.group(0)))
                    for issue, detail in issues:
                        quality_issues.append({
                            "source_file": relative_path,
                            "source_record_index": source_record_index,
                            "doc_id": doc_id,
                            "issue": issue,
                            "detail": detail,
                        })

                    row = {
                        "doc_id": doc_id,
                        "source_file": relative_path,
                        "source_record_index": source_record_index,
                        "council": council,
                        "committee": committee,
                        "meeting_name": meeting_name,
                        "meeting_date": meeting_date,
                        "meeting_year": meeting_year,
                        "meeting_month": meeting_month,
                        "meeting_type_derived": meeting_type,
                        "raw_html_chars": len(raw_html),
                        "extracted_text_chars": extracted_text_chars,
                        "clean_text_chars": clean_text_chars,
                        "removed_text_chars": removed_text_chars,
                        "replacement_char_count": replacement_char_count,
                        "speaker_marker_count": speaker_marker_count,
                        "raw_html_sha256": raw_html_sha256,
                        "content_sha256": content_sha256,
                        "clean_text": clean_text,
                    }
                    batch.append(row)
                    total_output_records += 1
                    total_raw_html_chars += len(raw_html)
                    total_clean_text_chars += clean_text_chars
                    council_counts[council] += 1
                    meeting_type_counts[meeting_type] += 1

                    if sample_counts[council] < args.sample_per_council:
                        cleaning_samples.append({
                            "doc_id": doc_id,
                            "council": council,
                            "meeting_name": meeting_name,
                            "meeting_date": meeting_date,
                            "before_preview": normalize_text_basic(BeautifulSoup(raw_html, "html.parser").get_text(" "))[:800],
                            "after_preview": clean_text[:800].replace("\n", " | "),
                            "clean_text_chars": clean_text_chars,
                            "speaker_marker_count": speaker_marker_count,
                        })
                        sample_counts[council] += 1

                    if len(batch) >= args.batch_size:
                        flush_batch(writer, batch)
            except Exception as error:
                file_errors.append({
                    "source_file": relative_path,
                    "error": f"{type(error).__name__}: {error}",
                })
                print(f"  오류: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        flush_batch(writer, batch)
        writer.close()

    exact_fields = [
        "duplicate_doc_id", "duplicate_source_file", "duplicate_source_record_index",
        "kept_doc_id", "kept_source_file", "kept_source_record_index", "council",
        "committee", "meeting_name", "meeting_date", "content_sha256",
    ]
    collision_fields = [
        "metadata_key_sha256", "occurrence", "first_doc_id", "first_source",
        "current_doc_id", "current_source", "same_clean_text", "council",
        "committee", "meeting_name", "meeting_date",
    ]
    issue_fields = ["source_file", "source_record_index", "doc_id", "issue", "detail"]
    sample_fields = [
        "doc_id", "council", "meeting_name", "meeting_date", "before_preview",
        "after_preview", "clean_text_chars", "speaker_marker_count",
    ]
    write_csv(report_dir / "exact_duplicates.csv", exact_duplicates, exact_fields)
    write_csv(report_dir / "metadata_collisions.csv", metadata_collisions, collision_fields)
    write_csv(report_dir / "quality_issues.csv", quality_issues, issue_fields)
    write_csv(report_dir / "cleaning_samples.csv", cleaning_samples, sample_fields)
    write_csv(report_dir / "file_errors.csv", file_errors, ["source_file", "error"])

    summary = {
        "input_dir": str(input_dir),
        "documents_parquet": str(parquet_path),
        "report_dir": str(report_dir),
        "file_count": len(json_files),
        "input_record_count": total_input_records,
        "output_record_count": total_output_records,
        "cutoff_date": cutoff_date or None,
        "cutoff_excluded_count": total_cutoff_excluded,
        "exact_duplicate_count": total_exact_duplicates,
        "exact_duplicates_removed": 0 if args.keep_duplicates else total_exact_duplicates,
        "metadata_collision_count": len(metadata_collisions),
        "quality_issue_count": len(quality_issues),
        "file_error_count": len(file_errors),
        "raw_html_char_count": total_raw_html_chars,
        "clean_text_char_count": total_clean_text_chars,
        "clean_to_raw_html_ratio": round(total_clean_text_chars / total_raw_html_chars, 6)
        if total_raw_html_chars else None,
        "council_record_counts": dict(council_counts.most_common()),
        "meeting_type_counts": dict(meeting_type_counts.most_common()),
        "generated_files": [
            "documents.parquet",
            "preprocess_summary.json",
            "exact_duplicates.csv",
            "metadata_collisions.csv",
            "quality_issues.csv",
            "cleaning_samples.csv",
            "file_errors.csv",
        ],
    }
    (report_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n완료")
    print(f"입력 레코드: {total_input_records:,}")
    print(f"출력 문서: {total_output_records:,}")
    print(f"정확 중복: {total_exact_duplicates:,}")
    print(f"메타데이터 충돌: {len(metadata_collisions):,}")
    print(f"품질 이슈: {len(quality_issues):,}")
    print(f"파일 오류: {len(file_errors):,}")
    print(f"문서 데이터: {parquet_path}")
    print(f"보고서: {report_dir}")


if __name__ == "__main__":
    main()
