#!/usr/bin/env python3
"""광역 지방의회 회의록 원본 JSON 파일 전수 인벤토리.

원본 데이터는 수정하지 않는다. 파일별 구조, 날짜 범위, 메타데이터,
본문 품질, 중복 여부를 outputs/inventory에 저장한다.
"""

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import ijson
except ImportError as error:
    raise SystemExit("ijson이 필요합니다. 실행 전: pip install ijson") from error


EXPECTED_COLUMNS = ["RASMBLY_NM", "MTGNM", "MINTS_NM", "MTG_DE", "MINTS_HTML"]
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
DATE_RE = re.compile(r"^\d{8}$")


def parse_args():
    parser = argparse.ArgumentParser(description="회의록 JSON 파일 전수 인벤토리")
    parser.add_argument("--input-dir", default="data/raw", help="원본 JSON 폴더")
    parser.add_argument("--output-dir", default="outputs/inventory", help="결과 저장 폴더")
    parser.add_argument("--sample-per-file", type=int, default=3, help="파일별 샘플 레코드 수")
    return parser.parse_args()


def find_json_files(input_dir):
    extensions = {".json", ".jsonl", ".ndjson"}
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def detect_encoding(path):
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

    # JSON 구조 문자는 ASCII이므로, 일부 잘못된 UTF-8 바이트만 치환해 계속 분석한다.
    decoded = raw.decode("utf-8", errors="replace").lstrip()
    if decoded.startswith("[") or decoded.startswith("{"):
        return "utf-8"

    raise UnicodeError(f"지원하지 않는 인코딩: {path}")


def first_non_whitespace_char(path, encoding):
    with path.open("r", encoding=encoding, errors="replace") as file:
        while True:
            char = file.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def iter_records(path, encoding):
    first_char = first_non_whitespace_char(path, encoding)
    if first_char == "[":
        # binary 모드는 UTF-16 JSON을 처리하지 못할 수 있으므로 텍스트 모드로 연다.
        with path.open("r", encoding=encoding, errors="replace") as file:
            yield from ijson.items(file, "item")
        return
    if first_char == "{":
        raise ValueError("JSON 객체 형식입니다. 최상위 배열 형식이 아닙니다.")
    with path.open("r", encoding=encoding, errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON Lines 파싱 오류: {line_number}번째 줄") from error


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def html_text_length(raw_html):
    text = html.unescape(TAG_RE.sub(" ", safe_text(raw_html)))
    return len(SPACE_RE.sub(" ", text).strip())


def parse_date(value):
    value = safe_text(value)
    if not DATE_RE.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def metadata_hash(record):
    values = [safe_text(record.get(column)) for column in EXPECTED_COLUMNS[:4]]
    return hashlib.sha1("\x1f".join(values).encode("utf-8")).hexdigest()


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"입력 폴더가 없습니다: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(input_dir)
    if not json_files:
        raise SystemExit(f"JSON 파일을 찾지 못했습니다: {input_dir}")

    file_rows = []
    schema_counter = defaultdict(lambda: {"present_count": 0, "non_empty_count": 0, "types": Counter()})
    council_counter = Counter()
    committee_counter = Counter()
    month_counter = Counter()
    global_seen = set()
    global_duplicates = 0
    samples = []
    total_records = 0
    total_empty_html = 0
    total_invalid_date = 0
    total_html_chars = 0

    for file_index, path in enumerate(json_files, start=1):
        relative_path = path.relative_to(input_dir).as_posix()
        print(f"[{file_index}/{len(json_files)}] {relative_path}", flush=True)
        encoding = detect_encoding(path)
        record_count = 0
        empty_html_count = 0
        invalid_date_count = 0
        duplicate_count = 0
        html_chars = 0
        min_date = None
        max_date = None
        file_columns = Counter()
        sample_count = 0
        error_message = ""

        try:
            for record in iter_records(path, encoding):
                if not isinstance(record, dict):
                    raise ValueError("레코드가 JSON 객체가 아닙니다.")

                record_count += 1
                total_records += 1
                for column, value in record.items():
                    file_columns[column] += 1
                    schema_counter[column]["present_count"] += 1
                    schema_counter[column]["types"][type(value).__name__] += 1
                    if safe_text(value):
                        schema_counter[column]["non_empty_count"] += 1

                council = safe_text(record.get("RASMBLY_NM"))
                committee = safe_text(record.get("MTGNM"))
                if council:
                    council_counter[council] += 1
                if committee:
                    committee_counter[committee] += 1

                meeting_date = parse_date(record.get("MTG_DE"))
                if meeting_date is None:
                    invalid_date_count += 1
                    total_invalid_date += 1
                else:
                    min_date = meeting_date if min_date is None else min(min_date, meeting_date)
                    max_date = meeting_date if max_date is None else max(max_date, meeting_date)
                    month_counter[meeting_date.strftime("%Y-%m")] += 1

                text_length = html_text_length(record.get("MINTS_HTML"))
                html_chars += text_length
                total_html_chars += text_length
                if text_length == 0:
                    empty_html_count += 1
                    total_empty_html += 1

                key = metadata_hash(record)
                if key in global_seen:
                    duplicate_count += 1
                    global_duplicates += 1
                else:
                    global_seen.add(key)

                if sample_count < args.sample_per_file:
                    raw_html = safe_text(record.get("MINTS_HTML"))
                    preview = SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", raw_html))).strip()[:500]
                    samples.append({
                        "source_file": relative_path,
                        "council": council,
                        "committee": committee,
                        "meeting_name": safe_text(record.get("MINTS_NM")),
                        "meeting_date": safe_text(record.get("MTG_DE")),
                        "html_preview": preview,
                        "html_text_length": text_length,
                    })
                    sample_count += 1
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"

        missing_columns = [column for column in EXPECTED_COLUMNS if column not in file_columns]
        file_rows.append({
            "source_file": relative_path,
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "encoding": encoding,
            "record_count": record_count,
            "min_date": min_date.isoformat() if min_date else "",
            "max_date": max_date.isoformat() if max_date else "",
            "empty_html_count": empty_html_count,
            "empty_html_ratio": round(empty_html_count / record_count, 6) if record_count else "",
            "invalid_date_count": invalid_date_count,
            "duplicate_count_global": duplicate_count,
            "avg_text_length": round(html_chars / record_count, 1) if record_count else "",
            "column_count": len(file_columns),
            "columns": "; ".join(sorted(file_columns)),
            "missing_expected_columns": "; ".join(missing_columns),
            "error": error_message,
        })

    schema_rows = []
    for column, info in sorted(schema_counter.items()):
        schema_rows.append({
            "column": column,
            "present_count": info["present_count"],
            "present_ratio": round(info["present_count"] / total_records, 6) if total_records else 0,
            "non_empty_count": info["non_empty_count"],
            "non_empty_ratio": round(info["non_empty_count"] / total_records, 6) if total_records else 0,
            "observed_types": "; ".join(f"{name}:{count}" for name, count in info["types"].most_common()),
        })

    write_csv(output_dir / "file_inventory.csv", file_rows, list(file_rows[0].keys()))
    write_csv(output_dir / "schema_summary.csv", schema_rows, list(schema_rows[0].keys()))
    write_csv(output_dir / "council_summary.csv", [{"council": key, "record_count": value} for key, value in council_counter.most_common()], ["council", "record_count"])
    write_csv(output_dir / "committee_summary.csv", [{"committee": key, "record_count": value} for key, value in committee_counter.most_common()], ["committee", "record_count"])
    write_csv(output_dir / "month_summary.csv", [{"month": key, "record_count": value} for key, value in sorted(month_counter.items())], ["month", "record_count"])
    write_csv(output_dir / "sample_records.csv", samples, ["source_file", "council", "committee", "meeting_name", "meeting_date", "html_preview", "html_text_length"])

    valid_dates = [row["min_date"] for row in file_rows if row["min_date"]]
    max_dates = [row["max_date"] for row in file_rows if row["max_date"]]
    summary = {
        "input_dir": str(input_dir),
        "file_count": len(json_files),
        "total_records": total_records,
        "total_councils": len(council_counter),
        "total_committees": len(committee_counter),
        "min_date": min(valid_dates) if valid_dates else None,
        "max_date": max(max_dates) if max_dates else None,
        "empty_html_count": total_empty_html,
        "empty_html_ratio": round(total_empty_html / total_records, 6) if total_records else None,
        "invalid_date_count": total_invalid_date,
        "duplicate_count_global": global_duplicates,
        "average_text_length": round(total_html_chars / total_records, 1) if total_records else None,
        "expected_columns": EXPECTED_COLUMNS,
        "found_columns": sorted(schema_counter),
        "generated_files": [
            "file_inventory.csv", "schema_summary.csv", "council_summary.csv",
            "committee_summary.csv", "month_summary.csv", "sample_records.csv",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n완료")
    print(f"파일 수: {len(json_files):,}")
    print(f"회의록 수: {total_records:,}")
    print(f"지방의회 수: {len(council_counter):,}")
    print(f"위원회 수: {len(committee_counter):,}")
    print(f"기간: {summary['min_date']} ~ {summary['max_date']}")
    print(f"결과 폴더: {output_dir}")


if __name__ == "__main__":
    main()
