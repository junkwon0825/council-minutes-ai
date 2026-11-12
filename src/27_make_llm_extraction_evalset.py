#!/usr/bin/env python3
"""LLM 메타데이터 추출 모델 비교용 고정 평가 표본을 생성한다.

최종 canonical utterance에서 지역·연도(source_key)별로 동일한 수를 뽑고,
정책성 여부·정책분야·문장 길이가 한쪽에 치우치지 않도록 층화한다.
기존 LLM 메타데이터는 정답이 아니라 candidate_* 참고값으로만 제공한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "utterances.parquet",
        help="최종 canonical 발화 Parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "llm_extraction",
    )
    parser.add_argument("--rows-per-source", type=int, default=20)
    parser.add_argument("--substantive-ratio", type=float, default=0.60)
    parser.add_argument("--max-input-chars", type=int, default=4_000)
    parser.add_argument("--seed", default="council-llm-extraction-eval-v1-20261023")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def choose_column(columns: set[str], *candidates: str, required: bool = False) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"필수 열이 없습니다. 후보={candidates}, 실제={sorted(columns)}")
    return None


def column_expr(
    columns: set[str],
    candidates: tuple[str, ...],
    alias: str,
    sql_type: str = "VARCHAR",
    required: bool = False,
) -> str:
    selected = choose_column(columns, *candidates, required=required)
    if selected is None:
        return f"CAST(NULL AS {sql_type}) AS {quote_identifier(alias)}"
    return f"CAST({quote_identifier(selected)} AS {sql_type}) AS {quote_identifier(alias)}"


def normalize_text(value: Any) -> str:
    return SPACE_PATTERN.sub(" ", "" if value is None else str(value)).strip()


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "출력 파일이 이미 있습니다. 교체하려면 --overwrite를 사용하세요: "
            + ", ".join(existing)
        )


def value_or_empty(value: Any) -> Any:
    return "" if value is None else value


def main() -> None:
    args = parse_args()
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "duckdb가 필요합니다. 활성화된 .venv에서 "
            "'pip install -r requirements-base.txt'를 실행하세요."
        ) from exc
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_jsonl = output_dir / "llm_eval_input_v1.jsonl"
    review_csv = output_dir / "llm_eval_review_v1.csv"
    manifest_json = output_dir / "llm_eval_manifest_v1.json"
    strata_csv = output_dir / "llm_eval_sample_strata_v1.csv"
    output_paths = [input_jsonl, review_csv, manifest_json, strata_csv]

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.rows_per_source < 2:
        raise ValueError("--rows-per-source는 2 이상이어야 합니다")
    if not 0.05 <= args.substantive_ratio <= 0.95:
        raise ValueError("--substantive-ratio는 0.05~0.95 사이여야 합니다")
    if args.max_input_chars < 200:
        raise ValueError("--max-input-chars는 200 이상이어야 합니다")
    ensure_writable(output_paths, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    source = str(input_path)
    description = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [source]
    ).fetchall()
    source_columns = {str(row[0]) for row in description}

    source_key_col = choose_column(source_columns, "source_key", required=True)
    row_id_col = choose_column(source_columns, "source_row_id", "row_id", "llm_row_id", required=True)
    text_col = choose_column(source_columns, "utterance", "text", required=True)
    utterance_id_col = choose_column(source_columns, "utterance_id", "chunk_id")

    source_key_sql = quote_identifier(source_key_col)
    row_id_sql = quote_identifier(row_id_col)
    text_sql = quote_identifier(text_col)
    utterance_id_sql = (
        f"CAST({quote_identifier(utterance_id_col)} AS VARCHAR)"
        if utterance_id_col
        else f"CAST({source_key_sql} AS VARCHAR) || ':' || CAST({row_id_sql} AS VARCHAR)"
    )

    projection = [
        f"{utterance_id_sql} AS utterance_id",
        f"CAST({source_key_sql} AS VARCHAR) AS source_key",
        f"CAST({row_id_sql} AS BIGINT) AS source_row_id",
        column_expr(source_columns, ("council", "assembly"), "council"),
        column_expr(source_columns, ("meeting_date", "date"), "meeting_date"),
        column_expr(source_columns, ("year",), "year", "INTEGER"),
        column_expr(source_columns, ("committee",), "committee"),
        column_expr(source_columns, ("meeting_name",), "meeting_name"),
        column_expr(source_columns, ("agenda_title", "agenda", "llm_agenda"), "agenda_title"),
        column_expr(source_columns, ("speaker_name", "speaker", "llm_speaker"), "speaker_name"),
        column_expr(source_columns, ("speaker_role",), "speaker_role"),
        column_expr(source_columns, ("speaker_type",), "speaker_type"),
        f"CAST({text_sql} AS VARCHAR) AS utterance",
        column_expr(source_columns, ("is_substantive",), "candidate_is_substantive", "BOOLEAN"),
        column_expr(source_columns, ("policy_category",), "candidate_policy_category"),
        column_expr(source_columns, ("core_issue",), "candidate_core_issue"),
        column_expr(source_columns, ("keywords_json", "keywords"), "candidate_keywords_json"),
        column_expr(source_columns, ("summary",), "candidate_summary"),
        column_expr(source_columns, ("parse_status",), "candidate_parse_status"),
    ]
    con.execute(
        f"CREATE TEMP TABLE standardized AS SELECT {', '.join(projection)} "
        "FROM read_parquet(?) WHERE "
        f"{text_sql} IS NOT NULL AND length(trim(CAST({text_sql} AS VARCHAR))) > 0",
        [source],
    )

    total_rows = int(con.execute("SELECT count(*) FROM standardized").fetchone()[0])
    source_keys = [
        str(row[0]) for row in con.execute(
            "SELECT DISTINCT source_key FROM standardized ORDER BY source_key"
        ).fetchall()
    ]
    if not source_keys:
        raise RuntimeError("평가 표본을 만들 canonical 행이 없습니다")

    substantive_quota = round(args.rows_per_source * args.substantive_ratio)
    substantive_quota = max(1, min(args.rows_per_source - 1, substantive_quota))
    non_substantive_quota = args.rows_per_source - substantive_quota

    con.execute(
        """
        CREATE TEMP TABLE ranked AS
        WITH deduplicated AS (
          SELECT *,
            CASE
              WHEN length(utterance) < 200 THEN 'short_under_200'
              WHEN length(utterance) < 800 THEN 'medium_200_799'
              ELSE 'long_800_plus'
            END AS length_bucket,
            coalesce(nullif(trim(candidate_policy_category), ''), '미분류') AS category_bucket,
            row_number() OVER (
              PARTITION BY source_key,
                regexp_replace(lower(trim(utterance)), '\\s+', '', 'g')
              ORDER BY hash(utterance_id || ?)
            ) AS duplicate_rank
          FROM standardized
        ), strata AS (
          SELECT *,
            row_number() OVER (
              PARTITION BY source_key, coalesce(candidate_is_substantive, false),
                category_bucket, length_bucket
              ORDER BY hash(utterance_id || ?)
            ) AS diversity_rank
          FROM deduplicated
          WHERE duplicate_rank = 1
        )
        SELECT *,
          row_number() OVER (
            PARTITION BY source_key, coalesce(candidate_is_substantive, false)
            ORDER BY diversity_rank, hash(utterance_id || ?)
          ) AS class_rank
        FROM strata
        """,
        [args.seed + "-dedup", args.seed + "-strata", args.seed + "-class"],
    )

    con.execute(
        """
        CREATE TEMP TABLE sample AS
        SELECT * FROM ranked
        WHERE (coalesce(candidate_is_substantive, false) = true AND class_rank <= ?)
           OR (coalesce(candidate_is_substantive, false) = false AND class_rank <= ?)
        """,
        [substantive_quota, non_substantive_quota],
    )

    con.execute(
        """
        CREATE TEMP TABLE remaining_ranked AS
        WITH needs AS (
          SELECT k.source_key, greatest(0, ? - count(s.utterance_id)) AS needed
          FROM (SELECT DISTINCT source_key FROM ranked) k
          LEFT JOIN sample s USING (source_key)
          GROUP BY k.source_key
        ), remaining AS (
          SELECT r.*,
            row_number() OVER (
              PARTITION BY r.source_key ORDER BY hash(r.utterance_id || ?)
            ) AS fill_rank
          FROM ranked r
          ANTI JOIN sample s USING (utterance_id)
        )
        SELECT r.* EXCLUDE (fill_rank)
        FROM remaining r
        JOIN needs n USING (source_key)
        WHERE r.fill_rank <= n.needed
        """,
        [args.rows_per_source, args.seed + "-topup"],
    )
    con.execute("INSERT INTO sample SELECT * FROM remaining_ranked")

    selected_columns = [
        "utterance_id", "source_key", "source_row_id", "council", "meeting_date", "year",
        "committee", "meeting_name", "agenda_title", "speaker_name", "speaker_role",
        "speaker_type", "utterance", "candidate_is_substantive",
        "candidate_policy_category", "candidate_core_issue", "candidate_keywords_json",
        "candidate_summary", "candidate_parse_status", "length_bucket",
    ]
    query = "SELECT " + ", ".join(selected_columns) + " FROM sample ORDER BY source_key, source_row_id"
    cursor = con.execute(query)
    names = [item[0] for item in cursor.description]
    rows = [dict(zip(names, row)) for row in cursor.fetchall()]

    with input_jsonl.open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            utterance = normalize_text(row["utterance"])
            input_text = utterance[: args.max_input_chars]
            record = {
                "eval_id": f"E{index:04d}",
                "utterance_id": row["utterance_id"],
                "source_key": row["source_key"],
                "row_id": int(row["source_row_id"]),
                "date": value_or_empty(row["meeting_date"]),
                "assembly": value_or_empty(row["council"]),
                "committee": value_or_empty(row["committee"]),
                "meeting_name": value_or_empty(row["meeting_name"]),
                "agenda": value_or_empty(row["agenda_title"]),
                "speaker_name": value_or_empty(row["speaker_name"]),
                "speaker_role": value_or_empty(row["speaker_role"]),
                "speaker_type": value_or_empty(row["speaker_type"]),
                "utterance": input_text,
                "original_char_length": len(utterance),
                "input_char_length": len(input_text),
                "was_truncated": len(input_text) < len(utterance),
            }
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    review_fields = [
        "eval_id", "utterance_id", "source_key", "row_id", "date", "assembly",
        "committee", "meeting_name", "agenda", "speaker_name", "speaker_role",
        "speaker_type", "utterance", "length_bucket", "original_char_length",
        "candidate_is_substantive", "candidate_policy_category", "candidate_core_issue",
        "candidate_keywords_json", "candidate_summary", "candidate_parse_status",
        "gold_is_substantive", "gold_policy_category", "gold_core_issue",
        "gold_keywords_json", "gold_summary", "review_status", "review_note",
    ]
    with review_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=review_fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            utterance = normalize_text(row["utterance"])
            writer.writerow({
                "eval_id": f"E{index:04d}",
                "utterance_id": row["utterance_id"],
                "source_key": row["source_key"],
                "row_id": int(row["source_row_id"]),
                "date": value_or_empty(row["meeting_date"]),
                "assembly": value_or_empty(row["council"]),
                "committee": value_or_empty(row["committee"]),
                "meeting_name": value_or_empty(row["meeting_name"]),
                "agenda": value_or_empty(row["agenda_title"]),
                "speaker_name": value_or_empty(row["speaker_name"]),
                "speaker_role": value_or_empty(row["speaker_role"]),
                "speaker_type": value_or_empty(row["speaker_type"]),
                "utterance": utterance[: args.max_input_chars],
                "length_bucket": row["length_bucket"],
                "original_char_length": len(utterance),
                "candidate_is_substantive": value_or_empty(row["candidate_is_substantive"]),
                "candidate_policy_category": value_or_empty(row["candidate_policy_category"]),
                "candidate_core_issue": value_or_empty(row["candidate_core_issue"]),
                "candidate_keywords_json": value_or_empty(row["candidate_keywords_json"]),
                "candidate_summary": value_or_empty(row["candidate_summary"]),
                "candidate_parse_status": value_or_empty(row["candidate_parse_status"]),
                "gold_is_substantive": "",
                "gold_policy_category": "",
                "gold_core_issue": "",
                "gold_keywords_json": "",
                "gold_summary": "",
                "review_status": "pending",
                "review_note": "",
            })

    source_counts = Counter(str(row["source_key"]) for row in rows)
    substantive_counts = Counter(str(bool(row["candidate_is_substantive"])).lower() for row in rows)
    category_counts = Counter(value_or_empty(row["candidate_policy_category"]) or "미분류" for row in rows)
    length_counts = Counter(str(row["length_bucket"]) for row in rows)
    source_key_row_pairs = {(row["source_key"], int(row["source_row_id"])) for row in rows}
    normalized_text_keys = {
        (row["source_key"], re.sub(r"\s+", "", normalize_text(row["utterance"]).lower()))
        for row in rows
    }
    expected_rows = len(source_keys) * args.rows_per_source
    checks = {
        "row_count_matches_target": len(rows) == expected_rows,
        "utterance_ids_unique": len({row["utterance_id"] for row in rows}) == len(rows),
        "source_row_keys_unique": len(source_key_row_pairs) == len(rows),
        "all_source_keys_present": len(source_counts) == len(source_keys),
        "all_sources_have_requested_rows": all(
            source_counts.get(key, 0) == args.rows_per_source for key in source_keys
        ),
        "no_empty_utterances": all(normalize_text(row["utterance"]) for row in rows),
        "no_same_source_exact_text_duplicates": len(normalized_text_keys) == len(rows),
    }
    passed = all(checks.values())

    with strata_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_key", "rows"])
        for key in source_keys:
            writer.writerow([key, source_counts.get(key, 0)])

    manifest = {
        "passed": passed,
        "input": str(input_path),
        "input_rows": total_rows,
        "source_key_count": len(source_keys),
        "expected_source_key_count": 34,
        "rows_per_source": args.rows_per_source,
        "target_rows": expected_rows,
        "sample_rows": len(rows),
        "substantive_ratio": args.substantive_ratio,
        "substantive_quota_per_source": substantive_quota,
        "non_substantive_quota_per_source": non_substantive_quota,
        "max_input_chars": args.max_input_chars,
        "seed": args.seed,
        "join_key": ["source_key", "row_id"],
        "candidate_label_warning": "candidate_*는 기존 LLM 출력이며 gold 정답이 아님",
        "candidate_substantive_counts": dict(substantive_counts),
        "candidate_policy_category_counts": dict(category_counts.most_common()),
        "length_bucket_counts": dict(length_counts),
        "source_key_counts": dict(source_counts),
        "checks": checks,
        "outputs": {
            "model_input_jsonl": str(input_jsonl),
            "human_review_csv": str(review_csv),
            "strata_csv": str(strata_csv),
        },
        "output_sha256": {
            "model_input_jsonl": file_sha256(input_jsonl),
            "human_review_csv": file_sha256(review_csv),
            "strata_csv": file_sha256(strata_csv),
        },
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== LLM 추출 성능 비교 평가 표본 생성 완료 ===")
    print(f"canonical: {total_rows:,}행 | source_key: {len(source_keys):,}")
    print(f"표본: {len(rows):,}/{expected_rows:,}행 | source별 {args.rows_per_source}행")
    print(f"후보 정책성 분포: {dict(substantive_counts)}")
    print(f"길이 분포: {dict(length_counts)}")
    for name, value in checks.items():
        print(f"{'PASS' if value else 'FAIL':4s} | {name}")
    if len(source_keys) != 34:
        print(f"주의 | 예상 source_key=34, 실제={len(source_keys)}")
    print(f"최종 결과: {'PASS' if passed else 'FAIL'}")
    print(f"모델 입력: {input_jsonl}")
    print(f"사람 검수: {review_csv}")
    print(f"요약: {manifest_json}")
    if not passed:
        raise RuntimeError("LLM 추출 평가 표본 검증에 실패했습니다. manifest를 확인하세요")


if __name__ == "__main__":
    main()
