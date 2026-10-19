#!/usr/bin/env python3
"""검수된 이슈 라벨과 군집 배정 결과로 최초 등장·전파·시계열을 만든다.

입력 데이터는 모두 로컬 파일이며 외부 API를 사용하지 않는다. 날짜와 전파는
제공 데이터 범위 안의 관측 결과일 뿐 실제 정책 인과관계를 단정하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error


SCRIPT_VERSION = "2026-10-19-issue-timeline-v2-evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이슈 최초 등장·전파·월별 변화 생성")
    parser.add_argument(
        "--assignments",
        default=(
            "data/index/arctic_ko_semantic_test_v2/clustering_k160/"
            "cluster_assignments.parquet"
        ),
    )
    parser.add_argument(
        "--labels",
        default="outputs/llm_labels_semantic_test_v2_k160_strict/llm_cluster_labels.csv",
    )
    parser.add_argument(
        "--evidence-metadata",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
        help=(
            "군집 배정 파일에 원문 텍스트가 없을 때 embedding_row로 결합할 의미 청크 "
            "메타데이터. 빈 문자열이면 결합하지 않음"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/issue_timeline_final")
    parser.add_argument(
        "--include-status",
        default="accepted_candidate",
        help="포함할 08 v2 review_status. 쉼표로 구분",
    )
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--evidence-per-council", type=int, default=3)
    return parser.parse_args()


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_date(value) -> str:
    digits = re.sub(r"\D", "", clean(value))[:8]
    if len(digits) != 8:
        return ""
    try:
        return datetime.strptime(digits, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return ""


def days_between(start: str, end: str) -> int:
    return (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_labels(path: Path, included_status: set[str]) -> tuple[dict[int, dict], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"라벨 파일이 비어 있습니다: {path}")
    if "review_status" not in rows[0]:
        raise SystemExit(
            "08 v2 라벨이 필요합니다. review_status가 없는 v1 결과는 타임라인에 사용하지 마세요."
        )

    selected = []
    by_cluster = {}
    seen_issue_ids = set()
    for row in rows:
        if clean(row.get("review_status")) not in included_status:
            continue
        if clean(row.get("cluster_quality")) != "coherent":
            continue
        issue_id = clean(row.get("issue_id"))
        if not issue_id:
            raise SystemExit("issue_id가 비어 있는 라벨이 있습니다.")
        if issue_id in seen_issue_ids:
            raise SystemExit(f"중복 issue_id: {issue_id}")
        seen_issue_ids.add(issue_id)
        cluster_id = safe_int(row.get("cluster_id"), -1)
        if cluster_id < 0:
            raise SystemExit(f"잘못된 cluster_id: {row.get('cluster_id')}")
        row["cluster_id"] = cluster_id
        selected.append(row)
        by_cluster[cluster_id] = row
    if not selected:
        raise SystemExit("조건을 만족하는 coherent 이슈가 없습니다. 08 v2 검수 결과를 확인하세요.")
    return by_cluster, selected


def main() -> None:
    args = parse_args()
    if not -1.0 <= args.min_similarity <= 1.0:
        raise SystemExit("--min-similarity는 -1 이상 1 이하여야 합니다.")
    if args.evidence_per_council < 1:
        raise SystemExit("--evidence-per-council은 1 이상이어야 합니다.")

    assignment_path = Path(args.assignments).expanduser().resolve()
    label_path = Path(args.labels).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (assignment_path, label_path):
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    statuses = {part.strip() for part in args.include_status.split(",") if part.strip()}
    labels, label_rows = load_labels(label_path, statuses)

    parquet = pq.ParquetFile(assignment_path)
    available = set(parquet.schema_arrow.names)
    required = {"cluster_id", "cluster_similarity", "council", "meeting_date"}
    missing = required - available
    if missing:
        raise SystemExit("군집 배정 파일 필수 컬럼 누락: " + ", ".join(sorted(missing)))
    wanted = [
        "embedding_row", "cluster_id", "cluster_similarity", "chunk_id",
        "parent_chunk_id", "doc_id", "council", "committee", "meeting_name",
        "meeting_date", "agenda_title", "speaker_label", "evidence_ref",
        "char_start", "char_end", "semantic_text", "issue_text", "chunk_text",
    ]
    columns = [column for column in wanted if column in available]

    # 과거 07 결과는 용량 절감을 위해 semantic_text를 배정 파일에서 제외했다.
    # 이 경우 04d 의미 청크 메타데이터와 embedding_row로 결합해 근거 원문을 복원한다.
    evidence_lookup: dict[int, dict] = {}
    evidence_metadata_path = (
        Path(args.evidence_metadata).expanduser().resolve()
        if clean(args.evidence_metadata) else None
    )
    text_columns = {"semantic_text", "issue_text", "chunk_text"}
    needs_evidence_join = not (text_columns & available)
    if needs_evidence_join and evidence_metadata_path and evidence_metadata_path.exists():
        evidence_parquet = pq.ParquetFile(evidence_metadata_path)
        evidence_available = set(evidence_parquet.schema_arrow.names)
        evidence_wanted = [
            "embedding_row", "char_start", "char_end", "evidence_ref",
            "semantic_text", "issue_text", "chunk_text",
        ]
        evidence_columns = [column for column in evidence_wanted if column in evidence_available]
        if "embedding_row" not in evidence_columns or not (text_columns & evidence_available):
            raise SystemExit(
                "근거 메타데이터에 embedding_row와 텍스트 컬럼이 필요합니다: "
                f"{evidence_metadata_path}"
            )
        for batch in evidence_parquet.iter_batches(batch_size=8192, columns=evidence_columns):
            for evidence_row in batch.to_pylist():
                evidence_lookup[safe_int(evidence_row.get("embedding_row"), -1)] = evidence_row
    elif needs_evidence_join and evidence_metadata_path:
        raise SystemExit(
            "군집 배정 파일에 원문 텍스트가 없고 근거 메타데이터도 없습니다: "
            f"{evidence_metadata_path}"
        )

    # 자식 의미 청크가 같은 부모 청크에서 여러 개 나온 경우 1회 관측으로 계산한다.
    observations: dict[tuple, dict] = {}
    scanned = 0
    skipped_similarity = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            scanned += 1
            if evidence_lookup:
                evidence_row = evidence_lookup.get(
                    safe_int(row.get("embedding_row"), -1), {}
                )
                for field in (
                    "char_start", "char_end", "evidence_ref",
                    "semantic_text", "issue_text", "chunk_text",
                ):
                    if not row.get(field) and evidence_row.get(field) is not None:
                        row[field] = evidence_row[field]
            cluster_id = safe_int(row.get("cluster_id"), -1)
            if cluster_id not in labels:
                continue
            similarity = safe_float(row.get("cluster_similarity"), -1.0)
            if similarity < args.min_similarity:
                skipped_similarity += 1
                continue
            date = normalize_date(row.get("meeting_date"))
            council = clean(row.get("council"))
            if not date or not council:
                continue
            issue_id = clean(labels[cluster_id]["issue_id"])
            source_unit = (
                clean(row.get("parent_chunk_id"))
                or clean(row.get("chunk_id"))
                or str(row.get("embedding_row", ""))
            )
            key = (issue_id, council, date, source_unit)
            item = {
                **row,
                "meeting_date": date,
                "council": council,
                "cluster_similarity": similarity,
                "issue_id": issue_id,
            }
            if key not in observations or similarity > observations[key]["cluster_similarity"]:
                observations[key] = item

    by_issue: dict[str, list[dict]] = defaultdict(list)
    for row in observations.values():
        by_issue[row["issue_id"]].append(row)

    registry_rows = []
    council_first_rows = []
    monthly_rows = []
    timelines = []
    for label in sorted(label_rows, key=lambda row: clean(row["issue_id"])):
        issue_id = clean(label["issue_id"])
        rows = by_issue.get(issue_id, [])
        if not rows:
            continue
        rows.sort(key=lambda row: (row["meeting_date"], row["council"], -row["cluster_similarity"]))
        first_date = rows[0]["meeting_date"]

        rows_by_council: dict[str, list[dict]] = defaultdict(list)
        rows_by_month: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            rows_by_council[row["council"]].append(row)
            rows_by_month[row["meeting_date"][:7]].append(row)

        firsts = []
        for council, council_rows in rows_by_council.items():
            council_rows.sort(
                key=lambda row: (row["meeting_date"], -row["cluster_similarity"])
            )
            council_first = council_rows[0]["meeting_date"]
            evidence_candidates = sorted(
                council_rows,
                key=lambda row: (row["meeting_date"] != council_first, -row["cluster_similarity"]),
            )[: args.evidence_per_council]
            firsts.append({
                "council": council,
                "first_date": council_first,
                "lag_days": days_between(first_date, council_first),
                "mention_count": len(council_rows),
                "mean_similarity": round(
                    sum(row["cluster_similarity"] for row in council_rows) / len(council_rows), 6
                ),
                "evidence": [
                    {
                        "meeting_date": row["meeting_date"],
                        "meeting_name": clean(row.get("meeting_name")),
                        "agenda_title": clean(row.get("agenda_title")),
                        "speaker_label": clean(row.get("speaker_label")),
                        "evidence_ref": clean(row.get("evidence_ref")),
                        "char_start": safe_int(row.get("char_start"), -1),
                        "char_end": safe_int(row.get("char_end"), -1),
                        "original_excerpt": clean(
                            row.get("semantic_text")
                            or row.get("issue_text")
                            or row.get("chunk_text")
                        )[:800],
                        "similarity": round(row["cluster_similarity"], 6),
                    }
                    for row in evidence_candidates
                ],
            })
        firsts.sort(key=lambda row: (row["first_date"], row["council"]))
        for order, event in enumerate(firsts, start=1):
            event["propagation_order"] = order
            council_first_rows.append({
                "issue_id": issue_id,
                "issue_name": clean(label.get("issue_name")),
                **{key: value for key, value in event.items() if key != "evidence"},
                "evidence_refs": " | ".join(
                    evidence["evidence_ref"] for evidence in event["evidence"]
                    if evidence["evidence_ref"]
                ),
                "evidence_excerpts": " || ".join(
                    evidence["original_excerpt"] for evidence in event["evidence"]
                    if evidence["original_excerpt"]
                ),
            })

        first_by_council = {row["council"]: row["first_date"] for row in firsts}
        previous_count = None
        series = []
        for month in sorted(rows_by_month):
            month_rows = rows_by_month[month]
            count = len(month_rows)
            councils = sorted({row["council"] for row in month_rows})
            new_councils = sorted(
                council for council in councils if first_by_council[council][:7] == month
            )
            delta = None if previous_count is None else count - previous_count
            growth = None
            if previous_count not in (None, 0):
                growth = round((count - previous_count) / previous_count, 4)
            if month == first_date[:7]:
                phase = "emergence"
            elif new_councils:
                phase = "diffusion"
            elif delta is not None and delta > 0:
                phase = "expansion"
            elif delta is not None and delta < 0:
                phase = "attenuation"
            else:
                phase = "persistence"
            point = {
                "month": month,
                "mention_count": count,
                "council_count": len(councils),
                "new_council_count": len(new_councils),
                "new_councils": new_councils,
                "mention_delta": delta,
                "growth_rate": growth,
                "phase": phase,
            }
            series.append(point)
            monthly_rows.append({
                "issue_id": issue_id,
                "issue_name": clean(label.get("issue_name")),
                **point,
                "new_councils": " | ".join(new_councils),
            })
            previous_count = count

        origin_councils = [row["council"] for row in firsts if row["first_date"] == first_date]
        registry = {
            "issue_id": issue_id,
            "cluster_id": safe_int(label.get("cluster_id")),
            "issue_name": clean(label.get("issue_name")),
            "issue_summary": clean(label.get("issue_summary")),
            "policy_domain": clean(label.get("policy_domain")),
            "keywords": clean(label.get("keywords")),
            "review_status": clean(label.get("review_status")),
            "confidence": safe_float(label.get("confidence")),
            "first_observed_date": first_date,
            "origin_councils": " | ".join(origin_councils),
            "observed_council_count": len(firsts),
            "observation_count": len(rows),
            "last_observed_date": rows[-1]["meeting_date"],
            "propagation_span_days": days_between(first_date, rows[-1]["meeting_date"]),
        }
        registry_rows.append(registry)
        timelines.append({
            **registry,
            "origin_councils": origin_councils,
            "propagation": firsts,
            "monthly_series": series,
            "interpretation_note": (
                "날짜 순서는 제공 데이터 내 관측 순서이며 실제 정책 인과관계를 의미하지 않습니다."
            ),
        })

    registry_fields = [
        "issue_id", "cluster_id", "issue_name", "issue_summary", "policy_domain",
        "keywords", "review_status", "confidence", "first_observed_date",
        "origin_councils", "observed_council_count", "observation_count",
        "last_observed_date", "propagation_span_days",
    ]
    council_fields = [
        "issue_id", "issue_name", "council", "first_date", "lag_days",
        "mention_count", "mean_similarity", "propagation_order", "evidence_refs",
        "evidence_excerpts",
    ]
    monthly_fields = [
        "issue_id", "issue_name", "month", "mention_count", "council_count",
        "new_council_count", "new_councils", "mention_delta", "growth_rate", "phase",
    ]
    write_csv(output_dir / "issue_registry.csv", registry_rows, registry_fields)
    write_csv(output_dir / "issue_council_firsts.csv", council_first_rows, council_fields)
    write_csv(output_dir / "issue_monthly.csv", monthly_rows, monthly_fields)
    atomic_json(output_dir / "issue_timelines.json", timelines)

    phase_counts = Counter(row["phase"] for row in monthly_rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "security": "로컬 파일만 사용하며 회의록 데이터 외부 전송 없음",
        "assignment_path": str(assignment_path),
        "label_path": str(label_path),
        "evidence_metadata_path": (
            str(evidence_metadata_path) if evidence_metadata_path else None
        ),
        "evidence_join_count": len(evidence_lookup),
        "included_review_status": sorted(statuses),
        "min_similarity": args.min_similarity,
        "scanned_assignment_count": scanned,
        "similarity_excluded_count": skipped_similarity,
        "deduplicated_observation_count": len(observations),
        "timeline_issue_count": len(timelines),
        "phase_counts": dict(phase_counts),
        "generated_files": [
            "issue_registry.csv", "issue_council_firsts.csv",
            "issue_monthly.csv", "issue_timelines.json",
        ],
    }
    atomic_json(output_dir / "timeline_summary.json", summary)
    print(f"완료: 이슈 {len(timelines):,}개, 관측 {len(observations):,}건")
    print(f"이슈 목록: {output_dir / 'issue_registry.csv'}")
    print(f"타임라인: {output_dir / 'issue_timelines.json'}")


if __name__ == "__main__":
    main()
