#!/usr/bin/env python3
"""검수된 이슈 ID를 고정하고 최초 관측·의회별 확산·월별 추이를 계산한다.

여기서 propagation은 회의록에 나타난 시간적 관측 순서이며 정책의 인과적
전파를 증명하지 않는다. 모든 날짜와 건수는 입력 군집의 근거 청크에서 계산한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from council_vector_runtime import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path,
        default=PROJECT_ROOT / "data" / "issues" / "issue_registry.json",
    )
    parser.add_argument("--issue-name", required=True)
    parser.add_argument("--aliases", nargs="*", default=[])
    parser.add_argument("--evidence-per-council", type=int, default=3)
    parser.add_argument(
        "--approve", action="store_true",
        help="검수된 issue_id를 registry에 fixed 상태로 기록",
    )
    return parser.parse_args()


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()[:10]
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def evidence_item(member: dict[str, Any]) -> dict[str, Any]:
    metadata = member.get("metadata") or {}
    return {
        "chunk_id": str(member.get("chunk_id") or ""),
        "parent_chunk_id": str(metadata.get("parent_chunk_id") or "") or None,
        "council": str(metadata.get("council") or ""),
        "meeting_date": str(metadata.get("meeting_date") or "")[:10],
        "committee": str(metadata.get("committee") or ""),
        "meeting_name": str(metadata.get("meeting_name") or ""),
        "agenda_title": str(metadata.get("agenda_title") or ""),
        "centroid_similarity": float(member.get("centroid_similarity") or 0.0),
        "text": str(member.get("text") or ""),
    }


def member_digest(members: list[dict[str, Any]]) -> str:
    stable = "\n".join(sorted(str(row.get("chunk_id") or "") for row in members))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"registry_version": 1, "issues": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
        raise ValueError(f"issue registry 형식이 올바르지 않습니다: {path}")
    return payload


def update_registry(
    path: Path,
    issue_id: str,
    issue_name: str,
    aliases: list[str],
    source_path: Path,
    members: list[dict[str, Any]],
    councils: list[str],
    first_date: str,
    last_date: str,
) -> None:
    registry = load_registry(path)
    issues = registry["issues"]
    existing = next((row for row in issues if row.get("issue_id") == issue_id), None)
    record = {
        "issue_id": issue_id,
        "issue_name": issue_name,
        "aliases": sorted({issue_name, *aliases}),
        "status": "fixed",
        "scope": "provided_2024_2025_council_minutes_only",
        "source_cluster": str(source_path),
        "member_digest_sha256": member_digest(members),
        "member_count": len(members),
        "council_count": len(councils),
        "first_observation_date": first_date,
        "last_observation_date": last_date,
    }
    if existing is None:
        issues.append(record)
    else:
        if existing.get("status") == "fixed" and (
            existing.get("member_digest_sha256") != record["member_digest_sha256"]
        ):
            raise RuntimeError(
                "이미 고정된 issue_id의 구성원이 달라졌습니다. 기존 registry를 "
                "덮어쓰지 말고 새 issue_id로 검수하세요."
            )
        existing.clear()
        existing.update(record)
    issues.sort(key=lambda row: str(row.get("issue_id") or ""))
    atomic_json(path, registry)


def propagation_stage(lag_days: int) -> str:
    if lag_days == 0:
        return "first_observation"
    if lag_days <= 30:
        return "early_followup_30d"
    if lag_days <= 90:
        return "expansion_90d"
    return "later_observation"


def main() -> None:
    args = parse_args()
    if args.evidence_per_council < 1:
        raise ValueError("evidence-per-council은 1 이상이어야 합니다")
    issue_name = args.issue_name.strip()
    if not issue_name:
        raise ValueError("issue-name이 비었습니다")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    issue = payload.get("issue") or {}
    issue_id = str(issue.get("issue_id") or "").strip()
    members = payload.get("members")
    if not issue_id.startswith("ISSUE_"):
        raise ValueError("입력 JSON에 유효한 issue_id가 없습니다")
    if not isinstance(members, list) or not members:
        raise ValueError("입력 JSON에 members가 없습니다")

    valid_members: list[dict[str, Any]] = []
    invalid_date_chunk_ids: list[str] = []
    seen_units: set[str] = set()
    seen_texts: set[str] = set()
    parent_duplicate_count = 0
    text_duplicate_count = 0
    for member in members:
        metadata = member.get("metadata") or {}
        current_date = parse_date(metadata.get("meeting_date"))
        if current_date is None:
            invalid_date_chunk_ids.append(str(member.get("chunk_id") or ""))
            continue
        parent_id = str(metadata.get("parent_chunk_id") or "")
        unit_id = parent_id or str(member.get("chunk_id") or "")
        if unit_id in seen_units:
            parent_duplicate_count += 1
            continue
        seen_units.add(unit_id)
        normalized_text = re.sub(
            r"\s+", "", str(member.get("text") or "")
        ).casefold()
        if normalized_text:
            text_key = hashlib.sha256(
                normalized_text.encode("utf-8")
            ).hexdigest()
            if text_key in seen_texts:
                text_duplicate_count += 1
                continue
            seen_texts.add(text_key)
        current = dict(member)
        current["_date"] = current_date
        valid_members.append(current)
    if not valid_members:
        raise RuntimeError("유효한 날짜를 가진 이슈 구성원이 없습니다")
    valid_members.sort(key=lambda row: (
        row["_date"],
        str((row.get("metadata") or {}).get("council") or ""),
        str(row.get("chunk_id") or ""),
    ))

    overall_first_date = valid_members[0]["_date"]
    overall_last_date = valid_members[-1]["_date"]
    by_council: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in valid_members:
        council = str((member.get("metadata") or {}).get("council") or "").strip()
        if council:
            by_council[council].append(member)
    if not by_council:
        raise RuntimeError("의회명이 있는 이슈 구성원이 없습니다")

    propagation = []
    for council, council_members in by_council.items():
        council_members.sort(key=lambda row: (
            row["_date"], -float(row.get("centroid_similarity") or 0.0)
        ))
        first = council_members[0]["_date"]
        lag_days = (first - overall_first_date).days
        first_day_members = [row for row in council_members if row["_date"] == first]
        first_day_members.sort(
            key=lambda row: float(row.get("centroid_similarity") or 0.0),
            reverse=True,
        )
        propagation.append({
            "council": council,
            "first_observation_date": first.isoformat(),
            "lag_days_from_first": lag_days,
            "stage": propagation_stage(lag_days),
            "mention_count": len(council_members),
            "last_observation_date": council_members[-1]["_date"].isoformat(),
            "evidence": [
                evidence_item(row)
                for row in first_day_members[:args.evidence_per_council]
            ],
        })
    propagation.sort(key=lambda row: (
        row["first_observation_date"], row["council"]
    ))
    for order, row in enumerate(propagation, start=1):
        row["observation_order"] = order

    co_first = [
        row for row in propagation if row["lag_days_from_first"] == 0
    ]

    monthly_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    first_month_by_council: dict[str, str] = {}
    for member in valid_members:
        month = member["_date"].strftime("%Y-%m")
        monthly_members[month].append(member)
        council = str((member.get("metadata") or {}).get("council") or "")
        if council and council not in first_month_by_council:
            first_month_by_council[council] = month

    cumulative_councils: set[str] = set()
    monthly_timeline = []
    for month in sorted(monthly_members):
        rows = monthly_members[month]
        councils = {
            str((row.get("metadata") or {}).get("council") or "")
            for row in rows if (row.get("metadata") or {}).get("council")
        }
        new_councils = sorted(
            council for council, first_month in first_month_by_council.items()
            if first_month == month
        )
        cumulative_councils.update(councils)
        similarities = [float(row.get("centroid_similarity") or 0.0) for row in rows]
        monthly_timeline.append({
            "month": month,
            "mention_count": len(rows),
            "active_council_count": len(councils),
            "active_councils": sorted(councils),
            "new_council_count": len(new_councils),
            "new_councils": new_councils,
            "cumulative_council_count": len(cumulative_councils),
            "centroid_similarity_mean": round(float(np.mean(similarities)), 6),
        })

    all_evidence = [evidence_item(row) for row in valid_members]
    clean_members = []
    for row in valid_members:
        clean = {key: value for key, value in row.items() if key != "_date"}
        clean_members.append(clean)

    report = {
        "status": "complete",
        "issue": {
            "issue_id": issue_id,
            "issue_name": issue_name,
            "aliases": sorted({issue_name, *args.aliases}),
            "id_status": "fixed" if args.approve else "provisional",
            "scope_note": "제공된 2024~2025 회의록 범위 내 관측",
        },
        "quality": {
            "input_member_count": len(members),
            "valid_unique_member_count": len(valid_members),
            "invalid_date_count": len(invalid_date_chunk_ids),
            "invalid_date_chunk_ids": invalid_date_chunk_ids,
            "parent_duplicates_removed": parent_duplicate_count,
            "exact_text_duplicates_removed": text_duplicate_count,
            "council_count": len(by_council),
        },
        "first_observation": {
            "date": overall_first_date.isoformat(),
            "councils": [row["council"] for row in co_first],
            "evidence": [item for row in co_first for item in row["evidence"]],
        },
        "last_observation_date": overall_last_date.isoformat(),
        "propagation_definition": (
            "각 의회의 최초 관측일을 정렬한 시간적 순서이며 인과적 전파를 의미하지 않음"
        ),
        "propagation_order": propagation,
        "monthly_timeline": monthly_timeline,
        "evidence": all_evidence,
        "members": clean_members,
        "source_cluster": str(input_path),
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, report)

    registry_path = args.registry.expanduser().resolve()
    if args.approve:
        update_registry(
            registry_path, issue_id, issue_name, args.aliases, input_path,
            clean_members, sorted(by_council), overall_first_date.isoformat(),
            overall_last_date.isoformat(),
        )

    print("\n=== 이슈 최초 관측·전파 타임라인 완료 ===")
    print(f"issue_id: {issue_id} | status={report['issue']['id_status']}")
    print(
        f"최초 관측: {overall_first_date.isoformat()} | "
        f"{', '.join(row['council'] for row in co_first)}"
    )
    print(
        f"의회: {len(by_council)} | 근거 단위: {len(valid_members)} | "
        f"월: {len(monthly_timeline)}"
    )
    print("\n관측 순서:")
    for row in propagation:
        print(
            f"{row['observation_order']:2d}. {row['council']} | "
            f"{row['first_observation_date']} | +{row['lag_days_from_first']}일 | "
            f"{row['mention_count']}건"
        )
    print(f"\nJSON: {output}")
    if args.approve:
        print(f"registry: {registry_path}")
    else:
        print("주의: --approve를 사용하지 않아 issue_id를 registry에 고정하지 않았습니다.")


if __name__ == "__main__":
    main()
