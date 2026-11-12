#!/usr/bin/env python3
"""고정 이슈의 지도·타임라인·예측·근거를 웹용 단일 JSON으로 결합한다."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


COUNCIL_COORDINATES = {
    "서울특별시의회": (37.5665, 126.9780),
    "부산광역시의회": (35.1796, 129.0756),
    "대구광역시의회": (35.8714, 128.6014),
    "인천광역시의회": (37.4563, 126.7052),
    "광주광역시의회": (35.1595, 126.8526),
    "대전광역시의회": (36.3504, 127.3845),
    "울산광역시의회": (35.5384, 129.3114),
    "세종특별자치시의회": (36.4800, 127.2890),
    "경기도의회": (37.4138, 127.5183),
    "강원특별자치도의회": (37.8228, 128.1555),
    "충청북도의회": (36.6357, 127.4917),
    "충청남도의회": (36.6588, 126.6728),
    "전북특별자치도의회": (35.8203, 127.1088),
    "전라북도의회": (35.8203, 127.1088),
    "전라남도의회": (34.8679, 126.9910),
    "경상북도의회": (36.5760, 128.5056),
    "경상남도의회": (35.4606, 128.2132),
    "제주특별자치도의회": (33.4996, 126.5312),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-per-council", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=1400)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 최상위는 객체여야 합니다: {resolved}")
    return payload


def safe_number(value: Any, integer: bool = False) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return int(round(number)) if integer else round(number, 6)


def add_month(month: str, amount: int = 1) -> str:
    year, number = map(int, month.split("-"))
    total = year * 12 + number - 1 + amount
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = add_month(current)
    return months


def read_monthly_features(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    numeric_integer = {
        "mention_count", "active_council_count", "new_council_count",
        "cumulative_council_count",
    }
    numeric_float = {
        "council_coverage_ratio", "mention_mean_3m", "mention_momentum_3m",
        "mention_slope_3m", "mention_slope_6m", "top_council_share",
        "council_entropy", "centroid_similarity_mean",
    }
    with resolved.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    result = []
    for row in rows:
        item: dict[str, Any] = {
            "month": row.get("feature_month", ""),
            "mention_count": safe_number(row.get("mention_count"), True),
            "active_council_count": safe_number(
                row.get("active_council_count"), True
            ),
            "new_council_count": safe_number(row.get("new_council_count"), True),
            "cumulative_council_count": safe_number(
                row.get("cumulative_council_count"), True
            ),
        }
        for name in numeric_float:
            if name in row:
                item[name] = safe_number(row.get(name))
        result.append(item)
    result.sort(key=lambda row: row["month"])
    return result


def complete_monthly_series(
    timeline: dict[str, Any], forecast: dict[str, Any],
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    first_date = str((timeline.get("first_observation") or {}).get("date") or "")
    target_month = str((forecast.get("forecast") or {}).get("target_month") or "")
    if len(first_date) < 7 or len(target_month) != 7:
        raise ValueError("전체 월 범위를 계산할 날짜가 없습니다")
    start_month = first_date[:7]
    end_month = add_month(target_month, -1)
    observed_by_month = {
        str(row.get("month")): row
        for row in timeline.get("monthly_timeline") or []
    }
    features_by_month = {row["month"]: row for row in feature_rows}
    result = []
    cumulative = 0
    for month in month_range(start_month, end_month):
        observed = observed_by_month.get(month) or {}
        features = features_by_month.get(month) or {}
        new_count = int(observed.get("new_council_count") or 0)
        cumulative += new_count
        item = {
            "month": month,
            "mention_count": int(observed.get("mention_count") or 0),
            "active_council_count": int(
                observed.get("active_council_count") or 0
            ),
            "new_council_count": new_count,
            "cumulative_council_count": int(
                observed.get("cumulative_council_count") or cumulative
            ),
            "feature_ready": bool(features),
        }
        for key, value in features.items():
            if key != "month":
                item[key] = value
        result.append(item)
    return result


def clean_evidence(item: dict[str, Any], max_chars: int) -> dict[str, Any]:
    text = " ".join(str(item.get("text") or "").split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return {
        "chunk_id": item.get("chunk_id"),
        "parent_chunk_id": item.get("parent_chunk_id"),
        "council": str(item.get("council") or ""),
        "meeting_date": str(item.get("meeting_date") or "")[:10],
        "committee": str(item.get("committee") or ""),
        "meeting_name": str(item.get("meeting_name") or ""),
        "agenda_title": str(item.get("agenda_title") or ""),
        "similarity": safe_number(item.get("centroid_similarity")),
        "text": text,
    }


def map_nodes(propagation: list[dict[str, Any]]) -> tuple[list[dict], list[str]]:
    nodes = []
    missing = []
    for row in propagation:
        council = str(row.get("council") or "")
        coordinate = COUNCIL_COORDINATES.get(council)
        if coordinate is None:
            missing.append(council)
            continue
        latitude, longitude = coordinate
        nodes.append({
            "order": int(row.get("observation_order") or len(nodes) + 1),
            "council": council,
            "latitude": latitude,
            "longitude": longitude,
            "first_observation_date": row.get("first_observation_date"),
            "lag_days": int(row.get("lag_days_from_first") or 0),
            "stage": row.get("stage"),
            "mention_count": int(row.get("mention_count") or 0),
        })
    nodes.sort(key=lambda row: row["order"])
    return nodes, missing


def map_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for source, target in zip(nodes, nodes[1:]):
        edges.append({
            "from_order": source["order"],
            "to_order": target["order"],
            "from_council": source["council"],
            "to_council": target["council"],
            "source": [source["longitude"], source["latitude"]],
            "target": [target["longitude"], target["latitude"]],
            "days_between_observations": max(
                0, target["lag_days"] - source["lag_days"]
            ),
        })
    return edges


def main() -> None:
    args = parse_args()
    if args.evidence_per_council < 1 or args.max_evidence_chars < 100:
        raise ValueError("evidence 개수는 1 이상, 최대 글자 수는 100 이상이어야 합니다")
    timeline = load_json(args.timeline)
    forecast = load_json(args.forecast)
    issue = timeline.get("issue") or {}
    forecast_issue = forecast.get("issue") or {}
    issue_id = str(issue.get("issue_id") or "")
    if not issue_id or forecast_issue.get("issue_id") != issue_id:
        raise ValueError("timeline과 forecast의 issue_id가 다릅니다")
    if issue.get("id_status") != "fixed":
        raise ValueError("fixed 상태의 이슈만 웹 결과로 만들 수 있습니다")

    propagation = list(timeline.get("propagation_order") or [])
    nodes, missing_coordinates = map_nodes(propagation)
    if not nodes:
        raise ValueError("지도에 표시할 의회 좌표가 없습니다")
    feature_rows = read_monthly_features(args.features)
    monthly = complete_monthly_series(timeline, forecast, feature_rows)

    evidence = []
    seen_ids = set()
    for row in propagation:
        council_items = row.get("evidence") or []
        for item in council_items[:args.evidence_per_council]:
            cleaned = clean_evidence(item, args.max_evidence_chars)
            identifier = cleaned["parent_chunk_id"] or cleaned["chunk_id"]
            if identifier and identifier in seen_ids:
                continue
            if identifier:
                seen_ids.add(identifier)
            evidence.append(cleaned)

    forecast_payload = forecast.get("forecast") or {}
    payload = {
        "schema_version": "council_issue_web_v1",
        "status": "complete",
        "issue": {
            "issue_id": issue_id,
            "issue_name": issue.get("issue_name"),
            "aliases": issue.get("aliases") or [],
            "status": "fixed",
            "scope": issue.get("scope_note"),
        },
        "summary": {
            "first_observation_date": (
                timeline.get("first_observation") or {}
            ).get("date"),
            "first_observation_councils": (
                timeline.get("first_observation") or {}
            ).get("councils") or [],
            "last_observation_date": timeline.get("last_observation_date"),
            "council_count": len(propagation),
            "evidence_unit_count": (
                timeline.get("quality") or {}
            ).get("valid_unique_member_count"),
            "calendar_month_count": len(monthly),
        },
        "map": {
            "center": {"latitude": 36.25, "longitude": 127.85},
            "zoom": 6.0,
            "nodes": nodes,
            "edges": map_edges(nodes),
            "missing_coordinate_councils": missing_coordinates,
            "interpretation": (
                "화살표는 의회별 최초 관측의 시간 순서를 연결하며 인과관계를 의미하지 않음"
            ),
        },
        "propagation": propagation,
        "monthly_series": monthly,
        "forecast": {
            **forecast_payload,
            "selected_baseline": forecast.get("selected_count_baseline"),
            "supporting_signals": forecast.get("supporting_signals"),
            "method": forecast.get("method"),
            "limitations": forecast.get("limitations"),
        },
        "evidence": evidence,
        "disclaimers": [
            "모든 결과는 제공된 2024~2025 광역 지방의회 회의록 범위에 한정됨",
            "지도 연결선은 최초 관측 순서이며 정책의 실제 인과적 전파를 증명하지 않음",
            "2026년 1월 예측은 단일 이슈 소표본 기준선이며 확률은 비보정 상대 지표임",
            "LLM은 수치 계산을 변경하지 않고 구조화 결과의 설명에만 사용",
        ],
        "source_files": {
            "timeline": str(args.timeline.expanduser().resolve()),
            "forecast": str(args.forecast.expanduser().resolve()),
            "features": str(args.features.expanduser().resolve()),
        },
    }
    output_path = args.output.expanduser().resolve()
    atomic_json(output_path, payload)

    print("\n=== 지도 웹용 이슈 결과 결합 완료 ===")
    print(f"issue_id: {issue_id}")
    print(
        f"지도 노드: {len(nodes)} | 연결선: {len(payload['map']['edges'])} | "
        f"월별 행: {len(monthly)} | 대표 근거: {len(evidence)}"
    )
    print(
        f"예측: {forecast_payload.get('target_month')} | "
        f"{forecast_payload.get('direction')} | "
        f"{forecast_payload.get('predicted_mention_count')}건"
    )
    if missing_coordinates:
        print(f"좌표 누락 의회: {', '.join(missing_coordinates)}")
    print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()
