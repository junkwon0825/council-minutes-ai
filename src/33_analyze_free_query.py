#!/usr/bin/env python3
"""임의 정책어를 검색·재순위·군집화하여 provisional 분석 결과를 만든다.

고정 이슈 registry를 변경하지 않는다. 최초 관측과 의회 순서는 동적 군집에서
계산하며, 관측 이력이 충분할 때만 설명용 기준선 전망을 함께 제공한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "dynamic_issues",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--expansion-k", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=500)
    parser.add_argument("--cluster-similarity", type=float, default=0.68)
    parser.add_argument("--min-centroid-similarity", type=float, default=0.45)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def run(command: list[str], expected: Path, reuse: bool) -> None:
    if reuse and expected.is_file():
        print(f"재사용: {expected}")
        return
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def add_month(month: str, amount: int = 1) -> str:
    year, number = map(int, month.split("-"))
    total = year * 12 + number - 1 + amount
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    result = []
    current = start
    while current <= end:
        result.append(current)
        current = add_month(current)
    return result


def provisional_forecast(
    monthly: list[dict[str, Any]], council_count: int,
) -> dict[str, Any]:
    if not monthly:
        return {"status": "insufficient_evidence", "reason": "월별 관측 없음"}
    counts = [int(row.get("mention_count") or 0) for row in monthly]
    active_months = sum(value > 0 for value in counts)
    if len(counts) < 6 or active_months < 4 or council_count < 2:
        return {
            "status": "insufficient_evidence",
            "reason": (
                f"최소 조건 미달: 달력월 {len(counts)}, 관측월 {active_months}, "
                f"의회 {council_count}"
            ),
            "minimum_requirements": {
                "calendar_months": 6, "active_months": 4, "councils": 2,
            },
        }
    recent = counts[-3:]
    weights = list(range(1, len(recent) + 1))
    prediction = sum(value * weight for value, weight in zip(recent, weights)) / sum(weights)
    differences = [counts[index] - counts[index - 1] for index in range(1, len(counts))]
    spread = float(max(1.0, (sum(value * value for value in differences) / max(1, len(differences))) ** 0.5))
    recent_mean = sum(recent) / len(recent)
    if prediction > recent_mean * 1.15:
        direction = "growth"
    elif prediction < recent_mean * 0.85:
        direction = "attenuation"
    else:
        direction = "persistence"
    return {
        "status": "provisional_baseline",
        "target_month": add_month(monthly[-1]["month"]),
        "direction": direction,
        "predicted_mention_count": int(round(max(0.0, prediction))),
        "prediction_interval_80": [
            int(max(0, math.floor(prediction - 1.282 * spread))),
            int(math.ceil(prediction + 1.282 * spread)),
        ],
        "method": "weighted_mean_3m_dynamic_baseline",
        "warning": "검수·고정 전 동적 이슈의 설명용 기준선이며 공식 예측이 아님",
    }


def build_map(propagation: list[dict[str, Any]]) -> dict[str, Any]:
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
    nodes.sort(key=lambda item: item["order"])
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
    return {
        "center": {"latitude": 36.25, "longitude": 127.85},
        "zoom": 6.0,
        "nodes": nodes,
        "edges": edges,
        "missing_coordinate_councils": missing,
        "interpretation": (
            "화살표는 의회별 최초 관측의 시간 순서이며 인과관계를 의미하지 않음"
        ),
    }


def build_result(query: str, timeline_path: Path, cluster_path: Path) -> dict[str, Any]:
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    cluster = json.loads(cluster_path.read_text(encoding="utf-8"))
    issue = timeline.get("issue") or {}
    monthly_sparse = {
        row["month"]: row for row in (timeline.get("monthly_timeline") or [])
    }
    first_month = str((timeline.get("first_observation") or {}).get("date"))[:7]
    last_month = str(timeline.get("last_observation_date") or "")[:7]
    monthly = []
    for month in month_range(first_month, last_month):
        row = monthly_sparse.get(month) or {}
        monthly.append({
            "month": month,
            "mention_count": int(row.get("mention_count") or 0),
            "active_council_count": int(row.get("active_council_count") or 0),
            "new_council_count": int(row.get("new_council_count") or 0),
        })
    propagation = timeline.get("propagation_order") or []
    evidence = []
    seen = set()
    for row in propagation:
        for item in (row.get("evidence") or [])[:2]:
            key = item.get("parent_chunk_id") or item.get("chunk_id")
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    council_count = int((timeline.get("quality") or {}).get("council_count") or 0)
    return {
        "schema_version": "council_dynamic_issue_v1",
        "status": "provisional_until_review",
        "issue": {
            "issue_id": issue.get("issue_id"),
            "issue_name": query,
            "status": "provisional",
        },
        "summary": {
            "first_observation_date": (
                timeline.get("first_observation") or {}
            ).get("date"),
            "first_observation_councils": (
                timeline.get("first_observation") or {}
            ).get("councils") or [],
            "last_observation_date": timeline.get("last_observation_date"),
            "council_count": council_count,
            "evidence_unit_count": (
                timeline.get("quality") or {}
            ).get("valid_unique_member_count"),
        },
        "propagation": propagation,
        "map": build_map(propagation),
        "monthly_series": monthly,
        "forecast": provisional_forecast(monthly, council_count),
        "evidence": evidence,
        "quality": {
            "seed_coverage": (
                (cluster.get("clustering") or {}).get("selected_issue") or {}
            ).get("seed_coverage"),
            "cluster_similarity": (
                cluster.get("clustering") or {}
            ).get("cluster_similarity"),
            **(timeline.get("quality") or {}),
        },
        "interpretation": (
            "의회 간 연결은 동적 군집의 최초 관측 시간 순서이며 인과관계가 아님"
        ),
        "source_files": {
            "cluster": str(cluster_path), "timeline": str(timeline_path),
        },
    }


def main() -> None:
    args = parse_args()
    query = " ".join(args.query.split())
    if not query:
        raise ValueError("query가 비었습니다")
    if not os.environ.get("PGVECTOR_DSN"):
        raise RuntimeError("PGVECTOR_DSN 환경변수가 필요합니다")
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    work_dir = args.output_dir.expanduser().resolve() / f"query_{digest}"
    retrieval = work_dir / "01_retrieval.json"
    reranked = work_dir / "02_reranked.json"
    cluster = work_dir / "03_cluster.json"
    timeline = work_dir / "04_timeline_provisional.json"
    result = work_dir / "05_dynamic_result.json"
    download = ["--allow-download"] if args.allow_download else []

    run([
        sys.executable, "-u", "src/18_pgvector_retrieval.py",
        "--query", query, "--top-k", str(args.rerank_top_k),
        "--candidate-k", str(args.candidate_k), "--ef-search",
        str(args.ef_search), "--device", args.device, "--output", str(retrieval),
        *download,
    ], retrieval, args.reuse)
    run([
        sys.executable, "-u", "src/19_rerank_candidates.py",
        "--input", str(retrieval), "--output", str(reranked),
        "--top-k", str(args.rerank_top_k), "--device", args.device, *download,
    ], reranked, args.reuse)
    run([
        sys.executable, "-u", "src/20_expand_cluster_issue.py",
        "--input", str(reranked), "--output", str(cluster),
        "--seed-count", str(args.rerank_top_k),
        "--expansion-k", str(args.expansion_k), "--ef-search", str(args.ef_search),
        "--min-centroid-similarity", str(args.min_centroid_similarity),
        "--cluster-similarity", str(args.cluster_similarity),
    ], cluster, args.reuse)
    run([
        sys.executable, "-u", "src/21_build_issue_timeline.py",
        "--input", str(cluster), "--output", str(timeline),
        "--issue-name", query,
    ], timeline, args.reuse)
    payload = build_result(query, timeline, cluster)
    atomic_json(result, payload)

    forecast = payload["forecast"]
    print("\n=== 자유검색 동적 이슈 분석 완료 ===")
    print(f"질의: {query}")
    print(f"상태: {payload['status']} (고정 registry 미변경)")
    print(
        f"최초 관측: {payload['summary']['first_observation_date']} | "
        f"의회: {payload['summary']['council_count']} | "
        f"근거: {payload['summary']['evidence_unit_count']}"
    )
    print(f"예측 상태: {forecast['status']}")
    print(f"JSON: {result}")


if __name__ == "__main__":
    main()
