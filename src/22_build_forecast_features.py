#!/usr/bin/env python3
"""고정 이슈 타임라인을 누수 없는 월별 예측 특징으로 변환한다.

입력 타임라인의 근거만 사용하며, 관측이 없는 달도 0으로 채운다. 각 행의
특징은 해당 월까지의 정보로 만들고 정답은 다음 달에서만 계산한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from council_vector_runtime import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-cutoff", default="2025-12-31",
        help="이 날짜가 속한 월까지만 사용 (YYYY-MM-DD)",
    )
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--long-lookback", type=int, default=6)
    parser.add_argument("--total-councils", type=int, default=17)
    return parser.parse_args()


def month_key(value: Any) -> str:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%Y-%m")
    except ValueError as error:
        raise ValueError(f"날짜 형식이 올바르지 않습니다: {value}") from error


def add_month(month: str, amount: int = 1) -> str:
    year, number = map(int, month.split("-"))
    total = year * 12 + number - 1 + amount
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    if start > end:
        raise ValueError(f"시작 월이 종료 월보다 늦습니다: {start} > {end}")
    result = []
    current = start
    while current <= end:
        result.append(current)
        current = add_month(current)
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def padded_tail(values: list[float], end: int, size: int) -> list[float]:
    selected = values[max(0, end - size + 1): end + 1]
    return [0.0] * (size - len(selected)) + [float(value) for value in selected]


def months_since_positive(values: list[int], end: int) -> int:
    for distance, value in enumerate(reversed(values[: end + 1])):
        if value > 0:
            return distance
    return end + 1


def concentration(counts: Counter[str]) -> tuple[float, float, float]:
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0, 0.0
    shares = np.asarray([value / total for value in counts.values()], dtype=np.float64)
    hhi = float(np.square(shares).sum())
    top_share = float(shares.max())
    if len(shares) <= 1:
        entropy = 0.0
    else:
        entropy = float(-(shares * np.log(shares)).sum() / np.log(len(shares)))
    return hhi, top_share, entropy


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("저장할 특징 행이 없습니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def member_date_and_council(member: dict[str, Any]) -> tuple[str, str] | None:
    metadata = member.get("metadata") or {}
    date_value = metadata.get("meeting_date") or member.get("meeting_date")
    council = str(metadata.get("council") or member.get("council") or "").strip()
    if not date_value or not council:
        return None
    return month_key(date_value), council


def main() -> None:
    args = parse_args()
    if args.lookback < 2 or args.long_lookback < args.lookback:
        raise ValueError("lookback은 2 이상, long-lookback은 lookback 이상이어야 합니다")
    if args.total_councils < 1:
        raise ValueError("total-councils는 1 이상이어야 합니다")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    issue = payload.get("issue") or {}
    issue_id = str(issue.get("issue_id") or "").strip()
    if not issue_id.startswith("ISSUE_") or issue.get("id_status") != "fixed":
        raise ValueError("fixed 상태의 issue_id가 필요합니다")
    members = payload.get("members") or []
    if not members:
        raise ValueError("타임라인에 members가 없습니다")

    cutoff_month = month_key(args.analysis_cutoff)
    first_month = month_key((payload.get("first_observation") or {}).get("date"))
    months = month_range(first_month, cutoff_month)
    counts_by_month: dict[str, Counter[str]] = defaultdict(Counter)
    similarities_by_month: dict[str, list[float]] = defaultdict(list)
    ignored_after_cutoff = 0
    for member in members:
        parsed = member_date_and_council(member)
        if parsed is None:
            continue
        month, council = parsed
        if month > cutoff_month:
            ignored_after_cutoff += 1
            continue
        if month < first_month:
            continue
        counts_by_month[month][council] += 1
        similarities_by_month[month].append(
            safe_float(member.get("centroid_similarity"))
        )

    first_month_by_council: dict[str, str] = {}
    for month in months:
        for council in counts_by_month[month]:
            first_month_by_council.setdefault(council, month)

    mentions = [sum(counts_by_month[month].values()) for month in months]
    active = [len(counts_by_month[month]) for month in months]
    new_counts = [
        sum(first == month for first in first_month_by_council.values())
        for month in months
    ]
    cumulative_values = []
    cumulative = 0
    for value in new_counts:
        cumulative += value
        cumulative_values.append(cumulative)

    rows: list[dict[str, Any]] = []
    minimum_index = args.long_lookback - 1
    for index in range(minimum_index, len(months)):
        recent_mentions = padded_tail(mentions, index, args.lookback)
        previous_mentions = padded_tail(
            mentions, index - args.lookback, args.lookback
        ) if index >= args.lookback else [0.0] * args.lookback
        long_mentions = padded_tail(mentions, index, args.long_lookback)
        recent_active = padded_tail(active, index, args.lookback)
        recent_new = padded_tail(new_counts, index, args.lookback)
        recent_mean = float(np.mean(recent_mentions))
        previous_mean = float(np.mean(previous_mentions))
        hhi, top_share, entropy = concentration(counts_by_month[months[index]])
        current_similarities = similarities_by_month[months[index]]
        next_known = index + 1 < len(months)
        next_mentions = mentions[index + 1] if next_known else None
        next_new = new_counts[index + 1] if next_known else None
        row = {
            "issue_id": issue_id,
            "issue_name": str(issue.get("issue_name") or ""),
            "feature_month": months[index],
            "target_month": add_month(months[index]),
            "mention_count": mentions[index],
            "active_council_count": active[index],
            "new_council_count": new_counts[index],
            "cumulative_council_count": cumulative_values[index],
            "council_coverage_ratio": round(
                cumulative_values[index] / args.total_councils, 6
            ),
            "mention_mean_3m": round(recent_mean, 6),
            "mention_mean_previous_3m": round(previous_mean, 6),
            "mention_momentum_3m": round(
                (recent_mean - previous_mean) / max(previous_mean, 1.0), 6
            ),
            "mention_slope_3m": round(slope(recent_mentions), 6),
            "mention_slope_6m": round(slope(long_mentions), 6),
            "active_council_mean_3m": round(float(np.mean(recent_active)), 6),
            "new_council_sum_3m": int(sum(recent_new)),
            "active_month_ratio_3m": round(
                sum(value > 0 for value in recent_mentions) / args.lookback, 6
            ),
            "months_since_last_mention": months_since_positive(mentions, index),
            "issue_age_months": index + 1,
            "council_hhi": round(hhi, 6),
            "top_council_share": round(top_share, 6),
            "council_entropy": round(entropy, 6),
            "centroid_similarity_mean": round(
                float(np.mean(current_similarities)), 6
            ) if current_similarities else 0.0,
            "next_month_mention_count": next_mentions if next_known else "",
            "target_discussion_active": int(next_mentions > 0) if next_known else "",
            "target_discussion_growth": int(
                next_mentions > recent_mean and next_mentions > 0
            ) if next_known else "",
            "target_new_council_spread": int(next_new > 0) if next_known else "",
        }
        rows.append(row)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "monthly_features.csv"
    forecast_path = output_dir / "forecast_input.json"
    summary_path = output_dir / "feature_summary.json"
    write_csv(features_path, rows)

    forecast_row = rows[-1]
    feature_names = [
        name for name in forecast_row
        if name not in {
            "issue_id", "issue_name", "feature_month", "target_month",
            "next_month_mention_count", "target_discussion_active",
            "target_discussion_growth", "target_new_council_spread",
        }
    ]
    atomic_json(forecast_path, {
        "issue_id": issue_id,
        "issue_name": str(issue.get("issue_name") or ""),
        "analysis_cutoff": args.analysis_cutoff,
        "feature_month": forecast_row["feature_month"],
        "forecast_target_month": forecast_row["target_month"],
        "feature_names": feature_names,
        "features": {name: forecast_row[name] for name in feature_names},
        "restriction": "제공 데이터의 분석 기준일까지의 정보만 사용",
    })
    known_rows = [row for row in rows if row["target_discussion_active"] != ""]
    summary = {
        "status": "complete",
        "issue_id": issue_id,
        "source": str(input_path),
        "analysis_cutoff": args.analysis_cutoff,
        "calendar_month_count": len(months),
        "zero_mention_month_count": sum(value == 0 for value in mentions),
        "feature_row_count": len(rows),
        "labeled_row_count": len(known_rows),
        "forecast_target_month": forecast_row["target_month"],
        "observed_council_count": len(first_month_by_council),
        "ignored_members_after_cutoff": ignored_after_cutoff,
        "targets": {
            "discussion_active_positive": sum(
                int(row["target_discussion_active"]) for row in known_rows
            ),
            "discussion_growth_positive": sum(
                int(row["target_discussion_growth"]) for row in known_rows
            ),
            "new_council_spread_positive": sum(
                int(row["target_new_council_spread"]) for row in known_rows
            ),
        },
        "leakage_control": (
            "특징은 feature_month까지, 정답은 그 다음 달에서만 계산"
        ),
        "training_readiness": (
            "single_issue_prototype_only" if len(known_rows) < 30
            else "minimum_sample_count_reached"
        ),
    }
    atomic_json(summary_path, summary)

    print("\n=== 월별 예측 특징 생성 완료 ===")
    print(f"issue_id: {issue_id}")
    print(
        f"달력 월: {len(months)} | 무관측 월: {summary['zero_mention_month_count']} | "
        f"학습 가능 행: {len(known_rows)}"
    )
    print(f"예측 대상 월: {forecast_row['target_month']}")
    print(f"학습 준비 상태: {summary['training_readiness']}")
    print(f"features: {features_path}")
    print(f"forecast input: {forecast_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
