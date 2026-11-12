#!/usr/bin/env python3
"""승인된 여러 이슈의 월별 특징을 하나의 누수 방지 패널로 결합한다."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = {
    "next_month_mention_count",
    "target_discussion_active",
    "target_discussion_growth",
    "target_new_council_spread",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "issues" / "final_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "forecast" / "multi_issue_v1",
    )
    parser.add_argument("--minimum-issues", type=int, default=5)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"특징 파일이 비었습니다: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("저장할 패널 행이 없습니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.minimum_issues < 2:
        raise ValueError("minimum-issues는 2 이상이어야 합니다")
    input_dir = args.input_dir.expanduser().resolve()
    feature_paths = sorted(input_dir.glob("*/02_forecast_features/monthly_features.csv"))
    if not feature_paths:
        raise FileNotFoundError(f"최종 이슈 특징 파일이 없습니다: {input_dir}")

    rows: list[dict[str, str]] = []
    sources = []
    schemas: set[tuple[str, ...]] = set()
    for path in feature_paths:
        current = read_rows(path)
        schemas.add(tuple(current[0]))
        issue_ids = sorted({row["issue_id"] for row in current})
        if len(issue_ids) != 1:
            raise ValueError(f"한 파일에 issue_id가 여러 개입니다: {path}")
        rows.extend(current)
        sources.append({
            "issue_key": path.parents[1].name,
            "issue_id": issue_ids[0],
            "row_count": len(current),
            "path": str(path),
        })
    if len(schemas) != 1:
        raise ValueError("이슈별 특징 CSV 스키마가 서로 다릅니다")

    issue_ids = sorted({row["issue_id"] for row in rows})
    if len(issue_ids) < args.minimum_issues:
        raise ValueError(
            f"다중 이슈 학습에 필요한 이슈가 부족합니다: {len(issue_ids)} < "
            f"{args.minimum_issues}"
        )
    keys = [(row["issue_id"], row["feature_month"]) for row in rows]
    if len(keys) != len(set(keys)):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise ValueError(f"issue_id×feature_month 중복: {duplicates[:10]}")
    rows.sort(key=lambda row: (row["target_month"], row["issue_id"]))

    latest_rows = []
    for issue_id in issue_ids:
        issue_rows = [row for row in rows if row["issue_id"] == issue_id]
        latest = max(issue_rows, key=lambda row: row["feature_month"])
        latest_rows.append(latest)

    labeled = [
        row for row in rows
        if all(str(row.get(column, "")).strip() != "" for column in TARGET_COLUMNS)
    ]
    feature_columns = [
        name for name in rows[0]
        if name not in {
            "issue_id", "issue_name", "feature_month", "target_month",
            *TARGET_COLUMNS,
        }
    ]
    output_dir = args.output_dir.expanduser().resolve()
    panel_path = output_dir / "multi_issue_panel.csv"
    latest_path = output_dir / "latest_forecast_inputs.json"
    summary_path = output_dir / "panel_summary.json"
    write_csv(panel_path, rows)
    atomic_json(latest_path, {
        "status": "complete",
        "issue_count": len(issue_ids),
        "feature_columns": feature_columns,
        "rows": latest_rows,
        "leakage_control": "각 특징은 feature_month까지, 정답은 target_month에서만 계산",
    })
    target_counts = {
        column: Counter(str(row[column]) for row in labeled)
        for column in sorted(TARGET_COLUMNS - {"next_month_mention_count"})
    }
    summary = {
        "status": "complete",
        "issue_count": len(issue_ids),
        "total_rows": len(rows),
        "labeled_rows": len(labeled),
        "feature_count": len(feature_columns),
        "first_target_month": min(row["target_month"] for row in labeled),
        "last_target_month": max(row["target_month"] for row in labeled),
        "target_counts": {
            key: dict(value) for key, value in target_counts.items()
        },
        "sources": sources,
        "panel": str(panel_path),
        "latest_forecast_inputs": str(latest_path),
    }
    atomic_json(summary_path, summary)

    print("\n=== 다중 이슈 월별 패널 생성 완료 ===")
    print(
        f"이슈: {len(issue_ids)} | 전체 행: {len(rows):,} | "
        f"학습 행: {len(labeled):,} | 특징: {len(feature_columns)}"
    )
    print(f"기간: {summary['first_target_month']} ~ {summary['last_target_month']}")
    print(f"panel: {panel_path}")
    print(f"forecast input: {latest_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
