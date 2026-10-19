#!/usr/bin/env python3
"""최종 타임라인·예측 결과의 연결성과 근거 누락을 검사한다."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


TIMELINES = Path("outputs/issue_timeline_final/issue_timelines.json")
FORECASTS = Path("outputs/issue_forecast_final/issue_forecasts.json")
MODEL_REPORT = Path("outputs/issue_forecast_final/model_comparison.json")


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"필수 결과 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    timelines = load(TIMELINES)
    forecasts = load(FORECASTS)
    model_report = load(MODEL_REPORT)
    if not timelines:
        raise SystemExit("타임라인이 비어 있습니다.")

    timeline_ids = [str(row.get("issue_id", "")) for row in timelines]
    forecast_ids = [str(row.get("issue_id", "")) for row in forecasts]
    duplicates = [key for key, count in Counter(timeline_ids).items() if key and count > 1]
    if duplicates:
        raise SystemExit(f"중복 issue_id: {duplicates}")
    missing_forecast = sorted(set(timeline_ids) - set(forecast_ids))
    if missing_forecast:
        raise SystemExit(f"예측이 없는 issue_id: {missing_forecast}")

    evidence_total = 0
    empty_excerpt = []
    invalid_dates = []
    for timeline in timelines:
        first_date = str(timeline.get("first_observed_date", ""))
        propagation = timeline.get("propagation", [])
        if not propagation:
            raise SystemExit(f"전파 관측이 없는 이슈: {timeline.get('issue_id')}")
        for event in propagation:
            if str(event.get("first_date", "")) < first_date:
                invalid_dates.append((timeline.get("issue_id"), event.get("council")))
            for evidence in event.get("evidence", []):
                evidence_total += 1
                if not str(evidence.get("original_excerpt", "")).strip():
                    empty_excerpt.append(
                        (timeline.get("issue_id"), event.get("council"), evidence.get("evidence_ref"))
                    )
    if invalid_dates:
        raise SystemExit(f"최초 관측일보다 이른 의회 이벤트: {invalid_dates[:5]}")
    if empty_excerpt:
        raise SystemExit(
            "원문 근거가 비어 있습니다. 09번의 --evidence-metadata 경로를 확인하세요: "
            f"{empty_excerpt[:5]}"
        )

    for target in ("growth", "spread"):
        report = model_report.get(target, {})
        if not report.get("selected_model"):
            raise SystemExit(f"{target} 선택 모델 정보가 없습니다.")

    print(f"검증 통과: 이슈 {len(timelines):,}개, 예측 {len(forecasts):,}개, 근거 {evidence_total:,}개")
    print(
        "선택 모델: 논의 증가="
        f"{model_report['growth']['selected_model']}, 신규 의회 확산="
        f"{model_report['spread']['selected_model']}"
    )


if __name__ == "__main__":
    main()

