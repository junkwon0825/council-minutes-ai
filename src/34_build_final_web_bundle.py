#!/usr/bin/env python3
"""10개 고정 이슈 웹 결과와 다중 이슈 예측을 최종 웹 번들로 결합한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--web-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "web",
    )
    parser.add_argument(
        "--multi-forecast", type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "forecast" / "multi_issue_v1" /
            "model" / "multi_issue_forecast.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "web_final",
    )
    parser.add_argument("--expected-issues", type=int, default=10)
    parser.add_argument(
        "--external-followup-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "external" / "followup_v1",
        help="32a 외부 후속 흐름 결과 폴더. 없으면 기존 데이터만으로 생성",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    web_dir = args.web_dir.expanduser().resolve()
    forecast_path = args.multi_forecast.expanduser().resolve()
    if not forecast_path.is_file():
        raise FileNotFoundError(forecast_path)
    model_report = load_json(forecast_path)
    if model_report.get("status") != "complete":
        raise ValueError("다중 이슈 예측 결과가 complete 상태가 아닙니다")
    forecasts = {
        str(row.get("issue_id")): row
        for row in (model_report.get("forecasts") or [])
        if row.get("issue_id")
    }
    if len(forecasts) < args.expected_issues:
        raise ValueError(
            f"다중 예측 이슈가 부족합니다: {len(forecasts)} < {args.expected_issues}"
        )

    payload_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(web_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("schema_version") != "council_issue_web_v1":
            continue
        issue_id = str((payload.get("issue") or {}).get("issue_id") or "")
        if issue_id not in forecasts:
            continue
        previous = payload_by_id.get(issue_id)
        if previous is None or path.stat().st_mtime_ns > previous[0].stat().st_mtime_ns:
            payload_by_id[issue_id] = (path, payload)

    missing_payloads = sorted(set(forecasts) - set(payload_by_id))
    if missing_payloads:
        raise ValueError(
            "다중 예측에 대응하는 웹 JSON이 없습니다: " + ", ".join(missing_payloads)
        )
    followup_dir = args.external_followup_dir.expanduser().resolve()
    followups: dict[str, dict[str, Any]] = {}
    for path in sorted(followup_dir.glob("*.json")) if followup_dir.is_dir() else []:
        try:
            followup = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if followup.get("schema_version") not in {
            "council_external_followup_v1", "council_external_followup_v2",
        }:
            continue
        followup_issue_id = str((followup.get("issue") or {}).get("issue_id") or "")
        if followup_issue_id:
            followups[followup_issue_id] = followup
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for issue_id, forecast in forecasts.items():
        source_path, payload = payload_by_id[issue_id]
        augmented = dict(payload)
        augmented["schema_version"] = "council_issue_web_v2"
        augmented["multi_issue_forecast"] = {
            **forecast,
            "model_selection": model_report.get("selection"),
            "validation_metrics": model_report.get("metrics"),
            "required_evaluation": model_report.get("required_evaluation"),
            "rolling_backtest": model_report.get("rolling_backtest"),
            "method": model_report.get("method"),
            "leakage_control": model_report.get("leakage_control"),
            "limitations": model_report.get("limitations"),
        }
        followup = followups.get(issue_id)
        if followup:
            augmented["external_followup"] = followup
        output_path = output_dir / source_path.name
        atomic_json(output_path, augmented)
        issue = augmented.get("issue") or {}
        summary = augmented.get("summary") or {}
        single = augmented.get("forecast") or {}
        entries.append({
            "issue_id": issue_id,
            "issue_key": source_path.stem,
            "issue_name": issue.get("issue_name"),
            "aliases": issue.get("aliases") or [],
            "first_observation_date": summary.get("first_observation_date"),
            "first_observation_councils": summary.get("first_observation_councils") or [],
            "council_count": summary.get("council_count"),
            "evidence_unit_count": summary.get("evidence_unit_count"),
            "target_month": forecast.get("target_month"),
            "discussion_active_probability": forecast.get(
                "discussion_active_probability"
            ),
            "predicted_mention_count": forecast.get("predicted_mention_count"),
            "selected_classifier": forecast.get("selected_classifier"),
            "selected_count_model": forecast.get("selected_count_model"),
            "single_issue_direction": single.get("direction"),
            "external_followup_candidates": (
                [
                    row.get("council")
                    for row in (followup.get("outlook") or {}).get("top_candidates") or []
                    if row.get("council")
                ] if followup else None
            ),
            "payload": str(output_path),
        })
    entries.sort(key=lambda row: (
        -float(row.get("discussion_active_probability") or 0.0),
        str(row.get("issue_name") or ""),
    ))
    metrics = model_report.get("metrics") or {}
    catalog = {
        "schema_version": "council_issue_catalog_v1",
        "status": "complete",
        "issue_count": len(entries),
        "analysis_scope": "provided_2024_2025_council_minutes_only",
        "model_summary": {
            "selection": model_report.get("selection"),
            "metrics": metrics,
            "required_evaluation": model_report.get("required_evaluation"),
            "rolling_backtest": model_report.get("rolling_backtest"),
            "method": model_report.get("method"),
            "leakage_control": model_report.get("leakage_control"),
            "training_rows": model_report.get("training_rows"),
            "validation_predictions": model_report.get("validation_predictions"),
        },
        "issues": entries,
        "source": {
            "web_dir": str(web_dir),
            "multi_forecast": str(forecast_path),
            "external_followup_dir": str(followup_dir) if followups else None,
        },
    }
    catalog_path = output_dir / "issue_catalog.json"
    atomic_json(catalog_path, catalog)

    print("\n=== 최종 웹 번들 생성 완료 ===")
    print(f"이슈: {len(entries)} | 예측 결합: {len(forecasts)}")
    print(
        "분류 선택: "
        f"{(model_report.get('selection') or {}).get('classifier')} | "
        "언급량 선택: "
        f"{(model_report.get('selection') or {}).get('count_model')}"
    )
    print(f"외부 후속 흐름 결합: {sum(issue_id in followups for issue_id in forecasts)}")
    print(f"catalog: {catalog_path}")
    print(f"payloads: {output_dir}")


if __name__ == "__main__":
    main()
