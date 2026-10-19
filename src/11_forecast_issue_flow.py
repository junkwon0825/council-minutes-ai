#!/usr/bin/env python3
"""검수된 이슈 타임라인으로 다음 달 논의 방향과 타 의회 확산 가능성을 예측한다.

외부 데이터와 외부 API를 사용하지 않는다. 제공 데이터의 분석 기준일까지 생성된
월별 관측값만 학습하며, 로지스틱 회귀와 LightGBM을 시간순 검증으로 비교한다.
학습 표본이 부족하거나 한 클래스만 존재하면 설명 가능한 추세 규칙으로 대체한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("numpy가 필요합니다. 실행 전: pip install numpy") from error

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    raise SystemExit("scikit-learn이 필요합니다. 실행 전: pip install scikit-learn") from error


SCRIPT_VERSION = "2026-10-19-issue-flow-forecast-v1.1-performance-gated"
FEATURE_NAMES = [
    "current_mentions",
    "recent_mean",
    "previous_mean",
    "mention_momentum",
    "recent_slope",
    "recent_active_ratio",
    "recent_council_mean",
    "recent_new_council_sum",
    "cumulative_council_ratio",
    "months_since_last_mention",
    "issue_age_months",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이슈 다음 달 논의·확산 예측")
    parser.add_argument(
        "--timelines", default="outputs/issue_timeline_final/issue_timelines.json"
    )
    parser.add_argument("--output-dir", default="outputs/issue_forecast_final")
    parser.add_argument(
        "--analysis-cutoff", default="2025-12-31",
        help="이 날짜까지의 관측만 사용. YYYY-MM-DD",
    )
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=4)
    parser.add_argument("--min-training-samples", type=int, default=30)
    parser.add_argument(
        "--min-validation-f1", type=float, default=0.25,
        help="이 값 미만인 ML 모델은 채택하지 않고 추세 규칙으로 대체",
    )
    parser.add_argument(
        "--min-balanced-accuracy", type=float, default=0.55,
        help="이 값 미만인 ML 모델은 채택하지 않고 추세 규칙으로 대체",
    )
    parser.add_argument("--total-councils", type=int, default=17)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--disable-lightgbm", action="store_true",
        help="LightGBM 없이 로지스틱 회귀와 추세 규칙만 사용",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError as error:
        raise SystemExit(f"날짜 형식은 YYYY-MM-DD여야 합니다: {value}") from error


def month_key(value: str) -> str:
    return str(value)[:7]


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


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sigmoid(value: float) -> float:
    value = min(30.0, max(-30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


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


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def months_since_positive(values: list[float]) -> int:
    for distance, value in enumerate(reversed(values)):
        if value > 0:
            return distance
    return len(values)


def timeline_arrays(timeline: dict, cutoff_month: str) -> dict:
    first_month = month_key(timeline.get("first_observed_date", cutoff_month))
    if not first_month or first_month > cutoff_month:
        return {}
    by_month = {str(row.get("month")): row for row in timeline.get("monthly_series", [])}
    months = month_range(first_month, cutoff_month)
    mentions = [safe_int(by_month.get(month, {}).get("mention_count")) for month in months]
    councils = [safe_int(by_month.get(month, {}).get("council_count")) for month in months]
    new_councils = [safe_int(by_month.get(month, {}).get("new_council_count")) for month in months]
    return {"months": months, "mentions": mentions, "councils": councils, "new": new_councils}


def make_features(arrays: dict, index: int, lookback: int, total_councils: int) -> list[float]:
    mentions = arrays["mentions"]
    councils = arrays["councils"]
    new_councils = arrays["new"]
    recent = mentions[max(0, index - lookback + 1): index + 1]
    previous = mentions[max(0, index - 2 * lookback + 1): max(0, index - lookback + 1)]
    recent = [0] * (lookback - len(recent)) + recent
    previous = [0] * (lookback - len(previous)) + previous
    recent_mean = float(np.mean(recent))
    previous_mean = float(np.mean(previous))
    momentum = (recent_mean - previous_mean) / max(1.0, previous_mean)
    recent_councils = councils[max(0, index - lookback + 1): index + 1]
    recent_new = new_councils[max(0, index - lookback + 1): index + 1]
    cumulative = min(total_councils, sum(new_councils[: index + 1]))
    return [
        float(mentions[index]),
        recent_mean,
        previous_mean,
        float(momentum),
        linear_slope(recent),
        sum(value > 0 for value in recent) / lookback,
        float(np.mean(recent_councils)) if recent_councils else 0.0,
        float(sum(recent_new)),
        cumulative / total_councils,
        float(months_since_positive(mentions[: index + 1])),
        float(index + 1),
    ]


def build_training_rows(
    timelines: list[dict], cutoff_month: str, lookback: int, total_councils: int
) -> tuple[list[dict], dict[str, tuple[dict, list[float]]]]:
    samples = []
    forecasts = {}
    minimum_history = lookback * 2
    for timeline in timelines:
        arrays = timeline_arrays(timeline, cutoff_month)
        if not arrays:
            continue
        months = arrays["months"]
        for index in range(minimum_history - 1, len(months) - 1):
            features = make_features(arrays, index, lookback, total_councils)
            next_count = arrays["mentions"][index + 1]
            recent_mean = features[1]
            samples.append({
                "issue_id": timeline.get("issue_id", ""),
                "target_month": months[index + 1],
                "features": features,
                "growth_target": int(next_count > recent_mean and next_count > 0),
                "spread_target": int(arrays["new"][index + 1] > 0),
            })
        last_index = len(months) - 1
        features = make_features(arrays, last_index, lookback, total_councils)
        forecasts[str(timeline.get("issue_id", ""))] = (arrays, features)
    return samples, forecasts


def lightgbm_factory(random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    return LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def candidate_models(random_state: int, disable_lightgbm: bool) -> dict:
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=random_state
            ),
        )
    }
    if not disable_lightgbm:
        lightgbm = lightgbm_factory(random_state)
        if lightgbm is not None:
            models["lightgbm"] = lightgbm
    return models


def train_task(
    samples: list[dict], target_key: str, validation_months: int,
    min_training_samples: int, random_state: int, disable_lightgbm: bool,
    min_validation_f1: float, min_balanced_accuracy: float,
) -> tuple[object | None, dict]:
    target_name = "논의 증가" if target_key == "growth_target" else "신규 의회 확산"
    report = {
        "target": target_name,
        "sample_count": len(samples),
        "positive_count": sum(row[target_key] for row in samples),
        "positive_rate": round(
            sum(row[target_key] for row in samples) / len(samples), 6
        ) if samples else 0.0,
        "selected_model": "trend_fallback",
        "candidates": [],
    }
    if len(samples) < min_training_samples:
        report["fallback_reason"] = "insufficient_training_samples"
        return None, report
    y_all = np.asarray([row[target_key] for row in samples], dtype=np.int64)
    if len(set(y_all.tolist())) < 2:
        report["fallback_reason"] = "single_target_class"
        return None, report

    target_months = sorted({row["target_month"] for row in samples})
    if len(target_months) <= validation_months:
        report["fallback_reason"] = "insufficient_time_periods"
        return None, report
    validation_set = set(target_months[-validation_months:])
    train_rows = [row for row in samples if row["target_month"] not in validation_set]
    valid_rows = [row for row in samples if row["target_month"] in validation_set]
    y_train = np.asarray([row[target_key] for row in train_rows], dtype=np.int64)
    y_valid = np.asarray([row[target_key] for row in valid_rows], dtype=np.int64)
    if len(train_rows) < min_training_samples or len(set(y_train.tolist())) < 2 or not valid_rows:
        report["fallback_reason"] = "invalid_chronological_split"
        return None, report

    x_train = np.asarray([row["features"] for row in train_rows], dtype=np.float64)
    x_valid = np.asarray([row["features"] for row in valid_rows], dtype=np.float64)
    best = None
    for name, model in candidate_models(random_state, disable_lightgbm).items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid)
        metrics = {
            "model": name,
            "f1": round(float(f1_score(y_valid, prediction, zero_division=0)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_valid, prediction)), 6),
            "train_count": len(train_rows),
            "validation_count": len(valid_rows),
            "validation_months": sorted(validation_set),
        }
        report["candidates"].append(metrics)
        score = (metrics["f1"], metrics["balanced_accuracy"], name == "logistic_regression")
        if best is None or score > best[0]:
            best = (score, name, model, metrics)

    selected_name = best[1]
    selected_metrics = best[3]
    report["selected_validation_metrics"] = selected_metrics
    if (
        selected_metrics["f1"] < min_validation_f1
        or selected_metrics["balanced_accuracy"] < min_balanced_accuracy
    ):
        report["rejected_best_model"] = selected_name
        report["selected_model"] = "trend_fallback"
        report["fallback_reason"] = "poor_validation_performance"
        report["performance_gate"] = {
            "min_validation_f1": min_validation_f1,
            "min_balanced_accuracy": min_balanced_accuracy,
        }
        return None, report

    final_model = candidate_models(random_state, disable_lightgbm)[selected_name]
    x_all = np.asarray([row["features"] for row in samples], dtype=np.float64)
    final_model.fit(x_all, y_all)
    report["selected_model"] = selected_name
    return final_model, report


def fallback_growth_probability(features: list[float]) -> float:
    momentum = max(-2.0, min(2.0, features[3]))
    slope_scale = features[4] / max(1.0, features[1])
    logit = -0.35 + 0.9 * momentum + 0.7 * slope_scale + 0.5 * features[5] - 0.35 * features[9]
    return sigmoid(logit)


def fallback_spread_probability(features: list[float]) -> float:
    saturation = features[8]
    logit = -1.15 + 0.65 * features[7] + 0.25 * features[6] + 0.7 * (1 - saturation) - 0.3 * features[9]
    return sigmoid(logit)


def predict_probability(model, features: list[float], fallback) -> float:
    if model is None:
        return float(fallback(features))
    return float(model.predict_proba(np.asarray([features], dtype=np.float64))[0, 1])


def direction_label(
    probability: float, recent_sum: int, months_since_last: int, predicted_count: int
) -> str:
    if recent_sum == 0:
        return "dormant"
    if predicted_count == 0 and months_since_last >= 1:
        return "attenuation"
    if probability >= 0.6:
        return "expansion"
    if probability <= 0.4:
        return "attenuation"
    return "persistence"


def confidence_label(timeline: dict, growth_report: dict, spread_report: dict) -> str:
    observations = safe_int(timeline.get("observation_count"))
    selected = {growth_report["selected_model"], spread_report["selected_model"]}
    if "trend_fallback" in selected:
        return "low"
    if observations >= 30 and "trend_fallback" not in selected:
        scores = [
            report.get("selected_validation_metrics", {}).get("balanced_accuracy", 0.0)
            for report in (growth_report, spread_report)
        ]
        if min(scores) >= 0.65:
            return "high"
    if observations >= 10:
        return "medium"
    return "low"


def main() -> None:
    args = parse_args()
    cutoff = parse_date(args.analysis_cutoff)
    cutoff_month = cutoff.strftime("%Y-%m")
    if args.lookback < 2 or args.validation_months < 1:
        raise SystemExit("--lookback은 2 이상, --validation-months는 1 이상이어야 합니다.")
    if args.min_training_samples < 10 or args.total_councils < 1:
        raise SystemExit("--min-training-samples는 10 이상, --total-councils는 1 이상이어야 합니다.")
    if not 0.0 <= args.min_validation_f1 <= 1.0:
        raise SystemExit("--min-validation-f1은 0 이상 1 이하여야 합니다.")
    if not 0.5 <= args.min_balanced_accuracy <= 1.0:
        raise SystemExit("--min-balanced-accuracy는 0.5 이상 1 이하여야 합니다.")

    timeline_path = Path(args.timelines).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not timeline_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {timeline_path}")
    timelines = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(timelines, list) or not timelines:
        raise SystemExit("예측 가능한 이슈 타임라인이 없습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, forecast_inputs = build_training_rows(
        timelines, cutoff_month, args.lookback, args.total_councils
    )
    growth_model, growth_report = train_task(
        samples, "growth_target", args.validation_months, args.min_training_samples,
        args.random_state, args.disable_lightgbm,
        args.min_validation_f1, args.min_balanced_accuracy,
    )
    spread_model, spread_report = train_task(
        samples, "spread_target", args.validation_months, args.min_training_samples,
        args.random_state, args.disable_lightgbm,
        args.min_validation_f1, args.min_balanced_accuracy,
    )

    forecast_month = add_month(cutoff_month)
    forecasts = []
    for timeline in timelines:
        issue_id = str(timeline.get("issue_id", ""))
        if issue_id not in forecast_inputs:
            continue
        arrays, features = forecast_inputs[issue_id]
        growth_probability = predict_probability(
            growth_model, features, fallback_growth_probability
        )
        spread_probability = predict_probability(
            spread_model, features, fallback_spread_probability
        )
        recent_mentions = arrays["mentions"][-args.lookback:]
        recent_sum = sum(recent_mentions)
        projected_trend = max(0.0, features[0] + features[4])
        base_count = 0.5 * features[1] + 0.3 * features[0] + 0.2 * projected_trend
        predicted_count = max(0, int(round(base_count * (0.75 + 0.5 * growth_probability))))
        direction = direction_label(
            growth_probability, recent_sum, safe_int(features[9]), predicted_count
        )
        spread_level = "high" if spread_probability >= 0.65 else "medium" if spread_probability >= 0.4 else "low"
        confidence = confidence_label(timeline, growth_report, spread_report)
        narrative = (
            f"분석 기준일 이전 최근 {args.lookback}개월 언급량은 "
            f"{recent_mentions}이며 월평균 {features[1]:.2f}건이다. "
            f"다음 달 논의 방향은 {direction}, 신규 의회 확산 가능성은 {spread_level}로 예측한다. "
            "이는 제공 회의록의 관측 패턴에 기반한 예측이며 실제 정책 인과관계를 의미하지 않는다."
        )
        forecasts.append({
            "issue_id": issue_id,
            "cluster_id": safe_int(timeline.get("cluster_id"), -1),
            "issue_name": str(timeline.get("issue_name", "")),
            "policy_domain": str(timeline.get("policy_domain", "")),
            "review_status": str(timeline.get("review_status", "")),
            "analysis_cutoff": args.analysis_cutoff,
            "forecast_month": forecast_month,
            "predicted_direction": direction,
            "growth_probability": round(growth_probability, 6),
            "predicted_mention_count": predicted_count,
            "spread_probability": round(spread_probability, 6),
            "spread_level": spread_level,
            "forecast_confidence": confidence,
            "recent_months": arrays["months"][-args.lookback:],
            "recent_mentions": recent_mentions,
            "recent_mean": round(features[1], 4),
            "previous_mean": round(features[2], 4),
            "recent_slope": round(features[4], 4),
            "recent_new_council_sum": safe_int(features[7]),
            "observed_council_count": safe_int(timeline.get("observed_council_count")),
            "observation_count": safe_int(timeline.get("observation_count")),
            "months_since_last_mention": safe_int(features[9]),
            "growth_model": growth_report["selected_model"],
            "spread_model": spread_report["selected_model"],
            "forecast_narrative": narrative,
            "interpretation_note": (
                "제공 데이터 범위 내 월별 관측 패턴만 사용한 다음 달 예측이며 "
                "분석 기준일 이후 외부 정보는 사용하지 않았다."
            ),
        })

    forecasts.sort(
        key=lambda row: (row["spread_probability"], row["growth_probability"]), reverse=True
    )
    csv_rows = []
    for row in forecasts:
        item = dict(row)
        item["recent_months"] = " | ".join(row["recent_months"])
        item["recent_mentions"] = " | ".join(map(str, row["recent_mentions"]))
        csv_rows.append(item)
    fields = [
        "issue_id", "cluster_id", "issue_name", "policy_domain", "review_status",
        "analysis_cutoff", "forecast_month", "predicted_direction",
        "growth_probability", "predicted_mention_count", "spread_probability",
        "spread_level", "forecast_confidence", "recent_months", "recent_mentions",
        "recent_mean", "previous_mean", "recent_slope", "recent_new_council_sum",
        "observed_council_count", "observation_count", "months_since_last_mention",
        "growth_model", "spread_model", "forecast_narrative", "interpretation_note",
    ]
    write_csv(output_dir / "issue_forecasts.csv", csv_rows, fields)
    atomic_json(output_dir / "issue_forecasts.json", forecasts)
    atomic_json(output_dir / "model_comparison.json", {
        "growth": growth_report,
        "spread": spread_report,
        "feature_names": FEATURE_NAMES,
        "selection_rule": "시간순 검증 F1, balanced accuracy 순. 동률이면 로지스틱 회귀 우선",
        "performance_gate": {
            "min_validation_f1": args.min_validation_f1,
            "min_balanced_accuracy": args.min_balanced_accuracy,
            "below_gate_action": "trend_fallback",
        },
    })
    summary = {
        "script_version": SCRIPT_VERSION,
        "security": "제공 회의록 파생 로컬 파일만 사용하며 외부 API와 외부 정보를 사용하지 않음",
        "timeline_path": str(timeline_path),
        "analysis_cutoff": args.analysis_cutoff,
        "forecast_month": forecast_month,
        "timeline_issue_count": len(timelines),
        "forecast_issue_count": len(forecasts),
        "historical_training_sample_count": len(samples),
        "direction_counts": dict(Counter(row["predicted_direction"] for row in forecasts)),
        "spread_level_counts": dict(Counter(row["spread_level"] for row in forecasts)),
        "growth_selected_model": growth_report["selected_model"],
        "spread_selected_model": spread_report["selected_model"],
        "generated_files": [
            "issue_forecasts.csv", "issue_forecasts.json",
            "model_comparison.json", "forecast_summary.json",
        ],
    }
    atomic_json(output_dir / "forecast_summary.json", summary)

    print(f"완료: 이슈 {len(forecasts):,}개 / 예측 월 {forecast_month}")
    print(f"논의 증가 모델: {growth_report['selected_model']}")
    print(f"신규 의회 확산 모델: {spread_report['selected_model']}")
    print(f"예측 CSV: {output_dir / 'issue_forecasts.csv'}")
    print(f"모델 비교: {output_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
