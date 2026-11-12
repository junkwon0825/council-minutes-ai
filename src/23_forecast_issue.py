#!/usr/bin/env python3
"""단일 고정 이슈의 다음 달 흐름을 설명 가능한 소표본 기준선으로 예측한다.

표본이 적은 단일 이슈에는 분류 모델을 학습하지 않는다. 여러 단순 시계열
후보를 rolling-origin 방식으로 백테스트하고 MAE가 가장 낮은 후보를 선택한다.
확률값은 단일 이슈 내부 빈도와 최근 신호를 결합한 비보정 지표다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--forecast-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backtest-output", type=Path, default=None)
    parser.add_argument("--minimum-history", type=int, default=6)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float = 0.02, upper: float = 0.98) -> float:
    return max(lower, min(upper, value))


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def prediction_last(history: list[float]) -> float:
    return history[-1]


def prediction_mean3(history: list[float]) -> float:
    return float(np.mean(history[-3:]))


def prediction_median3(history: list[float]) -> float:
    return float(np.median(history[-3:]))


def prediction_weighted3(history: list[float]) -> float:
    values = np.asarray(history[-3:], dtype=np.float64)
    weights = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(np.average(values, weights=weights))


def prediction_damped_trend6(history: list[float]) -> float:
    values = np.asarray(history[-6:], dtype=np.float64)
    if len(values) < 2:
        return float(values[-1])
    x = np.arange(len(values), dtype=np.float64)
    slope, intercept = np.polyfit(x, values, 1)
    raw_next = intercept + slope * len(values)
    damped = 0.65 * raw_next + 0.35 * float(np.mean(values[-3:]))
    return float(max(0.0, damped))


def count_direction(current: float, future: float) -> str:
    current_rounded = int(round(max(0.0, current)))
    future_rounded = int(round(max(0.0, future)))
    if future_rounded > current_rounded:
        return "growth"
    if future_rounded < current_rounded:
        return "attenuation"
    return "persistence"


def binary_f1(actual: list[int], predicted: list[int]) -> float:
    true_positive = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    false_positive = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    false_negative = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


MODEL_FUNCTIONS: dict[str, Callable[[list[float]], float]] = {
    "last_value": prediction_last,
    "mean_3m": prediction_mean3,
    "median_3m": prediction_median3,
    "weighted_mean_3m": prediction_weighted3,
    "damped_linear_trend_6m": prediction_damped_trend6,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"특징 CSV가 비었습니다: {path}")
    required = {
        "issue_id", "feature_month", "mention_count",
        "target_discussion_active", "target_discussion_growth",
        "target_new_council_spread",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"특징 CSV 필수 열 누락: {sorted(missing)}")
    rows.sort(key=lambda row: row["feature_month"])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def backtest(
    months: list[str], values: list[float], minimum_history: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(values) <= minimum_history:
        raise ValueError(
            f"백테스트 이력이 부족합니다: {len(values)} <= {minimum_history}"
        )
    details = []
    metrics = []
    for model_name, predictor in MODEL_FUNCTIONS.items():
        errors = []
        squared_errors = []
        direction_matches = []
        actual_active = []
        predicted_active = []
        for index in range(minimum_history, len(values)):
            prediction = max(0.0, predictor(values[:index]))
            actual = values[index]
            current = values[index - 1]
            error = actual - prediction
            errors.append(abs(error))
            squared_errors.append(error * error)
            actual_direction = count_direction(current, actual)
            predicted_direction = count_direction(current, prediction)
            direction_matches.append(actual_direction == predicted_direction)
            actual_active.append(int(actual > 0))
            predicted_active.append(int(round(prediction) > 0))
            details.append({
                "model": model_name,
                "target_month": months[index],
                "current_count": round(current, 6),
                "prediction": round(prediction, 6),
                "actual": round(actual, 6),
                "absolute_error": round(abs(error), 6),
                "actual_direction": actual_direction,
                "predicted_direction": predicted_direction,
                "direction_correct": int(actual_direction == predicted_direction),
                "actual_active": int(actual > 0),
                "predicted_active": int(round(prediction) > 0),
            })
        metrics.append({
            "model": model_name,
            "backtest_count": len(errors),
            "mae": round(float(np.mean(errors)), 6),
            "rmse": round(float(np.sqrt(np.mean(squared_errors))), 6),
            "median_absolute_error": round(float(np.median(errors)), 6),
            "directional_accuracy": round(float(np.mean(direction_matches)), 6),
            "discussion_active_f1": round(
                binary_f1(actual_active, predicted_active), 6
            ),
        })
    metrics.sort(key=lambda row: (row["mae"], row["rmse"], row["model"]))
    return metrics, details


def binary_values(rows: list[dict[str, str]], field: str) -> list[int]:
    values = []
    for row in rows:
        text = str(row.get(field, "")).strip()
        if text in {"0", "1"}:
            values.append(int(text))
    return values


def beta_rate(values: list[int]) -> float:
    return (sum(values) + 1.0) / (len(values) + 2.0)


def probability_indicators(
    rows: list[dict[str, str]], current: dict[str, Any], predicted_count: float,
) -> dict[str, float]:
    active_history = binary_values(rows, "target_discussion_active")
    growth_history = binary_values(rows, "target_discussion_growth")
    spread_history = binary_values(rows, "target_new_council_spread")

    active_recent = safe_float(current.get("active_month_ratio_3m"))
    months_since = safe_float(current.get("months_since_last_mention"))
    active_probability = clamp(
        0.65 * beta_rate(active_history)
        + 0.35 * active_recent
        - 0.08 * min(months_since, 3.0)
    )

    momentum = safe_float(current.get("mention_momentum_3m"))
    recent_mean = max(1.0, safe_float(current.get("mention_mean_3m")))
    slope_scaled = safe_float(current.get("mention_slope_3m")) / recent_mean
    count_growth_signal = sigmoid(1.1 * momentum + 0.8 * slope_scaled)
    predicted_growth_signal = sigmoid((predicted_count - recent_mean) / recent_mean)
    growth_probability = clamp(
        0.55 * beta_rate(growth_history)
        + 0.25 * count_growth_signal
        + 0.20 * predicted_growth_signal
    )

    coverage = clamp(
        safe_float(current.get("council_coverage_ratio")), 0.0, 1.0
    )
    recent_new = min(1.0, safe_float(current.get("new_council_sum_3m")) / 2.0)
    spread_probability = clamp(
        0.60 * beta_rate(spread_history)
        + 0.25 * recent_new
        + 0.15 * (1.0 - coverage)
    )
    return {
        "discussion_active_probability": round(active_probability, 4),
        "discussion_growth_probability": round(growth_probability, 4),
        "new_council_spread_probability": round(spread_probability, 4),
    }


def direction_label(
    predicted_count: float, recent_mean: float, active_probability: float,
) -> str:
    if active_probability < 0.4 or predicted_count < 0.5:
        return "attenuation"
    if predicted_count > max(1.0, recent_mean * 1.2):
        return "growth"
    if predicted_count < recent_mean * 0.7:
        return "attenuation"
    return "persistence"


def main() -> None:
    args = parse_args()
    if args.minimum_history < 3:
        raise ValueError("minimum-history는 3 이상이어야 합니다")
    features_path = args.features.expanduser().resolve()
    forecast_input_path = args.forecast_input.expanduser().resolve()
    if not features_path.is_file():
        raise FileNotFoundError(features_path)
    if not forecast_input_path.is_file():
        raise FileNotFoundError(forecast_input_path)

    rows = read_rows(features_path)
    forecast_input = json.loads(forecast_input_path.read_text(encoding="utf-8"))
    issue_id = str(forecast_input.get("issue_id") or "")
    if not issue_id.startswith("ISSUE_"):
        raise ValueError("forecast input의 issue_id가 올바르지 않습니다")
    if {row["issue_id"] for row in rows} != {issue_id}:
        raise ValueError("CSV와 forecast input의 issue_id가 다릅니다")

    months = [row["feature_month"] for row in rows]
    values = [safe_float(row["mention_count"]) for row in rows]
    metrics, details = backtest(months, values, args.minimum_history)
    selected = metrics[0]
    selected_name = selected["model"]
    last_value_metrics = next(
        row for row in metrics if row["model"] == "last_value"
    )
    predicted_count_raw = max(0.0, MODEL_FUNCTIONS[selected_name](values))

    selected_errors = [
        row["absolute_error"] for row in details if row["model"] == selected_name
    ]
    error_80 = float(np.quantile(selected_errors, 0.80))
    lower = max(0.0, predicted_count_raw - error_80)
    upper = predicted_count_raw + error_80
    current = dict(forecast_input.get("features") or {})
    probabilities = probability_indicators(rows, current, predicted_count_raw)
    recent_mean = safe_float(current.get("mention_mean_3m"))
    direction = direction_label(
        predicted_count_raw, recent_mean,
        probabilities["discussion_active_probability"],
    )

    report = {
        "status": "complete",
        "issue": {
            "issue_id": issue_id,
            "issue_name": str(forecast_input.get("issue_name") or ""),
        },
        "forecast": {
            "analysis_cutoff": forecast_input.get("analysis_cutoff"),
            "target_month": forecast_input.get("forecast_target_month"),
            "direction": direction,
            "predicted_mention_count": int(round(predicted_count_raw)),
            "predicted_mention_count_raw": round(predicted_count_raw, 4),
            "prediction_interval_80": [
                int(math.floor(lower)), int(math.ceil(upper)),
            ],
            **probabilities,
        },
        "selected_count_baseline": selected,
        "candidate_backtest_metrics": metrics,
        "required_evaluation": {
            "rolling_backtest_method": "rolling_origin_single_issue",
            "mention_count_mae": selected["mae"],
            "mention_count_rmse": selected["rmse"],
            "directional_accuracy": selected["directional_accuracy"],
            "discussion_active_f1": selected["discussion_active_f1"],
            "last_month_persistence_comparison": {
                "baseline_model": "last_value",
                "baseline_mae": last_value_metrics["mae"],
                "baseline_rmse": last_value_metrics["rmse"],
                "baseline_directional_accuracy": last_value_metrics[
                    "directional_accuracy"
                ],
                "baseline_discussion_active_f1": last_value_metrics[
                    "discussion_active_f1"
                ],
                "mae_improvement": round(
                    last_value_metrics["mae"] - selected["mae"], 6
                ),
                "rmse_improvement": round(
                    last_value_metrics["rmse"] - selected["rmse"], 6
                ),
                "directional_accuracy_delta": round(
                    selected["directional_accuracy"]
                    - last_value_metrics["directional_accuracy"], 6
                ),
                "discussion_active_f1_delta": round(
                    selected["discussion_active_f1"]
                    - last_value_metrics["discussion_active_f1"], 6
                ),
            },
        },
        "supporting_signals": {
            "current_mention_count": safe_float(current.get("mention_count")),
            "mention_mean_3m": recent_mean,
            "mention_momentum_3m": safe_float(
                current.get("mention_momentum_3m")
            ),
            "mention_slope_3m": safe_float(current.get("mention_slope_3m")),
            "mention_slope_6m": safe_float(current.get("mention_slope_6m")),
            "active_council_count": safe_float(
                current.get("active_council_count")
            ),
            "new_council_sum_3m": safe_float(current.get("new_council_sum_3m")),
            "cumulative_council_count": safe_float(
                current.get("cumulative_council_count")
            ),
            "council_coverage_ratio": safe_float(
                current.get("council_coverage_ratio")
            ),
            "top_council_share": safe_float(current.get("top_council_share")),
            "council_entropy": safe_float(current.get("council_entropy")),
        },
        "method": {
            "mode": "single_issue_explainable_baseline",
            "selection": (
                "rolling-origin 백테스트에서 MAE가 가장 낮은 건수 예측 후보 선택"
            ),
            "probability_note": (
                "확률은 단일 이슈 내부 관측 빈도와 최근 신호를 결합한 비보정 지표"
            ),
            "external_information_used": False,
            "llm_used_for_numeric_prediction": False,
        },
        "limitations": [
            "단일 이슈의 소표본 결과이므로 분류 모델 학습 결과가 아님",
            "예측 확률은 다중 이슈 검증 전까지 상대적 지표로만 해석",
            "의회 관측 순서는 정책의 인과적 전파를 증명하지 않음",
        ],
        "sources": {
            "features": str(features_path),
            "forecast_input": str(forecast_input_path),
        },
    }
    output_path = args.output.expanduser().resolve()
    atomic_json(output_path, report)
    backtest_path = (
        args.backtest_output.expanduser().resolve()
        if args.backtest_output else output_path.with_name("forecast_backtest.csv")
    )
    write_csv(backtest_path, details)

    print("\n=== 단일 이슈 예측 프로토타입 완료 ===")
    print(f"issue_id: {issue_id}")
    print(
        f"선택 기준선: {selected_name} | MAE={selected['mae']:.3f} | "
        f"RMSE={selected['rmse']:.3f} | "
        f"방향성 정확도={selected['directional_accuracy']:.3f} | "
        f"논의 지속 F1={selected['discussion_active_f1']:.3f}"
    )
    print(
        f"{report['forecast']['target_month']} 예측: {direction} | "
        f"언급 {report['forecast']['predicted_mention_count']}건 | "
        f"80% 경험적 구간 {report['forecast']['prediction_interval_80']}"
    )
    print(
        "지속 지표="
        f"{probabilities['discussion_active_probability']:.3f} | "
        "증가 지표="
        f"{probabilities['discussion_growth_probability']:.3f} | "
        "신규 의회 확산 지표="
        f"{probabilities['new_council_spread_probability']:.3f}"
    )
    print(f"JSON: {output_path}")
    print(f"backtest: {backtest_path}")


if __name__ == "__main__":
    main()
