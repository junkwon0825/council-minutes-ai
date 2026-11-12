#!/usr/bin/env python3
"""검수 승인된 이슈를 타임라인·예측·웹 결과까지 일괄 생성한다.

25번의 issue_review_queue.csv에서 approve 결정을 받은 행만 기본 처리한다.
CLI에서 직접 승인하려면 --keys와 --approve-selected를 함께 사용해야 한다.
기존 군집을 최신 중복 제거 코드와 검수 임계값으로 다시 만들려면
--refresh-clusters를 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_VALUES = {"approve", "approved", "accept", "accepted", "yes", "1", "승인"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-queue", type=Path,
        default=PROJECT_ROOT / "outputs" / "issues" / "candidates_v1" / "issue_review_queue.csv",
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "config" / "issue_seed_catalog_v1.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "issues" / "final_v1",
    )
    parser.add_argument(
        "--web-output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "web",
    )
    parser.add_argument(
        "--registry", type=Path,
        default=PROJECT_ROOT / "data" / "issues" / "issue_registry.json",
    )
    parser.add_argument("--keys", nargs="*", default=[])
    parser.add_argument(
        "--approve-selected", action="store_true",
        help="--keys로 지정한 이슈를 CSV 결정과 무관하게 명시 승인",
    )
    parser.add_argument(
        "--refresh-clusters", action="store_true",
        help="02_reranked.json부터 최신 20번 코드로 군집 재생성",
    )
    parser.add_argument("--min-centroid-similarity", type=float, default=0.45)
    parser.add_argument("--cluster-similarity", type=float, default=0.68)
    parser.add_argument("--expansion-k", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=500)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-legacy-cluster", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def catalog_by_key(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    required = {"issue_key", "issue_name", "aliases"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"이슈 카탈로그 열이 올바르지 않습니다: {path}")
    return {row["issue_key"]: row for row in rows}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def run(command: list[str], dry_run: bool) -> float:
    print("\n$ " + " ".join(command))
    if dry_run:
        return 0.0
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return time.perf_counter() - started


def find_cluster(review_row: dict[str, str], review_queue: Path) -> Path:
    recorded = Path(str(review_row.get("cluster_json") or "")).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    fallback = review_queue.parent / review_row["issue_key"] / "03_cluster.json"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        f"{review_row['issue_key']} 군집 JSON을 찾지 못했습니다: {recorded}, {fallback}"
    )


def cluster_is_current(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expansion = payload.get("expansion") or {}
    return "exact_text_duplicates_removed" in expansion


def choose_rows(
    rows: list[dict[str, str]], keys: set[str], approve_selected: bool,
) -> list[dict[str, str]]:
    by_key = {row.get("issue_key", ""): row for row in rows}
    if keys:
        missing = sorted(keys - set(by_key))
        if missing:
            raise ValueError(f"검수 대기열에 없는 issue_key: {missing}")
        selected = [by_key[key] for key in sorted(keys)]
        if approve_selected:
            return selected
        not_approved = [
            row["issue_key"] for row in selected
            if str(row.get("review_decision") or "").strip().lower()
            not in APPROVED_VALUES
        ]
        if not_approved:
            raise ValueError(
                "review_decision이 approve가 아닙니다: "
                f"{not_approved}. 직접 승인하려면 --approve-selected를 추가하세요."
            )
        return selected
    if approve_selected:
        raise ValueError("--approve-selected는 안전을 위해 --keys와 함께 사용해야 합니다")
    return [
        row for row in rows
        if str(row.get("review_decision") or "").strip().lower()
        in APPROVED_VALUES
    ]


def summarize_final(
    key: str, timeline_path: Path, forecast_path: Path, web_path: Path,
) -> dict[str, Any]:
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    return {
        "issue_key": key,
        "issue_id": (timeline.get("issue") or {}).get("issue_id"),
        "first_observation": (timeline.get("first_observation") or {}).get("date"),
        "council_count": (timeline.get("quality") or {}).get("council_count"),
        "member_count": (timeline.get("quality") or {}).get("valid_unique_member_count"),
        "exact_text_duplicates_removed": (
            timeline.get("quality") or {}
        ).get("exact_text_duplicates_removed"),
        "forecast_target": (forecast.get("forecast") or {}).get("target_month"),
        "forecast_direction": (forecast.get("forecast") or {}).get("direction"),
        "predicted_mentions": (
            forecast.get("forecast") or {}
        ).get("predicted_mention_count"),
        "timeline": str(timeline_path),
        "forecast": str(forecast_path),
        "web_payload": str(web_path),
    }


def main() -> None:
    args = parse_args()
    if args.approve_selected and not args.keys:
        raise ValueError("--approve-selected는 --keys와 함께 사용하세요")
    for name, value in (
        ("min-centroid-similarity", args.min_centroid_similarity),
        ("cluster-similarity", args.cluster_similarity),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}는 0~1 범위여야 합니다")
    if min(args.expansion_k, args.ef_search, args.seed_count) < 1:
        raise ValueError("개수 설정은 1 이상이어야 합니다")
    if args.ef_search < args.expansion_k:
        raise ValueError("ef-search는 expansion-k 이상이어야 합니다")
    if not args.dry_run and not os.environ.get("PGVECTOR_DSN") and args.refresh_clusters:
        raise RuntimeError("군집 재생성에는 PGVECTOR_DSN 환경변수가 필요합니다")

    review_queue = args.review_queue.expanduser().resolve()
    review_rows = read_csv(review_queue)
    if not review_rows or "issue_key" not in review_rows[0]:
        raise ValueError("검수 대기열 형식이 올바르지 않습니다")
    catalog = catalog_by_key(args.config)
    selected = choose_rows(review_rows, set(args.keys), args.approve_selected)
    if not selected:
        raise ValueError(
            "승인된 이슈가 없습니다. issue_review_queue.csv의 review_decision을 "
            "approve로 작성하거나 --keys ... --approve-selected를 사용하세요."
        )

    output_dir = args.output_dir.expanduser().resolve()
    web_output_dir = args.web_output_dir.expanduser().resolve()
    registry = args.registry.expanduser().resolve()
    manifest_rows = []

    for index, review_row in enumerate(selected, start=1):
        key = review_row["issue_key"]
        if key not in catalog:
            raise ValueError(f"카탈로그에 없는 issue_key: {key}")
        planned = catalog[key]
        issue_dir = output_dir / key
        issue_dir.mkdir(parents=True, exist_ok=True)
        cluster_path = find_cluster(review_row, review_queue)
        refreshed_cluster = issue_dir / "00_cluster_reviewed.json"
        timeline = issue_dir / "01_timeline.json"
        features_dir = issue_dir / "02_forecast_features"
        features = features_dir / "monthly_features.csv"
        forecast_input = features_dir / "forecast_input.json"
        forecast = issue_dir / "03_forecast.json"
        backtest = issue_dir / "03_forecast_backtest.csv"
        web_payload = web_output_dir / f"{key}.json"

        print(f"\n=== [{index}/{len(selected)}] {key} · {planned['issue_name']} ===")
        started = time.perf_counter()
        try:
            if args.refresh_clusters:
                reranked = cluster_path.parent / "02_reranked.json"
                if not reranked.is_file():
                    raise FileNotFoundError(reranked)
                run([
                    sys.executable, "-u", "src/20_expand_cluster_issue.py",
                    "--input", str(reranked), "--output", str(refreshed_cluster),
                    "--seed-count", str(args.seed_count),
                    "--expansion-k", str(args.expansion_k),
                    "--ef-search", str(args.ef_search),
                    "--min-centroid-similarity", str(args.min_centroid_similarity),
                    "--cluster-similarity", str(args.cluster_similarity),
                ], args.dry_run)
                cluster_path = refreshed_cluster
            elif not args.dry_run and not cluster_is_current(cluster_path):
                if not args.allow_legacy_cluster:
                    raise RuntimeError(
                        "기존 군집은 동일 원문 중복 제거 기록이 없습니다. "
                        "--refresh-clusters를 사용하세요."
                    )

            aliases = [
                value.strip() for value in str(planned.get("aliases") or "").split("|")
                if value.strip()
            ]
            timeline_command = [
                sys.executable, "-u", "src/21_build_issue_timeline.py",
                "--input", str(cluster_path), "--output", str(timeline),
                "--registry", str(registry), "--issue-name", planned["issue_name"],
                "--approve",
            ]
            if aliases:
                timeline_command.extend(["--aliases", *aliases])
            commands = [
                timeline_command,
                [
                    sys.executable, "-u", "src/22_build_forecast_features.py",
                    "--input", str(timeline), "--output-dir", str(features_dir),
                ],
                [
                    sys.executable, "-u", "src/23_forecast_issue.py",
                    "--features", str(features),
                    "--forecast-input", str(forecast_input),
                    "--output", str(forecast),
                    "--backtest-output", str(backtest),
                ],
                [
                    sys.executable, "-u", "src/24_build_web_payload.py",
                    "--timeline", str(timeline), "--forecast", str(forecast),
                    "--features", str(features), "--output", str(web_payload),
                ],
            ]
            if args.skip_existing and web_payload.is_file():
                print(f"기존 웹 결과 재사용: {web_payload}")
            else:
                for command in commands:
                    run(command, args.dry_run)
            if args.dry_run:
                result = {"issue_key": key, "status": "dry_run"}
            else:
                result = summarize_final(key, timeline, forecast, web_payload)
                result["status"] = "complete"
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            manifest_rows.append(result)
        except Exception as error:
            manifest_rows.append({
                "issue_key": key, "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            if not args.continue_on_error:
                atomic_json(output_dir / "finalize_manifest.json", {"issues": manifest_rows})
                raise
            print(f"실패 후 계속 진행: {type(error).__name__}: {error}")

    manifest = {
        "status": "complete" if all(
            row["status"] != "failed" for row in manifest_rows
        ) else "partial",
        "review_queue": str(review_queue),
        "approved_by": "explicit_cli" if args.approve_selected else "review_queue",
        "cluster_parameters": {
            "refreshed": args.refresh_clusters,
            "min_centroid_similarity": args.min_centroid_similarity,
            "cluster_similarity": args.cluster_similarity,
        },
        "issues": manifest_rows,
    }
    manifest_path = output_dir / "finalize_manifest.json"
    atomic_json(manifest_path, manifest)
    print("\n=== 승인 이슈 최종화 완료 ===")
    print(
        f"요청: {len(selected)} | 완료: "
        f"{sum(row['status'] == 'complete' for row in manifest_rows)} | "
        f"상태: {manifest['status']}"
    )
    print(f"manifest: {manifest_path}")
    print(f"웹 JSON: {web_output_dir}")


if __name__ == "__main__":
    main()
