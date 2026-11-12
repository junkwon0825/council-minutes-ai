#!/usr/bin/env python3
"""정책 이슈 카탈로그를 검색·재순위·군집화하고 검수 대기열을 만든다.

이 단계에서는 issue_id를 절대 fixed로 승인하지 않는다. 자동 결과의 품질 지표와
대표 근거를 사람이 확인한 뒤 21번 타임라인 단계에서만 --approve를 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "config" / "issue_seed_catalog_v1.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "issues" / "candidates_v1",
    )
    parser.add_argument("--keys", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--expansion-k", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=500)
    parser.add_argument("--min-centroid-similarity", type=float, default=0.45)
    parser.add_argument("--cluster-similarity", type=float, default=0.72)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_catalog(path: Path, keys: set[str], limit: int | None) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    required = {"issue_key", "issue_name", "query", "aliases", "category", "enabled"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"이슈 카탈로그 열이 올바르지 않습니다: {path}")
    selected = [
        row for row in rows
        if str(row["enabled"]).strip().lower() in {"1", "true", "yes"}
        and (not keys or row["issue_key"] in keys)
    ]
    if len({row["issue_key"] for row in selected}) != len(selected):
        raise ValueError("issue_key가 중복되었습니다")
    if keys:
        missing = sorted(keys - {row["issue_key"] for row in selected})
        if missing:
            raise ValueError(f"카탈로그에 없거나 비활성화된 key: {missing}")
    return selected[:limit] if limit else selected


def run_command(command: list[str], dry_run: bool) -> float:
    print("\n$ " + " ".join(command))
    if dry_run:
        return 0.0
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return time.perf_counter() - started


def cluster_metrics(path: Path, planned: dict[str, str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issue = payload.get("issue") or {}
    selected = (payload.get("clustering") or {}).get("selected_issue") or {}
    members = payload.get("members") or []
    councils = Counter(
        str((row.get("metadata") or {}).get("council") or "")
        for row in members
    )
    councils.pop("", None)
    top_council, top_count = councils.most_common(1)[0] if councils else ("", 0)
    member_count = len(members)
    top_share = top_count / member_count if member_count else 0.0
    similarities = [float(row.get("centroid_similarity") or 0.0) for row in members]
    flags = []
    seed_coverage = float(selected.get("seed_coverage") or 0.0)
    council_count = int(selected.get("council_count") or 0)
    if seed_coverage < 0.8:
        flags.append("low_seed_coverage")
    if member_count < 15:
        flags.append("small_cluster")
    if council_count < 2:
        flags.append("single_council")
    if top_share > 0.8:
        flags.append("council_concentration")
    if member_count > 350:
        flags.append("possibly_broad_cluster")
    return {
        "issue_key": planned["issue_key"],
        "planned_issue_name": planned["issue_name"],
        "query": planned["query"],
        "category": planned["category"],
        "aliases": planned["aliases"],
        "provisional_issue_id": issue.get("issue_id"),
        "member_count": member_count,
        "seed_coverage": round(seed_coverage, 6),
        "council_count": council_count,
        "first_date": selected.get("first_date"),
        "last_date": selected.get("last_date"),
        "top_council": top_council,
        "top_council_share": round(top_share, 6),
        "mean_centroid_similarity": round(
            sum(similarities) / len(similarities), 6
        ) if similarities else 0.0,
        "quality_flags": "|".join(flags),
        "review_status": "pending_human_review",
        "review_decision": "",
        "review_note": "",
        "cluster_json": str(path),
    }


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다")
    if not args.dry_run and not os.environ.get("PGVECTOR_DSN"):
        raise RuntimeError("PGVECTOR_DSN 환경변수가 필요합니다")
    catalog = read_catalog(
        args.config, set(args.keys), args.limit,
    )
    if not catalog:
        raise ValueError("실행할 이슈가 없습니다")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    runs = []

    for index, row in enumerate(catalog, start=1):
        key = row["issue_key"]
        current_dir = output_dir / key
        current_dir.mkdir(parents=True, exist_ok=True)
        retrieval = current_dir / "01_retrieval.json"
        reranked = current_dir / "02_reranked.json"
        cluster = current_dir / "03_cluster.json"
        print(f"\n=== [{index}/{len(catalog)}] {key} · {row['issue_name']} ===")
        if args.skip_existing and cluster.is_file():
            review_rows.append(cluster_metrics(cluster, row))
            runs.append({"issue_key": key, "status": "reused_existing"})
            continue
        common_download = ["--allow-download"] if args.allow_download else []
        commands = [
            [
                sys.executable, "-u", "src/18_pgvector_retrieval.py",
                "--query", row["query"], "--top-k", str(args.rerank_top_k),
                "--candidate-k", str(args.candidate_k), "--ef-search",
                str(args.ef_search), "--device", args.device,
                "--output", str(retrieval), *common_download,
            ],
            [
                sys.executable, "-u", "src/19_rerank_candidates.py",
                "--input", str(retrieval), "--output", str(reranked),
                "--top-k", str(args.rerank_top_k), "--device", args.device,
                *common_download,
            ],
            [
                sys.executable, "-u", "src/20_expand_cluster_issue.py",
                "--input", str(reranked), "--output", str(cluster),
                "--seed-count", str(args.rerank_top_k),
                "--expansion-k", str(args.expansion_k), "--ef-search",
                str(args.ef_search), "--min-centroid-similarity",
                str(args.min_centroid_similarity), "--cluster-similarity",
                str(args.cluster_similarity),
            ],
        ]
        issue_started = time.perf_counter()
        try:
            stage_seconds = [run_command(command, args.dry_run) for command in commands]
            if not args.dry_run:
                review_rows.append(cluster_metrics(cluster, row))
            runs.append({
                "issue_key": key, "status": "dry_run" if args.dry_run else "complete",
                "elapsed_seconds": round(time.perf_counter() - issue_started, 3),
                "stage_seconds": [round(value, 3) for value in stage_seconds],
            })
        except Exception as error:
            runs.append({
                "issue_key": key, "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
            if not args.continue_on_error:
                atomic_json(output_dir / "candidate_run_manifest.json", {"runs": runs})
                raise
            print(f"실패 후 계속 진행: {type(error).__name__}: {error}")

    review_path = output_dir / "issue_review_queue.csv"
    if review_rows:
        write_review_csv(review_path, review_rows)
    manifest = {
        "status": "complete" if all(row["status"] != "failed" for row in runs) else "partial",
        "catalog": str(args.config.expanduser().resolve()),
        "issue_count_requested": len(catalog),
        "issue_count_completed": sum(row["status"] in {"complete", "reused_existing"} for row in runs),
        "automatic_approval": False,
        "review_queue": str(review_path),
        "parameters": {
            "candidate_k": args.candidate_k,
            "rerank_top_k": args.rerank_top_k,
            "expansion_k": args.expansion_k,
            "ef_search": args.ef_search,
            "min_centroid_similarity": args.min_centroid_similarity,
            "cluster_similarity": args.cluster_similarity,
        },
        "runs": runs,
    }
    atomic_json(output_dir / "candidate_run_manifest.json", manifest)
    print("\n=== 다중 이슈 후보 생성 완료 ===")
    print(
        f"요청: {len(catalog)} | 완료: {manifest['issue_count_completed']} | "
        f"상태: {manifest['status']}"
    )
    if review_rows:
        print(f"검수 대기열: {review_path}")
    print(f"실행 기록: {output_dir / 'candidate_run_manifest.json'}")


if __name__ == "__main__":
    main()
