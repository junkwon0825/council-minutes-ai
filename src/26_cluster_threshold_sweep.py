#!/usr/bin/env python3
"""검수 대기 이슈의 군집 임계값을 비교하고 대표·경계 근거 표본을 만든다."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "issues" / "cluster_threshold_sweep_v1",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.68, 0.72, 0.76])
    parser.add_argument("--expansion-k", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=500)
    parser.add_argument("--min-centroid-similarity", type=float, default=0.45)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--sample-per-group", type=int, default=5)
    parser.add_argument("--max-text-chars", type=int, default=700)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    if not rows or "issue_key" not in rows[0] or "cluster_json" not in rows[0]:
        raise ValueError("검수 대기열 CSV 형식이 올바르지 않습니다")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"저장할 행이 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def metric_row(
    payload: dict[str, Any], source: dict[str, str], threshold: float,
    output_path: Path,
) -> dict[str, Any]:
    members = payload.get("members") or []
    selected = (payload.get("clustering") or {}).get("selected_issue") or {}
    councils = Counter(
        str((row.get("metadata") or {}).get("council") or "")
        for row in members
    )
    councils.pop("", None)
    top_council, top_count = councils.most_common(1)[0] if councils else ("", 0)
    similarities = [float(row.get("centroid_similarity") or 0.0) for row in members]
    return {
        "issue_key": source["issue_key"],
        "planned_issue_name": source.get("planned_issue_name", ""),
        "threshold": threshold,
        "provisional_issue_id": (payload.get("issue") or {}).get("issue_id"),
        "member_count": len(members),
        "seed_coverage": float(selected.get("seed_coverage") or 0.0),
        "selected_subcluster_count": int(selected.get("selected_subcluster_count") or 0),
        "council_count": len(councils),
        "top_council": top_council,
        "top_council_share": round(top_count / len(members), 6) if members else 0.0,
        "mean_centroid_similarity": round(
            sum(similarities) / len(similarities), 6
        ) if similarities else 0.0,
        "minimum_centroid_similarity": round(min(similarities), 6) if similarities else 0.0,
        "first_date": selected.get("first_date"),
        "last_date": selected.get("last_date"),
        "cluster_json": str(output_path),
    }


def sample_rows(
    payload: dict[str, Any], issue_key: str, issue_name: str,
    threshold: float, count: int, max_chars: int,
) -> list[dict[str, Any]]:
    members = list(payload.get("members") or [])
    members.sort(
        key=lambda row: float(row.get("centroid_similarity") or 0.0),
        reverse=True,
    )
    groups = {
        "representative_high": members[:count],
        "boundary_low": list(reversed(members[-count:])),
    }
    rows = []
    seen = set()
    for group_name, selected in groups.items():
        for member in selected:
            chunk_id = str(member.get("chunk_id") or "")
            key = (group_name, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            metadata = member.get("metadata") or {}
            text = " ".join(str(member.get("text") or "").split())
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "…"
            rows.append({
                "issue_key": issue_key,
                "planned_issue_name": issue_name,
                "threshold": threshold,
                "sample_group": group_name,
                "chunk_id": chunk_id,
                "is_seed": bool(member.get("is_seed")),
                "similarity": float(member.get("centroid_similarity") or 0.0),
                "council": metadata.get("council", ""),
                "meeting_date": str(metadata.get("meeting_date") or "")[:10],
                "meeting_name": metadata.get("meeting_name", ""),
                "agenda_title": metadata.get("agenda_title", ""),
                "text": text,
                "human_relevance_0_1_2": "",
                "human_note": "",
            })
    return rows


def main() -> None:
    args = parse_args()
    if not os.environ.get("PGVECTOR_DSN"):
        raise RuntimeError("PGVECTOR_DSN 환경변수가 필요합니다")
    if not args.thresholds or any(not 0.0 < value < 1.0 for value in args.thresholds):
        raise ValueError("thresholds는 0과 1 사이여야 합니다")
    if args.sample_per_group < 1:
        raise ValueError("sample-per-group은 1 이상이어야 합니다")
    source_rows = read_csv(args.review_queue)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    evidence_rows = []

    for source in source_rows:
        key = source["issue_key"]
        original_cluster = Path(source["cluster_json"]).expanduser().resolve()
        reranked = original_cluster.with_name("02_reranked.json")
        if not reranked.is_file():
            raise FileNotFoundError(f"재순위 결과가 없습니다: {reranked}")
        issue_dir = output_dir / key
        issue_dir.mkdir(parents=True, exist_ok=True)
        for threshold in sorted(set(args.thresholds)):
            threshold_name = f"{threshold:.2f}".replace(".", "p")
            output_path = issue_dir / f"cluster_t{threshold_name}.json"
            print(f"\n=== {key} | cluster_similarity={threshold:.2f} ===")
            if not (args.skip_existing and output_path.is_file()):
                command = [
                    sys.executable, "-u", "src/20_expand_cluster_issue.py",
                    "--input", str(reranked), "--output", str(output_path),
                    "--seed-count", str(args.seed_count),
                    "--expansion-k", str(args.expansion_k),
                    "--ef-search", str(args.ef_search),
                    "--min-centroid-similarity", str(args.min_centroid_similarity),
                    "--cluster-similarity", str(threshold),
                ]
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            metric_rows.append(metric_row(payload, source, threshold, output_path))
            evidence_rows.extend(sample_rows(
                payload, key, source.get("planned_issue_name", ""), threshold,
                args.sample_per_group, args.max_text_chars,
            ))

    metrics_path = output_dir / "threshold_comparison.csv"
    evidence_path = output_dir / "threshold_evidence_review.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(evidence_path, evidence_rows)
    print("\n=== 군집 임계값 비교 완료 ===")
    print(f"이슈: {len(source_rows)} | 임계값: {len(set(args.thresholds))}")
    print(f"비교표: {metrics_path}")
    print(f"근거 검수표: {evidence_path}")
    print("검수표의 human_relevance_0_1_2 열에 0·1·2를 입력하세요.")


if __name__ == "__main__":
    main()
