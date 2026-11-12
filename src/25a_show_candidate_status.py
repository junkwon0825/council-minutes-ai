#!/usr/bin/env python3
"""다중 이슈 후보 생성 manifest와 검수 대기열 상태를 간단히 출력한다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "issues" / "candidates_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    manifest_path = candidate_dir / "candidate_run_manifest.json"
    queue_path = candidate_dir / "issue_review_queue.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = list(manifest.get("runs") or [])
    completed = [
        row for row in runs if row.get("status") in {"complete", "reused_existing"}
    ]
    failed = [row for row in runs if row.get("status") == "failed"]
    queue_rows = []
    if queue_path.is_file():
        with queue_path.open("r", encoding="utf-8-sig", newline="") as file:
            queue_rows = list(csv.DictReader(file))

    print("\n=== 대표 이슈 후보 생성 상태 ===")
    print(f"manifest: {manifest_path}")
    print(f"전체 요청: {len(runs)} | 완료: {len(completed)} | 실패: {len(failed)}")
    print(f"검수 대기열: {len(queue_rows)}행 | {queue_path}")
    if completed:
        print("\n완료 이슈:")
        for row in completed:
            print(f"  PASS | {row.get('issue_key')} | {row.get('status')}")
    if failed:
        print("\n실패 이슈와 원인:")
        for row in failed:
            print(f"  FAIL | {row.get('issue_key')}")
            print(f"         {row.get('error') or '오류 기록 없음'}")
        print("\n실패 원인을 수정한 뒤 아래 명령을 다시 실행하세요.")
        print("  bash scripts/run_representative_issues.sh candidates")
        print("성공한 이슈는 --skip-existing으로 재사용됩니다.")
    else:
        print("\n모든 대표 이슈 후보가 생성되었습니다.")


if __name__ == "__main__":
    main()
