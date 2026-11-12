#!/usr/bin/env python3
"""승인 이슈별로 타 지방의회의 후속 논의 흐름과 향후 후속 후보를 외부 공공데이터로 보강한다.

회의록에서 관측된 의회별 최초 논의(24번 웹 JSON)와 다중 이슈 예측(32번)을 읽고,
법제처 자치법규·열린국회정보 의안을 이슈 정책어로, KOSIS 시도별 통계를 지표 목록으로 조회해 다음을 만든다.

- followup_flow: 의회별 '회의록 논의 → 광역 조례화', '기초 조례 선행' 등 후속 단계
- outlook: 후속 조례화·논의 진입 후보 순위 (회의록 + 조례 + 지역 정책 수요, 국회 의안은 참고 근거)
- post_cutoff_check: backtest 모드에서 기준일 이후 실제 '제정' 조례로 후보 순위를 사후 점검

--mode backtest(대회 기본): 외부 데이터는 --as-of(기본 회의록 기준일)까지만, 통계는 그 시점에 공표된 값만
--mode live             : 별도 사후 탐색용. 조회일(--as-of, 기본 오늘)까지 전부 사용, 전망은 조회일 이후

계산 규칙은 external_followup.py에 있으며 Agent 도구(agent/followup_tools.py)와 공유한다.
외부지표는 이슈 탐지·군집화·예측 모델 입력에 쓰지 않는다(reference_only).
외부로 보내는 값은 짧은 정책어뿐이며 회의록 원문·발언자는 전송하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import date
from pathlib import Path
from typing import Any

import external_followup as FU
import external_policy_sources as EXT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-dir", type=Path, default=PROJECT_ROOT / "outputs" / "web")
    parser.add_argument(
        "--multi-forecast", type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "forecast" / "multi_issue_v1" /
            "model" / "multi_issue_forecast.json"
        ),
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "config" / "issue_seed_catalog_v1.csv",
        help="external_keywords 열(| 구분)이 있으면 우선 사용, 없으면 aliases·이슈명",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "external" / "followup_v1",
    )
    parser.add_argument("--mode", choices=FU.MODES, default=os.environ.get("COUNCIL_FOLLOWUP_MODE", "backtest"))
    parser.add_argument("--as-of", default=None, help="live 기본=오늘, backtest 기본=회의록 기준일")
    parser.add_argument(
        "--minutes-cutoff", "--analysis-cutoff", dest="minutes_cutoff",
        default=os.environ.get("COUNCIL_MINUTES_CUTOFF", FU.DEFAULT_MINUTES_CUTOFF),
        help="회의록 근거 기준일 (보유 회의록 범위)",
    )
    parser.add_argument("--max-keywords", type=int, default=3)
    parser.add_argument("--recent-months", type=int, default=6)
    parser.add_argument("--momentum-months", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--keys", nargs="*", default=[], help="특정 issue_key만 처리")
    parser.add_argument("--dry-run", action="store_true", help="외부 호출 없이 검색어만 확인")
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


def read_catalog(path: Path) -> dict[str, dict[str, str]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {}
    with resolved.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["issue_key"]: row for row in csv.DictReader(file) if row.get("issue_key")}


def latest_web_payloads(web_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """34번과 같은 규칙: issue_id별 최신 council_issue_web_v1 파일"""
    payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(web_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("schema_version") != "council_issue_web_v1":
            continue
        issue_id = str((payload.get("issue") or {}).get("issue_id") or "")
        if not issue_id:
            continue
        previous = payloads.get(issue_id)
        if previous is None or path.stat().st_mtime_ns > previous[0].stat().st_mtime_ns:
            payloads[issue_id] = (path, payload)
    return payloads


def main() -> None:
    args = parse_args()
    minutes_cutoff = FU.parse_day(args.minutes_cutoff)
    if minutes_cutoff is None:
        raise ValueError(f"minutes-cutoff 날짜 형식 오류: {args.minutes_cutoff!r}")
    default_as_of = date.today() if args.mode == "live" else minutes_cutoff
    as_of = FU.parse_day(args.as_of) if args.as_of else default_as_of
    if as_of is None:
        raise ValueError(f"as-of 날짜 형식 오류: {args.as_of!r}")
    if as_of < minutes_cutoff:
        raise ValueError("as-of는 회의록 기준일 이후여야 합니다")
    if min(args.max_keywords, args.recent_months, args.momentum_months, args.top_k) < 1:
        raise ValueError("개수·기간 설정은 1 이상이어야 합니다")
    web_dir = args.web_dir.expanduser().resolve()
    payloads = latest_web_payloads(web_dir)
    if not payloads:
        raise FileNotFoundError(f"council_issue_web_v1 웹 JSON이 없습니다: {web_dir}")

    forecast_path = args.multi_forecast.expanduser().resolve()
    forecasts: dict[str, dict[str, Any]] = {}
    if forecast_path.is_file():
        report = load_json(forecast_path)
        forecasts = {str(row.get("issue_id")): row for row in report.get("forecasts") or []}
    else:
        print(f"주의 | 다중 이슈 예측이 없어 이슈 확률 신호를 '해당 없음'으로 둡니다: {forecast_path}")
    catalog = read_catalog(args.config)
    available = {source: EXT.key_available(source) for source in EXT.DATA_SOURCES}
    if not args.dry_run and not any(available.values()):
        print("주의 | LAW_OC·ASSEMBLY_API_KEY·KOSIS_API_KEY가 모두 없어 외부 조회 없이 회의록 흐름만 기록합니다")
    print(f"모드: {args.mode} | 외부 데이터 기준: {as_of} | 회의록 기준일: {minutes_cutoff}")

    output_dir = args.output_dir.expanduser().resolve()
    summary_rows = []
    for issue_id, (source_path, payload) in sorted(payloads.items()):
        issue_key = source_path.stem
        if args.keys and issue_key not in args.keys:
            continue
        issue = payload.get("issue") or {}
        keywords, rejected_keywords = FU.external_keywords(
            issue, catalog.get(issue_key), args.max_keywords,
        )
        indicator_ids = FU.match_indicator_ids(
            keywords + [str(issue.get("issue_name") or "")] + list(issue.get("aliases") or [])
        )
        print(
            f"\n=== {issue_key} · {issue.get('issue_name')} | 검색어: {keywords} | "
            f"KOSIS 지표: {indicator_ids or '-'} ==="
        )
        if args.dry_run:
            summary_rows.append({
                "issue_key": issue_key, "status": "dry_run",
                "keywords": keywords, "indicators": indicator_ids,
            })
            continue

        result = FU.build_followup(
            {"issue_id": issue_id, "issue_key": issue_key, "issue_name": issue.get("issue_name")},
            payload.get("propagation") or [],
            keywords, as_of,
            mode=args.mode,
            minutes_cutoff=minutes_cutoff,
            issue_forecast=forecasts.get(issue_id),
            relevance_terms=keywords + list(issue.get("aliases") or []),
            indicator_ids=indicator_ids,
            recent_months=args.recent_months,
            momentum_months=args.momentum_months,
            top_k=args.top_k,
            rejected_keywords=rejected_keywords,
        )
        result["source_files"] = {"web_payload": str(source_path), "multi_forecast": str(forecast_path)}
        output_path = output_dir / f"{issue_key}.json"
        atomic_json(output_path, result)
        view = result["outlook"]
        top = ", ".join(
            f"{row['council']}({row['followup_score']:.2f})" for row in view["top_candidates"][:3]
        )
        counts = result["external_counts"]
        print(
            f"조례: {counts['ordinances_visible']} (무관 제외 {counts['ordinances_unrelated_excluded']}) | "
            f"광역 조례화: {len(view['already_institutionalized'])} "
            f"(회의록 이후 {len(view['followups_after_minutes_cutoff'])}) | "
            f"의안: {counts['national_bills_visible']} | "
            f"KOSIS: {len(result['regional_need']['indicators'])}개 | 후보: {top or '-'}"
        )
        if result["errors"]:
            print(f"외부 조회 오류 {len(result['errors'])}건 (결과는 partial로 저장)")
        summary_rows.append({
            "issue_key": issue_key,
            "issue_id": issue_id,
            "status": result["status"],
            "keywords": keywords,
            "already_institutionalized": view["already_institutionalized"],
            "top_candidates": [row["council"] for row in view["top_candidates"]],
            "post_cutoff_check": result["post_cutoff_check"],
            "payload": str(output_path),
        })

    if not summary_rows:
        raise ValueError(f"처리한 이슈가 없습니다 (--keys 확인): {args.keys}")
    checked = [row["post_cutoff_check"] for row in summary_rows
               if (row.get("post_cutoff_check") or {}).get("status") == "complete"]
    summary = {
        "schema_version": "council_external_followup_summary_v2",
        "mode": args.mode,
        "as_of": as_of.isoformat(),
        "minutes_cutoff": minutes_cutoff.isoformat(),
        "status": "dry_run" if args.dry_run else (
            "complete" if all(row["status"] == "complete" for row in summary_rows) else "partial"
        ),
        "sources_available": available,
        "issue_count": len(summary_rows),
        "mean_post_cutoff_precision_at_k": (
            round(statistics.mean(row["precision_at_k"] for row in checked), 4) if checked else None
        ),
        "issues": summary_rows,
    }
    if not args.dry_run:
        atomic_json(output_dir / "followup_summary.json", summary)
    print("\n=== 외부 공공데이터 후속 흐름·전망 생성 완료 ===")
    print(f"이슈: {len(summary_rows)} | 상태: {summary['status']} | 사용 가능 출처: {available}")
    print("용도: 화면 참고 비교 전용 (이슈 탐지·예측 입력 제외)")
    if not args.dry_run:
        print(f"summary: {output_dir / 'followup_summary.json'}")


if __name__ == "__main__":
    main()
