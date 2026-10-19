#!/usr/bin/env python3
"""자연어 검색어를 검수된 이슈와 연결하고 최초 등장·전파·시계열을 출력한다."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("numpy가 필요합니다. 실행 전: pip install numpy") from error

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit("sentence-transformers가 필요합니다.") from error


DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9"
SCRIPT_VERSION = "2026-10-19-issue-query-v2-with-forecast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정책 이슈 타임라인 검색")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--timelines", default="outputs/issue_timeline_final/issue_timelines.json"
    )
    parser.add_argument(
        "--forecasts", default="outputs/issue_forecast_final/issue_forecasts.json",
        help="11번 예측 JSON. 파일이 없으면 예측 없이 검색만 수행",
    )
    parser.add_argument(
        "--centers",
        default=(
            "data/index/arctic_ko_semantic_test_v2/clustering_k160/"
            "cluster_centers.f32.npy"
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/issue_search_final")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise SystemExit("CUDA를 사용할 수 없습니다.")
    return requested


def slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value).strip("_")
    return cleaned[:60] or "query"


def lexical_bonus(query: str, timeline: dict) -> float:
    query_key = re.sub(r"\s+", "", query.lower())
    haystack = " ".join(
        str(timeline.get(key, ""))
        for key in ("issue_name", "issue_summary", "keywords", "policy_domain")
    ).lower()
    haystack_key = re.sub(r"\s+", "", haystack)
    return 0.12 if query_key and query_key in haystack_key else 0.0


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k는 1 이상이어야 합니다.")
    timeline_path = Path(args.timelines).expanduser().resolve()
    centers_path = Path(args.centers).expanduser().resolve()
    for path in (timeline_path, centers_path):
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")

    timelines = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not timelines:
        raise SystemExit("검색 가능한 이슈 타임라인이 없습니다.")
    forecast_path = Path(args.forecasts).expanduser().resolve() if args.forecasts else None
    forecast_by_issue = {}
    if forecast_path and forecast_path.exists():
        forecasts = json.loads(forecast_path.read_text(encoding="utf-8"))
        forecast_by_issue = {
            str(row.get("issue_id")): row for row in forecasts if row.get("issue_id")
        }
    elif forecast_path:
        print(f"경고: 예측 파일이 없어 검색 결과에 예측을 포함하지 않습니다: {forecast_path}", file=sys.stderr)
    centers = np.asarray(np.load(centers_path), dtype=np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(norms, 1e-12)

    device = resolve_device(args.device)
    model = SentenceTransformer(
        args.model_name,
        revision=args.revision,
        cache_folder=str(Path(args.model_cache).expanduser().resolve()),
        device=device,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model.max_seq_length = 2048
    kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if "query" in (getattr(model, "prompts", {}) or {}):
        kwargs["prompt_name"] = "query"
    else:
        print("경고: query 프롬프트 없이 검색합니다.", file=sys.stderr)
    query_vector = np.asarray(model.encode([args.query], **kwargs)[0], dtype=np.float32)
    if query_vector.shape[0] != centers.shape[1]:
        raise SystemExit(
            f"임베딩 차원 불일치: query={query_vector.shape[0]}, centers={centers.shape[1]}"
        )

    scored = []
    for timeline in timelines:
        cluster_id = int(timeline["cluster_id"])
        if not 0 <= cluster_id < len(centers):
            continue
        semantic_score = float(np.dot(query_vector, centers[cluster_id]))
        bonus = lexical_bonus(args.query, timeline)
        scored.append((semantic_score + bonus, semantic_score, bonus, timeline))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)

    results = []
    print(f"\n검색어: {args.query}")
    for rank, (score, semantic_score, bonus, timeline) in enumerate(scored[: args.top_k], start=1):
        result = {
            "rank": rank,
            "search_score": round(score, 6),
            "semantic_score": round(semantic_score, 6),
            "lexical_bonus": round(bonus, 6),
            **timeline,
            "forecast": forecast_by_issue.get(str(timeline.get("issue_id"))),
        }
        results.append(result)
        route = " → ".join(
            f"{event['council']}({event['first_date']})"
            for event in timeline.get("propagation", [])[:8]
        )
        print(f"\n[{rank}] {timeline['issue_name']} ({timeline['issue_id']})")
        print(f"  검색 점수: {score:.4f}")
        print(
            f"  최초 관측: {timeline['first_observed_date']} / "
            f"{', '.join(timeline.get('origin_councils', []))}"
        )
        print(f"  관측 의회: {timeline['observed_council_count']}개")
        print(f"  전파 순서: {route or '후속 의회 없음'}")
        print(f"  요약: {timeline.get('issue_summary', '')}")
        forecast = result.get("forecast")
        if forecast:
            print(
                f"  다음 달 예측: {forecast.get('predicted_direction', '')} / "
                f"언급 {forecast.get('predicted_mention_count', '')}건 / "
                f"신규 의회 확산 {forecast.get('spread_level', '')} "
                f"({float(forecast.get('spread_probability', 0.0)):.3f})"
            )
            print(f"  예측 신뢰도: {forecast.get('forecast_confidence', '')}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug(args.query)}.json"
    payload = {
        "script_version": SCRIPT_VERSION,
        "query": args.query,
        "model": args.model_name,
        "timeline_source": str(timeline_path),
        "forecast_source": str(forecast_path) if forecast_path and forecast_path.exists() else None,
        "result_count": len(results),
        "results": results,
        "interpretation_note": (
            "전파 순서는 제공 데이터 내 최초 관측 순서이며 실제 정책 인과관계를 의미하지 않습니다."
        ),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON 결과: {output_path}")


if __name__ == "__main__":
    main()
