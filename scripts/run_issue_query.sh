#!/usr/bin/env bash
set -euo pipefail

query="${1:-저출생 출산율 인구감소}"

python src/10_issue_query.py \
  --query "$query" \
  --timelines outputs/issue_timeline_final/issue_timelines.json \
  --forecasts outputs/issue_forecast_final/issue_forecasts.json \
  --centers data/index/arctic_ko_semantic_test_v2/clustering_k160/cluster_centers.f32.npy \
  --output-dir outputs/issue_search_final \
  --local-files-only

