#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${PGVECTOR_DSN:-}" ]]; then
  echo "PGVECTOR_DSN이 필요합니다. .env.example을 참고하세요." >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1; then
  docker compose up -d pgvector >/dev/null
fi

RETRIEVAL_MODE="${COUNCIL_RETRIEVAL_MODE:-hybrid}"
BM25_INDEX="${COUNCIL_BM25_INDEX:-data/bm25/arctic_raw_v1/bm25.sqlite3}"
if [[ "$RETRIEVAL_MODE" == "hybrid" && ! -f "$BM25_INDEX" ]]; then
  echo "BM25 인덱스가 없습니다: $BM25_INDEX" >&2
  echo "python src/35_build_bm25_index.py 로 먼저 생성하세요." >&2
  exit 1
fi

exec python -m streamlit run demo/final_app.py   --server.port "${PORT:-8501}"   --server.address "${HOST:-127.0.0.1}"
