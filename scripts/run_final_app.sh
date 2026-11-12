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

RETRIEVAL_MODE="${COUNCIL_RETRIEVAL_MODE:-hybrid}"
BM25_INDEX="${COUNCIL_BM25_INDEX:-data/bm25/arctic_raw_v1/bm25.sqlite3}"
if [[ "$RETRIEVAL_MODE" == "hybrid" && ! -f "$BM25_INDEX" ]]; then
  echo "BM25 인덱스가 없습니다: $BM25_INDEX" >&2
  echo "먼저 실행하세요: bash scripts/build_bm25_hybrid.sh" >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1; then
  docker compose up -d pgvector >/dev/null
fi

if [[ -n "${LAW_OC:-}" || -n "${ASSEMBLY_API_KEY:-}" || -n "${KOSIS_API_KEY:-}" ]]; then
  python -u src/32a_external_followup_flow.py     --mode "${COUNCIL_FOLLOWUP_MODE:-backtest}"     || echo "안내: 외부 후속 흐름 계산 실패 — 기존 결과로 계속합니다." >&2
fi

python -u src/34_build_final_web_bundle.py

OLLAMA_ENDPOINT="${OLLAMA_URL:-http://127.0.0.1:11434}"
if ! curl -fsS "$OLLAMA_ENDPOINT/api/tags" >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    mkdir -p logs
    nohup ollama serve >logs/ollama.log 2>&1 &
    for _ in $(seq 1 20); do
      curl -fsS "$OLLAMA_ENDPOINT/api/tags" >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
fi
if ! curl -fsS "$OLLAMA_ENDPOINT/api/tags" >/dev/null 2>&1; then
  echo "안내: Ollama를 자동 시작하지 못했습니다. Agent는 근거 구조화 요약으로 동작합니다." >&2
fi

exec python -m streamlit run demo/final_app.py   --server.port "${PORT:-8501}"   --server.address "${HOST:-127.0.0.1}"
