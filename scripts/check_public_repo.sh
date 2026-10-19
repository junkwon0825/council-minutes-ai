#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Git 저장소가 아닙니다. git init 후 다시 실행하세요."
  exit 1
fi

tracked=$(git ls-files)
forbidden=$(printf '%s\n' "$tracked" | grep -E \
  '^(data/(raw|processed|index|external)|models|outputs|logs|runs)/|\.(parquet|npy|npz|faiss|index|joblib|pkl|gguf|safetensors|pt|pth)$' \
  || true)

if [[ -n "$forbidden" ]]; then
  echo "공개 금지 파일이 Git 추적 대상에 포함되어 있습니다:"
  printf '%s\n' "$forbidden"
  exit 2
fi

large_files=$(git ls-files -z | xargs -0 -r du -b 2>/dev/null | awk '$1 > 20000000 {print $0}')
if [[ -n "$large_files" ]]; then
  echo "20MB를 넘는 추적 파일을 확인하세요:"
  printf '%s\n' "$large_files"
  exit 3
fi

echo "공개 저장소 검사 통과: 데이터·모델·대용량 파생 파일이 추적되지 않습니다."

