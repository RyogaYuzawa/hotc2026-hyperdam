#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (($# == 0)) || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: run_sample.sh VIDEO_DIR [run_inference options]

VIDEO_DIR is the validation directory containing HSI modality directories, or
its parent directory containing validation/ and sample_submisson.csv.

Example:
  bash scripts/run_sample.sh \
    data/HOTC2026 \
    --sample data/HOTC2026/sample_submisson.csv

The final CSV is written to result/submission.csv.
EOF
  (($# == 0)) && exit 2
  exit 0
fi

video_dir="$1"
shift

exec bash "${project_root}/scripts/run_inference.sh" \
  --data "${video_dir}" \
  --output "${project_root}/result" \
  --devices "${HOTC_DEVICES:-${HOTC_DEVICE:-cuda:0}}" \
  --overwrite \
  "$@"
