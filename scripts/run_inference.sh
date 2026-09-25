#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${project_root}/.venv/bin/python"
data_root=""
sample=""
output="${project_root}/result"
devices="${HOTC_DEVICES:-cuda:0}"
overwrite=0
temporary_sample=""
work_dir=""

cleanup() {
  local status=$?
  if [[ -n "${temporary_sample}" ]]; then
    rm -f -- "${temporary_sample}"
  fi
  if [[ -n "${work_dir}" ]]; then
    if ((status == 0)); then
      rm -rf -- "${work_dir}"
    else
      printf 'Failed-run diagnostics preserved at: %s\n' "${work_dir}" >&2
    fi
  fi
  return "${status}"
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage: run_inference.sh --data PATH [options]
       run_inference.sh PATH [options]

Run HSI v3 + amodal-v10 + empty-mask RTS on every sequence listed in the sample
CSV. PATH may be a validation directory or a directory containing validation/.

Options:
  --data PATH          Dataset or validation directory.
  --sample PATH        Submission schema/ID CSV. Auto-detected or generated.
  --output PATH        Output directory (default: result/).
  --devices LIST       Comma-separated CUDA devices (default: cuda:0).
  --overwrite          Delete and recreate a nonempty output directory.
  -h, --help           Show this help.

Expected dataset layout:

  PATH/
    sample_submisson.csv       # sample_submission.csv is also accepted
    validation/
      HSI-NIR-FalseColor/...   # as applicable
      HSI-NIR/...
      HSI-RedNIR-FalseColor/...
      HSI-RedNIR/...
      HSI-VIS-FalseColor/...
      HSI-VIS/...

Each sequence must include its official init_rect.txt. If no sample CSV is
found, IDs are generated from all numeric JPEG frames under the modality roots.
The final CSV is written as PATH/submission.csv (default: result/submission.csv).
EOF
}

while (($#)); do
  case "$1" in
    --data|--data-root)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      data_root="$2"
      shift 2
      ;;
    --sample)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      sample="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      output="$2"
      shift 2
      ;;
    --devices)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      devices="$2"
      shift 2
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${data_root}" && "$1" != -* ]]; then
        data_root="$1"
        shift
      else
        printf 'Unknown option: %s\n' "$1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${data_root}" || ! -d "${data_root}" ]]; then
  printf 'Dataset directory does not exist: %s\n' "${data_root:-<missing>}" >&2
  usage >&2
  exit 2
fi
if [[ ! -x "${python}" ]]; then
  printf 'Missing environment. Run: bash scripts/setup.sh\n' >&2
  exit 1
fi

data_root="$(cd "${data_root}" && pwd)"
if [[ -d "${data_root}/validation" ]]; then
  validation_root="${data_root}/validation"
else
  validation_root="${data_root}"
fi

if [[ -n "${sample}" ]]; then
  if [[ ! -f "${sample}" ]]; then
    printf 'Sample CSV does not exist: %s\n' "${sample}" >&2
    exit 2
  fi
  sample="$(cd "$(dirname "${sample}")" && pwd)/$(basename "${sample}")"
else
  sample_candidates=(
    "${data_root}/sample_submisson.csv"
    "${data_root}/sample_submission.csv"
    "${validation_root}/sample_submisson.csv"
    "${validation_root}/sample_submission.csv"
    "$(dirname "${validation_root}")/sample_submisson.csv"
    "$(dirname "${validation_root}")/sample_submission.csv"
  )
  for candidate in "${sample_candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      sample="${candidate}"
      break
    fi
  done
  if [[ -z "${sample}" ]]; then
    temporary_sample="$(mktemp --suffix=.csv)"
    PYTHONPATH="${project_root}" "${python}" \
      "${project_root}/scripts/prepare_sample.py" \
      --validation-root "${validation_root}" \
      --output "${temporary_sample}"
    sample="${temporary_sample}"
  fi
fi

mkdir -p "$(dirname "${output}")"
output="$(cd "$(dirname "${output}")" && pwd)/$(basename "${output}")"
case "${output}" in
  /|"${project_root}"|"${data_root}"|"${validation_root}")
    printf 'Refusing unsafe output directory: %s\n' "${output}" >&2
    exit 2
    ;;
esac
if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  if ((overwrite == 0)); then
    printf 'Output directory is not empty: %s\nUse --overwrite to run fresh.\n' \
      "${output}" >&2
    exit 2
  fi
  rm -rf -- "${output}"
fi
mkdir -p "${output}"
work_dir="$(mktemp -d "$(dirname "${output}")/.hotc2026-inference.XXXXXX")"
mkdir -p "${work_dir}/logs"
if [[ -n "${temporary_sample}" ]]; then
  cp "${temporary_sample}" "${work_dir}/generated-sample.csv"
  sample="${work_dir}/generated-sample.csv"
fi

IFS=',' read -r -a device_list <<< "${devices}"
if ((${#device_list[@]} == 0)); then
  printf 'At least one CUDA device is required.\n' >&2
  exit 2
fi
for device in "${device_list[@]}"; do
  if [[ ! "${device}" =~ ^cuda:[0-9]+$ ]]; then
    printf 'Invalid CUDA device %q; expected cuda:N.\n' "${device}" >&2
    exit 2
  fi
done
num_shards="${#device_list[@]}"

sequence_summary="$(PYTHONPATH="${project_root}" "${python}" - "${sample}" <<'PY'
import sys
from pathlib import Path

from hotc.submission import read_sample_ids, split_identifier

identifiers = read_sample_ids(Path(sys.argv[1]))
sequences = {split_identifier(identifier)[0] for identifier in identifiers}
print(f"sequences={len(sequences)} frames={len(identifiers)}")
PY
)"

"${python}" "${project_root}/scripts/verify_artifacts.py" --profile runtime \
  >"${work_dir}/logs/artifact-verification.json"

v10_checkpoint="${project_root}/weights/amodal-v10.pt"
hsi_script="${project_root}/hotc/pipeline.py"

tracking="${work_dir}/tracking"
tracking_csv="${work_dir}/tracking.csv"
v10_cache="${work_dir}/amodal-v10"
submission_candidate="${work_dir}/submission.csv"
final_csv="${output}/submission.csv"

run_sharded() {
  local label="$1"
  shift
  local pids=()
  local index
  for index in "${!device_list[@]}"; do
    "$@" \
      --device "${device_list[${index}]}" \
      --shard-index "${index}" \
      --num-shards "${num_shards}" \
      >"${work_dir}/logs/${label}-shard-${index}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  if ((status != 0)); then
    printf '%s inference failed. Logs: %s/logs/%s-shard-*.log\n' \
      "${label}" "${work_dir}" "${label}" >&2
    return "${status}"
  fi
}

printf 'Fresh model inference\nData: %s\nSample: %s\n%s\nDevices: %s\n' \
  "${validation_root}" "${sample}" "${sequence_summary}" "${devices}"

run_sharded hsi \
  "${python}" "${hsi_script}" \
  --validation-root "${validation_root}" \
  --sample "${sample}" \
  --amodal-head-checkpoint "${v10_checkpoint}" \
  --amodal-cache-root "${v10_cache}" \
  --amodal-amp \
  --output-dir "${tracking}" \
  --submission "${tracking_csv}"

"${python}" "${hsi_script}" \
  --validation-root "${validation_root}" \
  --sample "${sample}" \
  --finalize \
  --output-dir "${tracking}" \
  --submission "${tracking_csv}" \
  >"${work_dir}/logs/hsi-finalize.log" 2>&1

"${python}" \
  "${project_root}/hotc/compose.py" \
  --baseline "${tracking_csv}" \
  --amodal-cache-root "${v10_cache}" \
  --sample "${sample}" \
  --output "${submission_candidate}" \
  >"${work_dir}/logs/final-composition.log" 2>&1

"${python}" "${project_root}/scripts/validate_submission.py" \
  "${submission_candidate}" --sample "${sample}"

PYTHONPATH="${project_root}" "${python}" - \
  "${sample}" "${tracking}" "${v10_cache}" <<'PY'
import json
import sys
from pathlib import Path

from hotc.submission import read_sample_ids, split_identifier

(
  sample,
  tracking,
  v10_cache,
) = map(Path, sys.argv[1:])
identifiers = read_sample_ids(sample)
sequences = sorted({split_identifier(identifier)[0] for identifier in identifiers})
shards = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(tracking.glob("shard-*.json"))]
records = [record for shard in shards for record in shard["records"]]
if len(records) != len(sequences) or any(record.get("cached") for record in records):
    raise RuntimeError("fresh HSI inference evidence is incomplete or contains cached sequences")
if any(record.get("sam3_image_encoder_calls") != record["frames"] for record in records):
  raise RuntimeError("shared SAM3 image encoder did not run exactly once per frame")
cache_count = len(list(v10_cache.glob("*.npz")))
if cache_count != len(sequences):
    raise RuntimeError(
        f"fresh amodal-v10 cache count mismatch: {cache_count} != {len(sequences)}"
    )
PY

mv -- "${submission_candidate}" "${final_csv}"

printf 'Model inference: PASS\n%s\nOutput: %s\n' \
  "${sequence_summary}" "${final_csv}"
