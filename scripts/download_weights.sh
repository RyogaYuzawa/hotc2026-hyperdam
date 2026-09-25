#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_cli="${project_root}/.venv/bin/hf"

sam_repo="facebook/sam3"
sam_revision="3c879f39826c281e95690f02c7821c4de09afae7"
sam_file="sam3.pt"
sam_path="${project_root}/weights/sam3.pt"
sam_sha256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"

head_repo="ryo818/HOTC2026-Amodal"
head_revision="fff9b5e564f661bd520196bf526d8ead5d097403"
head_file="amodal-v10.pt"
head_path="${project_root}/weights/amodal-v10.pt"
head_sha256="718f34dd59f828762b8b0ca33be66516f26968c1d77893a0ac58b2e3d35c3966"

sam_license="${project_root}/THIRD_PARTY_LICENSES/SAM_LICENSE"
sam_license_sha256="4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be"

if (($#)); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  [[ -f "${path}" ]] || return 1
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

if [[ ! -x "${hf_cli}" ]]; then
  printf 'Missing Hugging Face CLI. Run: bash scripts/setup.sh\n' >&2
  exit 1
fi
if ! check_sha256 "${sam_license_sha256}" "${sam_license}"; then
  printf 'Missing or modified SAM License: %s\n' "${sam_license}" >&2
  exit 1
fi

for target_and_sha in "${sam_path}:${sam_sha256}" "${head_path}:${head_sha256}"; do
  target="${target_and_sha%:*}"
  expected="${target_and_sha##*:}"
  if [[ -e "${target}" || -L "${target}" ]] && \
     ! check_sha256 "${expected}" "${target}"; then
    printf 'Refusing to overwrite an unverified checkpoint: %s\n' "${target}" >&2
    exit 1
  fi
done

if check_sha256 "${sam_sha256}" "${sam_path}" && \
   check_sha256 "${head_sha256}" "${head_path}"; then
  printf 'Model checkpoints already installed and verified.\n'
  exit 0
fi

download_dir="$(mktemp -d "${TMPDIR:-/tmp}/hotc2026-weights.XXXXXX")"
trap 'rm -rf "${download_dir}"' EXIT

download_checkpoint() {
  local repo="$1"
  local revision="$2"
  local filename="$3"
  local expected="$4"
  local target="$5"
  local label="$6"
  local staging="${download_dir}/${label}"
  local temporary_target="${target}.tmp.$$"

  if check_sha256 "${expected}" "${target}"; then
    printf '%s already installed and verified.\n' "${label}"
    return
  fi

  mkdir -p "${staging}"
  printf 'Downloading %s from https://huggingface.co/%s ...\n' "${label}" "${repo}"
  if ! "${hf_cli}" download "${repo}" "${filename}" \
      --revision "${revision}" --local-dir "${staging}" --quiet; then
    if [[ "${repo}" == "facebook/sam3" ]]; then
      printf '%s\n' \
        'SAM3 requires access approval and authentication.' \
        'Accept the license at https://huggingface.co/facebook/sam3' \
        "Then run: ${hf_cli} auth login" >&2
    fi
    exit 1
  fi
  if ! check_sha256 "${expected}" "${staging}/${filename}"; then
    printf 'SHA-256 mismatch for %s.\n' "${label}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${target}")"
  install -m 0644 "${staging}/${filename}" "${temporary_target}"
  if ! check_sha256 "${expected}" "${temporary_target}"; then
    rm -f -- "${temporary_target}"
    printf 'Installed file verification failed for %s.\n' "${label}" >&2
    exit 1
  fi
  mv -- "${temporary_target}" "${target}"
}

printf 'SAM3 materials are governed by: %s\n' "${sam_license}"
download_checkpoint \
  "${sam_repo}" "${sam_revision}" "${sam_file}" "${sam_sha256}" \
  "${sam_path}" "SAM3"
download_checkpoint \
  "${head_repo}" "${head_revision}" "${head_file}" "${head_sha256}" \
  "${head_path}" "amodal-v10"

printf 'Model checkpoint installation: PASS\n'
