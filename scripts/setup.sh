#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${project_root}/.venv"
source_root="${project_root}/external/sam3"
runtime_file="${source_root}/sam3/model_builder.py"
upstream_repository="https://github.com/facebookresearch/sam3.git"
upstream_commit="20dba30a35a497606b06cf241f5b5605ea10e77e"
runtime_sha="faba3d0099f3da6d63d3ca49a8bf945d4014657b813e76ad082602d3922d3ff6"
python="${PYTHON:-python3}"
use_system_torch=0
bootstrap_pip_args=(--timeout 120 --retries 10)
pip_args=(--timeout 120 --retries 10 --resume-retries 20)

if (($#)); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

if "${python}" -c '
import torch, torchvision
assert torch.__version__.startswith("2.5.1+")
assert torchvision.__version__.startswith("0.20.1+")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
' >/dev/null 2>&1; then
  use_system_torch=1
fi

if [[ ! -f "${runtime_file}" ]]; then
  if [[ -e "${source_root}" ]]; then
    printf 'Refusing to replace incomplete upstream source: %s\n' "${source_root}" >&2
    exit 1
  fi
  command -v git >/dev/null 2>&1 || {
    printf 'git is required to fetch the official SAM 3 source.\n' >&2
    exit 1
  }
  mkdir -p "$(dirname "${source_root}")" "${source_root}"
  git -C "${source_root}" init
  git -C "${source_root}" remote add origin "${upstream_repository}"
  git -C "${source_root}" fetch --depth 1 origin "${upstream_commit}"
  git -C "${source_root}" checkout --detach FETCH_HEAD
fi
if [[ ! -d "${source_root}/.git" ]]; then
  printf 'Upstream source is not a verifiable Git checkout: %s\n' "${source_root}" >&2
  exit 1
fi
actual_commit="$(git -C "${source_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${upstream_commit}" ]]; then
  printf 'Unexpected official SAM 3 commit: %s\n' "${actual_commit}" >&2
  exit 1
fi
actual_runtime_sha="$(sha256sum "${runtime_file}" | awk '{print $1}')"
if [[ "${actual_runtime_sha}" != "${runtime_sha}" ]]; then
  printf 'Unexpected official SAM 3 runtime SHA-256: %s\n' "${actual_runtime_sha}" >&2
  exit 1
fi

venv_args=()
if ((use_system_torch)); then
  venv_args+=(--system-site-packages)
fi
"${python}" -m venv "${venv_args[@]}" "${venv}"
"${venv}/bin/python" -m pip install "${bootstrap_pip_args[@]}" \
  --upgrade pip==25.2 setuptools==75.8.0 wheel==0.45.1

if ((use_system_torch == 0)); then
  "${venv}/bin/python" -m pip install "${pip_args[@]}" \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}" \
    torch==2.5.1 torchvision==0.20.1
fi
"${venv}/bin/python" -m pip install "${pip_args[@]}" \
  -r "${project_root}/requirements.txt"

"${venv}/bin/python" -c '
import numpy, scipy
print("Environment setup: PASS")
'
