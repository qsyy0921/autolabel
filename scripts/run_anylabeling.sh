#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/home/panjunhao/miniconda3/etc/profile.d/conda.sh"
ENV_PREFIX="${ROOT_DIR}/.conda/autolabel"
PY_SITE="${ENV_PREFIX}/lib/python3.11/site-packages"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda activation script: ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "Missing local env: ${ENV_PREFIX}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"

export LD_LIBRARY_PATH="${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

exec python "${ROOT_DIR}/tmp/anylabeling/app.py" "$@"
