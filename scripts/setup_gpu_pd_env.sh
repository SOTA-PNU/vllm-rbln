#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PDD_VLLM_RBLN_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
venv_root="${PDD_VENV_ROOT:-${repo_dir}/.venvs}"
gpu_pd_venv="${PDD_GPU_PD_VENV:-${venv_root}/gpu-pd}"
python_bin="${PDD_BOOTSTRAP_PYTHON:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[setup:error] Python executable not found: ${python_bin}" >&2
    exit 1
fi

python_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${python_version}" in
    3.10|3.11|3.12|3.13) ;;
    *)
        echo "[setup:error] Python 3.10-3.13 is required; found ${python_version}." >&2
        exit 1
        ;;
esac

if [[ ! -x "${gpu_pd_venv}/bin/python3" ]]; then
    echo "[setup] Creating ${gpu_pd_venv}"
    "${python_bin}" -m venv "${gpu_pd_venv}"
fi

"${gpu_pd_venv}/bin/python3" -m pip install --upgrade "pip>=24" setuptools wheel

echo "[setup] Installing CUDA P/D dependencies"
"${gpu_pd_venv}/bin/python3" -m pip install \
    -r "${repo_dir}/requirements/pdd-prefill.txt"

echo "[setup] GPU P/D environment is ready: ${gpu_pd_venv}"
