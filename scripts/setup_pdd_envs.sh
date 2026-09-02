#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PDD_VLLM_RBLN_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
venv_root="${PDD_VENV_ROOT:-${repo_dir}/.venvs}"
prefill_venv="${PDD_PREFILL_VENV:-${venv_root}/pdd-prefill}"
decode_venv="${PDD_DECODE_VENV:-${venv_root}/pdd-decode}"
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

create_venv() {
    local target="$1"
    if [[ ! -x "${target}/bin/python3" ]]; then
        echo "[setup] Creating ${target}"
        "${python_bin}" -m venv "${target}"
    fi
    "${target}/bin/python3" -m pip install --upgrade "pip>=24" setuptools wheel
}

link_system_rbln_sdk() {
    local target="$1"
    local sdk_dir
    sdk_dir="$("${python_bin}" - <<'FIND_SDK'
import importlib.util, pathlib
spec = importlib.util.find_spec("rebel")
print(pathlib.Path(spec.origin).parent.parent if spec and spec.origin else "", end="")
FIND_SDK
)"
    if [[ -z "${sdk_dir}" ]]; then
        echo "[setup:error] rebel-compiler is not installed in ${python_bin}." >&2
        exit 1
    fi

    local site_dir
    site_dir="$("${target}/bin/python3" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

    local entries
    entries="$("${target}/bin/python3" - "${sdk_dir}" <<'OWNED'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
record = next(root.glob("rebel_compiler-*.dist-info/RECORD"), None)
if record is None:
    raise SystemExit("no rebel_compiler dist-info/RECORD under " + str(root))
tops = set()
for line in record.read_text().splitlines():
    path = line.split(",")[0]
    if not path or path.startswith("../"):
        continue
    top = path.split("/")[0]
    if top != "__pycache__":
        tops.add(top)
print("\n".join(sorted(tops)), end="")
OWNED
)"

    echo "[setup] Linking system RBLN SDK from ${sdk_dir} into ${target}"
    local entry
    while IFS= read -r entry; do
        [[ -n "${entry}" && -e "${sdk_dir}/${entry}" ]] || continue
        echo "[setup]   link ${entry}"
        ln -sfn "${sdk_dir}/${entry}" "${site_dir}/${entry}"
    done <<< "${entries}"

    local deps
    deps="$("${target}/bin/python3" - "${sdk_dir}" <<'READ_DEPS'
import pathlib, re, sys
names = []
for meta in pathlib.Path(sys.argv[1]).glob("rebel_compiler-*.dist-info/METADATA"):
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        req = line.split(":", 1)[1].strip()
        if ";" in req:          # skip `extra == "..."` markers
            continue
        names.append(re.split(r"[\s<>=!~\[;]", req, maxsplit=1)[0])
print(" ".join(sorted(set(names))), end="")
READ_DEPS
)"
    if [[ -n "${deps}" ]]; then
        echo "[setup] Installing rebel-compiler runtime deps: ${deps}"
        # shellcheck disable=SC2086
        "${target}/bin/python3" -m pip install ${deps}
    fi
}

create_venv "${prefill_venv}"
echo "[setup] Installing CUDA prefill dependencies"
"${prefill_venv}/bin/python3" -m pip install \
    -r "${repo_dir}/requirements/pdd-prefill.txt"

create_venv "${decode_venv}"
echo "[setup] Installing RBLN decode dependencies and this checkout"
"${decode_venv}/bin/python3" -m pip install \
    -r "${repo_dir}/requirements/pdd-decode.txt" \
    --extra-index-url https://wheels.vllm.ai/0.18.0/cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
"${decode_venv}/bin/python3" -m pip install \
    --editable "${repo_dir}" \
    --extra-index-url https://wheels.vllm.ai/0.18.0/cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

link_system_rbln_sdk "${decode_venv}"

echo "[setup] Environments are ready:"
echo "[setup]   prefill: ${prefill_venv}"
echo "[setup]   decode:  ${decode_venv}"
