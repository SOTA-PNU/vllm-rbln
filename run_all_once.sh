#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR=/home/jiwon_lee/sota/profile-handoff-0.11.1-2_native_dtype
readonly REPO=/home/jiwon_lee/sota/vllm-rbln-0.11.1-2_native_dtype
readonly RESULT_ROOT=/home/jiwon_lee/sota/profile-results

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$REPO"
export VLLM_RBLN_PDD_LAYOUT_REORDER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export RBLN_DEVICES=0

latest_result_dir() {
    local latest=
    local candidate
    [[ -d "$RESULT_ROOT" ]] || return 1
    while IFS= read -r -d '' candidate; do
        if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
            latest=$candidate
        fi
    done < <(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name 'native-dtype-*' -print0)
    [[ -n "$latest" ]] || return 1
    printf '%s\n' "$latest"
}

BEFORE_RESULT=$(latest_result_dir || true)
RESULT_DIR=

cleanup() {
    local status=$?
    local trigger=${1:-EXIT}
    local stop_status=0
    local candidate=
    trap - EXIT INT TERM
    if [[ "$trigger" == INT ]]; then
        status=130
    elif [[ "$trigger" == TERM ]]; then
        status=143
    fi
    if [[ -z "$RESULT_DIR" ]]; then
        candidate=$(latest_result_dir || true)
        if [[ -n "$candidate" && "$candidate" != "$BEFORE_RESULT" && \
              -f "$candidate/pids.env" ]]; then
            RESULT_DIR=$candidate
        fi
    fi
    if [[ -n "$RESULT_DIR" && -f "$RESULT_DIR/pids.env" ]]; then
        export RESULT_DIR
        "$SCRIPT_DIR/stop_pdd.sh" || stop_status=$?
    fi
    if (( status == 0 && stop_status != 0 )); then
        status=$stop_status
    fi
    exit "$status"
}
trap 'cleanup EXIT' EXIT
trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

# For manual profiler attachment, run start/check/workload/stop individually.
START_OUTPUT=$("$SCRIPT_DIR/start_pdd.sh")
printf '%s\n' "$START_OUTPUT"
RESULT_DIR=$(printf '%s\n' "$START_OUTPUT" | awk -F= '$1 == "RESULT_DIR" { print substr($0, index($0, "=") + 1) }')
[[ -n "$RESULT_DIR" && -d "$RESULT_DIR" ]] || {
    printf 'ERROR: start_pdd.sh did not return a valid RESULT_DIR\n' >&2
    exit 1
}
export RESULT_DIR

"$SCRIPT_DIR/check_correctness.sh"
"$SCRIPT_DIR/run_workload.sh"
