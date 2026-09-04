#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd -P
)"
REPO="${REPO:-$SCRIPT_DIR}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/vllm-rbln-profile-results}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

mkdir -p -- "$RESULT_ROOT" || die "cannot create RESULT_ROOT: $RESULT_ROOT"
RESULT_ROOT=$(cd -- "$RESULT_ROOT" >/dev/null 2>&1 && pwd -P) || \
    die "cannot resolve RESULT_ROOT: $RESULT_ROOT"
export REPO RESULT_ROOT

PDD_RUN_TOKEN="runall-$$-${RANDOM}"
export PDD_RUN_TOKEN

owned_result_dir() {
    local owned=
    local candidate
    [[ -d "$RESULT_ROOT" ]] || return 1
    while IFS= read -r -d '' candidate; do
        if [[ -z "$owned" || "$candidate" -nt "$owned" ]]; then
            owned=$candidate
        fi
    done < <(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "native-dtype-*-$PDD_RUN_TOKEN" -print0)
    [[ -n "$owned" ]] || return 1
    printf '%s\n' "$owned"
}

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
        candidate=$(owned_result_dir || true)
        if [[ -n "$candidate" && -f "$candidate/pids.env" ]]; then
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
