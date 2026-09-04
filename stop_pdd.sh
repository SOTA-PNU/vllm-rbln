#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd -P
)"
REPO="${REPO:-$SCRIPT_DIR}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/vllm-rbln-profile-results}"
EXPECTED_BRANCH=0.11.1-2_native_dtype
REQUIRED_LAYOUT_COMMIT=972ce20cf53b0d2d50c4155f9d44be6879ede966

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

proc_fields() {
    local pid=$1
    local stat_line tail
    [[ -r "/proc/$pid/stat" ]] || return 1
    IFS= read -r stat_line 2>/dev/null <"/proc/$pid/stat" || return 1
    tail=${stat_line##*) }
    set -- $tail
    printf '%s %s %s %s %s\n' "$1" "$2" "$3" "$4" "${20}"
}

recorded_pid_active() {
    local pid=$1
    local expected_starttime=$2
    local state ppid pgid sid starttime
    read -r state ppid pgid sid starttime < <(proc_fields "$pid") || return 1
    [[ "$starttime" == "$expected_starttime" ]] || return 1
    [[ "$state" != Z && "$state" != X ]] || return 1
}

group_exists() {
    local pgid=$1
    kill -0 -- "-$pgid" 2>/dev/null
}

group_session_valid() {
    local pgid=$1
    local sid=$2
    ps -eo pgid=,sid= | awk -v pgid="$pgid" -v sid="$sid" '
        $1 == pgid { found = 1; if ($2 != sid) bad = 1 }
        END { exit(found && !bad ? 0 : 1) }
    '
}

is_recorded_descendant() {
    local candidate=$1
    local current=$candidate
    local state ppid pgid sid starttime
    local depth
    for ((depth = 0; depth < 128; depth++)); do
        if [[ "$current" == "${PREFILL_PID:-}" || \
              "$current" == "${DECODE_PID:-}" || \
              "$current" == "${PROXY_PID:-}" ]]; then
            return 0
        fi
        read -r state ppid pgid sid starttime < <(proc_fields "$current") || return 1
        [[ "$ppid" =~ ^[0-9]+$ && "$ppid" -gt 1 && "$ppid" != "$current" ]] || return 1
        current=$ppid
    done
    return 1
}

snapshot_unrelated() {
    local destination=$1
    local pid pgid args state ppid actual_pgid sid starttime
    : >"$destination"
    while read -r pid pgid args; do
        [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ ]] || continue
        case "$args" in
            *vllm.entrypoints.cli.main*serve*|*pdd_profile_proxy.py*) ;;
            *) continue ;;
        esac
        if [[ "$pgid" == "${PREFILL_PGID:-}" || \
              "$pgid" == "${DECODE_PGID:-}" || \
              "$pgid" == "${PROXY_PGID:-}" ]]; then
            continue
        fi
        if is_recorded_descendant "$pid"; then
            continue
        fi
        read -r state ppid actual_pgid sid starttime < <(proc_fields "$pid") || continue
        [[ "$state" != Z && "$state" != X ]] || continue
        printf '%s\t%s\t%s\t%s\n' "$pid" "$starttime" "$pgid" "$args" >>"$destination"
    done < <(ps -eo pid=,pgid=,args=)
}

validate_role() {
    local role=$1
    local pid_name="${role}_PID"
    local pgid_name="${role}_PGID"
    local sid_name="${role}_SID"
    local start_name="${role}_STARTTIME"
    local pid=${!pid_name:-}
    local pgid=${!pgid_name:-}
    local sid=${!sid_name:-}
    local expected_starttime=${!start_name:-}
    local state ppid actual_pgid actual_sid actual_starttime
    local self_pgid

    printf -v "${role}_AUTHORIZED" '%s' 0
    if [[ -z "$pid" && -z "$pgid" && -z "$sid" && -z "$expected_starttime" ]]; then
        printf -v "${role}_RECORDED" '%s' 0
        return 0
    fi
    printf -v "${role}_RECORDED" '%s' 1
    [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || {
        printf 'ERROR: invalid %s PID record\n' "$role" >&2
        return 1
    }
    [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] || {
        printf 'ERROR: invalid %s PGID record\n' "$role" >&2
        return 1
    }
    [[ "$sid" =~ ^[0-9]+$ && "$sid" -gt 1 ]] || {
        printf 'ERROR: invalid %s SID record\n' "$role" >&2
        return 1
    }
    [[ "$expected_starttime" =~ ^[0-9]+$ ]] || {
        printf 'ERROR: invalid %s start-time record\n' "$role" >&2
        return 1
    }
    [[ "$pid" == "$pgid" && "$pid" == "$sid" ]] || {
        printf 'ERROR: %s was not recorded as an isolated session\n' "$role" >&2
        return 1
    }
    self_pgid=$(ps -o pgid= -p $$ | tr -d '[:space:]')
    [[ "$pgid" != "$self_pgid" ]] || {
        printf 'ERROR: refusing to signal the stop shell process group\n' >&2
        return 1
    }

    if read -r state ppid actual_pgid actual_sid actual_starttime < <(proc_fields "$pid"); then
        if [[ "$actual_starttime" != "$expected_starttime" ]]; then
            printf 'ERROR: %s PID %s was reused; its process group will not be signaled\n' \
                "$role" "$pid" >&2
            return 1
        fi
        [[ "$actual_pgid" == "$pgid" && "$actual_sid" == "$sid" ]] || {
            printf 'ERROR: %s process identity no longer matches its recorded group/session\n' \
                "$role" >&2
            return 1
        }
    fi

    if group_exists "$pgid"; then
        group_session_valid "$pgid" "$sid" || {
            printf 'ERROR: %s group/session membership is inconsistent\n' "$role" >&2
            return 1
        }
        printf -v "${role}_AUTHORIZED" '%s' 1
    fi
}

signal_role() {
    local role=$1
    local signal_name=$2
    local auth_name="${role}_AUTHORIZED"
    local pgid_name="${role}_PGID"
    local sid_name="${role}_SID"
    local pid_name="${role}_PID"
    local start_name="${role}_STARTTIME"
    local authorized=${!auth_name:-0}
    local pgid=${!pgid_name:-}
    local sid=${!sid_name:-}
    local pid=${!pid_name:-}
    local expected_starttime=${!start_name:-}
    local state ppid actual_pgid actual_sid actual_starttime

    [[ "$authorized" == 1 ]] || return 0
    group_exists "$pgid" || return 0
    if read -r state ppid actual_pgid actual_sid actual_starttime < <(proc_fields "$pid"); then
        if [[ "$actual_starttime" != "$expected_starttime" || \
              "$actual_pgid" != "$pgid" || "$actual_sid" != "$sid" ]]; then
            printf 'ERROR: refusing %s for reused or changed %s process identity\n' \
                "$signal_name" "$role" >&2
            return 1
        fi
    fi
    group_session_valid "$pgid" "$sid" || {
        printf 'ERROR: refusing %s for changed %s group/session\n' \
            "$signal_name" "$role" >&2
        return 1
    }
    if ! kill "-$signal_name" -- "-$pgid" 2>/dev/null; then
        group_exists "$pgid" || return 0
        printf 'ERROR: failed to send %s to recorded %s process group %s\n' \
            "$signal_name" "$role" "$pgid" >&2
        return 1
    fi
}

verify_unrelated() {
    local source=$1
    local pid expected_starttime pgid args
    local state ppid actual_pgid sid actual_starttime
    local failed=0
    while IFS=$'\t' read -r pid expected_starttime pgid args; do
        [[ -n "$pid" ]] || continue
        if ! read -r state ppid actual_pgid sid actual_starttime < <(proc_fields "$pid"); then
            printf 'ERROR: unrelated process %s is no longer present\n' "$pid" >&2
            failed=1
            continue
        fi
        if [[ "$actual_starttime" != "$expected_starttime" || "$state" == Z || "$state" == X ]]; then
            printf 'ERROR: unrelated process identity %s did not survive unchanged\n' "$pid" >&2
            failed=1
        fi
    done <"$source"
    return "$failed"
}

[[ -n "${RESULT_DIR:-}" ]] || \
    die "RESULT_DIR is required for PID-scoped cleanup"
RESULT_DIR=$(realpath -e -- "$RESULT_DIR") || die "RESULT_DIR does not exist"
[[ -f "$RESULT_DIR/runtime.env" ]] || die "runtime.env is missing in RESULT_DIR"
[[ -f "$RESULT_DIR/pids.env" ]] || die "pids.env is missing in RESULT_DIR"
SELECTED_RESULT_DIR=$RESULT_DIR
command -v flock >/dev/null 2>&1 || die "flock is required for start/stop serialization"

set -a
# shellcheck disable=SC1090
source "$RESULT_DIR/runtime.env"
# shellcheck disable=SC1090
source "$RESULT_DIR/pids.env"
set +a
[[ "${RESULT_DIR:-}" == "$SELECTED_RESULT_DIR" ]] || die "recorded RESULT_DIR mismatch"
[[ -d "${RESULT_ROOT:-}" ]] || die "recorded RESULT_ROOT does not exist"
[[ -d "${REPO:-}" ]] || die "recorded repository does not exist"
ACTUAL_COMMIT=$(git -C "$REPO" rev-parse HEAD) || die "cannot read recorded repository HEAD"
[[ "$ACTUAL_COMMIT" == "${COMMIT:-}" ]] || \
    die "recorded commit does not match repository HEAD"
ACTUAL_BRANCH=$(git -C "$REPO" branch --show-current) || \
    die "cannot read recorded repository branch"
if [[ -n "$ACTUAL_BRANCH" && "$ACTUAL_BRANCH" != "$EXPECTED_BRANCH" ]]; then
    die "branch mismatch: expected $EXPECTED_BRANCH or detached HEAD, got $ACTUAL_BRANCH"
fi
git -C "$REPO" merge-base --is-ancestor \
    "$REQUIRED_LAYOUT_COMMIT" "$ACTUAL_COMMIT" || \
    die "required layout commit is not an ancestor"
[[ -x "${CUDA_PY:-}" ]] || die "recorded CUDA_PY is unavailable"
[[ "${RBLN_DEVICE:-}" =~ ^[0-9]+$ ]] || die "recorded RBLN_DEVICE is invalid"

exec {START_LOCK_FD}<"$RESULT_ROOT"
flock --exclusive "$START_LOCK_FD"

FAILURE=0
validate_role PROXY || FAILURE=1
validate_role PREFILL || FAILURE=1
validate_role DECODE || FAILURE=1

UNRELATED_BEFORE="$RESULT_DIR/unrelated_processes_before_stop.txt"
UNRELATED_AFTER="$RESULT_DIR/unrelated_processes_after_stop.txt"
snapshot_unrelated "$UNRELATED_BEFORE"

signal_role PROXY TERM || FAILURE=1
signal_role PREFILL TERM || FAILURE=1
signal_role DECODE TERM || FAILURE=1

TERM_DEADLINE=$((SECONDS + 30))
while (( SECONDS < TERM_DEADLINE )); do
    ANY_GROUP=0
    for role in PROXY PREFILL DECODE; do
        auth_name="${role}_AUTHORIZED"
        pgid_name="${role}_PGID"
        if [[ "${!auth_name:-0}" == 1 ]] && group_exists "${!pgid_name}"; then
            ANY_GROUP=1
            break
        fi
    done
    (( ANY_GROUP == 0 )) && break
    sleep 0.2
done

signal_role PROXY KILL || FAILURE=1
signal_role PREFILL KILL || FAILURE=1
signal_role DECODE KILL || FAILURE=1

KILL_DEADLINE=$((SECONDS + 5))
while (( SECONDS < KILL_DEADLINE )); do
    ANY_GROUP=0
    for role in PROXY PREFILL DECODE; do
        auth_name="${role}_AUTHORIZED"
        pgid_name="${role}_PGID"
        if [[ "${!auth_name:-0}" == 1 ]] && group_exists "${!pgid_name}"; then
            ANY_GROUP=1
            break
        fi
    done
    (( ANY_GROUP == 0 )) && break
    sleep 0.1
done

for role in PROXY PREFILL DECODE; do
    pid_name="${role}_PID"
    start_name="${role}_STARTTIME"
    auth_name="${role}_AUTHORIZED"
    pgid_name="${role}_PGID"
    recorded_name="${role}_RECORDED"
    [[ "${!recorded_name:-0}" == 1 ]] || continue
    pid=${!pid_name:-}
    expected_starttime=${!start_name:-}
    pgid=${!pgid_name:-}
    if [[ "$pid" =~ ^[0-9]+$ && "$expected_starttime" =~ ^[0-9]+$ ]] && \
       recorded_pid_active "$pid" "$expected_starttime"; then
        printf 'ERROR: recorded %s PID %s is still active\n' "$role" "$pid" >&2
        FAILURE=1
    fi
    if [[ "${!auth_name:-0}" == 1 && "$pgid" =~ ^[0-9]+$ ]] && group_exists "$pgid"; then
        printf 'ERROR: recorded %s process group %s is still active\n' \
            "$role" "$pgid" >&2
        FAILURE=1
    fi
done

snapshot_unrelated "$UNRELATED_AFTER"
verify_unrelated "$UNRELATED_BEFORE" || FAILURE=1

command -v rbln-smi >/dev/null 2>&1 || {
    printf 'ERROR: rbln-smi is unavailable for post-stop verification\n' >&2
    FAILURE=1
}
RBLN_CLEAR=0
RBLN_STATE_FILE="$RESULT_DIR/rbln_smi_after_stop.json"
if command -v rbln-smi >/dev/null 2>&1; then
    for ((attempt = 0; attempt < 10; attempt++)); do
        if rbln-smi --json >"$RBLN_STATE_FILE" 2>/dev/null && \
           "$CUDA_PY" - "${DECODE_PID:-}" "$RBLN_STATE_FILE" "$RBLN_DEVICE" <<'PY'
import json
import sys

decode_pid = str(sys.argv[1])
device_id = str(sys.argv[3])
with open(sys.argv[2], encoding="utf-8") as handle:
    doc = json.load(handle)
for context in doc.get("contexts", []):
    marker = str(context.get("npu", context.get("device", ""))).lower()
    serialized = json.dumps(context, sort_keys=True)
    if marker in {device_id, f"rbln{device_id}"} or (
        decode_pid and decode_pid in serialized
    ):
        raise SystemExit(1)
PY
        then
            RBLN_CLEAR=1
            break
        fi
        sleep 1
    done
fi
if (( RBLN_CLEAR == 0 )); then
    printf 'ERROR: an RBLN device-%s context remains after recorded process cleanup\n' \
        "$RBLN_DEVICE" >&2
    FAILURE=1
fi

if (( FAILURE != 0 )); then
    exit 1
fi

printf 'PID_SCOPED_CLEANUP=PASS\n'
printf 'UNRELATED_PROCESSES_UNTOUCHED=PASS\n'
printf 'RBLN_CONTEXT_CLEAR=PASS\n'
printf 'STOP_COMPLETE\n'
printf 'RESULT_DIR=%s\n' "$RESULT_DIR"
