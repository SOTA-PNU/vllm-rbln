#!/usr/bin/env bash
set -Eeuo pipefail

umask 022

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd -P
)"
REPO="${REPO:-$SCRIPT_DIR}"
EXPECTED_BRANCH=0.11.1-2_native_dtype
REQUIRED_LAYOUT_COMMIT=972ce20cf53b0d2d50c4155f9d44be6879ede966

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

missing_python() {
    local name=$1
    printf 'ERROR: %s is required.\n' "$name" >&2
    printf 'Example:\n' >&2
    printf 'CUDA_PY=$HOME/.venvs/pdd-cuda/bin/python3 \\\n' >&2
    printf 'RBLN_PY=$HOME/.venvs/pdd-rbln/bin/python3 \\\n' >&2
    printf './start_pdd.sh\n' >&2
    exit 1
}

absolute_executable() {
    local supplied=$1
    local directory basename
    if [[ "$supplied" != */* ]]; then
        command -v -- "$supplied" || return 1
        return
    fi
    directory=$(dirname -- "$supplied")
    basename=$(basename -- "$supplied")
    directory=$(cd -- "$directory" >/dev/null 2>&1 && pwd -P) || return 1
    printf '%s/%s\n' "$directory" "$basename"
}

write_env() {
    local destination=$1
    local name=$2
    local value=$3
    printf '%s=%q\n' "$name" "$value" >>"$destination"
}

proc_starttime() {
    local pid=$1
    local stat_line tail
    IFS= read -r stat_line <"/proc/$pid/stat" || return 1
    tail=${stat_line##*) }
    set -- $tail
    printf '%s\n' "${20}"
}

wait_http() {
    local url=$1
    local pid=$2
    local timeout_seconds=$3
    local label=$4
    local deadline=$((SECONDS + timeout_seconds))

    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            printf '%s exited before readiness.\n' "$label" >&2
            return 1
        fi
        if curl --silent --show-error --fail --max-time 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    printf '%s readiness timed out after %s seconds: %s\n' \
        "$label" "$timeout_seconds" "$url" >&2
    return 1
}

record_process() {
    local role=$1
    local pid=$2
    local pgid sid starttime

    read -r pgid sid < <(ps -o pgid=,sid= -p "$pid") || \
        die "cannot inspect $role process $pid"
    pgid=${pgid//[[:space:]]/}
    sid=${sid//[[:space:]]/}
    starttime=$(proc_starttime "$pid") || die "cannot read $role process identity"
    [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ && "$sid" =~ ^[0-9]+$ ]] || \
        die "invalid $role process identity"
    [[ "$pgid" == "$pid" && "$sid" == "$pid" ]] || \
        die "$role was not isolated in its own process group and session"

    write_env "$PIDS_FILE" "${role}_PID" "$pid"
    write_env "$PIDS_FILE" "${role}_PGID" "$pgid"
    write_env "$PIDS_FILE" "${role}_SID" "$sid"
    write_env "$PIDS_FILE" "${role}_STARTTIME" "$starttime"

    printf -v "${role}_PID" '%s' "$pid"
    printf -v "${role}_PGID" '%s' "$pgid"
}

launch_group() {
    local role=$1
    local logfile=$2
    shift 2

    setsid "$@" </dev/null >>"$logfile" 2>&1 &
    local pid=$!
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        tail -n 80 "$logfile" >&2 || true
        die "$role failed immediately; recorded processes, if any, remain available to stop_pdd.sh"
    fi
    record_process "$role" "$pid"
}

[[ -n "${CUDA_PY:-}" ]] || missing_python CUDA_PY
[[ -n "${RBLN_PY:-}" ]] || missing_python RBLN_PY
[[ -d "$REPO" ]] || die "repository directory is missing: $REPO"
REPO=$(cd -- "$REPO" >/dev/null 2>&1 && pwd -P) || \
    die "cannot resolve repository path: $REPO"
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "not a Git repository: $REPO"

CUDA_PY=$(absolute_executable "$CUDA_PY") || die "cannot resolve CUDA_PY"
RBLN_PY=$(absolute_executable "$RBLN_PY") || die "cannot resolve RBLN_PY"
[[ -f "$CUDA_PY" && -x "$CUDA_PY" ]] || \
    die "CUDA Python is not an executable file: $CUDA_PY"
[[ -f "$RBLN_PY" && -x "$RBLN_PY" ]] || \
    die "RBLN Python is not an executable file: $RBLN_PY"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/vllm-rbln-profile-results}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RBLN_DEVICE="${RBLN_DEVICE:-0}"
EXPECTED_COMPILER="${EXPECTED_COMPILER:-0.11.1.post2.dev2+g2995098f.prod}"
EXPECTED_CUDA_VLLM="${EXPECTED_CUDA_VLLM:-0.22.0}"
EXPECTED_RBLN_VLLM="${EXPECTED_RBLN_VLLM:-0.22.0+cpu}"
OFFLINE="${OFFLINE:-0}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-$MODEL_DTYPE}"
VLLM_RBLN_PDD_LAYOUT_REORDER="${VLLM_RBLN_PDD_LAYOUT_REORDER:-0}"
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
PDD_RUN_TOKEN="${PDD_RUN_TOKEN:-}"
PROXY_SCRIPT="$REPO/pdd_profile_proxy.py"

[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || die "CUDA_DEVICE must be a nonnegative integer"
[[ "$RBLN_DEVICE" =~ ^[0-9]+$ ]] || die "RBLN_DEVICE must be a nonnegative integer"
[[ "$OFFLINE" == 0 || "$OFFLINE" == 1 ]] || die "OFFLINE must be 0 or 1"
[[ "$VLLM_RBLN_PDD_LAYOUT_REORDER" == 0 || \
   "$VLLM_RBLN_PDD_LAYOUT_REORDER" == 1 ]] || \
    die "VLLM_RBLN_PDD_LAYOUT_REORDER must be 0 or 1"
[[ "$MODEL_DTYPE" == float16 || "$MODEL_DTYPE" == bfloat16 ]] || \
    die "MODEL_DTYPE must be float16 or bfloat16"
[[ "$KV_CACHE_DTYPE" == float16 || "$KV_CACHE_DTYPE" == bfloat16 ]] || \
    die "KV_CACHE_DTYPE must be float16 or bfloat16"
if [[ -n "$PDD_RUN_TOKEN" && ! "$PDD_RUN_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "PDD_RUN_TOKEN may contain only letters, digits, dot, underscore, and hyphen"
fi
if [[ "$MODEL_PATH" == /* ]]; then
    [[ -d "$MODEL_PATH" ]] || die "local MODEL_PATH directory is missing: $MODEL_PATH"
    MODEL_PATH=$(realpath -e -- "$MODEL_PATH") || die "cannot resolve MODEL_PATH"
fi
[[ -f "$PROXY_SCRIPT" ]] || die "repository proxy is missing: $PROXY_SCRIPT"
command -v curl >/dev/null 2>&1 || die "curl is required for readiness checks"
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group isolation"
command -v flock >/dev/null 2>&1 || die "flock is required for launch serialization"
command -v git >/dev/null 2>&1 || die "git is required"

CURRENT_COMMIT=$(git -C "$REPO" rev-parse HEAD) || die "cannot resolve current commit"
BRANCH=$(git -C "$REPO" branch --show-current) || die "cannot resolve current branch"
if [[ -n "$BRANCH" && "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
    die "branch mismatch: expected $EXPECTED_BRANCH or detached HEAD, got $BRANCH"
fi
git -C "$REPO" merge-base --is-ancestor \
    "$REQUIRED_LAYOUT_COMMIT" "$CURRENT_COMMIT" || \
    die "required layout commit is not an ancestor: $REQUIRED_LAYOUT_COMMIT"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
    die "repository working tree is not clean"

if ! COMPILER_VERSION=$(
    "$RBLN_PY" -c '
import importlib.metadata as md
print(md.version("rebel-compiler"))
'
); then
    die "cannot read the installed rebel-compiler distribution from RBLN_PY"
fi
CUDA_VLLM_VERSION=$(
    "$CUDA_PY" -c 'import importlib.metadata as md; print(md.version("vllm"))'
) || die "cannot read CUDA vLLM version"
RBLN_VLLM_VERSION=$(
    "$RBLN_PY" -c 'import importlib.metadata as md; print(md.version("vllm"))'
) || die "cannot read RBLN vLLM version"
[[ "$COMPILER_VERSION" == "$EXPECTED_COMPILER" ]] || \
    die "compiler mismatch: expected $EXPECTED_COMPILER, got $COMPILER_VERSION"
[[ "$CUDA_VLLM_VERSION" == "$EXPECTED_CUDA_VLLM" ]] || \
    die "CUDA vLLM mismatch: expected $EXPECTED_CUDA_VLLM, got $CUDA_VLLM_VERSION"
[[ "$RBLN_VLLM_VERSION" == "$EXPECTED_RBLN_VLLM" ]] || \
    die "RBLN vLLM mismatch: expected $EXPECTED_RBLN_VLLM, got $RBLN_VLLM_VERSION"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
CUDA_GPU_LINE=$(nvidia-smi -i "$CUDA_DEVICE" \
    --query-gpu=index,uuid,name --format=csv,noheader,nounits) || \
    die "CUDA device $CUDA_DEVICE is unavailable"
CUDA_GPU_UUID=$(printf '%s\n' "$CUDA_GPU_LINE" | cut -d, -f2 | tr -d ' ')
[[ -n "$CUDA_GPU_UUID" ]] || die "CUDA device query returned an unexpected result"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | \
    tr -d ' ' | grep -Fxq "$CUDA_GPU_UUID"; then
    die "CUDA device $CUDA_DEVICE already has a compute process; no existing process was changed"
fi

command -v rbln-smi >/dev/null 2>&1 || die "rbln-smi is unavailable"
RBLN_STATE=$(rbln-smi --json) || die "RBLN device query failed"
printf '%s' "$RBLN_STATE" | "$CUDA_PY" -c '
import json
import sys

device_id = sys.argv[1]
doc = json.load(sys.stdin)
devices = [d for d in doc.get("devices", []) if str(d.get("npu")) == device_id]
if len(devices) != 1 or devices[0].get("status") != "normal":
    raise SystemExit(f"RBLN device {device_id} is unavailable")
for context in doc.get("contexts", []):
    marker = str(context.get("npu", context.get("device", ""))).lower()
    if marker in {device_id, f"rbln{device_id}"}:
        raise SystemExit(f"RBLN device {device_id} already has a context")
' "$RBLN_DEVICE" || \
    die "RBLN device $RBLN_DEVICE is not available for an isolated run"

mkdir -p -- "$RESULT_ROOT"
RESULT_ROOT=$(cd -- "$RESULT_ROOT" >/dev/null 2>&1 && pwd -P) || \
    die "cannot resolve RESULT_ROOT"
if [[ "${PDD_START_LOCK_HELD:-0}" != 1 ]]; then
    SELF=$(realpath -e -- "$0") || die "cannot resolve start_pdd.sh path"
    exec flock --exclusive --nonblock --close "$RESULT_ROOT" \
        env PDD_START_LOCK_HELD=1 bash "$SELF" "$@"
fi
unset PDD_START_LOCK_HELD

STAMP=$(date +%Y%m%d-%H%M%S)
RESULT_DIR="$RESULT_ROOT/native-dtype-$STAMP"
if [[ -n "$PDD_RUN_TOKEN" ]]; then
    RESULT_DIR+="-$PDD_RUN_TOKEN"
fi
if [[ -e "$RESULT_DIR" ]]; then
    RESULT_DIR="$RESULT_ROOT/native-dtype-$STAMP-$$"
    if [[ -n "$PDD_RUN_TOKEN" ]]; then
        RESULT_DIR+="-$PDD_RUN_TOKEN"
    fi
fi
mkdir -- "$RESULT_DIR"
mkdir -- "$RESULT_DIR/cache" "$RESULT_DIR/cache/cuda" "$RESULT_DIR/cache/rbln"

read -r PREFILL_PORT DECODE_PORT PROXY_PORT PREFILL_SIDE_PORT DECODE_SIDE_PORT < <(
    "$CUDA_PY" -c '
import socket

sockets = []
try:
    for _ in range(5):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        sockets.append(sock)
    ports = [sock.getsockname()[1] for sock in sockets]
    if len(set(ports)) != 5:
        raise RuntimeError("duplicate port allocation")
    print(*ports)
finally:
    for sock in sockets:
        sock.close()
'
)
for port in "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT" \
            "$PREFILL_SIDE_PORT" "$DECODE_SIDE_PORT"; do
    [[ "$port" =~ ^[0-9]+$ ]] || die "free-port allocation failed"
done

RUN_ID="pdd-$STAMP-$$"
PREFILL_ENGINE_ID="cuda-prefill-$RUN_ID"
DECODE_ENGINE_ID="rbln-decode-$RUN_ID"
PRODUCER_CONFIG=$(printf \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail","engine_id":"%s","kv_connector_extra_config":{"kv_recompute_threshold":0}}' \
    "$PREFILL_ENGINE_ID")
CONSUMER_CONFIG=$(printf \
    '{"kv_connector":"RblnNixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","engine_id":"%s","kv_connector_extra_config":{"kv_recompute_threshold":0,"remote_nixl_memory_type":"VRAM"}}' \
    "$DECODE_ENGINE_ID")

RUNTIME_FILE="$RESULT_DIR/runtime.env"
PIDS_FILE="$RESULT_DIR/pids.env"
: >"$RUNTIME_FILE"
: >"$PIDS_FILE"
for item in \
    "RESULT_DIR=$RESULT_DIR" \
    "RESULT_ROOT=$RESULT_ROOT" \
    "REPO=$REPO" \
    "BRANCH=$BRANCH" \
    "COMMIT=$CURRENT_COMMIT" \
    "REQUIRED_LAYOUT_COMMIT=$REQUIRED_LAYOUT_COMMIT" \
    "CUDA_PY=$CUDA_PY" \
    "RBLN_PY=$RBLN_PY" \
    "CUDA_VLLM_VERSION=$CUDA_VLLM_VERSION" \
    "RBLN_VLLM_VERSION=$RBLN_VLLM_VERSION" \
    "COMPILER_VERSION=$COMPILER_VERSION" \
    "MODEL_PATH=$MODEL_PATH" \
    "MODEL_NAME=$MODEL_NAME" \
    "MODEL_DTYPE=$MODEL_DTYPE" \
    "KV_CACHE_DTYPE=$KV_CACHE_DTYPE" \
    "OFFLINE=$OFFLINE" \
    "PDD_RUN_TOKEN=$PDD_RUN_TOKEN" \
    "CUDA_DEVICE=$CUDA_DEVICE" \
    "RBLN_DEVICE=$RBLN_DEVICE" \
    "PREFILL_PORT=$PREFILL_PORT" \
    "DECODE_PORT=$DECODE_PORT" \
    "PROXY_PORT=$PROXY_PORT" \
    "PREFILL_SIDE_PORT=$PREFILL_SIDE_PORT" \
    "DECODE_SIDE_PORT=$DECODE_SIDE_PORT" \
    "PREFILL_ENGINE_ID=$PREFILL_ENGINE_ID" \
    "DECODE_ENGINE_ID=$DECODE_ENGINE_ID" \
    "PRODUCER_CONFIG=$PRODUCER_CONFIG" \
    "CONSUMER_CONFIG=$CONSUMER_CONFIG" \
    "VLLM_RBLN_PDD_LAYOUT_REORDER=$VLLM_RBLN_PDD_LAYOUT_REORDER" \
    "VLLM_KV_CACHE_LAYOUT=HND" \
    "UCX_NET_DEVICES=$UCX_NET_DEVICES" \
    "PRODUCER_KV_BUFFER_DEVICE=cuda" \
    "CONSUMER_KV_BUFFER_DEVICE=cpu" \
    "PRODUCER_LOCAL_MEMORY_TYPE=VRAM" \
    "CONSUMER_LOCAL_MEMORY_TYPE=DRAM" \
    "CONSUMER_REMOTE_MEMORY_TYPE=VRAM" \
    "TRANSFER_PATH=CUDA_VRAM_TO_RBLN_HOST_DRAM" \
    "PRODUCER_USE_HOST_BUFFER=false" \
    "PRODUCER_APPLICATION_D2H_EXPECTED=0" \
    "PYTHON_BIT_TRANSFORM_EXPECTED=0" \
    "DTYPE_CAST_EXPECTED=0" \
    "BLOCK_SIZE=128" \
    "MAX_NUM_BATCHED_TOKENS=128" \
    "STARTED_AT=$STAMP"; do
    write_env "$RUNTIME_FILE" "${item%%=*}" "${item#*=}"
done
write_env "$PIDS_FILE" RESULT_DIR "$RESULT_DIR"

COMMON_SERVER_ARGS=(
    "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host 127.0.0.1
    --dtype "$MODEL_DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len 1024
    --max-num-batched-tokens 128
    --max-num-seqs 1
    --tensor-parallel-size 1
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --seed 0
)

PREFILL_CMD=(
    "$CUDA_PY" -m vllm.entrypoints.cli.main serve
    "${COMMON_SERVER_ARGS[@]}"
    --port "$PREFILL_PORT"
    --block-size 128
    --num-gpu-blocks-override 32
    --kv-transfer-config "$PRODUCER_CONFIG"
)
DECODE_CMD=(
    "$RBLN_PY" -m vllm.entrypoints.cli.main serve
    "${COMMON_SERVER_ARGS[@]}"
    --port "$DECODE_PORT"
    --block-size 128
    --num-gpu-blocks-override 32
    --kv-transfer-config "$CONSUMER_CONFIG"
)
PROXY_CMD=(
    "$CUDA_PY" "$PROXY_SCRIPT"
    --host 127.0.0.1
    --port "$PROXY_PORT"
    --prefill-host 127.0.0.1
    --prefill-port "$PREFILL_PORT"
    --decode-host 127.0.0.1
    --decode-port "$DECODE_PORT"
)

OFFLINE_ENV=()
if [[ "$OFFLINE" == 1 ]]; then
    OFFLINE_ENV+=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
fi

CUDA_ENV=(env)
while IFS= read -r inherited_name; do
    if [[ "$inherited_name" == RBLN_* || "$inherited_name" == VLLM_RBLN_* ]]; then
        CUDA_ENV+=(-u "$inherited_name")
    fi
done < <(compgen -e)
CUDA_ENV+=(
    -u VLLM_PLUGINS
    -u VLLM_RBLN_ENFORCE_MODEL_FP32
    -u HF_HUB_OFFLINE
    -u TRANSFORMERS_OFFLINE
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    PYTHONPATH="$REPO"
    TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    VLLM_KV_CACHE_LAYOUT=HND
    UCX_NET_DEVICES="$UCX_NET_DEVICES"
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
    VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT"
    VLLM_CACHE_ROOT="$RESULT_DIR/cache/cuda"
    "${OFFLINE_ENV[@]}"
)
RBLN_ENV=(env)
while IFS= read -r inherited_name; do
    if [[ "$inherited_name" == RBLN_* || "$inherited_name" == VLLM_RBLN_* ]]; then
        RBLN_ENV+=(-u "$inherited_name")
    fi
done < <(compgen -e)
RBLN_ENV+=(
    -u VLLM_PLUGINS
    -u VLLM_RBLN_ENFORCE_MODEL_FP32
    -u HF_HUB_OFFLINE
    -u TRANSFORMERS_OFFLINE
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    PYTHONPATH="$REPO"
    TOKENIZERS_PARALLELISM=false
    # UCX must see the source GPU to read the producer's remote VRAM while
    # the RBLN connector still registers its local receive buffers as DRAM.
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    RBLN_DEVICES="$RBLN_DEVICE"
    VLLM_KV_CACHE_LAYOUT=HND
    UCX_NET_DEVICES="$UCX_NET_DEVICES"
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
    VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT"
    VLLM_CACHE_ROOT="$RESULT_DIR/cache/rbln"
    VLLM_RBLN_USE_VLLM_MODEL=1
    VLLM_RBLN_USE_DEVICE_TENSOR=1
    VLLM_RBLN_COMPILE_MODEL=1
    VLLM_RBLN_COMPILE_STRICT_MODE=1
    VLLM_RBLN_SAMPLER=0
    VLLM_RBLN_PDD_LAYOUT_REORDER="$VLLM_RBLN_PDD_LAYOUT_REORDER"
    RBLN_USE_CUSTOM_KERNEL=0
    RBLN_ROOT_IP=127.0.0.1
    RBLN_LOCAL_IP=127.0.0.1
    "${OFFLINE_ENV[@]}"
)

PREFILL_PID=
PREFILL_PGID=
DECODE_PID=
DECODE_PGID=
PROXY_PID=
PROXY_PGID=

launch_group PREFILL "$RESULT_DIR/prefill.log" "${CUDA_ENV[@]}" "${PREFILL_CMD[@]}"
if ! wait_http "http://127.0.0.1:$PREFILL_PORT/health" "$PREFILL_PID" 900 \
        'CUDA prefiller'; then
    tail -n 120 "$RESULT_DIR/prefill.log" >&2 || true
    die "CUDA prefiller did not become ready; run RESULT_DIR=$RESULT_DIR ./stop_pdd.sh"
fi

launch_group DECODE "$RESULT_DIR/decode.log" "${RBLN_ENV[@]}" "${DECODE_CMD[@]}"
if ! wait_http "http://127.0.0.1:$DECODE_PORT/health" "$DECODE_PID" 2400 \
        'RBLN decoder'; then
    tail -n 120 "$RESULT_DIR/decode.log" >&2 || true
    die "RBLN decoder did not become ready; run RESULT_DIR=$RESULT_DIR ./stop_pdd.sh"
fi

launch_group PROXY "$RESULT_DIR/proxy.log" "${CUDA_ENV[@]}" "${PROXY_CMD[@]}"
if ! wait_http "http://127.0.0.1:$PROXY_PORT/healthcheck" "$PROXY_PID" 60 \
        'PDD proxy'; then
    tail -n 120 "$RESULT_DIR/proxy.log" >&2 || true
    die "PDD proxy did not become ready; run RESULT_DIR=$RESULT_DIR ./stop_pdd.sh"
fi

wait_http "http://127.0.0.1:$PREFILL_PORT/health" "$PREFILL_PID" 10 \
    'CUDA prefiller final check' || \
    die "CUDA prefiller did not survive full startup; run RESULT_DIR=$RESULT_DIR ./stop_pdd.sh"
wait_http "http://127.0.0.1:$DECODE_PORT/health" "$DECODE_PID" 10 \
    'RBLN decoder final check' || \
    die "RBLN decoder did not survive full startup; run RESULT_DIR=$RESULT_DIR ./stop_pdd.sh"
wait_http "http://127.0.0.1:$PROXY_PORT/healthcheck" "$PROXY_PID" 10 \
    'PDD proxy final check' || \
    die "PDD proxy did not survive full startup; run RESULT_DIR=$RESULT_DIR ./stop_pdd.sh"

printf 'RESULT_DIR=%s\n' "$RESULT_DIR"
printf 'PREFILL_PID=%s\n' "$PREFILL_PID"
printf 'DECODE_PID=%s\n' "$DECODE_PID"
printf 'PROXY_PID=%s\n' "$PROXY_PID"
printf 'PREFILL_PORT=%s\n' "$PREFILL_PORT"
printf 'DECODE_PORT=%s\n' "$DECODE_PORT"
printf 'PROXY_PORT=%s\n' "$PROXY_PORT"
printf 'COMPILER_VERSION=%s\n' "$COMPILER_VERSION"
printf 'COMMIT=%s\n' "$CURRENT_COMMIT"
printf 'LAYOUT_REORDER_ENABLED=%s\n' "$VLLM_RBLN_PDD_LAYOUT_REORDER"
