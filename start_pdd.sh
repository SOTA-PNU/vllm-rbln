#!/usr/bin/env bash
set -Eeuo pipefail

umask 022

readonly REPO=/home/jiwon_lee/sota/vllm-rbln-0.11.1-2_native_dtype
readonly EXPECTED_BRANCH=0.11.1-2_native_dtype
readonly EXPECTED_COMMIT=972ce20cf53b0d2d50c4155f9d44be6879ede966
readonly CUDA_VENV=/home/jiwon_lee/.venvs/official0111-cuda-20260903-121127
readonly RBLN_VENV=/home/jiwon_lee/.venvs/official0111-rbln-20260903-121127
readonly CUDA_PY="$CUDA_VENV/bin/python3"
readonly RBLN_PY="$RBLN_VENV/bin/python3"
readonly EXPECTED_CUDA_VLLM=0.22.0
readonly EXPECTED_RBLN_VLLM=0.22.0+cpu
readonly EXPECTED_COMPILER=0.11.1.post2.dev2+g2995098f.prod
readonly MODEL_NAME=Qwen/Qwen3-0.6B
readonly MODEL_SNAPSHOT=/home/jiwon_lee/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
readonly PROXY_SCRIPT=/home/jiwon_lee/sota/experiments/official0111-layout-only-20260903-151328/layout_proxy.py
readonly EXPECTED_PROXY_SHA256=40325ec2df38868e4f1bbdecb5e0ec7ec32b8d4f62a05ed50e04221f8d908d18
readonly FP16_CONFIG=/home/jiwon_lee/sota/experiments/official0111-layout-only-20260903-151328/layout_fp16_effective_config.json
readonly EXPECTED_FP16_CONFIG_SHA256=350dcb296f155225aa2a5bffaa489264c660a7eb4e999c4c3d8512dc0f29952c
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

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
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

[[ -d "$REPO/.git" ]] || die "candidate repository is missing: $REPO"
[[ -x "$CUDA_PY" ]] || die "CUDA Python is missing: $CUDA_PY"
[[ -x "$RBLN_PY" ]] || die "RBLN Python is missing: $RBLN_PY"
[[ -d "$MODEL_SNAPSHOT" ]] || die "model snapshot is missing: $MODEL_SNAPSHOT"
[[ -f "$PROXY_SCRIPT" ]] || die "validated proxy is missing: $PROXY_SCRIPT"
[[ -f "$FP16_CONFIG" ]] || die "validated FP16 config is missing: $FP16_CONFIG"
command -v curl >/dev/null 2>&1 || die "curl is required for readiness checks"
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group isolation"
command -v flock >/dev/null 2>&1 || die "flock is required for launch serialization"
[[ "$(sha256sum "$PROXY_SCRIPT" | awk '{print $1}')" == "$EXPECTED_PROXY_SHA256" ]] || \
    die "validated proxy checksum mismatch"
[[ "$(sha256sum "$FP16_CONFIG" | awk '{print $1}')" == "$EXPECTED_FP16_CONFIG_SHA256" ]] || \
    die "validated FP16 config checksum mismatch"

mkdir -p "$RESULT_ROOT"
if [[ "${PDD_START_LOCK_HELD:-0}" != 1 ]]; then
    SELF=$(realpath -e -- "$0") || die "cannot resolve start_pdd.sh path"
    exec flock --exclusive --nonblock --close "$RESULT_ROOT" \
        env PDD_START_LOCK_HELD=1 bash "$SELF" "$@"
fi
unset PDD_START_LOCK_HELD

COMMIT=$(git -C "$REPO" rev-parse HEAD)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ "$COMMIT" == "$EXPECTED_COMMIT" ]] || \
    die "candidate commit mismatch: expected $EXPECTED_COMMIT, got $COMMIT"
[[ "$BRANCH" == "$EXPECTED_BRANCH" ]] || \
    die "candidate branch mismatch: expected $EXPECTED_BRANCH, got $BRANCH"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
    die "candidate repository working tree is not clean"

COMPILER_VERSION=$(
    "$RBLN_PY" -c 'import importlib.metadata as m; print(m.version("rebel-compiler"))'
)
CUDA_VLLM_VERSION=$(
    "$CUDA_PY" -c 'import importlib.metadata as m; print(m.version("vllm"))'
)
RBLN_VLLM_VERSION=$(
    "$RBLN_PY" -c 'import importlib.metadata as m; print(m.version("vllm"))'
)
[[ "$COMPILER_VERSION" == "$EXPECTED_COMPILER" ]] || \
    die "compiler mismatch: expected $EXPECTED_COMPILER, got $COMPILER_VERSION"
[[ "$CUDA_VLLM_VERSION" == "$EXPECTED_CUDA_VLLM" ]] || \
    die "CUDA vLLM mismatch: expected $EXPECTED_CUDA_VLLM, got $CUDA_VLLM_VERSION"
[[ "$RBLN_VLLM_VERSION" == "$EXPECTED_RBLN_VLLM" ]] || \
    die "RBLN vLLM mismatch: expected $EXPECTED_RBLN_VLLM, got $RBLN_VLLM_VERSION"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
CUDA_GPU_LINE=$(nvidia-smi -i 0 --query-gpu=index,uuid,name --format=csv,noheader,nounits) || \
    die "CUDA device 0 is unavailable"
[[ "$CUDA_GPU_LINE" == 0,* ]] || die "CUDA device 0 query returned an unexpected result"
CUDA_GPU_UUID=$(printf '%s\n' "$CUDA_GPU_LINE" | cut -d, -f2 | tr -d ' ')
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | \
    tr -d ' ' | grep -Fxq "$CUDA_GPU_UUID"; then
    die "CUDA device 0 already has a compute process; no existing process was changed"
fi

command -v rbln-smi >/dev/null 2>&1 || die "rbln-smi is unavailable"
RBLN_STATE=$(rbln-smi --json) || die "RBLN device query failed"
printf '%s' "$RBLN_STATE" | "$CUDA_PY" -c '
import json, sys
doc = json.load(sys.stdin)
devices = [d for d in doc.get("devices", []) if str(d.get("npu")) == "0"]
if len(devices) != 1 or devices[0].get("status") != "normal":
    raise SystemExit("RBLN device 0 is unavailable")
for context in doc.get("contexts", []):
    marker = str(context.get("npu", context.get("device", ""))).lower()
    if marker in {"0", "rbln0"}:
        raise SystemExit("RBLN device 0 already has a context")
' || die "RBLN device 0 is not available for an isolated run"

STAMP=$(date +%Y%m%d-%H%M%S)
RESULT_DIR="$RESULT_ROOT/native-dtype-$STAMP"
if [[ -e "$RESULT_DIR" ]]; then
    RESULT_DIR="$RESULT_ROOT/native-dtype-$STAMP-$$"
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

RUN_ID="profile-native-dtype-$STAMP-$$"
PREFILL_ENGINE_ID="official0111-cuda-prefill-$RUN_ID"
DECODE_ENGINE_ID="official0111-rbln-decode-$RUN_ID"
PRODUCER_CONFIG=$(printf \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","engine_id":"%s","kv_connector_extra_config":{"kv_recompute_threshold":0}}' \
    "$PREFILL_ENGINE_ID")
CONSUMER_CONFIG=$(printf \
    '{"kv_connector":"RblnNixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","engine_id":"%s","kv_connector_extra_config":{"kv_recompute_threshold":0}}' \
    "$DECODE_ENGINE_ID")

RUNTIME_FILE="$RESULT_DIR/runtime.env"
PIDS_FILE="$RESULT_DIR/pids.env"
: >"$RUNTIME_FILE"
: >"$PIDS_FILE"
for item in \
    "RESULT_DIR=$RESULT_DIR" \
    "REPO=$REPO" \
    "BRANCH=$BRANCH" \
    "COMMIT=$COMMIT" \
    "CUDA_VENV=$CUDA_VENV" \
    "RBLN_VENV=$RBLN_VENV" \
    "CUDA_PY=$CUDA_PY" \
    "RBLN_PY=$RBLN_PY" \
    "CUDA_VLLM_VERSION=$CUDA_VLLM_VERSION" \
    "RBLN_VLLM_VERSION=$RBLN_VLLM_VERSION" \
    "COMPILER_VERSION=$COMPILER_VERSION" \
    "MODEL_NAME=$MODEL_NAME" \
    "MODEL_SNAPSHOT=$MODEL_SNAPSHOT" \
    "MODEL_DTYPE=float16" \
    "KV_CACHE_DTYPE=float16" \
    "PREFILL_PORT=$PREFILL_PORT" \
    "DECODE_PORT=$DECODE_PORT" \
    "PROXY_PORT=$PROXY_PORT" \
    "PREFILL_SIDE_PORT=$PREFILL_SIDE_PORT" \
    "DECODE_SIDE_PORT=$DECODE_SIDE_PORT" \
    "PREFILL_ENGINE_ID=$PREFILL_ENGINE_ID" \
    "DECODE_ENGINE_ID=$DECODE_ENGINE_ID" \
    "PYTHONNOUSERSITE=1" \
    "PYTHONPATH=$REPO" \
    "VLLM_RBLN_PDD_LAYOUT_REORDER=1" \
    "HF_HUB_OFFLINE=1" \
    "TRANSFORMERS_OFFLINE=1" \
    "TOKENIZERS_PARALLELISM=false" \
    "CUDA_VISIBLE_DEVICES=0" \
    "RBLN_DEVICES=0" \
    "VLLM_KV_CACHE_LAYOUT=HND" \
    "UCX_NET_DEVICES=all" \
    "PRODUCER_KV_BUFFER_DEVICE=cpu" \
    "CONSUMER_KV_BUFFER_DEVICE=cpu" \
    "BLOCK_SIZE=128" \
    "MAX_NUM_BATCHED_TOKENS=128" \
    "STARTED_AT=$STAMP"; do
    write_env "$RUNTIME_FILE" "${item%%=*}" "${item#*=}"
done
write_env "$PIDS_FILE" RESULT_DIR "$RESULT_DIR"

COMMON_SERVER_ARGS=(
    "$MODEL_SNAPSHOT"
    --served-model-name "$MODEL_NAME"
    --host 127.0.0.1
    --dtype float16
    --kv-cache-dtype float16
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
    --prefill-url "http://127.0.0.1:$PREFILL_PORT"
    --decode-url "http://127.0.0.1:$DECODE_PORT"
    --event-log "$RESULT_DIR/proxy_events.jsonl"
)

CUDA_ENV=(env)
while IFS='=' read -r inherited_name inherited_value; do
    if [[ "$inherited_name" == RBLN_* || "$inherited_name" == VLLM_RBLN_* ]]; then
        CUDA_ENV+=(-u "$inherited_name")
    fi
done < <(env)
CUDA_ENV+=(
    -u VLLM_PLUGINS
    -u VLLM_RBLN_ENFORCE_MODEL_FP32
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    PYTHONPATH="$REPO"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES=0
    VLLM_KV_CACHE_LAYOUT=HND
    UCX_NET_DEVICES=all
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
    VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT"
    VLLM_CACHE_ROOT="$RESULT_DIR/cache/cuda"
)
RBLN_ENV=(env
    -u VLLM_PLUGINS
    -u VLLM_RBLN_ENFORCE_MODEL_FP32
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    PYTHONPATH="$REPO"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES=
    RBLN_DEVICES=0
    VLLM_KV_CACHE_LAYOUT=HND
    UCX_NET_DEVICES=all
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
    VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT"
    VLLM_CACHE_ROOT="$RESULT_DIR/cache/rbln"
    VLLM_RBLN_USE_VLLM_MODEL=1
    VLLM_RBLN_USE_DEVICE_TENSOR=1
    VLLM_RBLN_COMPILE_MODEL=1
    VLLM_RBLN_COMPILE_STRICT_MODE=1
    VLLM_RBLN_SAMPLER=0
    VLLM_RBLN_PDD_LAYOUT_REORDER=1
    RBLN_USE_CUSTOM_KERNEL=0
    RBLN_ROOT_IP=127.0.0.1
    RBLN_LOCAL_IP=127.0.0.1
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
    die "CUDA prefiller did not become ready; use stop_pdd.sh with RESULT_DIR=$RESULT_DIR"
fi

launch_group DECODE "$RESULT_DIR/decode.log" "${RBLN_ENV[@]}" "${DECODE_CMD[@]}"
if ! wait_http "http://127.0.0.1:$DECODE_PORT/health" "$DECODE_PID" 2400 \
        'RBLN decoder'; then
    tail -n 120 "$RESULT_DIR/decode.log" >&2 || true
    die "RBLN decoder did not become ready; use stop_pdd.sh with RESULT_DIR=$RESULT_DIR"
fi

launch_group PROXY "$RESULT_DIR/proxy.log" "${CUDA_ENV[@]}" "${PROXY_CMD[@]}"
if ! wait_http "http://127.0.0.1:$PROXY_PORT/healthcheck" "$PROXY_PID" 60 \
        'PDD proxy'; then
    tail -n 120 "$RESULT_DIR/proxy.log" >&2 || true
    die "PDD proxy did not become ready; use stop_pdd.sh with RESULT_DIR=$RESULT_DIR"
fi

wait_http "http://127.0.0.1:$PREFILL_PORT/health" "$PREFILL_PID" 10 \
    'CUDA prefiller final check' || \
    die "CUDA prefiller did not survive full startup; use stop_pdd.sh with RESULT_DIR=$RESULT_DIR"
wait_http "http://127.0.0.1:$DECODE_PORT/health" "$DECODE_PID" 10 \
    'RBLN decoder final check' || \
    die "RBLN decoder did not survive full startup; use stop_pdd.sh with RESULT_DIR=$RESULT_DIR"
wait_http "http://127.0.0.1:$PROXY_PORT/healthcheck" "$PROXY_PID" 10 \
    'PDD proxy final check' || \
    die "PDD proxy did not survive full startup; use stop_pdd.sh with RESULT_DIR=$RESULT_DIR"

printf 'RESULT_DIR=%s\n' "$RESULT_DIR"
printf 'PREFILL_PID=%s\n' "$PREFILL_PID"
printf 'DECODE_PID=%s\n' "$DECODE_PID"
printf 'PROXY_PID=%s\n' "$PROXY_PID"
printf 'PREFILL_PORT=%s\n' "$PREFILL_PORT"
printf 'DECODE_PORT=%s\n' "$DECODE_PORT"
printf 'PROXY_PORT=%s\n' "$PROXY_PORT"
printf 'COMPILER_VERSION=%s\n' "$COMPILER_VERSION"
printf 'COMMIT=%s\n' "$COMMIT"
printf 'LAYOUT_REORDER_ENABLED=%s\n' "$VLLM_RBLN_PDD_LAYOUT_REORDER"
