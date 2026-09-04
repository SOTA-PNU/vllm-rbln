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
PROMPT_ID="${PROMPT_ID:-2}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

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

[[ "$PROMPT_ID" == 2 || "$PROMPT_ID" == 4 ]] || \
    die "PROMPT_ID must be 2 or 4"
if [[ -z "${RESULT_DIR:-}" ]]; then
    RESULT_DIR=$(latest_result_dir) || die "no native-dtype result directory exists"
fi
RESULT_DIR=$(realpath -e -- "$RESULT_DIR") || die "RESULT_DIR does not exist"
[[ -f "$RESULT_DIR/runtime.env" ]] || die "runtime.env is missing in RESULT_DIR"
[[ -f "$RESULT_DIR/pids.env" ]] || die "pids.env is missing in RESULT_DIR"
SELECTED_RESULT_DIR=$RESULT_DIR

set -a
# shellcheck disable=SC1090
source "$RESULT_DIR/runtime.env"
# shellcheck disable=SC1090
source "$RESULT_DIR/pids.env"
set +a
[[ "${RESULT_DIR:-}" == "$SELECTED_RESULT_DIR" ]] || die "recorded RESULT_DIR mismatch"
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
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
    die "recorded repository working tree is not clean"

[[ "${PRODUCER_KV_BUFFER_DEVICE:-}" == cuda ]] || die "producer buffer is not CUDA"
[[ "${CONSUMER_KV_BUFFER_DEVICE:-}" == cpu ]] || die "consumer buffer is not CPU"
[[ "${PRODUCER_LOCAL_MEMORY_TYPE:-}" == VRAM ]] || die "producer memory is not VRAM"
[[ "${CONSUMER_LOCAL_MEMORY_TYPE:-}" == DRAM ]] || die "consumer local memory is not DRAM"
[[ "${CONSUMER_REMOTE_MEMORY_TYPE:-}" == VRAM ]] || die "consumer remote memory is not VRAM"
[[ "${TRANSFER_PATH:-}" == CUDA_VRAM_TO_RBLN_HOST_DRAM ]] || die "transfer path mismatch"
[[ "${PRODUCER_USE_HOST_BUFFER:-}" == false ]] || die "producer host buffer must be disabled"
[[ "${VLLM_RBLN_PDD_LAYOUT_REORDER:-}" == 0 || \
   "${VLLM_RBLN_PDD_LAYOUT_REORDER:-}" == 1 ]] || die "invalid layout reorder mode"
[[ "${PROXY_PORT:-}" =~ ^[0-9]+$ ]] || die "invalid proxy port"
[[ -x "${CUDA_PY:-}" ]] || die "recorded CUDA_PY is unavailable"
for pid in "${PREFILL_PID:-}" "${DECODE_PID:-}" "${PROXY_PID:-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || die "pids.env contains an invalid PID"
    kill -0 "$pid" 2>/dev/null || die "recorded server PID $pid is not running"
done

successful_transfer_reports_before=$(
    awk \
        '/KV Transfer metrics: Num successful transfers=[1-9][0-9]*/ {count++}
         END {print count + 0}' \
        "$RESULT_DIR/decode.log"
)

export RESULT_DIR PROXY_PORT MODEL_NAME MODEL_DTYPE PROMPT_ID
"$CUDA_PY" - <<'PY'
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

result_dir = os.environ["RESULT_DIR"]
proxy_port = int(os.environ["PROXY_PORT"])
model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
model_dtype = os.environ.get("MODEL_DTYPE", "unknown")
prompt_id = int(os.environ["PROMPT_ID"])

if prompt_id == 2:
    prompt = "Complete the sequence: 1, 1, 2, 3, 5,"
    prompt_sha256 = "99c6c5a7e43544bac9573914b469b0391ad40d529fb1ad137276bef173dc7116"
    expected_text = " 8, 13, 21, 34, 55, 89, 144, 233"
    expected_ids = [
        220, 23, 11, 220, 16, 18, 11, 220,
        17, 16, 11, 220, 18, 19, 11, 220,
        20, 20, 11, 220, 23, 24, 11, 220,
        16, 19, 19, 11, 220, 17, 18, 18,
    ]
else:
    prompt = (
        "A clear glass of water rests on the wooden table. " * 29
        + "Continue with the next three words only:"
    )
    prompt_sha256 = "57242b91f80632b4eec84b172bfe170fa52b0d33940af0b59a9a43be51889921"
    expected_text = None
    expected_ids = [
        279, 1790, 2326, 4244, 525, 1112, 30, 576,
        4226, 374, 330, 1782, 1790, 2326, 4244, 525,
        1112, 3263, 576, 4226, 374, 330, 1782, 1790,
        2326, 4244, 525, 1112, 3263, 576, 4226, 374,
    ]

actual_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
if actual_prompt_sha256 != prompt_sha256:
    raise SystemExit("embedded prompt checksum mismatch")

output_path = os.path.join(
    result_dir, f"correctness_{model_dtype}_prompt{prompt_id}_response.json"
)
body = {
    "model": model,
    "prompt": prompt,
    "temperature": 0,
    "top_p": 1,
    "seed": 0,
    "max_tokens": 32,
    "logprobs": 5,
    "n": 1,
    "return_token_ids": True,
    "return_tokens_as_token_ids": True,
    "stream": False,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{proxy_port}/v1/completions",
    data=json.dumps(body).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
        "X-Request-Id": f"pdd-correctness-p{prompt_id}-{os.getpid()}",
        "X-Prompt-Sha256": prompt_sha256,
    },
    method="POST",
)

status = 0
payload = None
try:
    with urllib.request.urlopen(request, timeout=900) as response:
        status = response.status
        raw = response.read()
except urllib.error.HTTPError as exc:
    status = exc.code
    raw = exc.read()
except Exception as exc:
    raw = b""
    payload = {"transport_error": f"{type(exc).__name__}: {exc}"}

if payload is None:
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {
            "http_status": status,
            "non_json_body": raw.decode("utf-8", errors="replace"),
        }

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

errors = []
if not 200 <= status < 300:
    errors.append(f"HTTP status was {status}")
choices = payload.get("choices") if isinstance(payload, dict) else None
choice = choices[0] if isinstance(choices, list) and choices else {}
actual_text = choice.get("text") if isinstance(choice, dict) else None
actual_ids = choice.get("token_ids") if isinstance(choice, dict) else None
if not isinstance(actual_ids, list) or not all(
    isinstance(token_id, int) and not isinstance(token_id, bool)
    for token_id in actual_ids
):
    errors.append("choices[0].token_ids is not an integer array")
    actual_ids = []
if not actual_ids or actual_ids[0] != expected_ids[0]:
    errors.append(f"first token was {actual_ids[0] if actual_ids else None!r}")
if actual_ids != expected_ids:
    matching = sum(a == b for a, b in zip(actual_ids, expected_ids))
    errors.append(f"token IDs were not exact ({matching}/32 positions matched)")
if expected_text is not None and actual_text != expected_text:
    errors.append(f"text mismatch: {actual_text!r}")

if errors:
    for error in errors:
        print(f"CORRECTNESS_FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)

print("CORRECTNESS_PASS")
print(f"PROMPT_ID={prompt_id}")
print(f"FIRST_TOKEN={expected_ids[0]}")
print("TOKEN_EXACT=32/32")
if expected_text is not None:
    print("TEXT_EXACT=PASS")
print(f"RESPONSE_FILE={output_path}")
PY

transfer_deadline=$((SECONDS + 30))
transfer_completed=0
while (( SECONDS < transfer_deadline )); do
    successful_transfer_reports_after=$(
        awk \
            '/KV Transfer metrics: Num successful transfers=[1-9][0-9]*/ {count++}
             END {print count + 0}' \
            "$RESULT_DIR/decode.log"
    )
    if (( successful_transfer_reports_after > successful_transfer_reports_before )); then
        transfer_completed=1
        break
    fi
    sleep 1
done
(( transfer_completed == 1 )) || \
    die "decode log does not confirm a successful NIXL transfer"

if grep -nEi \
    'NIXL_ERR_NOT_FOUND|DRAM_SEG lookup for CUDA|remote VRAM registration.*fail|backend mismatch|failed receive|recompute fallback|NIXL transfer failure' \
    "$RESULT_DIR/prefill.log" "$RESULT_DIR/decode.log" "$RESULT_DIR/proxy.log"; then
    die "transfer logs contain a prohibited error"
fi
printf 'NIXL_TRANSFER_COMPLETION=PASS\n'
