#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_REPO=/home/jiwon_lee/sota/vllm-rbln-0.11.1-2_native_dtype
readonly EXPECTED_COMMIT=972ce20cf53b0d2d50c4155f9d44be6879ede966
readonly RESULT_ROOT=/home/jiwon_lee/sota/profile-results
readonly DEFAULT_CUDA_PY=/home/jiwon_lee/.venvs/official0111-cuda-20260903-121127/bin/python3

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$EXPECTED_REPO"
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

if [[ -z "${RESULT_DIR:-}" ]]; then
    RESULT_DIR=$(latest_result_dir) || die "no native-dtype result directory exists"
fi
RESULT_DIR=$(realpath -e -- "$RESULT_DIR") || die "RESULT_DIR does not exist"
[[ "$RESULT_DIR" == "$RESULT_ROOT"/native-dtype-* ]] || \
    die "RESULT_DIR is outside the native-dtype result root"
[[ -f "$RESULT_DIR/runtime.env" ]] || die "runtime.env is missing in RESULT_DIR"
SELECTED_RESULT_DIR=$RESULT_DIR

set -a
# shellcheck disable=SC1090
source "$RESULT_DIR/runtime.env"
set +a
[[ "$RESULT_DIR" == "$SELECTED_RESULT_DIR" ]] || die "runtime.env RESULT_DIR mismatch"
[[ "${REPO:-}" == "$EXPECTED_REPO" ]] || die "runtime candidate repository mismatch"
[[ "${COMMIT:-}" == "$EXPECTED_COMMIT" ]] || die "runtime candidate commit mismatch"
[[ "${VLLM_RBLN_PDD_LAYOUT_REORDER:-}" == 1 ]] || die "layout reorder is not enabled"
[[ "${PROXY_PORT:-}" =~ ^[0-9]+$ ]] || die "invalid proxy port"
CUDA_PY=${CUDA_PY:-$DEFAULT_CUDA_PY}
[[ -x "$CUDA_PY" ]] || die "CUDA Python is unavailable"

export RESULT_DIR PROXY_PORT MODEL_NAME
"$CUDA_PY" - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

result_dir = os.environ["RESULT_DIR"]
proxy_port = int(os.environ["PROXY_PORT"])
model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
output_path = os.path.join(result_dir, "correctness_response.json")

prompt = "Complete the sequence: 1, 1, 2, 3, 5,"
prompt_sha256 = "99c6c5a7e43544bac9573914b469b0391ad40d529fb1ad137276bef173dc7116"
expected_text = " 8, 13, 21, 34, 55, 89, 144, 233"
expected_ids = [
    220, 23, 11, 220, 16, 18, 11, 220,
    17, 16, 11, 220, 18, 19, 11, 220,
    20, 20, 11, 220, 23, 24, 11, 220,
    16, 19, 19, 11, 220, 17, 18, 18,
]
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
        "X-Request-Id": f"profile-correctness-{os.getpid()}",
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
if not actual_ids or actual_ids[0] != 220:
    errors.append(f"first token was {actual_ids[0] if actual_ids else None!r}")
if actual_ids != expected_ids:
    errors.append(f"token IDs were not exact ({len(actual_ids)}/32 returned)")
if actual_text != expected_text:
    errors.append(f"text mismatch: {actual_text!r}")

if errors:
    for error in errors:
        print(f"CORRECTNESS_FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)

print("CORRECTNESS_PASS")
print("FIRST_TOKEN=220")
print("TOKEN_EXACT=32/32")
PY
