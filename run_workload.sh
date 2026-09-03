#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_REPO=/home/jiwon_lee/sota/vllm-rbln-0.11.1-2_native_dtype
readonly EXPECTED_COMMIT=972ce20cf53b0d2d50c4155f9d44be6879ede966
readonly RESULT_ROOT=/home/jiwon_lee/sota/profile-results
readonly PROMPT_SOURCE=/home/jiwon_lee/sota/experiments/official0111-native-kv-20260903-121127/12_prompts.json
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

WARMUP_REQUESTS=${WARMUP_REQUESTS:-3}
MEASURE_REQUESTS=${MEASURE_REQUESTS:-20}
CONCURRENCY=${CONCURRENCY:-1}
MAX_TOKENS=${MAX_TOKENS:-128}
PROMPT_MODE=${PROMPT_MODE:-long}

[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be a nonnegative integer"
[[ "$MEASURE_REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "MEASURE_REQUESTS must be a positive integer"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || die "CONCURRENCY must be a positive integer"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS must be a positive integer"
[[ "$PROMPT_MODE" == short || "$PROMPT_MODE" == long ]] || \
    die "PROMPT_MODE must be short or long"
[[ -f "$PROMPT_SOURCE" ]] || die "prompt source is missing"

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
for port in "${PREFILL_PORT:-}" "${DECODE_PORT:-}" "${PROXY_PORT:-}"; do
    [[ "$port" =~ ^[0-9]+$ ]] || die "runtime.env contains an invalid HTTP port"
done
CUDA_PY=${CUDA_PY:-$DEFAULT_CUDA_PY}
[[ -x "$CUDA_PY" ]] || die "CUDA Python is unavailable"

export RESULT_DIR PREFILL_PORT DECODE_PORT PROXY_PORT MODEL_NAME PROMPT_SOURCE
export WARMUP_REQUESTS MEASURE_REQUESTS CONCURRENCY MAX_TOKENS PROMPT_MODE
"$CUDA_PY" - <<'PY'
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

result_dir = os.environ["RESULT_DIR"]
prefill_port = int(os.environ["PREFILL_PORT"])
decode_port = int(os.environ["DECODE_PORT"])
proxy_port = int(os.environ["PROXY_PORT"])
model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
prompt_source = os.environ["PROMPT_SOURCE"]
warmup_requests = int(os.environ["WARMUP_REQUESTS"])
measure_requests = int(os.environ["MEASURE_REQUESTS"])
concurrency = int(os.environ["CONCURRENCY"])
max_tokens = int(os.environ["MAX_TOKENS"])
prompt_mode = os.environ["PROMPT_MODE"]
results_path = os.path.join(result_dir, "workload_results.jsonl")
summary_path = os.path.join(result_dir, "workload_summary.txt")
run_id = f"profile-workload-{uuid.uuid4()}"

with open(prompt_source, encoding="utf-8") as handle:
    prompt_doc = json.load(handle)
prompt_id = 2 if prompt_mode == "short" else 4
matches = [row for row in prompt_doc.get("prompts", []) if int(row.get("id", -1)) == prompt_id]
if len(matches) != 1:
    raise SystemExit(f"prompt {prompt_id} is missing or duplicated in {prompt_source}")
prompt_row = matches[0]
prompt = prompt_row["prompt"]
prompt_sha256 = prompt_row["sha256_utf8"]
expected_prompt_sha256 = {
    2: "99c6c5a7e43544bac9573914b469b0391ad40d529fb1ad137276bef173dc7116",
    4: "57242b91f80632b4eec84b172bfe170fa52b0d33940af0b59a9a43be51889921",
}[prompt_id]
actual_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
if prompt_sha256 != expected_prompt_sha256 or actual_prompt_sha256 != expected_prompt_sha256:
    raise SystemExit(f"prompt {prompt_id} content/checksum differs from the validated source")

def get_ready(url: str, label: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(f"{label} readiness failed at {url}: {exc}") from exc

get_ready(f"http://127.0.0.1:{prefill_port}/health", "CUDA prefiller")
get_ready(f"http://127.0.0.1:{decode_port}/health", "RBLN decoder")
get_ready(f"http://127.0.0.1:{proxy_port}/healthcheck", "PDD proxy")

body = {
    "model": model,
    "prompt": prompt,
    "temperature": 0,
    "top_p": 1,
    "seed": 0,
    "max_tokens": max_tokens,
    "logprobs": 5,
    "n": 1,
    "return_token_ids": True,
    "return_tokens_as_token_ids": True,
    "stream": False,
}
body_bytes = json.dumps(body).encode("utf-8")

def issue_request(phase: str, index: int) -> dict[str, object]:
    request_id = f"{run_id}-{phase}-{index}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/completions",
        data=body_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "X-Request-Id": request_id,
            "X-Prompt-Sha256": prompt_sha256,
        },
        method="POST",
    )
    started = time.monotonic()
    status = 0
    payload: object = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            status = response.status
            raw = response.read()
        payload = json.loads(raw)
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"non_json_body": raw.decode("utf-8", errors="replace")}
        error = f"HTTP {status}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()

    token_ids = None
    generated_text = None
    finish_reason = None
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            generated_text = choices[0].get("text")
            finish_reason = choices[0].get("finish_reason")
            candidate = choices[0].get("token_ids")
            if isinstance(candidate, list) and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in candidate
            ):
                token_ids = candidate
    if 200 <= status < 300 and (token_ids is None or not token_ids):
        error = "HTTP response omitted choices[0].token_ids"

    return {
        "run_id": run_id,
        "phase": phase,
        "request_index": index,
        "request_id": request_id,
        "prompt_mode": prompt_mode,
        "prompt_id": prompt_id,
        "started_monotonic": started,
        "ended_monotonic": ended,
        "latency_seconds": ended - started,
        "http_status": status,
        "generated_token_count": len(token_ids) if token_ids is not None else None,
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "finish_reason": finish_reason,
        "success": error is None and 200 <= status < 300,
        "error": error,
    }

def run_batch(phase: str, count: int) -> list[dict[str, object]]:
    if count == 0:
        return []
    workers = min(concurrency, count)
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(issue_request, phase, index) for index in range(count)]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: int(row["request_index"]))

with open(results_path, "w", encoding="utf-8"):
    pass

def append_rows(rows: list[dict[str, object]]) -> None:
    with open(results_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()

started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
warmup_rows = run_batch("warmup", warmup_requests)
append_rows(warmup_rows)
warmup_failures = [row for row in warmup_rows if not row["success"]]
if warmup_failures:
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"RESULT_DIR={result_dir}\n")
        handle.write(f"RUN_ID={run_id}\n")
        handle.write("STATUS=WARMUP_FAILED\n")
        handle.write(f"WARMUP_FAILED={len(warmup_failures)}\n")
    print(f"WORKLOAD_FAIL: {len(warmup_failures)} warm-up request(s) failed", file=sys.stderr)
    raise SystemExit(1)

print("READY_FOR_PROFILING", flush=True)
time.sleep(5)
measure_wall_started = time.monotonic()
measure_rows = run_batch("measure", measure_requests)
measure_wall_ended = time.monotonic()
append_rows(measure_rows)

passed = [row for row in measure_rows if row["success"]]
failed = [row for row in measure_rows if not row["success"]]
latencies = [float(row["latency_seconds"]) for row in passed]
token_counts = [int(row["generated_token_count"]) for row in passed]
sorted_latencies = sorted(latencies)

def percentile_nearest_rank(values: list[float], percentile: float) -> str:
    if not values:
        return "NA"
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return f"{values[index]:.9f}"

summary = {
    "RESULT_DIR": result_dir,
    "RUN_ID": run_id,
    "STARTED_UTC": started_utc,
    "STATUS": "PASS" if not failed else "FAILED",
    "PROMPT_MODE": prompt_mode,
    "PROMPT_ID": prompt_id,
    "WARMUP_REQUESTS": warmup_requests,
    "MEASURE_REQUESTS": measure_requests,
    "CONCURRENCY": concurrency,
    "MAX_TOKENS": max_tokens,
    "READY_DELAY_SECONDS": 5,
    "MEASURE_SUCCEEDED": len(passed),
    "MEASURE_FAILED": len(failed),
    "MEASURE_WALL_SECONDS": f"{measure_wall_ended - measure_wall_started:.9f}",
    "GENERATED_TOKEN_TOTAL": sum(token_counts),
    "LATENCY_MEAN_SECONDS": f"{statistics.mean(latencies):.9f}" if latencies else "NA",
    "LATENCY_P50_SECONDS": percentile_nearest_rank(sorted_latencies, 0.50),
    "LATENCY_P95_SECONDS": percentile_nearest_rank(sorted_latencies, 0.95),
}
with open(summary_path, "w", encoding="utf-8") as handle:
    for key, value in summary.items():
        handle.write(f"{key}={value}\n")

if failed:
    print(f"WORKLOAD_FAIL: {len(failed)} measurement request(s) failed", file=sys.stderr)
    raise SystemExit(1)
print("WORKLOAD_COMPLETE")
print(f"RESULTS_FILE={results_path}")
print(f"SUMMARY_FILE={summary_path}")
PY
