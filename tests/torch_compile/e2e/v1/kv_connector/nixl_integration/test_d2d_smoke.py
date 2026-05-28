# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0

# SPDX-License-Identifier: Apache-2.0
"""Smoke check for ``RblnNixlDirectConnector`` D2D RDMA KV transfer.

Same prefill / decode / proxy topology that ``test_accuracy.py`` runs
under, but skips the ``lm_eval`` gsm8k pass and just sends one greedy
completion against the proxy. If D2D RDMA truncated or corrupted any
bytes of the transferred KV cache the decode worker will produce
garbage logits — greedy decoding off a deterministic prompt makes that
observable without scoring an accuracy benchmark.

Use this as the fast PR gate; keep ``test_accuracy.py`` as the
nightly regressing on real gsm8k accuracy.

Invoke after the same prefill+decode+proxy stack ``test_accuracy.py``
expects (i.e. proxy listening on :8192) is already up — typically by
running whichever ``run_accuracy_test.*.sh`` flavor brought it up and
then pointing pytest at this file:

    TEST_MODEL=Qwen/Qwen3-0.6B python3 -m pytest -s -x test_d2d_smoke.py
"""

import os

import openai

BASE_URL = "http://localhost:8192/v1"

# Same env var the accuracy test reads, so the shell harness can keep
# passing TEST_MODEL=<model> unchanged.
MODEL_NAME = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")

# A factual, common-knowledge prompt where any 0.6B-scale chat/base LM
# fine-tuned on web text will greedy-decode the same continuation.
# Sensitivity to RDMA byte corruption: any KV cache mismatch makes the
# first emitted token diverge from the expected word.
PROMPT = "The capital city of France is"
EXPECTED_WORD = "Paris"
MAX_TOKENS = 16


def test_d2d_kv_transfer_smoke():
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,  # greedy -> deterministic; same prompt -> same tokens
    )
    text = resp.choices[0].text
    print("-" * 50)
    print(f"Model:      {MODEL_NAME}")
    print(f"Prompt:     {PROMPT!r}")
    print(f"Completion: {text!r}")
    print("-" * 50)

    assert text, (
        "Empty completion text — the prefill -> decode pipeline returned "
        "no tokens. Check the proxy + worker logs; usually means an "
        "earlier RDMA failure aborted the request mid-flight."
    )
    assert EXPECTED_WORD in text, (
        f"Greedy continuation of {PROMPT!r} doesn't contain "
        f"{EXPECTED_WORD!r}; D2D KV transfer most likely corrupted the "
        f"prefill cache before the decode worker read it.\n"
        f"  got: {text!r}"
    )
