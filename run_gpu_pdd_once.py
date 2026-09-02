#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path

import run_pdd_once as runner

def _replace(cmd: list[str], option: str, value: str) -> None:
    cmd[cmd.index(option) + 1] = value

def _gpu_decode_cmd(*args):
    cmd = runner._build_prefill_cmd(*args[:6])
    _replace(cmd, "--port", str(runner.DECODE_PORT))
    _replace(cmd, "--gpu-memory-utilization", str(args[7]))
    _replace(cmd, "--tensor-parallel-size", str(args[6]))
    config = cmd[cmd.index("--kv-transfer-config") + 1]
    _replace(cmd, "--kv-transfer-config", config.replace("kv_producer", "kv_consumer"))
    return cmd

def _configure() -> None:
    gpu_pd_venv = Path(
        os.environ.get("PDD_GPU_PD_VENV", runner.VENV_ROOT / "gpu-pd")
    ).expanduser().resolve()
    runner.CUDA_PREFILL_VENV = gpu_pd_venv
    runner.CUDA_VLLM_BIN = gpu_pd_venv / "bin/vllm"
    runner.RBLN_VLLM_BIN = runner.CUDA_VLLM_BIN
    runner.RBLN_PYTHON_BIN = gpu_pd_venv / "bin/python3"
    runner.RBLN_DECODE_VENV = gpu_pd_venv
    runner.SETUP_SCRIPT = runner.SCRIPT_DIR / "scripts/setup_gpu_pd_env.sh"
    runner.DECODE_KV_CONNECTOR_NAME = "NixlConnector"
    runner.BLOCK_SIZE = int(os.environ.get("PDD_GPU_BLOCK_SIZE", "16"))
    runner._build_decode_cmd = _gpu_decode_cmd

    start = runner.start_process

    def start_on_gpu(name, cmd, env, cwd, log_path):
        env = dict(env or {})
        if name in {"prefill", "decode"}:
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                f"PDD_GPU_{name.upper()}_DEVICE", "0"
            )
        if name == "decode":
            env = {key: value for key, value in env.items() if "RBLN" not in key}
        return start(name, cmd, env, cwd, log_path)

    runner.start_process = start_on_gpu

def main() -> int:
    _configure()
    return runner.main()

if __name__ == "__main__":
    raise SystemExit(main())
