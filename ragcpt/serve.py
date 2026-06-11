"""Serve a model as an OpenAI-compatible vLLM endpoint. GPU host.

Prints the command and the base_url to drop into .env (TUNED_BASE_URL / BASE_BASE_URL) so eval +
chat can reach it. Use --base to serve the untuned Qwen3-14B for the baseline.
"""
from __future__ import annotations

import subprocess

from .config import load_config


def serve(model_path: str | None = None, port: int = 8000, base: bool = False,
          dry_run: bool = False) -> None:
    cfg = load_config()
    model = model_path or (cfg.base_model if base else None)
    if not model:
        raise ValueError("Pass model_path (the run's merged/ dir) or --base for the untuned model.")
    which = "base" if base else "tuned"
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--served-model-name", cfg.base_model,
    ]
    url = f"http://0.0.0.0:{port}/v1"
    print(f"[serve:{which}] cmd: {' '.join(cmd)}")
    print(f"[serve:{which}] once up, set in .env:  "
          f"{'BASE_BASE_URL' if base else 'TUNED_BASE_URL'}=http://<this-host>:{port}/v1")
    if dry_run:
        print("[serve] --dry-run: not executing.")
        return
    subprocess.run(cmd, check=True)
