"""Merge a LoRA adapter into bf16 base weights via LLaMA-Factory export. GPU host."""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from .config import load_config


def merge(run_path: str, dry_run: bool = False) -> Path:
    cfg = load_config()
    run = Path(run_path)
    adapter = run / "adapter"
    merged = run / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    export_cfg = {
        "model_name_or_path": cfg.base_model,
        "adapter_name_or_path": str(adapter),
        "template": "qwen3",
        "finetuning_type": "lora",
        "export_dir": str(merged),
        "export_size": 5,
        "export_legacy_format": False,
    }
    cfg_path = run / "merge_config.yaml"
    cfg_path.write_text(yaml.safe_dump(export_cfg, sort_keys=False))
    cmd = ["llamafactory-cli", "export", str(cfg_path)]
    print(f"[merge] cmd: {' '.join(cmd)}")
    if dry_run:
        print("[merge] --dry-run: config written, not executing.")
        return merged
    subprocess.run(cmd, check=True)
    print(f"[merge] merged bf16 -> {merged}")
    return merged
