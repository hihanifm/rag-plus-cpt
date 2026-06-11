"""LoRA SFT via LLaMA-Factory. Runs on the GPU host (Lambda H100 / DGX), NOT the Mac.

Generates a run-versioned config from pipeline.yaml + configs/sft_lora.yaml, then calls
`llamafactory-cli train <config>`. Writes everything under models/<run_name>/run_<ts>/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from .config import load_config


def run_dir(run_name: str) -> Path:
    cfg = load_config()
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = cfg.models_dir / run_name / f"run_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def train(run_name: str = "mvp", smoke: bool = False, dry_run: bool = False) -> Path:
    cfg = load_config()
    base_cfg = yaml.safe_load((Path(__file__).parent.parent / "configs" / "sft_lora.yaml").read_text())
    out = run_dir(run_name)

    t = cfg.get("train", default={})
    base_cfg.update({
        "model_name_or_path": cfg.base_model,
        "dataset": "carrier_sft",
        "dataset_dir": str(cfg.data_dir),
        "output_dir": str(out / "adapter"),
        "lora_rank": t.get("lora_rank", 64),
        "lora_alpha": t.get("lora_alpha", 128),
        "lora_target": t.get("lora_target", "all"),
        "cutoff_len": t.get("cutoff_len", 4096),
        "learning_rate": t.get("learning_rate", 7e-5),
        "num_train_epochs": t.get("epochs", 3),
        "per_device_train_batch_size": t.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": t.get("gradient_accumulation_steps", 16),
    })
    if smoke:
        base_cfg["max_samples"] = t.get("smoke_max_samples", 16)
        base_cfg["max_steps"] = t.get("smoke_max_steps", 20)
        base_cfg["num_train_epochs"] = 1

    cfg_path = out / "train_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False))
    _snapshot(cfg, out)

    cmd = ["llamafactory-cli", "train", str(cfg_path)]
    print(f"[train] run dir: {out}\n[train] cmd: {' '.join(cmd)}")
    if dry_run:
        print("[train] --dry-run: config written, not executing.")
        return out
    subprocess.run(cmd, check=True)
    print(f"[train] done. adapter -> {out / 'adapter'}")
    return out


def _snapshot(cfg, out: Path):
    """Record provenance into the run dir."""
    man = cfg.data_dir / "dataset_manifest.json"
    if man.exists():
        shutil.copy(man, out / "dataset_manifest.json")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "no-git"
    (out / "git_commit.txt").write_text(commit + "\n")
    (out / "pipeline_snapshot.json").write_text(json.dumps(cfg.raw, indent=2))
